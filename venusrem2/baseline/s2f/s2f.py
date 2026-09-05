"""
S2F / S3F baseline adapter for VenusREM-Orbit.

S2F (Sequence-to-Function) uses ESM-2 fine-tuned with a classification head
predicting 20 amino acid types. The scoring is delta-log-likelihood:
  log P(mutant AA) - log P(WT AA) per position.

S3F extends S2F with a GVP structure encoder on top of ESM-2 features.

Both require TorchDrug and PyTorch Geometric (heavy dependencies).
When TorchDrug is unavailable, the adapter falls back to a lightweight
ESM-2-only mode that reproduces the S2F scoring from raw ESM-2 logits
(without the fine-tuned classification head -- requires a checkpoint
trained through the TorchDrug pipeline for full accuracy).

Reference:
  - S3F: https://github.com/DeepGraphLearning/S3F
  - ProteinGym S3F baseline
"""

import os
import pickle
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# -- TorchDrug residue ordering used by S2F/S3F model outputs --
STANDARD_AAS = [
    "G", "A", "S", "P", "V", "T", "C", "I", "L", "N",
    "D", "Q", "K", "E", "M", "H", "F", "R", "Y", "W",
]

# Check for torchdrug availability
try:
    import torchdrug
    from torchdrug import core, data, utils
    HAS_TORCHDRUG = True
except ImportError:
    HAS_TORCHDRUG = False


def _build_aa20_to_esm_vocab_projection(esm_tokenizer) -> torch.Tensor:
    """
    Build a projection matrix from 20 standard AAs to ESM-2 vocab.

    The S2F/S3F model produces logits of shape [L, 20] corresponding to
    the 20 standard amino acids (in residue_constants order). To make
    these compatible with the VenusREM scoring pipeline (which expects
    [L, esm_vocab_size] log-probs), we project: for each of the 20 AAs,
    we place its log-prob at the corresponding ESM token index, and fill
    all other positions with a large negative value.

    Returns:
        mapping: LongTensor [20] of ESM vocab indices for each standard AA.
    """
    vocab = esm_tokenizer.get_vocab()
    indices = []
    for aa in STANDARD_AAS:
        if aa in vocab:
            indices.append(vocab[aa])
        else:
            raise ValueError(
                f"Amino acid '{aa}' not found in ESM tokenizer vocabulary. "
                f"Available tokens: {sorted(vocab.keys())}"
            )
    return torch.tensor(indices, dtype=torch.long)


def _project_aa20_to_esm_vocab(
    logits_20: torch.Tensor,
    aa_to_esm_indices: torch.Tensor,
    esm_vocab_size: int,
) -> torch.Tensor:
    """
    Project [L, 20] log-probs to [L, esm_vocab_size] log-probs.

    Non-AA positions are filled with -inf (effectively zero probability).
    """
    L = logits_20.shape[0]
    device = logits_20.device
    full_logits = torch.full(
        (L, esm_vocab_size), float("-inf"), device=device, dtype=logits_20.dtype
    )
    aa_to_esm_indices = aa_to_esm_indices.to(device)
    full_logits[:, aa_to_esm_indices] = logits_20
    return full_logits


def load_s2f_model(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
    surface_path: Optional[str] = None,
    structure_path: Optional[str] = None,
) -> Tuple:
    """
    Load S2F/S3F model from config and checkpoint.

    This function requires TorchDrug to be installed. The config file
    should be one of the YAML files from the S3F config/evaluate/ directory.

    Args:
        config_path: Path to the S2F/S3F YAML config file.
        checkpoint_path: Path to the model checkpoint (.pt file).
        device: Device to load the model onto.
        esm_model_name: HuggingFace model name for ESM-2 tokenizer
            (used for vocab projection only).
        surface_path: Directory of precomputed surface ``.pkl`` files (S3F).
        structure_path: Directory of PDB structures (optional metadata).

    Returns:
        Tuple of (task, esm_tokenizer, config_dict) where:
        - task: The TorchDrug ResidueTypePrediction task with loaded weights.
        - esm_tokenizer: HuggingFace ESM-2 tokenizer for vocab mapping.
        - config_dict: The parsed config dictionary.
    """
    if not HAS_TORCHDRUG:
        raise ImportError(
            "TorchDrug is required for S2F/S3F model loading. "
            "Install it with: pip install torchdrug\n"
            "Additional dependencies: torch-geometric, torch-scatter, torch-cluster\n"
            "For S3F (structure variant), also: pykeops, robust-laplacian\n"
            "Note: official torchdrug wheels support Python <3.11 only; on 3.11+ "
            "install from source with --ignore-requires-python --no-deps and "
            "rdkit==2023.9.6 (see README)."
        )

    from transformers import AutoTokenizer

    # Import model/task registrations so torchdrug can instantiate them
    from venus_orbit.baseline.s2f import model as _model_reg  # noqa: F401
    from venus_orbit.baseline.s2f import task as _task_reg  # noqa: F401
    from venus_orbit.baseline.s2f import gvp as _gvp_reg  # noqa: F401
    from venus_orbit.baseline.esm_if.esm_if import _fair_esm_modules
    from venus_orbit.models.weights import ensure_fair_esm_source

    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Bundled YAML uses {{ surfdir }} / {{ structdir }} placeholders.
    def _resolve_path(value, override):
        if override:
            return os.path.expanduser(override)
        if value is None:
            return None
        text = str(value)
        if "{{" in text:
            return None
        return os.path.expanduser(text)

    cfg["surface_path"] = _resolve_path(cfg.get("surface_path"), surface_path)
    cfg["structure_path"] = _resolve_path(cfg.get("structure_path"), structure_path)

    # TorchDrug MyESM imports facebookresearch ``esm.pretrained`` at module level;
    # isolate from EvolutionaryScale ESM3 and rebind torchdrug.models.esm.esm.
    fair_esm_repo = ensure_fair_esm_source()
    task_cfg = cfg["task"]
    _orig_load = torch.load

    def _torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _torch_load  # type: ignore[assignment]
    try:
        with _fair_esm_modules(fair_esm_repo):
            import esm as fair_esm  # facebookresearch esm
            import torchdrug.models.esm as td_esm_mod

            _prev_td_esm = getattr(td_esm_mod, "esm", None)
            td_esm_mod.esm = fair_esm
            try:
                task_obj = core.Configurable.load_config_dict(task_cfg)
                task_obj.preprocess(None, None, None)

                checkpoint_path = os.path.expanduser(checkpoint_path)
                if os.path.exists(checkpoint_path):
                    model_dict = torch.load(checkpoint_path, map_location="cpu")
                    if "model" in model_dict:
                        model_dict = model_dict["model"]
                    task_obj.load_state_dict(model_dict)
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            finally:
                if _prev_td_esm is not None:
                    td_esm_mod.esm = _prev_td_esm
    finally:
        torch.load = _orig_load  # type: ignore[assignment]

    task_obj = task_obj.to(device)
    task_obj.eval()

    esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)

    return task_obj, esm_tokenizer, cfg


def load_s2f_model_lightweight(
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
    device: str = "cuda",
):
    """
    Load a lightweight S2F-equivalent model using only ESM-2.

    This does NOT require TorchDrug. It uses raw ESM-2 as the backbone
    and produces per-position amino acid log-probabilities directly from
    ESM-2's masked language model head. This reproduces the S2F scoring
    approach but WITHOUT the fine-tuned classification head weights.

    For the full S2F model with trained weights, use load_s2f_model()
    with the TorchDrug pipeline.

    Args:
        esm_model_name: HuggingFace model name/path for ESM-2.
        device: Device to load the model onto.

    Returns:
        Tuple of (model, tokenizer, max_residue_len) compatible with
        the standard baseline adapter interface.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from venus_orbit.backbone.forward_utils import infer_model_max_residue_len

    model = AutoModelForMaskedLM.from_pretrained(esm_model_name, trust_remote_code=True)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(esm_model_name, trust_remote_code=True)
    max_residue_len = infer_model_max_residue_len(model, tokenizer)

    return model, tokenizer, max_residue_len



def _s3f_official_window(mutation_position_relative: int, seq_len: int, model_window: int = 1022):
    """ProteinGym / Tranception-style site-centered window."""
    half_model_window = model_window // 2
    if seq_len <= model_window:
        return 0, seq_len
    if mutation_position_relative < half_model_window:
        return 0, model_window
    if mutation_position_relative >= seq_len - half_model_window:
        return seq_len - model_window, seq_len
    return (
        max(0, mutation_position_relative - half_model_window),
        min(seq_len, mutation_position_relative + half_model_window),
    )


def _s3f_slice_sequence_graph(protein, start: int, end: int):
    """Clone protein to residue window [start, end) and set graph start/end."""
    window = protein.clone()
    if end - start < window.num_residue:
        residue_mask = torch.zeros(window.num_residue, dtype=torch.bool)
        residue_mask[start:end] = True
        window = window.subresidue(residue_mask)
    with window.graph():
        window.start = torch.as_tensor(start)
        window.end = torch.as_tensor(end)
    return window


def _s3f_load_wild_type_and_surface(
    pdb_file,
    config,
    logger=None,
    protein_name=None,
):
    """Load full-structure WT graph + optional surface (start=0, end=L)."""
    from venus_orbit.baseline.s2f import dataset as s2f_dataset

    if pdb_file is None or not os.path.exists(pdb_file):
        return None, None

    try:
        wild_type_loaded = s2f_dataset.bio_load_pdb(pdb_file)[0]
        ca_index = wild_type_loaded.atom_name == wild_type_loaded.atom_name2id["CA"]
        wild_type = wild_type_loaded.subgraph(ca_index)
        wild_type.view = "residue"
        with wild_type.graph():
            wild_type.start = torch.as_tensor(0)
            wild_type.end = torch.as_tensor(wild_type.num_residue)

        surf_graph = None
        surface_path = config.get("surface_path") if config else None
        if surface_path is not None:
            pdb_basename = os.path.splitext(os.path.basename(pdb_file))[0]
            surf_pkl = os.path.join(
                os.path.expanduser(surface_path), pdb_basename + ".pkl"
            )
            if os.path.exists(surf_pkl):
                with open(surf_pkl, "rb") as fin:
                    surf_dict = pickle.load(fin)
                surf_graph = s2f_dataset.load_surface(surf_dict)
                res2surf = torch.as_tensor(surf_dict["res2surf"])
                with wild_type.residue():
                    wild_type.res2surf = res2surf
            else:
                msg = f"S3F: surface pkl not found: {surf_pkl}"
                if logger is not None:
                    logger.warn(msg, protein=protein_name)
                else:
                    warnings.warn(msg)
        return wild_type, surf_graph
    except Exception as e:
        msg = f"S2F/S3F: Failed to load structure from {pdb_file}: {e}"
        if logger is not None:
            logger.warn(msg, protein=protein_name)
        else:
            warnings.warn(msg)
        return None, None


def _s3f_infer_sequence_graphs(
    task,
    sequence_graphs,
    wild_type,
    surf_graph,
    device_obj,
    batch_size: int = 8,
):
    """Run S3F inference over a list of sequence graphs; return list of [L_i, 20] preds."""
    from venus_orbit.baseline.s2f import dataset as s2f_dataset

    if wild_type is not None:
        _dataset = s2f_dataset.MutantDataset(sequence_graphs, wild_type, surf_graph=surf_graph)
    else:
        _dataset = [{"graph": g} for g in sequence_graphs]

    dataloader = data.DataLoader(_dataset, batch_size, shuffle=False, num_workers=0)
    task.eval()
    preds = []
    for batch in dataloader:
        batch = utils.cuda(batch, device=device_obj)
        with torch.no_grad():
            pred, sizes = task.inference(batch)
        cum_sizes = sizes.cumsum(dim=0)
        for k in range(len(sizes)):
            batch_start = (cum_sizes[k] - sizes[k]).item()
            batch_end = cum_sizes[k].item()
            preds.append(pred[batch_start:batch_end])
    return preds


@torch.no_grad()
def forward_s2f(
    task,
    esm_tokenizer,
    sequence: str,
    device: torch.device,
    pdb_file: Optional[str] = None,
    config: Optional[dict] = None,
    logger=None,
    protein_name: Optional[str] = None,
    long_seq_overlap: int = 256,
) -> torch.Tensor:
    """
    Score a single protein sequence using S2F/S3F (full TorchDrug pipeline).

    wt-marginals style: one forward over the WT sequence (no masking).
    For L > 1022, use overlapping sliding windows (same idea as ESM2 wt)
    and average logits in the overlap.

    Returns:
        Tensor of shape [L, esm_vocab_size] with log-probabilities.
    """
    if not HAS_TORCHDRUG:
        raise ImportError(
            "TorchDrug is required for forward_s2f. "
            "Use forward_s2f_lightweight for a dependency-free alternative."
        )

    seq_len = len(sequence)
    esm_vocab_size = esm_tokenizer.vocab_size
    aa_to_esm = _build_aa20_to_esm_vocab_projection(esm_tokenizer)
    device_obj = torch.device(device) if isinstance(device, str) else device
    max_len = 1022
    if config is not None:
        max_len = int(config.get("max_residue_len", max_len))
        long_seq_overlap = int(config.get("long_seq_overlap", long_seq_overlap))

    protein = data.Protein.from_sequence(
        sequence, atom_feature=None, bond_feature=None
    )
    protein.view = "residue"

    wild_type, surf_graph = _s3f_load_wild_type_and_surface(
        pdb_file, config, logger=logger, protein_name=protein_name
    )
    batch_size = 1
    if config is not None:
        batch_size = int(config.get("batch_size", 1))

    if seq_len <= max_len:
        window = _s3f_slice_sequence_graph(protein, 0, seq_len)
        preds = _s3f_infer_sequence_graphs(
            task, [window], wild_type, surf_graph, device_obj, batch_size=batch_size
        )
        logits_20 = preds[0]
    else:
        overlap = max(0, int(long_seq_overlap))
        if overlap >= max_len:
            overlap = max_len // 2
        step = max(1, max_len - overlap)
        msg = (
            f"S3F wt-marginals: L={seq_len} > {max_len}; "
            f"sliding-window (window={max_len}, overlap={overlap})"
        )
        if logger is not None:
            logger.warn(msg, protein=protein_name)
        else:
            print(f">>> {msg}")

        aggregated = None
        counts = None
        start = 0
        while start < seq_len:
            end = min(seq_len, start + max_len)
            window = _s3f_slice_sequence_graph(protein, start, end)
            preds = _s3f_infer_sequence_graphs(
                task, [window], wild_type, surf_graph, device_obj, batch_size=1
            )
            window_logits = preds[0]
            window_len = min(window_logits.size(0), end - start)
            window_logits = window_logits[:window_len]
            if aggregated is None:
                aggregated = torch.zeros(
                    seq_len, window_logits.size(-1),
                    device=window_logits.device, dtype=window_logits.dtype,
                )
                counts = torch.zeros(
                    seq_len, 1, device=window_logits.device, dtype=window_logits.dtype,
                )
            aggregated[start:start + window_len] += window_logits
            counts[start:start + window_len] += 1.0
            if end >= seq_len:
                break
            start += step
        logits_20 = aggregated / counts.clamp_min(1.0)

    log_probs_20 = F.log_softmax(logits_20, dim=-1)
    return _project_aa20_to_esm_vocab(log_probs_20, aa_to_esm, esm_vocab_size)


@torch.no_grad()
def forward_s3f_masked_marginal(
    task,
    esm_tokenizer,
    sequence: str,
    device: torch.device,
    pdb_file: Optional[str] = None,
    config: Optional[dict] = None,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    """
    S3F masked-marginals: mask each position independently, collect [L, V] logits.

    For L > 1022, use per-position site-centered 1022 windows (same geometry as
    ProteinGym S3F official native / ESM2 masked long-seq), then stitch to full L.
    Positions that share a window are batched together.
    """
    if not HAS_TORCHDRUG:
        raise ImportError("TorchDrug is required for forward_s3f_masked_marginal.")

    seq_len = len(sequence)
    esm_vocab_size = esm_tokenizer.vocab_size
    aa_to_esm = _build_aa20_to_esm_vocab_projection(esm_tokenizer)
    device_obj = torch.device(device) if isinstance(device, str) else device
    max_len = 1022
    if config is not None:
        max_len = int(config.get("max_residue_len", max_len))

    protein = data.Protein.from_sequence(
        sequence, atom_feature=None, bond_feature=None
    )
    protein.view = "residue"

    wild_type, surf_graph = _s3f_load_wild_type_and_surface(
        pdb_file, config, logger=logger, protein_name=protein_name
    )
    if wild_type is None:
        raise RuntimeError(
            f"S3F masked-marginals requires PDB structure but loading failed "
            f"for {protein_name}"
        )

    mask_id = task.model.sequence_model.alphabet.get_idx("<mask>")
    batch_size = 8
    if config is not None:
        batch_size = int(config.get("batch_size", 8))

    logits_20 = torch.zeros(seq_len, 20, device=device_obj)

    # Group positions by their scoring window for efficient batching.
    from collections import OrderedDict
    window_to_positions = OrderedDict()
    for pos in range(seq_len):
        if seq_len <= max_len:
            start, end = 0, seq_len
        else:
            start, end = _s3f_official_window(pos, seq_len, max_len)
        window_to_positions.setdefault((start, end), []).append(pos)

    if seq_len > max_len:
        msg = (
            f"S3F mask: L={seq_len} > {max_len}; "
            f"per-position windowed masking ({len(window_to_positions)} unique windows)"
        )
        if logger is not None:
            logger.warn(msg, protein=protein_name)
        else:
            print(f">>> {msg}")

    for (start, end), positions in window_to_positions.items():
        base = _s3f_slice_sequence_graph(protein, start, end)
        masked_sequences = []
        local_indices = []
        for pos in positions:
            local = pos - start
            masked_seq = base.clone()
            with masked_seq.residue():
                masked_seq.residue_feature[local] = 0
                masked_seq.residue_type[local] = mask_id
            masked_sequences.append(masked_seq)
            local_indices.append(local)

        preds = _s3f_infer_sequence_graphs(
            task, masked_sequences, wild_type, surf_graph, device_obj,
            batch_size=batch_size,
        )
        for pos, local, pred in zip(positions, local_indices, preds):
            logits_20[pos] = pred[local]

    log_probs_20 = F.log_softmax(logits_20, dim=-1)
    return _project_aa20_to_esm_vocab(log_probs_20, aa_to_esm, esm_vocab_size)


def _s3f_parse_mutant(mutant: str):
    sites = []
    muts = []
    for sub in str(mutant).split(":"):
        sites.append(int(sub[1:-1]) - 1)
        muts.append(sub)
    return tuple(sites), muts


@torch.no_grad()
def score_s3f_official_native(
    task,
    sequence: str,
    mutant_df,
    device: torch.device,
    pdb_file: Optional[str] = None,
    config: Optional[dict] = None,
    logger=None,
    protein_name: Optional[str] = None,
):
    """ProteinGym S3F native per-mutant scoring.

    This mirrors ProteinGym's S3F baseline: build one masked sequence per
    unique mutation-site tuple, choose the 1022-aa window around that site tuple,
    run S3F, and compute sum(log P(mutant) - log P(wild type)).
    """
    if not HAS_TORCHDRUG:
        raise ImportError("TorchDrug is required for score_s3f_official_native.")
    if pdb_file is None or not os.path.exists(pdb_file):
        raise RuntimeError(f"S3F native scoring requires PDB structure for {protein_name}")

    from venus_orbit.baseline.s2f import dataset as s2f_dataset

    device_obj = torch.device(device) if isinstance(device, str) else device
    seq_len = len(sequence)
    protein = data.Protein.from_sequence(sequence, atom_feature=None, bond_feature=None)
    protein.view = "residue"

    wild_type_loaded = s2f_dataset.bio_load_pdb(pdb_file)[0]
    ca_index = wild_type_loaded.atom_name == wild_type_loaded.atom_name2id["CA"]
    wild_type = wild_type_loaded.subgraph(ca_index)
    wild_type.view = "residue"
    with wild_type.graph():
        wild_type.start = torch.as_tensor(0)
        wild_type.end = torch.as_tensor(wild_type.num_residue)

    surf_graph = None
    surface_path = config.get("surface_path") if config else None
    if surface_path is not None:
        pdb_basename = os.path.splitext(os.path.basename(pdb_file))[0]
        surf_pkl = os.path.join(os.path.expanduser(surface_path), pdb_basename + ".pkl")
        if os.path.exists(surf_pkl):
            with open(surf_pkl, "rb") as fin:
                surf_dict = pickle.load(fin)
            surf_graph = s2f_dataset.load_surface(surf_dict)
            res2surf = torch.as_tensor(surf_dict["res2surf"])
            with wild_type.residue():
                wild_type.res2surf = res2surf
        elif logger is not None:
            logger.warn(f"S3F native: surface pkl not found: {surf_pkl}", protein=protein_name)

    parsed = []
    for row_idx, mutant in enumerate(mutant_df["mutant"].tolist()):
        sites, muts = _s3f_parse_mutant(mutant)
        parsed.append((sites, muts, row_idx))

    unique_sites = sorted({sites for sites, _muts, _idx in parsed})
    mask_id = task.model.sequence_model.alphabet.get_idx("<mask>")
    masked_sequences = []
    offsets = []
    for sites in unique_sites:
        masked_seq = protein.clone()
        if protein_name == "POLG_HCVJF_Qi_2014":
            start, end = 1981, 2225
        elif protein_name == "A0A140D2T1_ZIKV_Sourisseau_2019":
            start, end = 290, 794
        elif protein_name == "B2L11_HUMAN_Dutta_2010_binding-Mcl-1":
            start, end = 119, 197
        elif masked_seq.num_residue > 1022:
            start, end = _s3f_official_window(sites[0], masked_seq.num_residue, 1022)
        else:
            start, end = 0, masked_seq.num_residue
        if protein_name == "BRCA2_HUMAN_Erwood_2022_HEK293T" and end > 2832:
            start, end = 1820, 2832

        node_index = torch.tensor(sites, dtype=torch.long) - start
        if (node_index < 0).any() or (node_index >= end - start).any():
            raise RuntimeError(
                f"S3F native window {start}:{end} does not cover sites {sites} "
                f"for {protein_name}"
            )
        residue_mask = torch.zeros((masked_seq.num_residue,), dtype=torch.bool)
        residue_mask[start:end] = True
        masked_seq = masked_seq.subresidue(residue_mask)
        with masked_seq.graph():
            masked_seq.start = torch.as_tensor(start)
            masked_seq.end = torch.as_tensor(end)
        with masked_seq.residue():
            masked_seq.residue_feature[node_index] = 0
            masked_seq.residue_type[node_index] = mask_id
        masked_sequences.append(masked_seq)
        offsets.append(start)

    batch_size = config.get("batch_size", 8) if config is not None else 8
    dataset_obj = s2f_dataset.MutantDataset(masked_sequences, wild_type, surf_graph=surf_graph)
    dataloader = data.DataLoader(dataset_obj, batch_size, shuffle=False, num_workers=0)
    task.eval()
    seq_prob = []
    for batch in dataloader:
        batch = utils.cuda(batch, device=device_obj)
        prob, sizes = task.inference(batch)
        cum_sizes = sizes.cumsum(dim=0)
        for i in range(len(sizes)):
            seq_prob.append(prob[cum_sizes[i] - sizes[i]:cum_sizes[i]])

    site_to_pred_index = {sites: i for i, sites in enumerate(unique_sites)}
    scores = [float("nan")] * len(parsed)
    for sites, muts, row_idx in parsed:
        pred_idx = site_to_pred_index[sites]
        offset = offsets[pred_idx]
        node_index = torch.tensor(sites, dtype=torch.long, device=seq_prob[pred_idx].device) - offset
        mt_target = [data.Protein.residue_symbol2id.get(mut[-1], -1) for mut in muts]
        wt_target = [data.Protein.residue_symbol2id.get(mut[0], -1) for mut in muts]
        mt_target = torch.tensor(mt_target, dtype=torch.long, device=seq_prob[pred_idx].device)
        wt_target = torch.tensor(wt_target, dtype=torch.long, device=seq_prob[pred_idx].device)
        log_prob = F.log_softmax(seq_prob[pred_idx], dim=-1)
        score = (log_prob[node_index, mt_target] - log_prob[node_index, wt_target]).sum()
        scores[row_idx] = float(score.detach().cpu())

    return scores


@torch.no_grad()
def forward_s2f_lightweight(
    model,
    tokenizer,
    sequence: str,
    device: torch.device,
    max_residue_len: Optional[int] = None,
    long_seq_mode: str = "auto_window",
    long_seq_overlap: int = 256,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Lightweight S2F-equivalent scoring using raw ESM-2 (no TorchDrug).

    This produces per-position log-probabilities using ESM-2's wildtype
    marginals, which approximates the S2F scoring approach. For the full
    S2F model with the trained classification head, use forward_s2f()
    with TorchDrug.

    The forward pass uses single-pass wildtype marginals (not masked
    marginals). The output is [L, esm_vocab_size] log-probs, identical
    to the standard ESM-2 baseline.

    Args:
        model: HuggingFace ESM-2 model.
        tokenizer: HuggingFace ESM-2 tokenizer.
        sequence: Wild-type protein sequence.
        device: Torch device.
        max_residue_len: Maximum residue length for the model.
        long_seq_mode: How to handle long sequences.
        long_seq_overlap: Overlap for sliding window.
        logger: Optional logger.
        protein_name: Optional protein name for logging.

    Returns:
        Tensor of shape [L, esm_vocab_size] with log-probabilities.
    """
    from venus_orbit.backbone.forward_utils import forward_sequence_logits

    return forward_sequence_logits(
        model=model,
        tokenizer=tokenizer,
        sequence=sequence,
        device=device,
        use_structure=False,
        structure_sequence=None,
        max_residue_len=max_residue_len,
        long_seq_mode=long_seq_mode,
        long_seq_overlap=long_seq_overlap,
        logger=logger,
        protein_name=protein_name,
    )
