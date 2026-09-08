"""
ProteinMPNN adapter for the VenusREM2 pipeline.

Extracts per-position log probabilities from ProteinMPNN's teacher-forced
decoder and projects the 21-dim output to the ESM-2 tokenizer vocabulary.

Reference: ProteinGym baselines/protein_mpnn/compute_fitness.py
"""

import os
import torch
import numpy as np

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


@torch.no_grad()
def forward_protein_mpnn(
    sequence,
    pdb_file,
    device,
    esm_tokenizer,
    mpnn_model,
    chain_id="A",
    scoring_mode="teacher_force",
    random_orders=1,
    logger=None,
    protein_name=None,
) -> torch.Tensor:
    """ProteinMPNN scoring → [L, V_esm] logits.

    ``teacher_force`` keeps the historical deterministic decoding order used by
    this repo. ``random_order`` uses ProteinMPNN's official random decoding-order
    path controlled by the model's ``randn`` argument, optionally averaging over
    multiple orders.
    """
    from venusrem2.baseline.protein_mpnn.protein_mpnn_utils import (
        parse_PDB,
        StructureDatasetPDB,
        tied_featurize,
    )

    pdb_dict_list = parse_PDB(pdb_file, input_chain_list=[chain_id])
    dataset = StructureDatasetPDB(
        pdb_dict_list, verbose=False, max_length=100000
    )

    pdb_name = pdb_dict_list[0]["name"]
    chain_id_dict = {pdb_name: ([chain_id], [])}

    batch = [dataset[0]]
    (
        X, S, mask, lengths, chain_M, chain_encoding_all,
        _letter_list_list, _visible_list_list, _masked_list_list,
        _masked_chain_length_list_list, chain_M_pos, _omit_AA_mask,
        residue_idx, _dihedral_mask, _tied_pos,
        _pssm_coef, _pssm_bias, _pssm_log_odds,
        _bias_by_res, _tied_beta,
    ) = tied_featurize(batch, device, chain_id_dict)

    seq_len = len(sequence)
    alphabet_dict = {aa: i for i, aa in enumerate(MPNN_ALPHABET)}
    seq_indices = torch.tensor(
        [alphabet_dict.get(aa, 20) for aa in sequence],
        dtype=torch.long, device=device,
    )
    S[:, :seq_len] = seq_indices

    score_mask = chain_M * chain_M_pos
    if scoring_mode == "teacher_force":
        randn = torch.zeros(chain_M.shape, device=device)
        log_probs = mpnn_model(
            X, S, mask, score_mask,
            residue_idx, chain_encoding_all, randn,
        )
    elif scoring_mode == "random_order":
        n_orders = max(int(random_orders), 1)
        log_prob_list = []
        for _ in range(n_orders):
            randn = torch.randn(chain_M.shape, device=device)
            log_prob_list.append(mpnn_model(
                X, S, mask, score_mask,
                residue_idx, chain_encoding_all, randn,
            ))
        if len(log_prob_list) == 1:
            log_probs = log_prob_list[0]
        else:
            # Average probabilities across decoding orders, then return log-probs.
            log_probs = torch.logsumexp(torch.stack(log_prob_list, dim=0), dim=0) - np.log(len(log_prob_list))
    else:
        raise ValueError(f"Unknown ProteinMPNN scoring_mode: {scoring_mode}")

    pos_log_probs = log_probs[0, :seq_len, :]

    esm_vocab = esm_tokenizer.get_vocab()
    projected = torch.full((seq_len, len(esm_vocab)), -1e9, device=device)

    for aa in AA_LIST:
        mpnn_idx = alphabet_dict[aa]
        if aa in esm_vocab:
            projected[:, esm_vocab[aa]] = pos_log_probs[:, mpnn_idx]

    if logger and protein_name:
        logger.debug(
            f"ProteinMPNN logits projected: [{seq_len}, {len(esm_vocab)}], scoring_mode={scoring_mode}",
            protein=protein_name,
        )
    return projected




def _ensure_mpnn_checkpoint(checkpoint_dir, cache_dir=None, logger=None):
    """Resolve ProteinMPNN v_48_020 from cache, or prompt to download."""
    from venusrem2.models.weights import (
        ensure_url_file,
        log_cache_hit,
        resolve_existing_weight,
        weight_candidates,
    )

    if os.path.isfile(checkpoint_dir):
        return log_cache_hit("ProteinMPNN", checkpoint_dir, logger)
    ckpt_path = os.path.join(checkpoint_dir, "v_48_020.pt")
    if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        return log_cache_hit("ProteinMPNN", ckpt_path, logger)
    existing = resolve_existing_weight("protein_mpnn", "v_48_020.pt", cache_dir=cache_dir)
    if existing:
        return log_cache_hit("ProteinMPNN", existing, logger)
    url = (
        "https://raw.githubusercontent.com/dauparas/ProteinMPNN/"
        "main/vanilla_model_weights/v_48_020.pt"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    return ensure_url_file(
        url,
        ckpt_path,
        logger=logger,
        name="ProteinMPNN v_48_020.pt",
        looked_in=weight_candidates("protein_mpnn", "v_48_020.pt", cache_dir=cache_dir),
    )


def load_protein_mpnn_model(checkpoint_path, device, cache_dir=None, logger=None):
    """Load ProteinMPNN model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file, or a directory
            (reads cache, otherwise asks before downloading v_48_020.pt).
        device: torch device.

    Returns:
        (model, esm_tokenizer)
    """
    from transformers import AutoTokenizer
    from venusrem2.baseline.protein_mpnn.protein_mpnn_utils import ProteinMPNN as ProteinMPNNModel

    if os.path.isdir(checkpoint_path) or not os.path.isfile(checkpoint_path):
        checkpoint_path = _ensure_mpnn_checkpoint(
            checkpoint_path, cache_dir=cache_dir, logger=logger
        )
    else:
        from venusrem2.models.weights import log_cache_hit

        log_cache_hit("ProteinMPNN", checkpoint_path, logger)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ProteinMPNNModel(
        ca_only=False,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,
        k_neighbors=checkpoint.get("num_edges", 48),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.eval().to(device)

    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    return model, esm_tokenizer
