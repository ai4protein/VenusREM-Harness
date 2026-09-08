"""
MIF-ST baseline adapter for VenusREM2.

Masked Inverse Folding with Sequence Transfer (Yang et al.): structure-conditioned
MLM that transfers sequence features from CARP-640M into a GNN decoder.

Weights: Zenodo 6573779 (`mifst.pt`) + CARP-640M (Zenodo 6564798), auto-downloaded
via ``sequence_models`` / torch.hub.

Scoring uses WT-marginals (single forward; matches ProteinGym carp_mif's
effective behavior) and projects PROTEIN_ALPHABET log-probs onto the ESM-2
vocab so MSA/CCD/RSA/pLDDT heads apply unchanged.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

MIF_ZENODO_URL = "https://zenodo.org/record/6573779/files/"
CARP_ZENODO_URL = "https://zenodo.org/record/6564798/files/"


def ensure_mifst_checkpoint(cache_dir: Optional[str] = None, logger=None) -> str:
    """Return path to ``mifst.pt`` from cache, or prompt to download."""
    from venusrem2.models.weights import (
        default_cache_dir,
        ensure_dir,
        ensure_url_file,
        log_cache_hit,
        resolve_existing_weight,
        weight_candidates,
    )

    existing = resolve_existing_weight("mifst", "mifst.pt", cache_dir=cache_dir)
    if existing:
        return log_cache_hit("MIF-ST", existing, logger)
    cache = ensure_dir(os.path.join(default_cache_dir(cache_dir), "mifst"))
    dest = os.path.join(cache, "mifst.pt")
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return log_cache_hit("MIF-ST", dest, logger)

    hub = os.path.join(
        os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "mifst.pt"
    )
    if os.path.isfile(hub) and os.path.getsize(hub) > 0:
        import shutil

        shutil.copy2(hub, dest)
        return log_cache_hit("MIF-ST", dest, logger)

    return ensure_url_file(
        MIF_ZENODO_URL + "mifst.pt?download=1",
        dest,
        logger=logger,
        name="MIF-ST mifst.pt",
        looked_in=weight_candidates("mifst", "mifst.pt", cache_dir=cache_dir) + [hub],
    )


def load_mifst_model(
    device: torch.device,
    cache_dir: Optional[str] = None,
    logger=None,
) -> Tuple:
    """
    Load MIF-ST (+ CARP-640M encoder) and an ESM-2 tokenizer for vocab projection.

    Returns: (model, structure_collater, esm_tokenizer, protein_alphabet)
    """
    from sequence_models.constants import PROTEIN_ALPHABET
    from sequence_models.pretrained import load_model_and_alphabet

    ckpt = ensure_mifst_checkpoint(cache_dir=cache_dir, logger=logger)
    model, collater = load_model_and_alphabet(ckpt)
    model = model.to(device).eval()

    esm_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t6_8M_UR50D", trust_remote_code=True
    )
    return model, collater, esm_tokenizer, PROTEIN_ALPHABET


def _build_esm_projection_map(protein_alphabet: str, esm_tokenizer):
    esm_vocab = esm_tokenizer.get_vocab()
    idx_to_esm = {}
    for aa in STANDARD_AA:
        src_idx = protein_alphabet.index(aa)
        if aa in esm_vocab:
            idx_to_esm[src_idx] = esm_vocab[aa]
    return idx_to_esm


def _structure_batch(sequence: str, pdb_file: str, collater, device: torch.device):
    from sequence_models.pdb_utils import parse_PDB, process_coords

    coords, _wt, _ = parse_PDB(pdb_file)
    coords = {
        "N": coords[:, 0],
        "CA": coords[:, 1],
        "C": coords[:, 2],
    }
    dist, omega, theta, phi = process_coords(coords)
    batch = [[
        sequence,
        torch.tensor(dist, dtype=torch.float),
        torch.tensor(omega, dtype=torch.float),
        torch.tensor(theta, dtype=torch.float),
        torch.tensor(phi, dtype=torch.float),
    ]]
    input_ids, nodes, edges, connections, edge_mask = collater(batch)
    return (
        input_ids.to(device),
        nodes.to(device),
        edges.to(device),
        connections.to(device),
        edge_mask.to(device),
    )


@torch.no_grad()
def forward_mifst(
    sequence: str,
    pdb_file: str,
    device: torch.device,
    model,
    collater,
    esm_tokenizer,
    protein_alphabet: str,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    """
    WT-marginals scoring for MIF-ST → ``[L, V_esm]`` log-probs.

    ProteinGym's ``carp_mif/compute_fitness.py`` labels the loop as
    ``masked_marginals`` but passes the *unmasked* ids into MIF, i.e. a
    single teacher-forced / WT forward. We match that effective behavior
    (one forward) so scores stay comparable and runtime stays practical.
    """
    if not pdb_file or not os.path.isfile(pdb_file):
        raise FileNotFoundError(f"MIF-ST requires a PDB file; got {pdb_file!r}")

    L = len(sequence)
    esm_vocab_size = esm_tokenizer.vocab_size

    if logger is not None:
        logger.debug(f"MIF-ST forward: seq_len={L} pdb={pdb_file}", protein=protein_name)

    input_ids, nodes, edges, connections, edge_mask = _structure_batch(
        sequence, pdb_file, collater, device
    )
    struct_len = int(input_ids.size(1))
    if struct_len != L:
        raise ValueError(
            f"MIF-ST length mismatch for {protein_name}: seq_len={L} pdb_len={struct_len} "
            f"(pdb={pdb_file})"
        )

    logits = model(input_ids, nodes, edges, connections, edge_mask, result="logits")
    # logits: [1, L, alphabet]
    log_probs = F.log_softmax(logits[0], dim=-1)

    proj = _build_esm_projection_map(protein_alphabet, esm_tokenizer)
    projected = torch.full((L, esm_vocab_size), -1e9, device=device)
    for src_idx, esm_id in proj.items():
        projected[:, esm_id] = log_probs[:, src_idx]

    if logger is not None:
        logger.debug(f"MIF-ST done: output shape {projected.shape}", protein=protein_name)

    return projected
