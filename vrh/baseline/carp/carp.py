"""
CARP baseline adapter for VenusREM2.

CARP (Convolutional Autoregressive Protein) is a masked protein language model
based on ByteNet with dilated convolutions. This adapter scores single protein
sequences using masked-marginals, then projects per-residue log-probs to ESM2
vocabulary for VenusREM2 compatibility.

Models are hosted on Zenodo and auto-downloaded via torch.hub on first use.

Reference: Yang et al., "Convolutions are competitive with transformers for
protein sequence modeling", Cell Systems 2024.
"""

from typing import Optional, Tuple

import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

CARP_ZENODO_URL = "https://zenodo.org/record/6564798/files/"

CARP_MODELS = {
    "carp_600k": "carp_600k",
    "carp_38M": "carp_38M",
    "carp_76M": "carp_76M",
    "carp_640M": "carp_640M",
}


def load_carp_model(
    model_name: str,
    device: torch.device,
    cache_dir: Optional[str] = None,
    logger=None,
) -> Tuple:
    """
    Load a CARP model from cache, or prompt to download from Zenodo.

    Args:
        model_name: One of "carp_600k", "carp_38M", "carp_76M", "carp_640M"
        device: torch device

    Returns: (model, collater, esm_tokenizer, protein_alphabet)
    """
    from sequence_models.pretrained import load_carp
    from sequence_models.collaters import SimpleCollater
    from sequence_models.constants import PROTEIN_ALPHABET
    from vrh.models.weights import (
        default_cache_dir,
        ensure_url_file,
        log_cache_hit,
        resolve_existing_weight,
        weight_candidates,
    )

    name = CARP_MODELS.get(model_name, model_name)
    fname = f"{name}.pt"
    existing = resolve_existing_weight("carp", fname, cache_dir=cache_dir)
    hub = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", fname)
    if existing:
        path = log_cache_hit(f"CARP {name}", existing, logger)
    elif os.path.isfile(hub) and os.path.getsize(hub) > 0:
        path = log_cache_hit(f"CARP {name}", hub, logger)
    else:
        dest = os.path.join(default_cache_dir(cache_dir), "carp", fname)
        path = ensure_url_file(
            CARP_ZENODO_URL + f"{fname}?download=1",
            dest,
            logger=logger,
            name=f"CARP {name}",
            looked_in=weight_candidates("carp", fname, cache_dir=cache_dir) + [hub],
        )
    model_data = torch.load(path, map_location="cpu", weights_only=False)
    model = load_carp(model_data)
    model = model.to(device).eval()

    collater = SimpleCollater(PROTEIN_ALPHABET, pad=True)

    esm_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t6_8M_UR50D", trust_remote_code=True
    )

    return model, collater, esm_tokenizer, PROTEIN_ALPHABET


def _build_esm_projection_map(protein_alphabet: str, esm_tokenizer):
    """
    Build mapping from CARP PROTEIN_ALPHABET indices to ESM2 token IDs
    for standard amino acids only.
    """
    esm_vocab = esm_tokenizer.get_vocab()
    carp_idx_to_esm_id = {}
    for aa in STANDARD_AA:
        carp_idx = protein_alphabet.index(aa)
        if aa in esm_vocab:
            carp_idx_to_esm_id[carp_idx] = esm_vocab[aa]
    return carp_idx_to_esm_id


@torch.no_grad()
def forward_carp(
    model,
    collater,
    esm_tokenizer,
    protein_alphabet: str,
    sequence: str,
    device: torch.device,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Score a protein sequence using CARP with masked-marginals.

    For each position i, mask that position and run a forward pass.
    Collect the logits at the masked position. This gives per-position
    marginal probabilities conditioned on the rest of the sequence.

    Returns: torch.Tensor of shape [L, esm_vocab_size] with log-probabilities
             projected to ESM2 vocab.
    """
    L = len(sequence)
    esm_vocab_size = esm_tokenizer.vocab_size
    mask_idx = protein_alphabet.index("#")

    if logger is not None:
        logger.debug(f"CARP forward: seq_len={L}", protein=protein_name)

    input_ids = collater([[sequence]])[0].to(device)  # [1, L]

    all_logits = torch.zeros(L, len(protein_alphabet), device=device)
    for i in range(L):
        masked = input_ids.clone()
        masked[0, i] = mask_idx
        output = model(masked, logits=True)
        all_logits[i] = output["logits"][0, i]

    log_probs = F.log_softmax(all_logits, dim=-1)  # [L, 30]

    carp_to_esm = _build_esm_projection_map(protein_alphabet, esm_tokenizer)
    projected = torch.full((L, esm_vocab_size), -1e9, device=device)
    for carp_idx, esm_id in carp_to_esm.items():
        projected[:, esm_id] = log_probs[:, carp_idx]

    if logger is not None:
        logger.debug(f"CARP done: output shape {projected.shape}", protein=protein_name)

    return projected
