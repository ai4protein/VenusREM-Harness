"""
ESM-IF1 inverse folding adapter for the VenusREM2 pipeline.

Extracts per-position logits from teacher-forced decoding and projects
them to the ESM-2 tokenizer vocabulary so the full VenusREM2 recipe
(MSA fusion, CCD, RSA, pLDDT) can run unchanged.

Loads facebookresearch/esm sources in isolation so it does not conflict
with the EvolutionaryScale ``esm`` (ESM3) package.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch

AA_LIST = "ACDEFGHIKLMNPQRSTVWY"


@contextmanager
def _fair_esm_modules(repo_dir: str):
    """Import facebookresearch ``esm`` without clashing with ESM3's ``esm``."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "esm" or k.startswith("esm.")}
    inserted = False
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
        inserted = True
    try:
        yield
    finally:
        # Drop fair-esm modules we just loaded.
        for k in list(sys.modules):
            if k == "esm" or k.startswith("esm."):
                sys.modules.pop(k, None)
        if inserted:
            try:
                sys.path.remove(repo_dir)
            except ValueError:
                pass
        sys.modules.update(saved)


def _patch_biotite():
    import biotite.structure

    if not hasattr(biotite.structure, "filter_backbone"):
        biotite.structure.filter_backbone = biotite.structure.filter_peptide_backbone


@torch.no_grad()
def forward_esm_if(
    sequence,
    pdb_file,
    device,
    esm_tokenizer,
    esm_if_model,
    esm_if_alphabet,
    chain_id="A",
    logger=None,
    protein_name=None,
    load_coords=None,
    CoordBatchConverter=None,
) -> torch.Tensor:
    """Teacher-forced inverse folding scoring → [L, V_esm] logits."""
    _patch_biotite()
    if load_coords is None or CoordBatchConverter is None:
        raise RuntimeError(
            "ESM-IF helpers missing; reload the model via load_esm_if_model()."
        )

    coords, _native_seq = load_coords(pdb_file, chain_id)

    batch_converter = CoordBatchConverter(esm_if_alphabet)
    batch = [(coords, None, sequence)]
    coords_batch, confidence, _strs, tokens, padding_mask = batch_converter(
        batch, device=device
    )

    prev_output_tokens = tokens[:, :-1].to(device)
    logits, _ = esm_if_model.forward(
        coords_batch, padding_mask, confidence, prev_output_tokens
    )

    seq_len = len(sequence)
    # logits shape: [B, V_if, L] (PyTorch cross-entropy convention)
    pos_logits = logits[0, :, :seq_len].T  # → [L, V_if]

    esm_vocab = esm_tokenizer.get_vocab()
    projected = torch.full((seq_len, len(esm_vocab)), -1e9, device=device)

    for aa in AA_LIST:
        if_idx = esm_if_alphabet.get_idx(aa)
        if aa in esm_vocab:
            projected[:, esm_vocab[aa]] = pos_logits[:, if_idx]

    if logger and protein_name:
        logger.debug(
            f"ESM-IF logits projected: [{seq_len}, {len(esm_vocab)}]",
            protein=protein_name,
        )
    return projected


def load_esm_if_model(
    device,
    cache_dir: Optional[str] = None,
    logger=None,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Load ESM-IF1 and ESM-2 tokenizer; returns (model, alphabet, tokenizer, helpers)."""
    from transformers import AutoTokenizer

    from venusrem2.models.weights import ensure_esm_if_checkpoint, ensure_fair_esm_source

    _patch_biotite()
    repo_dir = ensure_fair_esm_source(cache_dir=cache_dir, logger=logger)
    ckpt = ensure_esm_if_checkpoint(cache_dir=cache_dir, logger=logger)

    with _fair_esm_modules(repo_dir):
        import esm  # noqa: F401  # facebookresearch esm
        from esm.inverse_folding.util import CoordBatchConverter, load_coords
        from esm.pretrained import load_model_and_alphabet_local

        # fair-esm checkpoints pickle argparse.Namespace; PyTorch>=2.6 defaults
        # weights_only=True and rejects that global.
        _orig_load = torch.load

        def _torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _torch_load  # type: ignore[assignment]
        try:
            model, alphabet = load_model_and_alphabet_local(ckpt)
        finally:
            torch.load = _orig_load  # type: ignore[assignment]

        helpers = {
            "CoordBatchConverter": CoordBatchConverter,
            "load_coords": load_coords,
            "fair_esm_repo": repo_dir,
        }

    model = model.eval().to(device)
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    return model, alphabet, esm_tokenizer, helpers
