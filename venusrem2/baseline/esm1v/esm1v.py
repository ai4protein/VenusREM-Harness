import os
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from venus_orbit.backbone.forward_utils import (
    force_config_max_residue_len,
    forward_masked_marginal,
    forward_sequence_logits,
    infer_model_max_residue_len,
)

ESM1V_MODEL_TEMPLATE = "facebook/esm1v_t33_650M_UR90S_{seed}"
_ESM1V_MODEL_CACHE: Dict[Tuple[int, str], Tuple[AutoModelForMaskedLM, AutoTokenizer]] = {}


def _cache_enabled() -> bool:
    return os.environ.get("VENUSREM_ESM1V_CACHE", "1") != "0"


def _load_esm1v_seed(
    seed: int,
    device: torch.device,
    logger=None,
    protein_name: Optional[str] = None,
    masked_marginal: bool = False,
) -> Tuple[AutoModelForMaskedLM, AutoTokenizer]:
    model_name = ESM1V_MODEL_TEMPLATE.format(seed=seed)
    cache_key = (seed, str(device))

    if _cache_enabled() and cache_key in _ESM1V_MODEL_CACHE:
        if logger is not None:
            logger.debug(f"Reusing cached ESM-1v seed {seed}: {model_name}", protein=protein_name)
        return _ESM1V_MODEL_CACHE[cache_key]

    if logger is not None:
        mode = " (masked marginal)" if masked_marginal else ""
        logger.debug(f"Loading ESM-1v seed {seed}{mode}: {model_name}", protein=protein_name)

    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if _cache_enabled():
        _ESM1V_MODEL_CACHE[cache_key] = (model, tokenizer)

    return model, tokenizer


@torch.no_grad()
def forward_esm1v_ensemble(
    sequence: str,
    device: torch.device,
    seeds: List[int] = None,
    max_residue_len: Optional[int] = None,
    long_seq_mode: str = "auto_window",
    long_seq_overlap: int = 256,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    all_logits = []
    for seed in seeds:
        model, tokenizer = _load_esm1v_seed(
            seed=seed,
            device=device,
            logger=logger,
            protein_name=protein_name,
        )

        seed_max_len = max_residue_len
        if seed_max_len is None:
            pad_idx = getattr(model.config, "pad_token_id", 1)
            seed_max_len = model.config.max_position_embeddings - pad_idx - 3

        logits = forward_sequence_logits(
            model=model,
            tokenizer=tokenizer,
            sequence=sequence,
            device=device,
            use_structure=False,
            structure_sequence=None,
            max_residue_len=seed_max_len,
            long_seq_mode=long_seq_mode,
            long_seq_overlap=long_seq_overlap,
            logger=logger,
            protein_name=protein_name,
        )
        all_logits.append(logits.cpu())
        if not _cache_enabled():
            del model
            torch.cuda.empty_cache()

    avg_logits = torch.stack(all_logits).mean(dim=0).to(device)
    return avg_logits


@torch.no_grad()
def forward_esm1v_masked_marginal(
    sequence: str,
    device: torch.device,
    seeds: List[int] = None,
    max_residue_len: Optional[int] = None,
    long_seq_mode: str = "auto_window",
    long_seq_overlap: int = 256,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    all_logits = []
    for seed in seeds:
        model, tokenizer = _load_esm1v_seed(
            seed=seed,
            device=device,
            logger=logger,
            protein_name=protein_name,
            masked_marginal=True,
        )

        seed_max_len = max_residue_len
        if seed_max_len is None:
            pad_idx = getattr(model.config, "pad_token_id", 1)
            seed_max_len = model.config.max_position_embeddings - pad_idx - 3

        logits = forward_masked_marginal(
            model=model,
            tokenizer=tokenizer,
            sequence=sequence,
            device=device,
            max_residue_len=seed_max_len,
            long_seq_mode=long_seq_mode,
            long_seq_overlap=long_seq_overlap,
            logger=logger,
            protein_name=protein_name,
        )
        all_logits.append(logits.cpu())
        if not _cache_enabled():
            del model
            torch.cuda.empty_cache()

    avg_logits = torch.stack(all_logits).mean(dim=0).to(device)
    return avg_logits


def load_esm1v_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(
        ESM1V_MODEL_TEMPLATE.format(seed=1), trust_remote_code=True
    )
