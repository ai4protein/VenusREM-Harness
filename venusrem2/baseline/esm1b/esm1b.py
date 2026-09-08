from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from venusrem2.backbone.forward_utils import (
    force_config_max_residue_len,
    forward_masked_marginal,
    forward_sequence_logits,
    infer_model_max_residue_len,
)

DEFAULT_ESM1B_MODEL = "facebook/esm1b_t33_650M_UR50S"


def load_esm1b_model(model_name=None, device="cuda"):
    if model_name is None:
        model_name = DEFAULT_ESM1B_MODEL
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    max_residue_len = infer_model_max_residue_len(model, tokenizer)
    return model, tokenizer, max_residue_len


@torch.no_grad()
def forward_esm1b(
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


@torch.no_grad()
def forward_esm1b_masked_marginal(
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
    return forward_masked_marginal(
        model=model,
        tokenizer=tokenizer,
        sequence=sequence,
        device=device,
        max_residue_len=max_residue_len,
        long_seq_mode=long_seq_mode,
        long_seq_overlap=long_seq_overlap,
        logger=logger,
        protein_name=protein_name,
    )
