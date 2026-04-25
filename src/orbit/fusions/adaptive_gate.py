from typing import Optional

import torch

from src.orbit.fusions.base import BaseFusion
from src.orbit.types import RetrievalOutput


class AdaptiveGateFusion(BaseFusion):
    """
    Position-wise gate using retrieval confidence (top-1 probability).
    Keeps downstream scoring unchanged by producing fused logits.
    """

    def __init__(self, alpha: float, min_gate: float = 0.1, max_gate: float = 0.95) -> None:
        self.alpha = alpha
        lo = float(min_gate)
        hi = float(max_gate)
        if lo > hi:
            lo, hi = hi, lo
        self.min_gate = max(0.0, lo)
        self.max_gate = min(1.0, hi)

    def _gate_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.unsqueeze(-1)
        gate = self.alpha * confidence
        return gate.clamp(self.min_gate, self.max_gate)

    def _fuse_block(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gate = self._gate_from_logits(y)
        return (1.0 - gate) * x + gate * y

    def fuse(
        self,
        plm_logits: torch.Tensor,
        retrieval_output: RetrievalOutput,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        logits = plm_logits.clone()
        use_aa = retrieval_output.aa_seq_aln_logits is not None and "aa_seq_aln" in str(mode)
        use_struc = (
            retrieval_output.struc_seq_aln_logits is not None and "struc_seq_aln" in str(mode)
        )

        if not use_aa and not use_struc:
            return logits

        if use_struc:
            struc_logits = retrieval_output.struc_seq_aln_logits.to(logits.device)
            logits = self._fuse_block(logits, struc_logits)

        if use_aa:
            aln_start = retrieval_output.aa_aln_start
            aln_end = retrieval_output.aa_aln_end
            if aln_start is None or aln_end is None:
                raise ValueError("aa_aln_start/aa_aln_end are required for aa_seq_aln fusion")
            aa_logits = retrieval_output.aa_seq_aln_logits.to(logits.device)
            fused_block = self._fuse_block(logits[aln_start:aln_end, :], aa_logits)
            logits = torch.cat([logits[:aln_start], fused_block, logits[aln_end:]], dim=0)

        return logits
