from typing import Optional

import torch

from src.orbit.fusions.base import BaseFusion
from src.orbit.types import RetrievalOutput


class LinearAlphaFusion(BaseFusion):
    """Keep VenusREM's original linear alpha blending behaviour."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def _blend(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.alpha) * x + self.alpha * y

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
            logits = self._blend(logits, retrieval_output.struc_seq_aln_logits.to(logits.device))

        if use_aa:
            aln_start = retrieval_output.aa_aln_start
            aln_end = retrieval_output.aa_aln_end
            if aln_start is None or aln_end is None:
                raise ValueError("aa_aln_start/aa_aln_end are required for aa_seq_aln fusion")
            aa_logits = retrieval_output.aa_seq_aln_logits.to(logits.device)
            blended = self._blend(logits[aln_start:aln_end, :], aa_logits)
            logits = torch.cat([logits[:aln_start], blended, logits[aln_end:]], dim=0)

        return logits
