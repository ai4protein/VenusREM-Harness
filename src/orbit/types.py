from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class RetrievalOutput:
    """Container for retrieval-derived logits and metadata."""

    aa_seq_aln_logits: Optional[torch.Tensor] = None
    aa_aln_start: Optional[int] = None
    aa_aln_end: Optional[int] = None
    struc_seq_aln_logits: Optional[torch.Tensor] = None
    layer2_logits: Optional[torch.Tensor] = None
    layer2_position_weights: Optional[torch.Tensor] = None
    layer2_mask: Optional[torch.Tensor] = None
    layer2_name: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


def merge_retrieval_outputs(primary: RetrievalOutput, secondary: RetrievalOutput) -> RetrievalOutput:
    """Merge second-layer retrieval signal into a primary retrieval output."""
    merged = RetrievalOutput(
        aa_seq_aln_logits=primary.aa_seq_aln_logits,
        aa_aln_start=primary.aa_aln_start,
        aa_aln_end=primary.aa_aln_end,
        struc_seq_aln_logits=primary.struc_seq_aln_logits,
        layer2_logits=secondary.layer2_logits,
        layer2_position_weights=secondary.layer2_position_weights,
        layer2_mask=secondary.layer2_mask,
        layer2_name=secondary.layer2_name,
        metadata={**primary.metadata, **secondary.metadata},
    )
    return merged
