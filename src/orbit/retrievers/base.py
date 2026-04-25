from abc import ABC, abstractmethod
from typing import Optional

from transformers import AutoTokenizer

from src.orbit.types import RetrievalOutput


class BaseRetriever(ABC):
    """Base interface for retrieval plugins in VenusREM-Orbit."""

    @abstractmethod
    def retrieve(
        self,
        tokenizer: AutoTokenizer,
        aa_seq_aln_file: Optional[str] = None,
        struc_seq_aln_file: Optional[str] = None,
        protein_name: Optional[str] = None,
        **kwargs,
    ) -> RetrievalOutput:
        """Build retrieval logits for downstream fusion."""
