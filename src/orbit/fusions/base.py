from abc import ABC, abstractmethod
from typing import Optional

import torch

from src.orbit.types import RetrievalOutput


class BaseFusion(ABC):
    """Base interface for logits fusion plugins."""

    @abstractmethod
    def fuse(
        self,
        plm_logits: torch.Tensor,
        retrieval_output: RetrievalOutput,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Return fused logits."""
