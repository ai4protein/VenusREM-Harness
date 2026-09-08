"""Model adapter protocol for pluggable logit backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Optional

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a registered backbone."""

    name: str
    description: str = ""
    default_model_id: Optional[str] = None
    needs_pdb: bool = False
    auto_download: bool = True
    extras: str = ""
    notes: str = ""
    aliases: tuple = ()
    # Internal baseline_type key used by legacy dispatch.
    baseline_type: Optional[str] = None
    # True only if --scoring_strategy masked-marginals is implemented.
    supports_mask: bool = False


class _NullLogger:
    def info(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


class ModelAdapter(ABC):
    """Thin wrapper around a logit model used by VenusREM2 scoring.

    Custom backbones should return ``[L, V]`` log-probs from
    :meth:`forward_log_probs` (projected to the ESM vocab when needed).
    VenusREM2 calibration heads are applied on top by ``score_protein``.
    """

    spec: ClassVar[ModelSpec]

    def __init__(self, state: Any, args: Any, device: str):
        self.state = state
        self.args = args
        self.device = device

    @classmethod
    @abstractmethod
    def load(
        cls,
        model_id: Optional[str],
        device: str,
        cache_dir: str,
        args: Any,
        logger: Any,
    ) -> "ModelAdapter":
        raise NotImplementedError

    def create_forward_fn(
        self,
        protein_name: str,
        pdb_file: Optional[str],
        structure_fasta: Optional[str],
        idx: int,
        logger: Any,
    ) -> Optional[Callable]:
        from venusrem2.backbone.baseline_dispatch import create_baseline_forward_fn

        return create_baseline_forward_fn(
            self.state,
            self.args,
            protein_name,
            pdb_file,
            structure_fasta,
            idx,
            self.device,
            logger,
        )

    def forward_log_probs(
        self,
        sequence: str,
        *,
        pdb_path: Optional[str] = None,
        structure_fasta: Optional[str] = None,
        protein_name: str = "query",
        idx: int = 0,
        logger: Any = None,
        **_kw: Any,
    ):
        """Return ``[L, V]`` log-probs for ``sequence``."""
        if logger is None:
            logger = _NullLogger()
        fn = self.create_forward_fn(
            protein_name=protein_name,
            pdb_file=pdb_path,
            structure_fasta=structure_fasta,
            idx=idx,
            logger=logger,
        )
        if fn is None:
            raise RuntimeError(
                f"{self.spec.name}: no forward function (missing PDB/structure inputs?)"
            )
        return fn(sequence)

    def create_native_scorer_fn(
        self,
        protein_name: str,
        logger: Any,
    ) -> Optional[Callable]:
        from venusrem2.backbone.baseline_dispatch import create_native_scorer_fn

        return create_native_scorer_fn(
            self.state,
            self.args,
            protein_name,
            self.device,
            logger,
        )

    def native_score(
        self,
        sequence: str,
        mutants: Any,
        *,
        protein_name: str = "query",
        logger: Any = None,
        **_kw: Any,
    ) -> Optional[Any]:
        """Optional native scorer (Tranception / S3F). Returns ``None`` if unused."""
        if logger is None:
            logger = _NullLogger()
        fn = self.create_native_scorer_fn(protein_name=protein_name, logger=logger)
        if fn is None:
            return None
        return fn(sequence, mutants)
