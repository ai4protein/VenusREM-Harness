"""Pluggable logit-model adapters."""

from venusrem2.models.base import ModelAdapter, ModelSpec
from venusrem2.models.registry import get_model, list_models, register_model
from venusrem2.models.resolve import apply_model_defaults, resolve_model_name
from venusrem2.models.scoring_strategy import (
    UnsupportedScoringStrategy,
    require_scoring_strategy,
)
from venusrem2.models.download_policy import DownloadRefused
from venusrem2.models.weights import default_cache_dir

# Register built-in adapters on import.
from venusrem2.models import builtins as _builtins  # noqa: F401

__all__ = [
    "ModelAdapter",
    "ModelSpec",
    "register_model",
    "get_model",
    "list_models",
    "resolve_model_name",
    "apply_model_defaults",
    "default_cache_dir",
    "DownloadRefused",
    "UnsupportedScoringStrategy",
    "require_scoring_strategy",
]
