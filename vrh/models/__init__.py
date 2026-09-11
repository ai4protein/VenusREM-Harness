"""Pluggable logit-model adapters."""

from vrh.models.base import ModelAdapter, ModelSpec
from vrh.models.registry import get_model, list_models, register_model
from vrh.models.resolve import apply_model_defaults, resolve_model_name
from vrh.models.scoring_strategy import (
    UnsupportedScoringStrategy,
    normalize_scoring_strategy,
    require_scoring_strategy,
)
from vrh.models.download_policy import DownloadRefused
from vrh.models.weights import default_cache_dir

# Register built-in adapters on import.
from vrh.models import builtins as _builtins  # noqa: F401

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
    "normalize_scoring_strategy",
    "require_scoring_strategy",
]
