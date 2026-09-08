"""Pluggable logit-model adapters."""

from rem2.models.base import ModelAdapter, ModelSpec
from rem2.models.registry import get_model, list_models, register_model
from rem2.models.resolve import apply_model_defaults, resolve_model_name
from rem2.models.scoring_strategy import (
    UnsupportedScoringStrategy,
    normalize_scoring_strategy,
    require_scoring_strategy,
)
from rem2.models.download_policy import DownloadRefused
from rem2.models.weights import default_cache_dir

# Register built-in adapters on import.
from rem2.models import builtins as _builtins  # noqa: F401

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
