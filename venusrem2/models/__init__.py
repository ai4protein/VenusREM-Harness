"""Pluggable logit-model adapters."""

from venus_orbit.models.base import ModelAdapter, ModelSpec
from venus_orbit.models.registry import get_model, list_models, register_model
from venus_orbit.models.resolve import apply_model_defaults, resolve_model_name
from venus_orbit.models.weights import default_cache_dir

# Register built-in adapters on import.
from venus_orbit.models import builtins as _builtins  # noqa: F401

__all__ = [
    "ModelAdapter",
    "ModelSpec",
    "register_model",
    "get_model",
    "list_models",
    "resolve_model_name",
    "apply_model_defaults",
    "default_cache_dir",
]
