"""Model registry and entry-point loading."""

from __future__ import annotations

from typing import Dict, List, Type

from rem2.models.base import ModelAdapter, ModelSpec

_REGISTRY: Dict[str, Type[ModelAdapter]] = {}
_ENTRY_POINTS_LOADED = False


def register_model(cls: Type[ModelAdapter]) -> Type[ModelAdapter]:
    """Decorator / helper to register a ModelAdapter subclass."""
    if not hasattr(cls, "spec") or not isinstance(cls.spec, ModelSpec):
        raise TypeError(f"{cls.__name__} must define ClassVar[ModelSpec] spec")
    names = [cls.spec.name, *cls.spec.aliases]
    for name in names:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"Model name already registered: {name}")
        _REGISTRY[key] = cls
    return cls


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return
    for group in ("rem2.models", "venusrem2.models"):
        try:
            eps = entry_points(group=group)
        except TypeError:  # Python < 3.10
            eps = entry_points().get(group, [])
        for ep in eps:
            try:
                ep.load()
            except Exception:
                continue


def get_model(name: str) -> Type[ModelAdapter]:
    _load_entry_points()
    key = name.lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted({c.spec.name for c in _REGISTRY.values()}))
        raise KeyError(f"Unknown model '{name}'. Available: {known}")
    return _REGISTRY[key]


def list_models() -> List[ModelSpec]:
    _load_entry_points()
    seen = set()
    specs: List[ModelSpec] = []
    for cls in _REGISTRY.values():
        if cls.spec.name in seen:
            continue
        seen.add(cls.spec.name)
        specs.append(cls.spec)
    return sorted(specs, key=lambda s: s.name)
