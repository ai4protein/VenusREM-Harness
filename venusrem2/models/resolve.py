"""Resolve CLI model selection and apply per-model defaults."""

from __future__ import annotations

import os
from typing import Any, Optional

from venus_orbit.models.registry import get_model
from venus_orbit.models.weights import default_cache_dir, ensure_dir


def resolve_model_name(args: Any) -> str:
    """Prefer --model; fall back to --baseline_type; default prosst."""
    model = getattr(args, "model", None)
    if model:
        return model.lower()
    baseline = getattr(args, "baseline_type", None) or "auto"
    if baseline == "auto":
        # Heuristic: ProSST-style HF ids register as "prosst", others as "auto".
        names = getattr(args, "model_name", None) or []
        if names and "prosst" in str(names[0]).lower():
            return "prosst"
        if names:
            return "auto"
        return "prosst"
    return baseline.lower()


def apply_model_defaults(args: Any, model_name: str, cache_dir: Optional[str] = None) -> str:
    """Fill missing cache paths / model ids / baseline_type for a registered model."""
    cache = default_cache_dir(cache_dir or getattr(args, "cache_dir", None))
    args.cache_dir = cache

    cls = get_model(model_name)
    spec = cls.spec
    baseline_type = spec.baseline_type or spec.name
    args.baseline_type = baseline_type

    model_id = getattr(args, "model_id", None)
    default_prosst = "AI4Protein/ProSST-2048"
    names = list(getattr(args, "model_name", None) or [])
    on_legacy_default = (not names) or names == [default_prosst]

    if model_id:
        args.model_name = [model_id]
    elif on_legacy_default:
        # Always stamp a model-specific label so CLI score columns
        # (`{basename}__raw_backbone`) are not left as ProSST-2048.
        if spec.default_model_id:
            args.model_name = [spec.default_model_id]
        else:
            args.model_name = [model_name]

    if model_name == "prosst" and getattr(args, "backbone_mode", "auto") == "auto":
        args.backbone_mode = "prosst"

    # Dedicated legacy id flags.
    if baseline_type == "progen2":
        if not getattr(args, "progen2_model_name_or_path", None):
            args.progen2_model_name_or_path = model_id or spec.default_model_id
    elif baseline_type == "progen3":
        if not getattr(args, "progen3_model_name_or_path", None):
            args.progen3_model_name_or_path = model_id or spec.default_model_id
    elif baseline_type == "protgpt2":
        if not getattr(args, "protgpt2_model_name_or_path", None):
            args.protgpt2_model_name_or_path = model_id or spec.default_model_id
    elif baseline_type == "rita":
        if not getattr(args, "rita_model_name_or_path", None):
            args.rita_model_name_or_path = model_id or spec.default_model_id
    elif baseline_type == "esm3":
        if model_id:
            args.esm3_model_name = model_id
        elif not getattr(args, "esm3_model_name", None) and spec.default_model_id:
            args.esm3_model_name = spec.default_model_id
    elif baseline_type == "tranception":
        if model_id:
            args.tranception_checkpoint = model_id
        elif not getattr(args, "tranception_checkpoint", None):
            args.tranception_checkpoint = spec.default_model_id
    elif baseline_type == "carp":
        if model_id:
            args.carp_model_name = model_id

    if baseline_type == "protssn" and not getattr(args, "protssn_model_dir", None):
        args.protssn_model_dir = ensure_dir(os.path.join(cache, "protssn"))

    if baseline_type == "protein_mpnn" and not getattr(args, "protein_mpnn_checkpoint", None):
        args.protein_mpnn_checkpoint = ensure_dir(os.path.join(cache, "protein_mpnn"))

    if baseline_type == "saprot":
        from venus_orbit.models.weights import ensure_foldseek_bin

        args.foldseek_bin = ensure_foldseek_bin(
            cache_dir=cache,
            explicit=getattr(args, "foldseek_bin", None),
            logger=None,
        )

    if baseline_type == "s3f":
        from venus_orbit.models.weights import bundled_s3f_config, ensure_s3f_checkpoint

        if not getattr(args, "s2f_config", None):
            args.s2f_config = bundled_s3f_config()
        args.s2f_checkpoint = ensure_s3f_checkpoint(
            cache_dir=cache,
            explicit=getattr(args, "s2f_checkpoint", None),
            logger=None,
        )
        # Prefer AF2 assay-resolved surfaces under base_dir when present.
        if not getattr(args, "s3f_surface_dir", None) and getattr(args, "base_dir", None):
            cand = os.path.join(args.base_dir, "s3f_surfaces_af2_assay_resolved_full")
            if os.path.isdir(cand):
                args.s3f_surface_dir = cand

    if model_name in ("esm2", "esm1b", "esm1v") and getattr(args, "backbone_mode", "auto") == "auto":
        args.backbone_mode = "plain_mlm"

    return cache

