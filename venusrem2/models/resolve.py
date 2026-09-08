"""Resolve CLI model selection and apply per-model defaults."""

from __future__ import annotations

import os
from typing import Any, Optional

from venusrem2.models.registry import get_model
from venusrem2.models.weights import default_cache_dir, ensure_dir, resolve_existing_dir, resolve_existing_weight
from venusrem2.naming import DEFAULT_PROSST_ID, PROSST_ENSEMBLE_IDS, fill_model_out_names


def resolve_model_name(args: Any) -> str:
    """Prefer --model; fall back to --baseline_type; default esm2."""
    model = getattr(args, "model", None) or os.environ.get("VENUSREM2_MODEL") or os.environ.get("REM2_MODEL")
    if model:
        return model.lower()
    baseline = getattr(args, "baseline_type", None) or "auto"
    if baseline != "auto":
        return baseline.lower()
    names = getattr(args, "model_name", None) or []
    # argparse default ProSST-2048 is treated as unset.
    if names and names != [DEFAULT_PROSST_ID]:
        if "prosst" in str(names[0]).lower():
            return "prosst"
        return "auto"
    return "esm2"


def apply_model_defaults(
    args: Any, model_name: str, cache_dir: Optional[str] = None, logger: Any = None
) -> str:
    """Fill missing cache paths / model ids / baseline_type for a registered model."""
    cache = default_cache_dir(cache_dir or getattr(args, "cache_dir", None))
    args.cache_dir = cache

    cls = get_model(model_name)
    spec = cls.spec
    baseline_type = spec.baseline_type or spec.name
    args.baseline_type = baseline_type

    model_id = getattr(args, "model_id", None)
    names = list(getattr(args, "model_name", None) or [])
    on_legacy_default = (not names) or names == [DEFAULT_PROSST_ID]

    if model_name == "auto" and not model_id and (on_legacy_default or names == ["auto"]):
        raise SystemExit("--model auto requires --model_id (Hugging Face repo or local path)")

    if model_id:
        args.model_name = [model_id]
    elif model_name == "venusrem2" and on_legacy_default:
        args.model_name = list(PROSST_ENSEMBLE_IDS)
    elif on_legacy_default:
        # Always stamp a model-specific label so CLI score columns
        # (`{basename}__raw_backbone`) are not left as ProSST-2048.
        if spec.default_model_id:
            args.model_name = [spec.default_model_id]
        else:
            args.model_name = [model_name]

    if model_name in ("prosst", "venusrem", "venusrem2") and getattr(args, "backbone_mode", "auto") == "auto":
        args.backbone_mode = "prosst"

    fill_model_out_names(args, model_name)

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
        cached = resolve_existing_dir("protssn", cache_dir=cache)
        args.protssn_model_dir = cached or ensure_dir(os.path.join(cache, "protssn"))

    if baseline_type == "protein_mpnn" and not getattr(args, "protein_mpnn_checkpoint", None):
        cached = resolve_existing_weight("protein_mpnn", "v_48_020.pt", cache_dir=cache)
        args.protein_mpnn_checkpoint = cached or ensure_dir(os.path.join(cache, "protein_mpnn"))

    if baseline_type == "saprot":
        from venusrem2.models.weights import ensure_foldseek_bin

        args.foldseek_bin = ensure_foldseek_bin(
            cache_dir=cache,
            explicit=getattr(args, "foldseek_bin", None),
            logger=logger,
        )

    if baseline_type == "s3f":
        from venusrem2.models.weights import bundled_s3f_config, ensure_s3f_checkpoint

        if not getattr(args, "s2f_config", None):
            args.s2f_config = bundled_s3f_config()
        args.s2f_checkpoint = ensure_s3f_checkpoint(
            cache_dir=cache,
            explicit=getattr(args, "s2f_checkpoint", None),
            logger=logger,
        )
        # Prefer AF2 assay-resolved surfaces under base_dir when present.
        if not getattr(args, "s3f_surface_dir", None) and getattr(args, "base_dir", None):
            cand = os.path.join(args.base_dir, "s3f_surfaces_af2_assay_resolved_full")
            if os.path.isdir(cand):
                args.s3f_surface_dir = cand

    if baseline_type in ("esm2", "esm1b", "esm1v", "mifst", "mif_st", "mif-st") and getattr(args, "backbone_mode", "auto") == "auto":
        args.backbone_mode = "plain_mlm"

    return cache

