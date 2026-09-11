"""Resolve CLI model selection and apply per-model defaults."""

from __future__ import annotations

import os
from typing import Any, Optional

from vrh.models.registry import get_model
from vrh.models.weights import default_cache_dir, ensure_dir, resolve_existing_dir, resolve_existing_weight
from vrh.naming import (
    DEFAULT_PROSST_ID,
    PROSST_ENSEMBLE_IDS,
    fill_model_out_names,
    is_ensemble_model_key,
    is_prosst_key,
)


def resolve_model_name(args: Any) -> str:
    """Prefer --model; fall back to --baseline_type; default esm2."""
    from vrh.env import first_env

    model = getattr(args, "model", None) or first_env("VRH_MODEL", "REM2_MODEL", "VENUSREM2_MODEL")
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
    elif is_ensemble_model_key(model_name) and on_legacy_default:
        args.model_name = list(PROSST_ENSEMBLE_IDS)
    elif on_legacy_default:
        # Always stamp a model-specific label so CLI score columns
        # (`{basename}__raw_backbone`) are not left as ProSST-2048.
        if spec.default_model_id:
            args.model_name = [spec.default_model_id]
        else:
            args.model_name = [model_name]

    if is_prosst_key(model_name) and getattr(args, "backbone_mode", "auto") == "auto":
        args.backbone_mode = "prosst"

    fill_model_out_names(args, model_name)

    # --model KEY selects the checkpoint. Dedicated argparse flags
    # (e.g. --progen2_model_name_or_path) are overwritten unless the user
    # passed --model_id. Argparse family defaults like esmc_300m must not
    # leak onto --model esm3 / --model progen2-xl.
    dedicated = model_id or spec.default_model_id

    if baseline_type == "progen2" and dedicated:
        args.progen2_model_name_or_path = dedicated
    elif baseline_type == "progen3" and dedicated:
        args.progen3_model_name_or_path = dedicated
    elif baseline_type == "protgpt2" and dedicated:
        args.protgpt2_model_name_or_path = dedicated
    elif baseline_type == "rita" and dedicated:
        args.rita_model_name_or_path = dedicated
    elif baseline_type == "esm3" and dedicated:
        args.esm3_model_name = dedicated
    elif baseline_type == "carp" and dedicated:
        args.carp_model_name = dedicated

    if baseline_type == "protssn" and not getattr(args, "protssn_model_dir", None):
        cached = resolve_existing_dir("protssn", cache_dir=cache)
        args.protssn_model_dir = cached or ensure_dir(os.path.join(cache, "protssn"))

    if baseline_type == "protein_mpnn":
        from vrh.models.scoring_strategy import TEACHER_FORCE, normalize_scoring_strategy

        try:
            strategy = normalize_scoring_strategy(getattr(args, "scoring_strategy", None))
        except ValueError:
            strategy = None
        if strategy is not None:
            args.scoring_strategy = strategy
        if strategy == TEACHER_FORCE:
            args.protein_mpnn_scoring_mode = "teacher_force"
        ckpt_name = (dedicated or "v_48_020.pt").split("/")[-1]
        if not ckpt_name.endswith(".pt"):
            ckpt_name = "v_48_020.pt"
        if not getattr(args, "protein_mpnn_checkpoint", None) or (
            model_name not in {"protein_mpnn", "proteinmpnn"}
        ):
            cached = resolve_existing_weight("protein_mpnn", ckpt_name, cache_dir=cache)
            args.protein_mpnn_checkpoint = cached or os.path.join(
                ensure_dir(os.path.join(cache, "protein_mpnn")), ckpt_name
            )

    if baseline_type == "s3f":
        from vrh.models.weights import bundled_s3f_config

        if not getattr(args, "s2f_config", None):
            args.s2f_config = bundled_s3f_config()
        # Prefer AF2 assay-resolved surfaces under base_dir when present.
        if not getattr(args, "s3f_surface_dir", None) and getattr(args, "base_dir", None):
            cand = os.path.join(args.base_dir, "s3f_surfaces_af2_assay_resolved_full")
            if os.path.isdir(cand):
                args.s3f_surface_dir = cand

    if baseline_type in ("esm2", "esm1b", "esm1v", "mifst", "mif_st", "mif-st") and getattr(args, "backbone_mode", "auto") == "auto":
        args.backbone_mode = "plain_mlm"

    return cache

