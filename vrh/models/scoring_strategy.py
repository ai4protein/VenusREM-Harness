"""Forward-strategy capability checks.

Public names: ``wt`` / ``mask`` / ``tf`` (aliases: wt-marginals,
masked-marginals, teacher-force).

Masked-marginals needs a real mask token / per-site mask forward. Causal LMs
and inverse-folding models must refuse that flag instead of silently scoring
with a wild-type pass. ProSST is wt-only. ProteinMPNN is teacher-force.
"""

from __future__ import annotations

from typing import Any, Optional

MASKED_MARGINALS = "masked-marginals"
WT_MARGINALS = "wt-marginals"
TEACHER_FORCE = "teacher-force"

# Canonical values stored on args after normalize.
CANONICAL_STRATEGIES = (WT_MARGINALS, MASKED_MARGINALS, TEACHER_FORCE)

_ALIASES = {
    "wt": WT_MARGINALS,
    "wt-marginals": WT_MARGINALS,
    "wt_marginals": WT_MARGINALS,
    "wild-type": WT_MARGINALS,
    "wildtype": WT_MARGINALS,
    "mask": MASKED_MARGINALS,
    "masked": MASKED_MARGINALS,
    "masked-marginals": MASKED_MARGINALS,
    "masked_marginals": MASKED_MARGINALS,
    "tf": TEACHER_FORCE,
    "teacher-force": TEACHER_FORCE,
    "teacher_force": TEACHER_FORCE,
    "teacherforce": TEACHER_FORCE,
}

# baseline_type keys that actually implement a mask forward in dispatch.
_DISPATCH_MASK_TYPES = frozenset(
    {
        "auto",
        "esm1b",
        "esm1v",
        "esm2",
        "esm3",
        "saprot",
        "protssn",
        "carp",
        "s3f",
    }
)


class UnsupportedScoringStrategy(ValueError):
    """Requested --scoring_strategy is not implemented for this backbone."""


def normalize_scoring_strategy(raw: Optional[str]) -> str:
    """Map wt/mask/tf (and long aliases) to a canonical strategy name."""
    key = (raw or WT_MARGINALS).strip().lower().replace(" ", "-")
    key = key.replace("_", "-")
    # keep teacher-force / wt-marginals after underscore fold
    if key in {"wt-marginals", "masked-marginals", "teacher-force"}:
        return key
    # second pass for wt_marginals → wt-marginals already handled
    alias_key = (raw or WT_MARGINALS).strip().lower().replace(" ", "-")
    if alias_key in _ALIASES:
        return _ALIASES[alias_key]
    raise ValueError(
        f"Unknown --scoring_strategy {raw!r}. Use wt, mask, or tf "
        "(aliases: wt-marginals, masked-marginals, teacher-force)."
    )


def tokenizer_supports_mask(tokenizer: Any) -> bool:
    return getattr(tokenizer, "mask_token_id", None) is not None


def spec_supports_mask(model_key: Optional[str]) -> bool:
    if not model_key:
        return False
    from vrh.models.registry import get_model

    try:
        return bool(get_model(model_key).spec.supports_mask)
    except KeyError:
        return False


def spec_supports_tf(model_key: Optional[str]) -> bool:
    if not model_key:
        return False
    from vrh.models.registry import get_model

    try:
        return bool(getattr(get_model(model_key).spec, "supports_tf", False))
    except KeyError:
        return False


def models_supporting_mask() -> list[str]:
    from vrh.models.registry import list_models

    return [spec.name for spec in list_models() if spec.supports_mask]


def models_supporting_tf() -> list[str]:
    from vrh.models.registry import list_models

    return [spec.name for spec in list_models() if getattr(spec, "supports_tf", False)]


def dispatch_supports_mask(baseline_type: Optional[str]) -> bool:
    return (baseline_type or "").lower() in _DISPATCH_MASK_TYPES


def forward_modes_for_spec(spec: Any) -> list[str]:
    modes = ["wt"]
    if getattr(spec, "supports_mask", False):
        modes.append("mask")
    if getattr(spec, "supports_tf", False):
        modes.append("tf")
    return modes


def forward_modes_label(spec: Any) -> str:
    return ",".join(forward_modes_for_spec(spec))


def allowed_strategies(model_key: str) -> set[str]:
    allowed = {WT_MARGINALS}
    if spec_supports_mask(model_key):
        allowed.add(MASKED_MARGINALS)
    if spec_supports_tf(model_key):
        allowed.add(TEACHER_FORCE)
    return allowed


def strategy_short_name(strategy: str) -> str:
    if strategy == MASKED_MARGINALS:
        return "mask"
    if strategy == TEACHER_FORCE:
        return "tf"
    return "wt"


def unsupported_mask_message(model_key: str, *, extra: str = "") -> str:
    supported = ", ".join(models_supporting_mask()) or "(none registered)"
    bits = [
        f"--scoring_strategy mask is not supported by model '{model_key}'.",
        "This backbone has no mask token / per-site mask forward "
        "(ProSST is wt-only; causal LMs and inverse folding have no mask).",
        "Refusing to score rather than silently falling back to wt.",
        f"Use --scoring_strategy wt, or pick a mask-capable LM: {supported}.",
    ]
    if extra:
        bits.insert(1, extra)
    return " ".join(bits)


def unsupported_tf_message(model_key: str) -> str:
    supported = ", ".join(models_supporting_tf()) or "protein_mpnn"
    return (
        f"--scoring_strategy tf is not supported by model '{model_key}'. "
        f"teacher-force is for ProteinMPNN ({supported}). "
        "Use --scoring_strategy wt or mask."
    )


def require_scoring_strategy(model_key: str, scoring_strategy: Optional[str]) -> None:
    try:
        strategy = normalize_scoring_strategy(scoring_strategy)
    except ValueError as exc:
        raise UnsupportedScoringStrategy(str(exc)) from exc
    allowed = allowed_strategies(model_key)
    if strategy in allowed:
        return
    if strategy == MASKED_MARGINALS:
        raise UnsupportedScoringStrategy(unsupported_mask_message(model_key))
    if strategy == TEACHER_FORCE:
        raise UnsupportedScoringStrategy(unsupported_tf_message(model_key))
    raise UnsupportedScoringStrategy(
        f"--scoring_strategy {strategy_short_name(strategy)} is not supported by "
        f"model '{model_key}'."
    )


def require_tokenizer_mask(model_key: str, tokenizer: Any) -> None:
    if tokenizer_supports_mask(tokenizer):
        return
    name = getattr(tokenizer, "name_or_path", type(tokenizer).__name__)
    raise UnsupportedScoringStrategy(
        unsupported_mask_message(
            model_key,
            extra=f"Tokenizer {name!r} has no mask_token_id.",
        )
    )
