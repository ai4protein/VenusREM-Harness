"""Forward-strategy capability checks.

Masked-marginals needs a real mask token / per-site mask forward. Causal LMs
and inverse-folding models must refuse that flag instead of silently scoring
with a wild-type pass.
"""

from __future__ import annotations

from typing import Any, Optional

MASKED_MARGINALS = "masked-marginals"
WT_MARGINALS = "wt-marginals"

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


def tokenizer_supports_mask(tokenizer: Any) -> bool:
    return getattr(tokenizer, "mask_token_id", None) is not None


def spec_supports_mask(model_key: Optional[str]) -> bool:
    if not model_key:
        return False
    from venusrem2.models.registry import get_model

    try:
        return bool(get_model(model_key).spec.supports_mask)
    except KeyError:
        return False


def models_supporting_mask() -> list[str]:
    from venusrem2.models.registry import list_models

    return [spec.name for spec in list_models() if spec.supports_mask]


def dispatch_supports_mask(baseline_type: Optional[str]) -> bool:
    return (baseline_type or "").lower() in _DISPATCH_MASK_TYPES


def unsupported_mask_message(model_key: str, *, extra: str = "") -> str:
    supported = ", ".join(models_supporting_mask()) or "(none registered)"
    bits = [
        f"--scoring_strategy {MASKED_MARGINALS} is not supported by model '{model_key}'.",
        "This backbone has no mask token / per-site mask forward "
        "(causal LM or inverse folding).",
        "Refusing to score rather than silently falling back to wt-marginals.",
        f"Use --scoring_strategy {WT_MARGINALS}, or pick a masked LM: {supported}.",
    ]
    if extra:
        bits.insert(1, extra)
    return " ".join(bits)


def require_scoring_strategy(model_key: str, scoring_strategy: Optional[str]) -> None:
    strategy = (scoring_strategy or WT_MARGINALS).strip().lower()
    if strategy != MASKED_MARGINALS:
        return
    if spec_supports_mask(model_key):
        return
    raise UnsupportedScoringStrategy(unsupported_mask_message(model_key))


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
