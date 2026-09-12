"""Score-column / run labels.

**vrh** is the generic calibration recipe (MSA + CCD + RSA + pLDDT) and can
sit on any backbone.

**VenusREM2** is only the official system: vrh on a **ProSST ensemble**
(2+ ProSST checkpoints, typically different structure-vocab sizes).
"""

from __future__ import annotations

from typing import Any, Optional

RECIPE_NAME = "vrh"
OFFICIAL_SYSTEM_NAME = "VenusREM2"
VENUSREM_V1_NAME = "VenusREM"
VENUSREM_V1_KEYS = {"venusrem", "venusrem1", "venus-rem", "venusrem-v1"}
_PROSST_KEYS = {
    "prosst",
    "prosst-2048",
    "prosst_2048",
    "prosst2048",
    "venusrem",
    "venusrem1",
    "venus-rem",
    "venusrem-v1",
    "venusrem2",
    "prosst_ensemble",
}
ENSEMBLE_MODEL_KEYS = {"venusrem2", "prosst_ensemble", "prosst-ensemble"}
PROSST_ENSEMBLE_SIZES = (20, 128, 512, 1024, 2048, 4096)
PROSST_ENSEMBLE_IDS = tuple(f"AI4Protein/ProSST-{k}" for k in PROSST_ENSEMBLE_SIZES)
DEFAULT_PROSST_ID = "AI4Protein/ProSST-2048"


def _names(args: Any) -> list[str]:
    return [str(n) for n in (getattr(args, "model_name", None) or [])]


def _prosst_ids(args: Any) -> list[str]:
    return [n for n in _names(args) if "prosst" in n.lower()]


def is_venusrem_v1(model_key: Optional[str]) -> bool:
    key = (model_key or "").lower().replace("_", "-")
    return key in VENUSREM_V1_KEYS


def is_prosst_key(model_key: Optional[str]) -> bool:
    key = (model_key or "").lower().replace("_", "-")
    return key in _PROSST_KEYS or key.startswith("prosst-") or is_venusrem_v1(model_key)


def is_prosst_run(model_key: Optional[str], args: Any) -> bool:
    if is_prosst_key(model_key):
        return True
    return bool(_prosst_ids(args))


def is_prosst_ensemble(args: Any) -> bool:
    return len(_prosst_ids(args)) >= 2


def is_ensemble_model_key(model_key: Optional[str]) -> bool:
    return (model_key or "").lower() in ENSEMBLE_MODEL_KEYS


def is_official_venusrem2(model_key: Optional[str], args: Any) -> bool:
    """True only for a ProSST ensemble (the branded VenusREM2 system)."""
    return is_prosst_run(model_key, args) and is_prosst_ensemble(args)


def short_backbone_id(model_id: Optional[str], model_key: Optional[str] = None) -> str:
    raw = model_id or model_key or RECIPE_NAME
    return str(raw).split("/")[-1]


def default_score_label(
    model_key: Optional[str],
    args: Any,
    model_id: Optional[str] = None,
) -> str:
    """CSV / summary column for one backbone in this run."""
    short = short_backbone_id(model_id, model_key)
    if is_official_venusrem2(model_key, args):
        return f"{OFFICIAL_SYSTEM_NAME}__{short}"
    if is_venusrem_v1(model_key):
        return VENUSREM_V1_NAME
    return f"{short}__{RECIPE_NAME}"


def fill_model_out_names(args: Any, model_key: str) -> list[str]:
    """Keep an explicit ``--model_out_name``; otherwise stamp vrh / VenusREM2 labels."""
    names = _names(args)
    if not names:
        names = [model_key]
    explicit = getattr(args, "model_out_name", None)
    if explicit:
        if (
            len(explicit) == 1
            and explicit[0] == OFFICIAL_SYSTEM_NAME
            and not is_official_venusrem2(model_key, args)
        ):
            # User forced the branded name on a non-ensemble run — keep it,
            # caller should warn.
            args.model_out_name = list(explicit)
            return args.model_out_name
        padded = list(explicit)
        while len(padded) < len(names):
            padded.append(default_score_label(model_key, args, names[len(padded)]))
        args.model_out_name = padded[: len(names)]
        return args.model_out_name
    args.model_out_name = [default_score_label(model_key, args, mid) for mid in names]
    return args.model_out_name


def run_banner(model_key: Optional[str], args: Any) -> str:
    if is_official_venusrem2(model_key, args):
        n = len(_prosst_ids(args))
        return f"{OFFICIAL_SYSTEM_NAME} scoring run (ProSST ensemble, {n} checkpoints + {RECIPE_NAME})"
    if is_venusrem_v1(model_key):
        return f"{VENUSREM_V1_NAME} scoring run (ProSST-2048, fixed α=0.8, log_odds)"
    if is_prosst_run(model_key, args):
        return f"{RECIPE_NAME} scoring run on ProSST (not the official {OFFICIAL_SYSTEM_NAME} ensemble)"
    key = model_key or "plm"
    return f"{RECIPE_NAME} scoring run on {key}"
