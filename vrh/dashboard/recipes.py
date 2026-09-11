"""Dashboard scoring recipes (CLI-equivalent extra flags)."""

from __future__ import annotations

from typing import Any

RECIPES: list[dict[str, Any]] = [
    {"id": "full", "label": "Full vrh", "argv": [], "msa": "optional"},
    {
        "id": "raw",
        "label": "Raw backbone",
        "argv": [
            "--alpha",
            "0",
            "--scoring_mode",
            "log_odds",
            "--background_weight",
            "0",
            "--no_rsa_decay",
            "--no_plddt_decay",
        ],
        "msa": "off",
    },
    {
        "id": "msa",
        "label": "+ MSA",
        "argv": [
            "--alpha",
            "entropy",
            "--scoring_mode",
            "log_odds",
            "--background_weight",
            "0",
            "--no_rsa_decay",
            "--no_plddt_decay",
        ],
        "msa": "optional",
    },
    {
        "id": "ccd",
        "label": "+ MSA + CCD",
        "argv": ["--no_rsa_decay", "--no_plddt_decay"],
        "msa": "optional",
    },
]

_BY_ID = {item["id"]: item for item in RECIPES}


def recipe_argv(recipe_id: str) -> list[str]:
    spec = _BY_ID.get((recipe_id or "full").strip().lower())
    if spec is None:
        raise ValueError(f"Unknown recipe: {recipe_id}")
    return list(spec["argv"])


def recipe_public() -> list[dict[str, Any]]:
    return [dict(item) for item in RECIPES]
