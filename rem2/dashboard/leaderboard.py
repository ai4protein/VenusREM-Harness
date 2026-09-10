"""ProteinGym boards from the REM2 paper (docs/0overleaf Table 1 + staged VenusREM2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Official 217-assay Average Spearman and the five ProteinGym function types.
# Source: docs/0overleaf/table1_leaderboard.tex and table1_leaderboard_full.tex
# (ranked public rows from proteingym.org/benchmarks; VenusREM2 is the paper result).
_PAPER_ROWS = [
    {
        "name": "VenusREM2",
        "model": "venusrem2",
        "highlight": True,
        "inputs": ["seq", "str", "evo"],
        "note": "ProSST ensemble + rem2",
        "scores": {
            "average": 0.556,
            "activity": 0.541,
            "binding": 0.495,
            "expression": 0.557,
            "organismal": 0.494,
            "stability": 0.691,
        },
    },
    {
        "name": "AIDO Protein-RAG (16B)",
        "inputs": ["str", "evo"],
        "scores": {
            "average": 0.518,
            "activity": 0.517,
            "binding": 0.426,
            "expression": 0.522,
            "organismal": 0.491,
            "stability": 0.635,
        },
    },
    {
        "name": "VenusREM",
        "inputs": ["seq", "str", "evo"],
        "scores": {
            "average": 0.518,
            "activity": 0.495,
            "binding": 0.454,
            "expression": 0.533,
            "organismal": 0.459,
            "stability": 0.650,
        },
    },
    {
        "name": "ProSST (K=2048)",
        "inputs": ["seq", "str"],
        "scores": {
            "average": 0.507,
            "activity": 0.476,
            "binding": 0.445,
            "expression": 0.530,
            "organismal": 0.431,
            "stability": 0.653,
        },
    },
    {
        "name": "S3F-MSA",
        "inputs": ["str", "evo"],
        "scores": {
            "average": 0.496,
            "activity": 0.502,
            "binding": 0.440,
            "expression": 0.479,
            "organismal": 0.477,
            "stability": 0.581,
        },
    },
    {
        "name": "Protriever",
        "inputs": ["evo"],
        "scores": {
            "average": 0.479,
            "activity": 0.487,
            "binding": 0.396,
            "expression": 0.496,
            "organismal": 0.479,
            "stability": 0.537,
        },
    },
    {
        "name": "ESCOTT",
        "inputs": ["str", "evo"],
        "scores": {
            "average": 0.476,
            "activity": 0.499,
            "binding": 0.389,
            "expression": 0.468,
            "organismal": 0.466,
            "stability": 0.557,
        },
    },
    {
        "name": "PoET (200M)",
        "inputs": ["evo"],
        "scores": {
            "average": 0.470,
            "activity": 0.494,
            "binding": 0.396,
            "expression": 0.466,
            "organismal": 0.475,
            "stability": 0.519,
        },
    },
    {
        "name": "ESM3 open (1.4B)",
        "inputs": ["seq", "str"],
        "scores": {
            "average": 0.466,
            "activity": 0.430,
            "binding": 0.400,
            "expression": 0.470,
            "organismal": 0.389,
            "stability": 0.641,
        },
    },
    {
        "name": "RSALOR",
        "inputs": ["str", "evo"],
        "scores": {
            "average": 0.465,
            "activity": 0.479,
            "binding": 0.416,
            "expression": 0.427,
            "organismal": 0.426,
            "stability": 0.575,
        },
    },
    {
        "name": "VespaG",
        "inputs": ["seq"],
        "scores": {
            "average": 0.458,
            "activity": 0.493,
            "binding": 0.370,
            "expression": 0.456,
            "organismal": 0.437,
            "stability": 0.533,
        },
    },
    {
        "name": "SaProt (650M)",
        "model": "saprot",
        "inputs": ["seq", "str"],
        "scores": {
            "average": 0.457,
            "activity": 0.458,
            "binding": 0.378,
            "expression": 0.488,
            "organismal": 0.366,
            "stability": 0.592,
        },
    },
    {
        "name": "TranceptEVE-L",
        "inputs": ["seq", "evo"],
        "scores": {
            "average": 0.456,
            "activity": 0.487,
            "binding": 0.376,
            "expression": 0.457,
            "organismal": 0.459,
            "stability": 0.500,
        },
    },
    {
        "name": "GEMME",
        "inputs": ["evo"],
        "scores": {
            "average": 0.455,
            "activity": 0.482,
            "binding": 0.383,
            "expression": 0.438,
            "organismal": 0.452,
            "stability": 0.519,
        },
    },
    {
        "name": "ProtSSN ensemble",
        "inputs": ["seq", "str"],
        "scores": {
            "average": 0.449,
            "activity": 0.466,
            "binding": 0.366,
            "expression": 0.449,
            "organismal": 0.396,
            "stability": 0.568,
        },
    },
]

# docs/0overleaf/table_staged_per_config.tex — VenusREM2 row, 3-decimal paper values.
_VENUSREM2_STAGES = [
    {"name": "VenusREM2 · rem2", "score": 0.556, "note": "+ pLDDT (full)", "highlight": True},
    {"name": "VenusREM2 · + RSA", "score": 0.554, "note": "gated CCD + RSA"},
    {"name": "VenusREM2 · + gated CCD", "score": 0.550, "note": "adaptive mix + coherence gate"},
    {"name": "VenusREM2 · adaptive mix", "score": 0.542, "note": "entropy-α MSA"},
    {"name": "VenusREM2 · + ungated CCD", "score": 0.538, "note": "mix + κ=1"},
    {"name": "VenusREM2 · raw", "score": 0.524, "note": "uncalibrated ProSST ensemble", "recipe": "raw"},
]

_PROTEINGYM_REM2_DATA = Path(__file__).with_name("data") / "proteingym_rem2.json"


def _proteingym_pairs() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    raw_data = json.loads(_PROTEINGYM_REM2_DATA.read_text(encoding="utf-8"))
    pairs = []
    for raw in raw_data["pairs"]:
        spearman = raw["metrics"]["spearman"]
        pairs.append(
            {
                "key": raw["key"],
                "family": raw["base_name"],
                "base_name": raw["base_name"],
                "enhanced_name": raw["rem2_name"],
                "base": spearman["base"],
                "enhanced": spearman["rem2"],
                "delta": spearman["delta"],
                "protocol": raw["notes"],
                "metrics": raw["metrics"],
            }
        )
    return pairs, raw_data["metrics"]


def _proteingym_product_benchmark() -> dict[str, Any]:
    pairs, metrics = _proteingym_pairs()
    return {
        "id": "proteingym",
        "label": "ProteinGym",
        "title": "ProteinGym substitutions",
        "description": "Same-backbone comparison of raw model scores and the full REM2 recipe.",
        "status": "ready",
        "n": 217,
        "setting": "Zero-shot · substitutions",
        "metric": "Five paired metrics",
        "source": "PG_Raw_vs_REM2 · Feishu revision 461",
        "source_url": "https://proteingym.org/benchmarks",
        "model_count": len(pairs),
        "metrics": metrics,
        "pairs": pairs,
        "function_scores": {
            "activity": 0.539,
            "binding": 0.499,
            "expression": 0.562,
            "organismal fitness": 0.483,
            "stability": 0.698,
        },
    }

_CATEGORY_BOARDS = [
    ("substitutions", "Substitutions", "average", "ProteinGym substitutions"),
    ("stability", "Stability", "stability", "ProteinGym stability"),
    ("activity", "Activity", "activity", "ProteinGym activity"),
    ("binding", "Binding", "binding", "ProteinGym binding"),
    ("expression", "Expression", "expression", "ProteinGym expression"),
    ("organismal", "Organismal", "organismal", "ProteinGym organismal fitness"),
]


def _rank(rows: list[dict[str, Any]], key: str = "score") -> list[dict[str, Any]]:
    ranked = []
    for i, row in enumerate(sorted(rows, key=lambda item: -float(item[key])), start=1):
        item = dict(row)
        item["rank"] = i
        item[key] = round(float(item[key]), 3)
        ranked.append(item)
    return ranked


def _category_rows(metric: str) -> list[dict[str, Any]]:
    rows = []
    for raw in _PAPER_ROWS:
        scores = raw["scores"]
        item = {
            "name": raw["name"],
            "model": raw.get("model"),
            "inputs": list(raw.get("inputs") or []),
            "note": raw.get("note") or "",
            "highlight": bool(raw.get("highlight")),
            "score": float(scores[metric]),
            "average": float(scores["average"]),
            "activity": float(scores["activity"]),
            "binding": float(scores["binding"]),
            "expression": float(scores["expression"]),
            "organismal": float(scores["organismal"]),
            "stability": float(scores["stability"]),
        }
        rows.append(item)
    return _rank(rows)


def proteingym_board() -> dict[str, Any]:
    """Default (substitutions / Average Spearman) board."""
    return next(
        board for board in proteingym_catalog()["boards"] if board["id"] == "substitutions"
    )


def proteingym_catalog() -> dict[str, Any]:
    boards = []
    for board_id, label, metric, title in _CATEGORY_BOARDS:
        boards.append(
            {
                "id": board_id,
                "label": label,
                "title": title,
                "metric": "Mean Spearman",
                "metric_key": metric,
                "n": 217,
                "source": "REM2 paper Table 1; public rows from proteingym.org/benchmarks",
                "note": "",
                "rows": _category_rows(metric),
            }
        )
    boards.append(
        {
            "id": "ablations",
            "label": "Full rem2",
            "title": "VenusREM2 recipe ladder",
            "metric": "Mean Spearman",
            "metric_key": "average",
            "n": 217,
            "source": "REM2 paper staged ProteinGym scores (VenusREM2 row)",
            "note": "",
            "rows": _rank(
                [
                    {
                        "name": row["name"],
                        "model": "venusrem2",
                        "recipe": row.get("recipe", "full"),
                        "score": row["score"],
                        "inputs": ["seq", "str", "evo"],
                        "note": row["note"],
                        "highlight": bool(row.get("highlight")),
                    }
                    for row in _VENUSREM2_STAGES
                ]
            ),
        }
    )
    return {
        "default": "substitutions",
        "url": "https://proteingym.org/benchmarks",
        "metric": "Mean Spearman",
        "default_benchmark": "proteingym",
        "benchmarks": [_proteingym_product_benchmark()],
        "planned_benchmarks": ["VenusMutHub", "VenusViroHub"],
        "boards": boards,
    }
