"""ProteinGym boards from the REM2 paper (docs/0overleaf Table 1 + staged VenusREM2)."""

from __future__ import annotations

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

# Paired, same-pipeline ProteinGym results from the REM2 paper.  These are the
# values that answer the product question directly: what changes when REM2 is
# added to a fixed backbone?  ``official_reference`` is contextual only; it is
# deliberately kept separate because ProteinGym can use a different scoring
# protocol (most visibly for SaProt).
_PROTEINGYM_REM2_PAIRS = [
    {
        "family": "ProSST ensemble",
        "base_name": "ProSST ensemble",
        "enhanced_name": "VenusREM2",
        "inputs": ["seq", "str", "evo"],
        "base": 0.524,
        "enhanced": 0.556,
        "official_reference": None,
        "official_name": "No matching ensemble row",
        "protocol": "Six ProSST checkpoints · same internal pipeline",
        "comparison": "direct",
    },
    {
        "family": "ESM-2 650M",
        "base_name": "ESM-2 650M",
        "enhanced_name": "ESM-2 650M + REM2",
        "inputs": ["seq", "evo"],
        "base": 0.418,
        "enhanced": 0.468,
        "official_reference": 0.414,
        "official_name": "ESM2 (650M)",
        "protocol": "wt-marginals · same internal pipeline",
        "comparison": "direct",
    },
    {
        "family": "SaProt AF-650M",
        "base_name": "SaProt AF-650M",
        "enhanced_name": "SaProt AF-650M + REM2",
        "inputs": ["seq", "str", "evo"],
        "base": 0.424,
        "enhanced": 0.454,
        "official_reference": 0.457,
        "official_name": "SaProt (650M)",
        "protocol": "Internal: wt-marginals · ProteinGym: masked mutant positions",
        "comparison": "protocol_mismatch",
    },
    {
        "family": "ESM-1v ensemble",
        "base_name": "ESM-1v 5-seed ensemble",
        "enhanced_name": "ESM-1v ensemble + REM2",
        "inputs": ["seq", "evo"],
        "base": 0.410,
        "enhanced": 0.457,
        "official_reference": 0.407,
        "official_name": "ESM-1v (ensemble)",
        "protocol": "wt-marginals · same internal pipeline",
        "comparison": "direct",
    },
]


def _proteingym_product_benchmark() -> dict[str, Any]:
    pairs = []
    for raw in _PROTEINGYM_REM2_PAIRS:
        row = dict(raw)
        row["delta"] = round(float(row["enhanced"]) - float(row["base"]), 3)
        official = row.get("official_reference")
        row["official_gap"] = (
            round(float(row["base"]) - float(official), 3)
            if official is not None
            else None
        )
        pairs.append(row)
    return {
        "id": "proteingym",
        "label": "ProteinGym",
        "title": "ProteinGym substitutions",
        "description": "Same-backbone comparison of raw model scores and the full REM2 recipe.",
        "status": "ready",
        "n": 217,
        "setting": "Zero-shot · substitutions",
        "metric": "Mean Spearman",
        "source": "REM2 paper results; ProteinGym v1.3 values shown as reference only",
        "source_url": "https://proteingym.org/benchmarks",
        "pairs": pairs,
        "stages": [dict(row) for row in _VENUSREM2_STAGES],
        "function_scores": {
            "activity": 0.541,
            "binding": 0.495,
            "expression": 0.557,
            "organismal fitness": 0.494,
            "stability": 0.691,
        },
        "prior_method": {
            "name": "VenusREM",
            "score": 0.518,
            "base_name": "ProSST (K=2048)",
            "base_score": 0.507,
            "note": "Official ProteinGym row; fixed-method predecessor, not the ProSST ensemble used by VenusREM2.",
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
