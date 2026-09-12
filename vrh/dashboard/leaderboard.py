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
        "note": "ProSST ensemble + vrh",
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
    {"name": "VenusREM2 · vrh", "score": 0.556, "note": "+ pLDDT (full)", "highlight": True},
    {"name": "VenusREM2 · + RSA", "score": 0.554, "note": "gated CCD + RSA"},
    {"name": "VenusREM2 · + gated CCD", "score": 0.550, "note": "adaptive mix + coherence gate"},
    {"name": "VenusREM2 · adaptive mix", "score": 0.542, "note": "entropy-α MSA"},
    {"name": "VenusREM2 · + ungated CCD", "score": 0.538, "note": "mix + κ=1"},
    {"name": "VenusREM2 · raw", "score": 0.524, "note": "uncalibrated ProSST ensemble", "recipe": "raw"},
]

def _benchmark_inputs(model_key: str) -> list[str]:
    """Return the foundation model inputs; VRH adds evolutionary evidence."""
    key = model_key.lower()
    if key.startswith(("proteinmpnn", "pmpnn_")) or key == "esmif":
        return ["str"]
    if key.startswith(("prosst", "saprot", "s3f")) or key in {"protssn", "mifst"}:
        return ["seq", "str"]
    return ["seq"]


_DASHBOARD_DATA = Path(__file__).with_name("data")
_DEFAULT_METRICS = [
    {"id": "spearman", "label": "Spearman"},
    {"id": "ndcg", "label": "NDCG"},
    {"id": "auc", "label": "AUC"},
    {"id": "mcc", "label": "MCC"},
    {"id": "top_recall", "label": "Top recall"},
]
_MUTHUB_DEFAULT = {
    "id": "venusmuthub",
    "label": "VenusMutHub",
    "title": "VenusMutHub substitutions",
    "description": "905 substitution assays across stability, activity, PPI binding, selectivity, and DTI binding. Paired Raw / +VRH model scores will appear after the evaluation snapshot is packaged.",
    "status": "catalog",
    "n": 905,
    "setting": "Zero-shot · substitutions",
    "metric": "Average Spearman",
    "source": "VenusMutHub assay_manifest.csv",
    "manifest": "data/VenusMutHub/assay_manifest.csv",
    "n_mutants": 27846,
    "median_seq_len": 226,
    "paired_score_table": None,
    "properties": [
        {"id": "overall", "label": "Overall", "n": 905},
        {"id": "stability", "label": "Stability", "n": 540},
        {"id": "activity", "label": "Activity", "n": 175},
        {"id": "ppi_binding", "label": "PPI binding", "n": 100},
        {"id": "selectivity", "label": "Selectivity", "n": 47},
        {"id": "dti_binding", "label": "DTI binding", "n": 43},
    ],
    "metrics": list(_DEFAULT_METRICS),
    "pairs": [],
}
_VIROHUB_DEFAULT = {
    "id": "venusvirohub",
    "label": "VenusViroHub",
    "title": "VenusViroHub substitutions",
    "description": "89 viral DMS substitution assays with zero ProteinGym overlap, covering immune escape, cell entry, and receptor binding.",
    "status": "catalog",
    "n": 89,
    "setting": "Zero-shot · substitutions",
    "metric": "Average Spearman",
    "source": "VenusViroHub · 89 viral DMS assays",
    "properties": [
        {"id": "overall", "label": "Overall", "n": 89},
        {"id": "activity", "label": "Activity", "n": 1},
        {"id": "binding", "label": "Binding", "n": 17},
        {"id": "cell_entry", "label": "Cell entry", "n": 18},
        {"id": "expression", "label": "Expression", "n": 15},
        {"id": "fitness", "label": "Fitness", "n": 15},
        {"id": "immune_escape", "label": "Immune escape", "n": 21},
        {"id": "stability", "label": "Stability", "n": 2},
    ],
    "metrics": list(_DEFAULT_METRICS),
    "pairs": [],
}


def _public_score_triple(block: dict[str, Any]) -> dict[str, Any]:
    recipe = block.get("vrh", block.get("rem2", block.get("enhanced")))
    out: dict[str, Any] = {}
    if "base" in block:
        out["base"] = block["base"]
    if recipe is not None:
        out["vrh"] = recipe
    if "delta" in block:
        out["delta"] = block["delta"]
    return out


def _public_score_tree(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    if {"base", "vrh", "rem2", "enhanced"} & set(obj):
        return _public_score_triple(obj)
    return {key: _public_score_tree(value) for key, value in obj.items()}


def _public_recipe_label(name: Any) -> str:
    text = str(name or "")
    return (
        text.replace(", rem2)", ", vrh)")
        .replace(", REM2)", ", VRH)")
        .replace("(rem2)", "(vrh)")
        .replace("(REM2)", "(VRH)")
        .replace(" rem2", " vrh")
        .replace(" REM2", " VRH")
    )


def _pairs_from_raw(raw_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for raw in raw_pairs:
        metrics = _public_score_tree(raw.get("metrics") or {})
        spearman = metrics.get("spearman") or {}
        base = raw.get("base")
        if base is None:
            base = spearman.get("base")
        enhanced = raw.get("enhanced")
        if enhanced is None:
            enhanced = raw.get("vrh", raw.get("rem2", spearman.get("vrh")))
        delta = raw.get("delta", spearman.get("delta"))
        if delta is None and base is not None and enhanced is not None:
            delta = round(float(enhanced) - float(base), 3)
        pairs.append(
            {
                "key": raw["key"],
                "family": raw.get("family") or raw.get("base_name"),
                "base_name": raw.get("base_name") or raw.get("family"),
                "enhanced_name": _public_recipe_label(
                    raw.get("enhanced_name") or raw.get("vrh_name") or raw.get("rem2_name")
                ),
                "base": base,
                "enhanced": enhanced,
                "delta": delta,
                "protocol": raw.get("protocol") or raw.get("notes") or "",
                "metrics": metrics,
                "properties": _public_score_tree(raw.get("properties") or {}),
                "properties_by_metric": _public_score_tree(raw.get("properties_by_metric") or {}),
                "inputs": list(raw.get("inputs") or _benchmark_inputs(raw["key"])),
            }
        )
    return pairs


def _load_hub_snapshot(hub_id: str) -> dict[str, Any]:
    for name in (f"{hub_id}_vrh.json", f"{hub_id}_rem2.json"):
        path = _DASHBOARD_DATA / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    catalog_path = _DASHBOARD_DATA / f"{hub_id}_catalog.json"
    if catalog_path.is_file():
        return json.loads(catalog_path.read_text(encoding="utf-8"))
    return {}


def _hub_product_benchmark(default: dict[str, Any]) -> dict[str, Any]:
    raw = _load_hub_snapshot(default["id"])
    merged = dict(default)
    for key in (
        "id",
        "label",
        "title",
        "description",
        "status",
        "setting",
        "metric",
        "source",
        "source_url",
        "manifest",
        "n_mutants",
        "median_seq_len",
        "paired_score_table",
        "properties",
        "metrics",
    ):
        if raw.get(key) not in (None, ""):
            merged[key] = raw[key]
    if raw.get("n") not in (None, ""):
        merged["n"] = raw["n"]
    elif raw.get("assays") not in (None, ""):
        merged["n"] = raw["assays"]
    pairs = _pairs_from_raw(raw.get("pairs") or [])
    merged["pairs"] = pairs
    merged["model_count"] = len(pairs)
    if pairs and merged.get("status") != "ready":
        merged["status"] = "ready"
    elif not pairs and not merged.get("status"):
        merged["status"] = "catalog"
    return merged


def _proteingym_snapshot() -> dict[str, Any]:
    raw = _load_hub_snapshot("proteingym")
    if not raw:
        raise FileNotFoundError("missing proteingym_vrh.json or proteingym_rem2.json")
    return raw


def _proteingym_pairs() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    raw_data = _proteingym_snapshot()
    return _pairs_from_raw(raw_data["pairs"]), raw_data["properties"]


def _proteingym_product_benchmark() -> dict[str, Any]:
    pairs, properties = _proteingym_pairs()
    raw_data = _proteingym_snapshot()
    return {
        "id": "proteingym",
        "label": "ProteinGym",
        "title": "ProteinGym substitutions",
        "description": "Same-backbone comparison of raw model scores and the full vrh recipe.",
        "status": "ready",
        "n": 217,
        "setting": "Zero-shot · substitutions",
        "metric": "Average Spearman",
        "source": "PG_Raw_vs_REM2 · Feishu revision 461",
        "source_url": "https://proteingym.org/benchmarks",
        "model_count": len(pairs),
        "properties": properties,
        "metrics": raw_data["metrics"],
        "pairs": pairs,
        "function_scores": {
            "activity": 0.539,
            "binding": 0.499,
            "expression": 0.562,
            "organismal fitness": 0.483,
            "stability": 0.698,
        },
    }


def _planned_product_benchmark(benchmark_id: str, label: str) -> dict[str, Any]:
    return {
        "id": benchmark_id,
        "label": label,
        "title": label,
        "description": "Benchmark schema is ready; paired Raw and +VRH results will be added after validation.",
        "status": "planned",
        "n": "—",
        "setting": "Awaiting data",
        "metric": "—",
        "properties": [],
        "pairs": [],
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
            "label": "Full vrh",
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
    product_benchmarks = [
        _proteingym_product_benchmark(),
        _hub_product_benchmark(_MUTHUB_DEFAULT),
        _hub_product_benchmark(_VIROHUB_DEFAULT),
    ]
    return {
        "default": "substitutions",
        "url": "https://proteingym.org/benchmarks",
        "metric": "Mean Spearman",
        "default_benchmark": "proteingym",
        "benchmarks": product_benchmarks,
        "planned_benchmarks": [
            item["label"] for item in product_benchmarks if item.get("status") == "planned"
        ],
        "boards": boards,
    }
