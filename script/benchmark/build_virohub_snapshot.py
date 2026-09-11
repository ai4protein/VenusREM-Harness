#!/usr/bin/env python3
"""Build the packaged VenusViroHub Raw/REM2 dashboard snapshot.

Reads Feishu leaderboard CSVs and writes the compact JSON consumed by
``_load_hub_snapshot("venusvirohub")``. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES = REPO_ROOT / "data/venusvirohub/iclr_appendix/feishu_tables"
DEFAULT_SNAPSHOT = REPO_ROOT / "rem2/dashboard/data/venusvirohub_rem2.json"

COMPARE_METRICS = {
    "spearman": "Spearman",
    "ndcg": "NDCG",
    "auc": "AUC",
    "mcc": "MCC",
    "top_recall": "Top_recall",
}
METRIC_SHEETS = {
    "spearman": ("Leaderboard_Spearman.csv", "Average_Spearman_hier"),
    "ndcg": ("Leaderboard_NDCG.csv", "Average_NDCG_hier"),
    "auc": ("Leaderboard_AUC.csv", "Average_AUC_hier"),
    "mcc": ("Leaderboard_MCC.csv", "Average_MCC_hier"),
    "top_recall": ("Leaderboard_Top_recall.csv", "Average_Top_recall_hier"),
}
PROPERTY_COLUMNS = {
    "activity": "Function_activity",
    "binding": "Function_binding",
    "cell_entry": "Function_cell_entry",
    "expression": "Function_expression",
    "fitness": "Function_fitness",
    "immune_escape": "Function_immune_escape",
    "stability": "Function_stability",
}
PROPERTY_META = (
    ("overall", "Overall", 89),
    ("activity", "Activity", 1),
    ("binding", "Binding", 17),
    ("cell_entry", "Cell entry", 18),
    ("expression", "Expression", 15),
    ("fitness", "Fitness", 15),
    ("immune_escape", "Immune escape", 21),
    ("stability", "Stability", 2),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key.strip(): (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def score_triple(base: float, rem2: float) -> dict[str, float]:
    rounded_base = round(base, 3)
    rounded_rem2 = round(rem2, 3)
    return {
        "base": rounded_base,
        "rem2": rounded_rem2,
        "delta": round(rounded_rem2 - rounded_base, 3),
    }


def sheet_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index = {}
    for row in rows:
        variant = row.get("Variant", "")
        name = row.get("Model_name", "")
        if variant and name:
            index[(variant, name)] = row
    return index


def lookup_row(
    index: dict[tuple[str, str], dict[str, str]],
    variant: str,
    names: list[str],
    model_key: str,
    metric_key: str,
) -> dict[str, str]:
    seen: list[str] = []
    for name in names:
        if not name or name in seen:
            continue
        seen.append(name)
        row = index.get((variant, name))
        if row is not None:
            return row
    raise SystemExit(
        f"no {variant} row for {model_key} on {metric_key}; tried {seen or names}"
    )


def build_snapshot(tables: Path) -> dict:
    compare = read_csv(tables / "Leaderboard_Compare.csv")
    if len(compare) != 59:
        raise SystemExit(f"expected 59 Compare rows, got {len(compare)}")

    model_index = {row["model_key"]: row for row in read_csv(tables / "model_index.csv") if row.get("model_key")}
    summary_indexes = {
        metric_key: sheet_index(read_csv(tables / sheet_name))
        for metric_key, (sheet_name, _average_column) in METRIC_SHEETS.items()
    }

    join_fallbacks: list[str] = []
    pairs = []
    for row in compare:
        model_key = row["model_key"]
        index_row = model_index.get(model_key, {})
        raw_names = [row.get("raw_name", ""), index_row.get("raw_display", "")]
        rem2_names = [row.get("rem2_name", ""), index_row.get("rem2_display", "")]
        metrics = {}
        properties_by_metric = {}
        for metric_key, (_sheet_name, average_column) in METRIC_SHEETS.items():
            prefix = COMPARE_METRICS[metric_key]
            raw = lookup_row(summary_indexes[metric_key], "Raw", raw_names, model_key, metric_key)
            rem2 = lookup_row(summary_indexes[metric_key], "REM2", rem2_names, model_key, metric_key)
            if raw["Model_name"] != row.get("raw_name") or rem2["Model_name"] != row.get("rem2_name"):
                join_fallbacks.append(
                    f"{model_key}/{metric_key}: Compare ({row.get('raw_name')!r}, {row.get('rem2_name')!r}) "
                    f"-> sheets ({raw['Model_name']!r}, {rem2['Model_name']!r})"
                )
            properties = {
                "overall": score_triple(float(raw[average_column]), float(rem2[average_column])),
                **{
                    key: score_triple(float(raw[column]), float(rem2[column]))
                    for key, column in PROPERTY_COLUMNS.items()
                },
            }
            compare_triple = score_triple(
                float(row[f"{prefix}_raw"]),
                float(row[f"{prefix}_rem2"]),
            )
            if compare_triple != properties["overall"]:
                raise SystemExit(
                    f"{metric_key} overall mismatch for {model_key}: "
                    f"Compare {compare_triple} != sheet {properties['overall']}"
                )
            metrics[metric_key] = compare_triple
            properties_by_metric[metric_key] = properties
        pairs.append(
            {
                "key": model_key,
                "base_name": row["raw_name"],
                "rem2_name": row["rem2_name"],
                "notes": row.get("category") or index_row.get("category") or "",
                "metrics": metrics,
                "properties": properties_by_metric["spearman"],
                "properties_by_metric": properties_by_metric,
            }
        )

    snapshot = {
        "id": "venusvirohub",
        "benchmark": "VenusViroHub",
        "label": "VenusViroHub",
        "title": "VenusViroHub substitutions",
        "status": "ready",
        "assays": 89,
        "n": 89,
        "source": "VenusViroHub Feishu leaderboard · 89 viral DMS assays",
        "description": (
            "89 viral DMS substitution assays with zero ProteinGym overlap, "
            "covering immune escape, cell entry, and receptor binding."
        ),
        "metrics": [
            {"id": "spearman", "label": "Spearman"},
            {"id": "ndcg", "label": "NDCG"},
            {"id": "auc", "label": "AUC"},
            {"id": "mcc", "label": "MCC"},
            {"id": "top_recall", "label": "Top recall"},
        ],
        "properties": [
            {"id": prop_id, "label": label, "n": count} for prop_id, label, count in PROPERTY_META
        ],
        "pairs": pairs,
    }
    return snapshot, join_fallbacks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()

    snapshot, join_fallbacks = build_snapshot(args.tables)
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    top = snapshot["pairs"][0]
    venus = next(pair for pair in snapshot["pairs"] if pair["key"] == "prosst_ensemble")
    print(
        f"wrote {len(snapshot['pairs'])} pairs to {args.snapshot} "
        f"(top {top['key']} Spearman rem2={top['metrics']['spearman']['rem2']}, "
        f"prosst_ensemble rem2={venus['metrics']['spearman']['rem2']})"
    )
    if join_fallbacks:
        print(f"name-join fallbacks: {len(join_fallbacks)}")
        for item in join_fallbacks:
            print(f"  {item}")
    else:
        print("name joins: Compare raw_name/rem2_name matched sheet Model_name for all 59 pairs")


if __name__ == "__main__":
    main()
