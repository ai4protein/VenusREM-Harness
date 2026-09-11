#!/usr/bin/env python3
"""Build the VenusMutHub dashboard snapshot from the assay manifest.

A ProteinGym/ViroHub-style packaged Raw/+VRH ``Leaderboard_Compare.csv``
(Spearman / NDCG / AUC / MCC / Top_recall, ``*_raw`` + ``*_vrh`` columns)
was not found. An older Orbit hub at
``experiments/hubs/venusmuthub_raw_vs_orbit/tables/Leaderboard_Compare.csv``
uses Spearman / NDCG / Accuracy / F1 and is not treated as dashboard scores.

This script therefore emits a catalog-only snapshot. Counts are derived from
``data/VenusMutHub/assay_manifest.csv``; model pairs are never invented.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "data" / "VenusMutHub" / "assay_manifest.csv"
DASHBOARD_DATA = REPO / "vrh" / "dashboard" / "data"
CATALOG_PATH = DASHBOARD_DATA / "venusmuthub_catalog.json"
REM2_PATH = DASHBOARD_DATA / "venusmuthub_vrh.json"

METRICS = [
    {"id": "spearman", "label": "Spearman"},
    {"id": "ndcg", "label": "NDCG"},
    {"id": "auc", "label": "AUC"},
    {"id": "mcc", "label": "MCC"},
    {"id": "top_recall", "label": "Top recall"},
]
PROPERTY_LABELS = {
    "overall": "Overall",
    "stability": "Stability",
    "activity": "Activity",
    "ppi_binding": "PPI binding",
    "selectivity": "Selectivity",
    "dti_binding": "DTI binding",
}
PREFERRED_TASK_ORDER = (
    "stability",
    "activity",
    "ppi_binding",
    "selectivity",
    "dti_binding",
)
PG_COMPARE_METRICS = ("Spearman", "NDCG", "AUC", "MCC", "Top_recall")


def _int_field(row: dict[str, str], key: str) -> int | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    return int(float(raw))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def task_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        task = (row.get("task") or "").strip()
        if task:
            counts[task] += 1
    return dict(counts)


def property_entries(counts: dict[str, int], n_assays: int) -> list[dict[str, object]]:
    properties = [{"id": "overall", "label": PROPERTY_LABELS["overall"], "n": n_assays}]
    seen = set()
    for task_id in (*PREFERRED_TASK_ORDER, *sorted(counts)):
        if task_id in seen or task_id not in counts:
            continue
        seen.add(task_id)
        properties.append(
            {
                "id": task_id,
                "label": PROPERTY_LABELS.get(task_id, task_id.replace("_", " ")),
                "n": counts[task_id],
            }
        )
    return properties


def catalog_snapshot(rows: list[dict[str, str]], source: Path) -> dict[str, object]:
    counts = task_counts(rows)
    n_assays = len(rows)
    mutants = [_int_field(row, "n_mutants") for row in rows]
    lengths = [_int_field(row, "seq_len") for row in rows]
    mutant_values = [value for value in mutants if value is not None]
    length_values = [value for value in lengths if value is not None]
    properties = property_entries(counts, n_assays)
    task_labels = []
    for item in properties[1:]:
        label = str(item["label"])
        if label in {"PPI binding", "DTI binding"}:
            task_labels.append(label)
        else:
            task_labels.append(label[:1].lower() + label[1:] if label else label)
    if len(task_labels) == 1:
        task_phrase = task_labels[0]
    elif len(task_labels) == 2:
        task_phrase = " and ".join(task_labels)
    else:
        task_phrase = ", ".join(task_labels[:-1]) + f", and {task_labels[-1]}"
    description = (
        f"{n_assays} substitution assays across {task_phrase}. "
        "Paired Raw / +VRH model scores will appear after the evaluation "
        "snapshot is packaged."
    )
    return {
        "id": "venusmuthub",
        "benchmark": "VenusMutHub",
        "label": "VenusMutHub",
        "title": "VenusMutHub substitutions",
        "status": "catalog",
        "assays": n_assays,
        "n": n_assays,
        "source": "VenusMutHub assay_manifest.csv",
        "description": description,
        "metrics": METRICS,
        "properties": properties,
        "pairs": [],
        "task_counts": {item["id"]: item["n"] for item in properties[1:]},
        "n_mutants": sum(mutant_values),
        "median_seq_len": statistics.median(length_values) if length_values else None,
        "paired_score_table": None,
        "manifest": _relative_to_repo(source),
    }


def _header_map(fieldnames: list[str] | None) -> dict[str, str]:
    return {name.lower(): name for name in (fieldnames or [])}


def _has_pg_compare_schema(fieldnames: list[str] | None) -> bool:
    headers = _header_map(fieldnames)
    if "model_key" not in headers and "model" not in headers:
        return False
    for metric in PG_COMPARE_METRICS:
        raw = f"{metric}_raw".lower()
        recipe = f"{metric}_vrh".lower() if f"{metric}_vrh".lower() in headers else f"{metric}_rem2".lower()
        if raw not in headers or recipe not in headers:
            return False
    return True


def find_paired_compare(repo: Path) -> Path | None:
    """Return a packaged Raw/+VRH compare table, or None.

    Only a ProteinGym/ViroHub ``Leaderboard_Compare`` schema qualifies.
    Orbit-era Accuracy/F1 tables and figure caches are ignored.
    """
    candidates = [
        repo / "experiments" / "hubs" / "venusmuthub_raw_vs_vrh" / "tables" / "Leaderboard_Compare.csv",
        repo / "experiments" / "hubs" / "venusmuthub_raw_vs_rem2" / "tables" / "Leaderboard_Compare.csv",
        *sorted((repo / "experiments" / "hubs").glob("*/tables/Leaderboard_Compare.csv")),
        repo / "data" / "VenusMutHub" / "Leaderboard_Compare.csv",
    ]
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        if "venusmuthub" not in path.as_posix().lower() and "muthub" not in path.as_posix().lower():
            if path.parent.parent.name != "VenusMutHub":
                continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if _has_pg_compare_schema(reader.fieldnames):
                return path
    return None


def _metric_cell(row: dict[str, str], headers: dict[str, str], metric: str, side: str) -> float:
    key = f"{metric}_{side}".lower()
    if key not in headers and side == "vrh":
        key = f"{metric}_rem2".lower()
    return float(row[headers[key]])


def pairs_from_compare(path: Path) -> list[dict[str, object]]:
    pairs = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = _header_map(reader.fieldnames)
        for row in reader:
            key = (row.get(headers.get("model_key", ""), "") or row.get(headers.get("model", ""), "")).strip()
            if not key:
                continue
            metrics = {}
            for metric_id, column in (
                ("spearman", "Spearman"),
                ("ndcg", "NDCG"),
                ("auc", "AUC"),
                ("mcc", "MCC"),
                ("top_recall", "Top_recall"),
            ):
                base = _metric_cell(row, headers, column, "raw")
                vrh = _metric_cell(row, headers, column, "vrh")
                delta_key = f"{column}_delta".lower()
                if delta_key in headers and row.get(headers[delta_key], "").strip():
                    delta = float(row[headers[delta_key]])
                else:
                    delta = round(vrh - base, 3)
                metrics[metric_id] = {"base": base, "vrh": vrh, "delta": delta}
            base_name = (
                row.get(headers.get("raw_name", ""), "")
                or row.get(headers.get("display_name", ""), "")
                or key
            )
            vrh_name = (
                row.get(headers.get("vrh_name", ""), "")
                or row.get(headers.get("rem2_name", ""), "")
                or f"{base_name}, vrh"
            )
            pairs.append(
                {
                    "key": key,
                    "family": row.get(headers.get("category", ""), "")
                    or row.get(headers.get("family", ""), ""),
                    "base_name": base_name,
                    "vrh_name": vrh_name,
                    "notes": row.get(headers.get("notes", ""), ""),
                    "metrics": metrics,
                    "properties": {"overall": dict(metrics["spearman"])},
                    "properties_by_metric": {
                        metric_id: {"overall": dict(values)} for metric_id, values in metrics.items()
                    },
                }
            )
    return pairs


def vrh_snapshot(rows: list[dict[str, str]], compare: Path, source: Path) -> dict[str, object]:
    catalog = catalog_snapshot(rows, source)
    pairs = pairs_from_compare(compare)
    catalog["status"] = "ready"
    catalog["pairs"] = pairs
    catalog["source"] = _relative_to_repo(compare)
    catalog["paired_score_table"] = catalog["source"]
    catalog["description"] = (
        f"{len(rows)} substitution assays across stability, activity, PPI binding, "
        "selectivity, and DTI binding. Paired Raw / +VRH scores from the packaged "
        "evaluation snapshot."
    )
    return catalog


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if not rows:
        raise SystemExit(f"empty manifest: {args.manifest}")

    compare = find_paired_compare(REPO)
    if compare is not None:
        payload = vrh_snapshot(rows, compare, args.manifest)
        out = args.out or REM2_PATH
        write_json(out, payload)
        print(f"wrote ready snapshot ({len(payload['pairs'])} pairs) to {out}")
        print(f"paired table: {compare}")
        return

    payload = catalog_snapshot(rows, args.manifest)
    out = args.out or CATALOG_PATH
    write_json(out, payload)
    print(f"wrote catalog snapshot ({payload['n']} assays) to {out}")
    print("paired score table: none")


if __name__ == "__main__":
    main()
