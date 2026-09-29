#!/usr/bin/env python3
"""Package VenusMutHub Raw/+VRH pairs from the paper artifacts.

Sources (no invented numbers):
  - docs/figure/data/fig4_venusmuthub_multimetric.csv  (5 metrics, overall)
  - docs/0overleaf/tables/table_leaderboard_vmh_category.tex  (Spearman by task)
  - data/VenusMutHub/assay_manifest.csv  (assay counts)
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from vrh.dashboard.leaderboard import _benchmark_inputs


REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "docs" / "figure" / "data" / "fig4_venusmuthub_multimetric.csv"
TEX_PATH = REPO / "docs" / "0overleaf" / "tables" / "table_leaderboard_vmh_category.tex"
MANIFEST = REPO / "data" / "VenusMutHub" / "assay_manifest.csv"
PG_SNAPSHOT = REPO / "vrh" / "dashboard" / "data" / "proteingym_rem2.json"
OUT_PATH = REPO / "vrh" / "dashboard" / "data" / "venusmuthub_vrh.json"

METRICS = [
    {"id": "spearman", "label": "Spearman"},
    {"id": "ndcg", "label": "NDCG"},
    {"id": "auc", "label": "AUC"},
    {"id": "mcc", "label": "MCC"},
    {"id": "top_recall", "label": "Top recall"},
]
CSV_METRICS = (
    ("spearman", "Spearman"),
    ("ndcg", "NDCG"),
    ("auc", "AUC"),
    ("mcc", "MCC"),
    ("top_recall", "Top_recall"),
)
TASKS = (
    "overall",
    "stability",
    "activity",
    "ppi_binding",
    "selectivity",
    "dti_binding",
)
TASK_LABELS = {
    "overall": "Overall",
    "stability": "Stability",
    "activity": "Activity",
    "ppi_binding": "PPI binding",
    "selectivity": "Selectivity",
    "dti_binding": "DTI binding",
}
NAME_ALIASES = {
    "venusrem2": "prosst-ensemble k=all",
}


def _round3(value: float) -> float:
    return round(float(value), 3)


def _triple(base: float, vrh: float) -> dict[str, float]:
    return {"base": _round3(base), "vrh": _round3(vrh), "delta": _round3(vrh - base)}


def _norm_name(name: str) -> str:
    text = name.replace(r"\_", "_").replace(",", " ").replace("(", " ").replace(")", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return NAME_ALIASES.get(text, text)


def _slug(name: str) -> str:
    text = _norm_name(name).replace(" ", "_").replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return re.sub(r"_+", "_", text).strip("_")


def proteingym_name_keys() -> dict[str, str]:
    raw = json.loads(PG_SNAPSHOT.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for pair in raw["pairs"]:
        key = pair["key"]
        for label in (pair.get("base_name"), pair.get("family"), pair.get("key")):
            if label:
                mapping[_norm_name(str(label))] = key
    mapping["venusrem2"] = "prosst_ensemble"
    mapping["prosst-ensemble k=all"] = "prosst_ensemble"
    return mapping


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_tex_rows(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    row_re = re.compile(
        r"& (?P<name>.+?) & (?P<readout>\S+) & (?P<flag>\\checkmark|\$\\times\$) & "
        r"(?P<vals>.+?) \\\\"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = row_re.search(line)
        if not match:
            continue
        name = re.sub(r"\\textbf\{([^}]*)\}", r"\1", match.group("name")).strip()
        name = name.replace(r"\_", "_")
        side = "vrh" if match.group("flag") == r"\checkmark" else "raw"
        cells = []
        for cell in match.group("vals").split("&"):
            number = re.search(r"-?\d+\.\d+", cell)
            if number:
                cells.append(float(number.group(0)))
        if len(cells) != 6:
            raise SystemExit(f"expected 6 task scores, got {cells} for {name}")
        bucket = grouped.setdefault(_norm_name(name), {"name": name})
        bucket[side] = dict(zip(TASKS, cells))
    return grouped


def manifest_stats(path: Path) -> tuple[int, dict[str, int], int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts: Counter[str] = Counter()
    mutants: list[int] = []
    lengths: list[int] = []
    for row in rows:
        task = (row.get("task") or "").strip()
        if task:
            counts[task] += 1
        raw_n = (row.get("n_mutants") or row.get("n_variants") or "").strip()
        if raw_n:
            mutants.append(int(float(raw_n)))
        raw_len = (row.get("seq_len") or row.get("length") or "").strip()
        if raw_len:
            lengths.append(int(float(raw_len)))
    return (
        len(rows),
        dict(counts),
        sum(mutants),
        int(statistics.median(lengths)) if lengths else 0,
    )


def build() -> dict[str, object]:
    name_keys = proteingym_name_keys()
    csv_rows = read_csv_rows(CSV_PATH)
    tex = parse_tex_rows(TEX_PATH)
    n_assays, task_counts, n_mutants, median_len = manifest_stats(MANIFEST)

    pairs = []
    for row in csv_rows:
        display = row["Model"].strip()
        norm = _norm_name(display)
        key = name_keys.get(norm) or _slug(display)
        metrics = {}
        for metric_id, column in CSV_METRICS:
            metrics[metric_id] = _triple(float(row[f"{column}_Raw"]), float(row[f"{column}_REM2"]))
        properties = {"overall": dict(metrics["spearman"])}
        tex_row = tex.get(norm)
        if tex_row and "raw" in tex_row and "vrh" in tex_row:
            for task in TASKS:
                properties[task] = _triple(tex_row["raw"][task], tex_row["vrh"][task])
            # Paper leaderboard Spearman comes from the category table, not Fig.4 rounding.
            metrics["spearman"] = dict(properties["overall"])
        display_name = "ProSST-Ensemble (K=all)" if key == "prosst_ensemble" else display
        pairs.append(
            {
                "key": key,
                "family": display_name,
                "base_name": display_name,
                "enhanced_name": f"{display_name}, vrh",
                "vrh_name": f"{display_name}, vrh",
                "base": metrics["spearman"]["base"],
                "enhanced": metrics["spearman"]["vrh"],
                "delta": metrics["spearman"]["delta"],
                "notes": "Paper Fig.4 MutHub multimetric + Table leaderboard_vmh_category",
                "metrics": metrics,
                "properties": properties,
                "properties_by_metric": {
                    "spearman": {task: dict(values) for task, values in properties.items()},
                    **{
                        metric_id: {"overall": dict(values)}
                        for metric_id, values in metrics.items()
                        if metric_id != "spearman"
                    },
                },
                "inputs": _benchmark_inputs(key),
            }
        )

    properties = [{"id": "overall", "label": "Overall", "n": n_assays}]
    for task_id in TASKS[1:]:
        properties.append({"id": task_id, "label": TASK_LABELS[task_id], "n": task_counts[task_id]})

    return {
        "id": "venusmuthub",
        "benchmark": "VenusMutHub",
        "label": "VenusMutHub",
        "title": "VenusMutHub substitutions",
        "status": "ready",
        "assays": n_assays,
        "n": n_assays,
        "source": "assay_manifest.csv; fig4_venusmuthub_multimetric.csv; table_leaderboard_vmh_category.tex",
        "setting": "Zero-shot · substitutions",
        "description": (
            "Same-backbone comparison of raw model scores and the full vrh recipe "
            "on 905 VenusMutHub substitution assays."
        ),
        "metrics": METRICS,
        "properties": properties,
        "pairs": pairs,
        "n_mutants": n_mutants,
        "median_seq_len": median_len,
        "manifest": "data/VenusMutHub/assay_manifest.csv",
        "paired_score_table": "docs/figure/data/fig4_venusmuthub_multimetric.csv",
    }


def main() -> None:
    payload = build()
    missing_tasks = [
        pair["base_name"]
        for pair in payload["pairs"]
        if set(pair["properties"]) != set(TASKS)
    ]
    if missing_tasks:
        raise SystemExit(f"missing task scores for: {missing_tasks}")
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['pairs'])} pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
