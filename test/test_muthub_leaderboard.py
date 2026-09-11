"""VenusMutHub dashboard catalog.

No packaged 59-model Raw/+REM2 ``Leaderboard_Compare.csv`` matching the
ProteinGym/ViroHub schema (Spearman / NDCG / AUC / MCC / Top_recall) was
found. The older Orbit hub
``experiments/hubs/venusmuthub_raw_vs_orbit/tables/Leaderboard_Compare.csv``
is Spearman / NDCG / Accuracy / F1 and is not used as dashboard scores.
"""

from __future__ import annotations

import csv
from pathlib import Path

from rem2.dashboard.leaderboard import proteingym_catalog

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "VenusMutHub" / "assay_manifest.csv"
PROPERTY_IDS = (
    "overall",
    "stability",
    "activity",
    "ppi_binding",
    "selectivity",
    "dti_binding",
)


def _manifest_task_counts() -> dict[str, int]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[str, int] = {}
    for row in rows:
        task = (row.get("task") or "").strip()
        if task:
            counts[task] = counts.get(task, 0) + 1
    counts["overall"] = len(rows)
    return counts


def _muthub() -> dict:
    catalog = proteingym_catalog()
    return next(item for item in catalog["benchmarks"] if item["id"] == "venusmuthub")


def test_venusmuthub_benchmark_exists() -> None:
    catalog = proteingym_catalog()
    ids = [item["id"] for item in catalog["benchmarks"]]
    assert "venusmuthub" in ids
    muthub = _muthub()
    assert muthub["n"] == 905
    assert muthub["n"] == _manifest_task_counts()["overall"]


def test_venusmuthub_properties_match_manifest() -> None:
    muthub = _muthub()
    properties = {item["id"]: item["n"] for item in muthub["properties"]}
    assert tuple(item["id"] for item in muthub["properties"]) == PROPERTY_IDS
    expected = _manifest_task_counts()
    for property_id in PROPERTY_IDS:
        assert properties[property_id] == expected[property_id]


def test_venusmuthub_score_status() -> None:
    muthub = _muthub()
    rem2_path = REPO / "rem2" / "dashboard" / "data" / "venusmuthub_rem2.json"
    if rem2_path.is_file():
        assert muthub["status"] == "ready"
        assert muthub["pairs"]
        assert all(row.get("metrics") for row in muthub["pairs"])
        return
    assert muthub["status"] == "catalog"
    assert muthub["pairs"] == []


def test_venusmuthub_is_not_planned() -> None:
    catalog = proteingym_catalog()
    muthub = _muthub()
    assert muthub["status"] != "planned"
    assert "VenusMutHub" not in catalog["planned_benchmarks"]
