"""VenusMutHub dashboard snapshot from paper Fig.4 + category tex."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from vrh.dashboard.leaderboard import proteingym_catalog

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "VenusMutHub" / "assay_manifest.csv"
FIG4 = REPO / "docs" / "figure" / "data" / "fig4_venusmuthub_multimetric.csv"
TEX = REPO / "docs" / "0overleaf" / "tables" / "table_leaderboard_vmh_category.tex"
SNAPSHOT = REPO / "vrh" / "dashboard" / "data" / "venusmuthub_vrh.json"
PROPERTY_IDS = (
    "overall",
    "stability",
    "activity",
    "ppi_binding",
    "selectivity",
    "dti_binding",
)
METRICS = ("spearman", "ndcg", "auc", "mcc", "top_recall")


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


def _pairs() -> dict[str, dict]:
    return {row["key"]: row for row in _muthub()["pairs"]}


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
    assert SNAPSHOT.is_file()
    assert muthub["status"] == "ready"
    assert len(muthub["pairs"]) == 59
    assert all(row.get("metrics") for row in muthub["pairs"])
    assert all(len(row["metrics"]) == 5 for row in muthub["pairs"])


def test_venusmuthub_is_not_planned() -> None:
    catalog = proteingym_catalog()
    muthub = _muthub()
    assert muthub["status"] != "planned"
    assert "VenusMutHub" not in catalog["planned_benchmarks"]


def test_venusmuthub_catalog_fields() -> None:
    muthub = _muthub()
    assert muthub["manifest"] == "data/VenusMutHub/assay_manifest.csv"
    assert "fig4_venusmuthub_multimetric.csv" in muthub["source"]
    assert "table_leaderboard_vmh_category.tex" in muthub["source"]
    assert muthub["n_mutants"] == 27846
    assert muthub["median_seq_len"] == 226
    assert muthub.get("paired_score_table")
    assert muthub["setting"] == "Zero-shot · substitutions"


def test_venusmuthub_paper_spot_checks() -> None:
    pairs = _pairs()
    assert "prosst_ensemble" in pairs
    assert "proteinmpnn" in pairs
    rem2 = pairs["prosst_ensemble"]
    mpnn = pairs["proteinmpnn"]
    assert rem2["enhanced"] == 0.258
    assert rem2["base"] == 0.190
    assert rem2["metrics"]["spearman"]["vrh"] == 0.258
    assert rem2["properties"]["stability"]["vrh"] == 0.351
    assert rem2["properties"]["activity"]["vrh"] == 0.122
    assert mpnn["enhanced"] == 0.271
    assert mpnn["base"] == 0.221
    assert mpnn["properties"]["stability"]["vrh"] == 0.399
    assert mpnn["properties"]["overall"]["vrh"] == 0.271
    assert mpnn["metrics"]["ndcg"]["vrh"] == 0.850
    assert mpnn["inputs"] == ["str"]
    assert rem2["inputs"] == ["seq", "str"]


def test_venusmuthub_non_spearman_is_overall_only() -> None:
    for row in _muthub()["pairs"]:
        by_metric = row["properties_by_metric"]
        assert set(by_metric["spearman"]) == set(PROPERTY_IDS)
        for metric in METRICS:
            if metric == "spearman":
                continue
            assert list(by_metric[metric]) == ["overall"]


def test_venusmuthub_keys_match_proteingym() -> None:
    catalog = proteingym_catalog()
    pg = next(item for item in catalog["benchmarks"] if item["id"] == "proteingym")
    mut = _muthub()
    assert {row["key"] for row in mut["pairs"]} == {row["key"] for row in pg["pairs"]}
    assert "prosst_ensemble" in {row["key"] for row in mut["pairs"]}
    assert "esmif" in {row["key"] for row in mut["pairs"]}
    assert "esm1b_mask" in {row["key"] for row in mut["pairs"]}


def test_venusmuthub_snapshot_matches_fig4_ndcg() -> None:
    with FIG4.open(newline="", encoding="utf-8") as handle:
        fig4 = next(row for row in csv.DictReader(handle) if row["Model"].startswith("ProteinMPNN (v_48_020)"))
    mpnn = _pairs()["proteinmpnn"]
    assert mpnn["metrics"]["ndcg"]["base"] == round(float(fig4["NDCG_Raw"]), 3)
    assert mpnn["metrics"]["ndcg"]["vrh"] == round(float(fig4["NDCG_REM2"]), 3)
    assert TEX.is_file()
    packaged = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert packaged["status"] == "ready"
    assert len(packaged["pairs"]) == 59
