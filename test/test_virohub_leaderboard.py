"""VenusViroHub product-benchmark snapshot loaded by the vrh dashboard."""

from __future__ import annotations

from vrh.dashboard.leaderboard import proteingym_catalog


METRIC_IDS = ("spearman", "ndcg", "auc", "mcc", "top_recall")
PROPERTY_IDS = (
    "overall",
    "activity",
    "binding",
    "cell_entry",
    "expression",
    "fitness",
    "immune_escape",
    "stability",
)


def _virohub() -> dict:
    catalog = proteingym_catalog()
    return next(item for item in catalog["benchmarks"] if item["id"] == "venusvirohub")


def test_virohub_catalog_ready():
    catalog = proteingym_catalog()
    virohub = _virohub()
    assert virohub["status"] == "ready"
    assert virohub["n"] == 89
    assert len(virohub["pairs"]) == 71
    assert "VenusViroHub" not in catalog["planned_benchmarks"]


def test_virohub_public_recipe_labels_hide_rem2_suffix():
    from vrh.dashboard.leaderboard import _public_recipe_label

    assert _public_recipe_label("ESM-IF (REM2)") == "ESM-IF (VRH)"
    assert _public_recipe_label("SaProt (650M_PDB, mask, REM2)") == "SaProt (650M_PDB, mask, VRH)"
    assert _public_recipe_label("VenusREM2") == "VenusREM2"
    for pair in _virohub()["pairs"]:
        name = pair.get("enhanced_name") or ""
        assert "REM2)" not in name
        assert "rem2)" not in name
        assert "VenusVRH2" not in name


def test_virohub_pair_keys_and_spearman():
    pairs = {row["key"]: row for row in _virohub()["pairs"]}
    assert "prosst_ensemble" in pairs
    assert "saprot650m_pdb_mask" in pairs
    assert pairs["prosst_ensemble"]["enhanced"] == 0.297
    assert pairs["saprot650m_pdb_mask"]["enhanced"] == 0.320


def test_virohub_properties_and_metric_grid():
    virohub = _virohub()
    property_ids = [item["id"] for item in virohub["properties"]]
    assert "cell_entry" in property_ids
    assert "immune_escape" in property_ids
    for pair in virohub["pairs"]:
        assert set(pair["metrics"]) == set(METRIC_IDS)
        assert set(pair["properties_by_metric"]) == set(METRIC_IDS)
        assert set(pair["properties_by_metric"]["spearman"]) == set(PROPERTY_IDS)
        for metric_id in METRIC_IDS:
            if metric_id != "spearman":
                assert list(pair["properties_by_metric"][metric_id]) == ["overall"]
            for values in pair["properties_by_metric"][metric_id].values():
                assert values["delta"] == round(values["vrh"] - values["base"], 3)
        for values in pair["metrics"].values():
            assert values["delta"] == round(values["vrh"] - values["base"], 3)
