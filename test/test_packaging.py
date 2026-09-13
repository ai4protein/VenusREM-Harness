"""Keep research leftovers and README screenshots out of the pip artifact."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Standalone cluster / data-prep scripts. They live under script/ (pruned).
RESEARCH_LEFTOVERS = (
    "vrh/esmfold2.py",
    "vrh/single_config_monomer.txt",
    "vrh/data/plot_attention_map.py",
    "vrh/data/get_msa.py",
    "vrh/data/get_sav.py",
    "vrh/data/get_substitutions.py",
    "vrh/data/data_format_convert.py",
    "vrh/data/select_msa.py",
    "vrh/data/get_struc_aln.py",
    "vrh/data/get_struc_seq.py",
    "vrh/data/utils.py",
)


def test_research_scripts_are_outside_the_installable_package():
    for rel in RESEARCH_LEFTOVERS:
        assert not (REPO / rel).exists(), f"{rel} must not live under vrh/"


def test_manifest_prunes_non_install_trees():
    lines = {
        line.strip()
        for line in (REPO / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    }
    for name in ("img", "script", "test", "data", "docs"):
        assert f"prune {name}" in lines


def test_setuptools_does_not_collect_extra_trees():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'include = ["vrh*"]' in text
    assert "include-package-data = false" in text
