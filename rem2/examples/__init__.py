"""Bundled demo datasets."""

from __future__ import annotations

from pathlib import Path

# Official ProteinGym substitution assay (Tsuboyama 2023, PDB 1PV0, 44 aa).
DEMO_ASSAY = "SDA_BACSU_Tsuboyama_2023_1PV0"


def bundled_demo_dir() -> Path:
    return Path(__file__).resolve().parent / DEMO_ASSAY
