"""Bundled demo datasets shipped in the rem2 wheel."""

from __future__ import annotations

from pathlib import Path

DEMO_ASSAY = "HCP_LAMBD_Tsuboyama_2023_2L6Q"


def bundled_demo_dir() -> Path:
    return Path(__file__).resolve().parent / DEMO_ASSAY
