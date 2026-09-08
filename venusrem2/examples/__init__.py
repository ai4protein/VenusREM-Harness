"""Bundled demo datasets."""

from __future__ import annotations

from pathlib import Path


def bundled_trp_cage_dir() -> Path:
    return Path(__file__).resolve().parent / "trp_cage"
