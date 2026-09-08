#!/usr/bin/env python3
"""Fetch ProteinGym into data/proteingym_v1. Same as ``rem2 download``."""

from rem2.data.proteingym import run_download


if __name__ == "__main__":
    raise SystemExit(run_download())
