"""Small Python API on top of the rem2 CLI."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import pandas as pd


def _flag(name: str, value) -> list[str]:
    if value is None or value is False:
        return []
    if value is True:
        return [name]
    return [name, str(value)]


def build_score_argv(
    *,
    fasta: Optional[str] = None,
    base_dir: Optional[str] = None,
    mutants: Optional[str] = None,
    pdb: Optional[str] = None,
    model: str = "esm2",
    out_dir: str = "result",
    extra_argv: Optional[Sequence[str]] = None,
) -> list[str]:
    if base_dir and fasta:
        raise ValueError("Pass fasta= or base_dir=, not both")
    if not fasta and not base_dir and not pdb:
        raise ValueError("Pass fasta=, pdb=, or base_dir=")
    argv = ["--model", model, "--out_scores_dir", out_dir]
    if base_dir:
        argv += ["--base_dir", base_dir]
    else:
        argv += _flag("--fasta", fasta)
        argv += _flag("--mutants", mutants)
        argv += _flag("--pdb", pdb)
    if extra_argv:
        argv.extend(list(extra_argv))
    return argv


def score(
    fasta: Optional[str] = None,
    mutants: Optional[str] = None,
    *,
    model: str = "esm2",
    pdb: Optional[str] = None,
    base_dir: Optional[str] = None,
    out_dir: str = "result",
    extra_argv: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Run rem2 and return the scores table (one protein) or the summary (dataset)."""
    from rem2.cli import main

    argv = build_score_argv(
        fasta=fasta,
        base_dir=base_dir,
        mutants=mutants,
        pdb=pdb,
        model=model,
        out_dir=out_dir,
        extra_argv=extra_argv,
    )
    main(argv)
    scores_dir = os.path.join(out_dir, "scores")
    if (fasta or pdb) and not base_dir and os.path.isdir(scores_dir):
        tables = sorted(
            os.path.join(scores_dir, name)
            for name in os.listdir(scores_dir)
            if name.endswith(".csv")
        )
        if tables:
            return pd.read_csv(tables[0])
    summary = os.path.join(out_dir, "summary_performance.csv")
    if os.path.exists(summary):
        return pd.read_csv(summary)
    raise FileNotFoundError(f"No rem2 scores written under {out_dir}")
