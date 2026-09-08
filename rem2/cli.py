"""CLI entry. Help / doctor / download stay off the torch import path."""

from __future__ import annotations

import sys
from typing import Iterable, Optional

_LAZY_EXPORTS = {
    "_crystal_plddt_skip_paths",
    "_is_single_protein",
    "_numeric_series",
    "_prepare_single_protein",
    "_print_model_table",
    "_warn_crystal_no_plddt",
    "_zmean_columns",
    "finite_spearman",
    "run_score",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        from rem2 import cli_run

        return getattr(cli_run, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")


def main(argv: Optional[Iterable[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        from rem2.user_commands import GETTING_STARTED

        print(GETTING_STARTED, end="")
        return
    head = argv[0]
    if head == "doctor":
        from rem2.user_commands import run_doctor

        code = run_doctor(argv[1:])
        if code:
            raise SystemExit(code)
        return
    if head in {"download", "download-proteingym"}:
        from rem2.data.proteingym import run_download

        code = run_download(argv[1:])
        if code:
            raise SystemExit(code)
        return
    if any(item == "-h" or item == "--help" for item in argv):
        from rem2.config import create_parser

        create_parser().parse_args(argv)
        return
    if "--version" in argv:
        from rem2.config import create_parser

        create_parser().parse_args(argv)
        return
    if "--list-models" in argv:
        from rem2.user_commands import print_model_table

        print_model_table()
        return

    from rem2.cli_run import run_score

    return run_score(argv)
