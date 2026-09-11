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
        from vrh import cli_run

        return getattr(cli_run, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")


def main(argv: Optional[Iterable[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        from vrh.user_commands import GETTING_STARTED

        print(GETTING_STARTED, end="")
        return
    head = argv[0]
    if head == "doctor":
        from vrh.user_commands import run_doctor

        code = run_doctor(argv[1:])
        if code:
            raise SystemExit(code)
        return
    if head == "dashboard":
        from vrh.dashboard.server import run_dashboard

        code = run_dashboard(argv[1:])
        if code:
            raise SystemExit(code)
        return
    if head in {"download", "download-proteingym"}:
        from vrh.data.download import run_download

        code = run_download(argv[1:])
        if code:
            raise SystemExit(code)
        return
    if any(item == "-h" or item == "--help" for item in argv):
        from vrh.config import create_parser

        create_parser().parse_args(argv)
        return
    if "--version" in argv:
        from vrh.config import create_parser

        create_parser().parse_args(argv)
        return
    if "--list-models" in argv:
        from vrh.user_commands import print_model_table

        print_model_table()
        return

    from vrh.status import working

    if head == "demo":
        from vrh.user_commands import build_demo_argv

        sys.stderr.write("vrh demo ...\n")
        sys.stderr.flush()
        argv = build_demo_argv(argv[1:])
        with working("loading vrh"):
            from vrh.cli_run import run_score

        return run_score(argv)

    with working("loading vrh"):
        from vrh.cli_run import run_score

    return run_score(argv)
