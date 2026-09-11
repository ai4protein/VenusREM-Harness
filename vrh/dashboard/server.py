"""Launch ``vrh dashboard`` (uvicorn + FastAPI)."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from vrh.dashboard import DASHBOARD_DEFAULT_HOST, DASHBOARD_DEFAULT_PORT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vrh dashboard",
        description="Local vrh predict / select console (preview).",
    )
    parser.add_argument("--host", default=DASHBOARD_DEFAULT_HOST, help="bind address (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=DASHBOARD_DEFAULT_PORT,
        help=f"port (default: {DASHBOARD_DEFAULT_PORT})",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="run store directory (default: ~/.cache/vrh/dashboard)",
    )
    return parser


def run_dashboard(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv or []))
    try:
        import uvicorn
    except ImportError:
        print(
            "vrh dashboard needs FastAPI / uvicorn. Install with:\n"
            "  pip install 'vrh[dashboard]'",
            file=sys.stderr,
        )
        return 1
    from vrh.dashboard.app import create_app
    from vrh.dashboard.store import RunStore

    store = RunStore(args.root)
    app = create_app(store.root)
    print(f"vrh dashboard  http://{args.host}:{args.port}")
    print(f"runs            {store.runs_dir}")
    print("Prediction scores rank variants; higher is better. They are not physical ΔΔG values.")
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    return 0
