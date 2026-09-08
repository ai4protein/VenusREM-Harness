"""Unzipped single-assay example on the VenusREM2 Hugging Face dataset.

``rem2 demo`` fetches these loose files (DMS / PDB / MSA / FASTA) into
``~/.cache/rem2/examples/<assay>`` with one progress bar per file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from rem2.data.mirrors import VENUSREM2_REPOS, download_from_venusrem2
from rem2.download.progress import format_bytes, print_plan

# ProteinGym substitution assay (Tsuboyama 2023, PDB 2L6Q, 55 aa).
EXAMPLE_NAME = "HCP_LAMBD_Tsuboyama_2023_2L6Q"
EXAMPLE_PREFIX = f"example/{EXAMPLE_NAME}"
EXAMPLE_FILES = (
    f"{EXAMPLE_PREFIX}/aa_seq/{EXAMPLE_NAME}.fasta",
    f"{EXAMPLE_PREFIX}/substitutions/{EXAMPLE_NAME}.csv",
    f"{EXAMPLE_PREFIX}/pdbs/{EXAMPLE_NAME}.pdb",
    f"{EXAMPLE_PREFIX}/aa_seq_aln_a2m/{EXAMPLE_NAME}.a2m",
)
EXAMPLE_ALIASES = frozenset(
    {
        "example",
        "examples",
        "demo",
        "hcp",
        "hcplambd",
        "proteingymexample",
        "hcp_lambd_tsuboyama_2023_2l6q",
        "hcplambdtsuboyama20232l6q",
    }
)


def default_example_dir(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(os.path.expanduser(explicit))
    env = os.environ.get("REM2_CACHE") or os.environ.get("VENUSREM2_CACHE")
    if env:
        root = Path(os.path.expanduser(env))
        if root.name == "weights":
            root = root.parent
    else:
        root = Path.home() / ".cache" / "rem2"
    return root / "examples" / EXAMPLE_NAME


def _rel_from_remote(remote: str) -> str:
    prefix = EXAMPLE_PREFIX.rstrip("/") + "/"
    if remote.startswith(prefix):
        return remote[len(prefix) :]
    return Path(remote).name


def _example_complete(dest: Path) -> bool:
    dest = Path(dest)
    return all(
        (dest / _rel_from_remote(name)).is_file()
        and (dest / _rel_from_remote(name)).stat().st_size > 0
        for name in EXAMPLE_FILES
    )


def example_plan_rows(dest: Path) -> list[tuple[str, str, str]]:
    rows = []
    for remote in EXAMPLE_FILES:
        local = dest / _rel_from_remote(remote)
        if local.is_file() and local.stat().st_size > 0:
            size = format_bytes(local.stat().st_size)
        else:
            size = "loose file"
        rows.append((Path(remote).name, size, remote))
    return rows


def ensure_demo_dataset(
    dest: Optional[str] = None,
    *,
    force: bool = False,
    dry_run: bool = False,
    log=print,
) -> Path:
    """Download the ProteinGym demo assay from Hugging Face, or reuse cache."""
    out = default_example_dir(dest)
    print_plan(
        f"Demo example {EXAMPLE_NAME} → {out}",
        example_plan_rows(out),
        log=log,
    )
    log(f"VenusREM2 mirrors: {' then '.join(VENUSREM2_REPOS)}")
    if dry_run:
        return out
    if (not force) and _example_complete(out):
        log(f"Using cached demo: {out}")
        return out

    missing = [
        remote
        for remote in EXAMPLE_FILES
        if force
        or not (
            (out / _rel_from_remote(remote)).is_file()
            and (out / _rel_from_remote(remote)).stat().st_size > 0
        )
    ]
    failed: list[str] = []
    for i, remote in enumerate(missing, 1):
        local = out / _rel_from_remote(remote)
        log(f"[{i}/{len(missing)}] {remote}")
        got = download_from_venusrem2(
            remote,
            local,
            force=force,
            log=log,
            progress=True,
            desc=Path(remote).name,
        )
        if got is None:
            failed.append(remote)

    if _example_complete(out):
        log(f"Demo ready: {out}")
        return out
    raise SystemExit(
        "Could not download the rem2 demo example from "
        f"{' or '.join(VENUSREM2_REPOS)} ({EXAMPLE_PREFIX}/). "
        f"Missing: {', '.join(failed) or 'download failed'}. "
        "Private repos need a token: export HF_TOKEN=... "
        "or run `hf auth login` (saved at ~/.cache/huggingface/token)."
    )
