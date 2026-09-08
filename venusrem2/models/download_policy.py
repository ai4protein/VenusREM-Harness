"""Download consent for missing checkpoints.

Cache is always tried first. If the file is absent:

- ``--auto_download`` / ``VENUSREM2_AUTO_DOWNLOAD=1`` → download
- ``--no_auto_download`` / ``VENUSREM2_AUTO_DOWNLOAD=0`` → refuse
- interactive TTY → ask ``Download … into cache? [Y/n]`` (Enter = download)
- non-interactive without a flag → download
"""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Optional

_POLICY = "ask"  # yes | no | ask


class DownloadRefused(FileNotFoundError):
    """User declined, or downloads are disabled / non-interactive."""


def get_download_policy() -> str:
    return _POLICY


def set_download_policy(policy: str) -> str:
    global _POLICY
    value = str(policy).strip().lower()
    if value not in {"yes", "no", "ask"}:
        raise ValueError(f"download policy must be yes/no/ask, got {policy!r}")
    _POLICY = value
    return _POLICY


def policy_from_env() -> Optional[str]:
    raw = os.environ.get("VENUSREM2_AUTO_DOWNLOAD", "").strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return "yes"
    if raw in {"0", "false", "no", "n"}:
        return "no"
    return None


def apply_download_policy_from_args(args: Any) -> str:
    flag = getattr(args, "auto_download", None)
    if flag is True:
        return set_download_policy("yes")
    if flag is False:
        return set_download_policy("no")
    env = policy_from_env()
    if env:
        return set_download_policy(env)
    return set_download_policy("ask")


def format_missing_weight(
    name: str,
    dest: str,
    source: str,
    looked_in: Optional[Iterable[str]] = None,
) -> str:
    lines = [f"{name} not found."]
    seen: list[str] = []
    for path in list(looked_in or []) + [dest]:
        if path and path not in seen:
            seen.append(path)
    if seen:
        lines.append("  looked in:")
        for path in seen:
            lines.append(f"    {path}")
    lines.append(f"  source: {source}")
    lines.append(
        "Place the file in the cache (or pass --cache_dir / a --*-checkpoint path). "
        "Pass --no_auto_download to stay offline."
    )
    return "\n".join(lines)


def confirm_download(
    *,
    name: str,
    dest: str,
    source: str,
    looked_in: Optional[Iterable[str]] = None,
    logger: Any = None,
) -> None:
    """Return if download is allowed; raise ``DownloadRefused`` otherwise."""
    policy = get_download_policy()
    hint = format_missing_weight(name, dest, source, looked_in)
    if policy == "yes":
        if logger is not None:
            logger.info(f"Downloading {name}: {source} -> {dest}")
        else:
            print(f"Downloading {name}: {source} -> {dest}")
        return
    if policy == "no":
        raise DownloadRefused(hint + "\nDownload disabled (--no_auto_download / VENUSREM2_AUTO_DOWNLOAD=0).")
    if not sys.stdin.isatty():
        if logger is not None:
            logger.info(f"Downloading {name}: {source} -> {dest}")
        else:
            print(f"Downloading {name}: {source} -> {dest}")
        return
    print(hint, file=sys.stderr)
    try:
        answer = input(f"Download {name} into cache? [Y/n] ").strip().lower()
    except EOFError:
        answer = ""
    if answer in {"", "y", "yes"}:
        if logger is not None:
            logger.info(f"Downloading {name}: {source} -> {dest}")
        return
    raise DownloadRefused(hint + "\nDownload declined.")
