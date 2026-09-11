"""Progress bars and download plans for ``vrh download``."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

from tqdm import tqdm

_CHUNK = 1024 * 1024


def format_bytes(n: Optional[int]) -> str:
    if n is None or n < 0:
        return "?"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def format_hint(n: Optional[int]) -> str:
    if not n:
        return "?"
    return f"~{format_bytes(n)}"


def print_plan(title: str, rows: list[tuple[str, str, str]], log=print) -> None:
    """Print a table: index-free name / size / detail."""
    log(title)
    if not rows:
        return
    name_w = max(len(row[0]) for row in rows)
    size_w = max(len(row[1]) for row in rows)
    for i, (name, size, detail) in enumerate(rows, 1):
        extra = f"  {detail}" if detail else ""
        log(f"  [{i}/{len(rows)}]  {name:<{name_w}}  {size:>{size_w}}{extra}")


def tqdm_bar(desc: str, total: Optional[int], *, unit: str = "B"):
    kwargs = {
        "desc": desc[:48],
        "unit": unit,
        "dynamic_ncols": True,
        "leave": True,
        "mininterval": 0.2,
    }
    if unit == "B":
        kwargs.update(unit_scale=True, unit_divisor=1024)
    if total and total > 0:
        kwargs["total"] = total
    return tqdm(**kwargs)


def download_url_with_progress(
    url: str,
    dest: Path,
    *,
    desc: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    force: bool = False,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        with tqdm_bar(f"{label} (cached)", dest.stat().st_size) as bar:
            bar.update(dest.stat().st_size)
        return dest
    req = urllib.request.Request(url, headers=headers or {})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length") or 0)
            with tmp.open("wb") as handle, tqdm_bar(label, total or None) as bar:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def disable_hf_bars():
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        return True
    except Exception:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        return False


def enable_hf_bars():
    try:
        from huggingface_hub.utils import enable_progress_bars

        enable_progress_bars()
    except Exception:
        pass

