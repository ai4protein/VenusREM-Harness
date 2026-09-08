"""Hugging Face mirrors for rem2 download.

Try ``AI4Protein/VenusREM2`` first, then ``tyang816/VenusREM2``.
The first repo that has the file is used. Private repos need ``HF_TOKEN``.
"""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

VENUSREM2_REPOS = ("AI4Protein/VenusREM2", "tyang816/VenusREM2")
LEGACY_PROTEINGYM_REPO = "AI4Protein/VenusREM"


def hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def hf_headers() -> dict[str, str]:
    token = hf_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def hf_resolve_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"


def hf_file_available(repo: str, filename: str, timeout: float = 20) -> bool:
    try:
        from huggingface_hub import file_exists

        return bool(
            file_exists(repo_id=repo, filename=filename, repo_type="dataset")
        )
    except Exception:
        pass
    req = urllib.request.Request(
        hf_resolve_url(repo, filename), method="HEAD", headers=hf_headers()
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 400
    except urllib.error.HTTPError as exc:
        return exc.code == 200
    except Exception:
        return False


def download_hf_file(
    repo: str,
    filename: str,
    dest: Path,
    force: bool = False,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return _download_url(hf_resolve_url(repo, filename), dest, force=force)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest.parent),
        force_download=force,
    )
    downloaded = Path(path)
    if downloaded.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, dest)
    return dest


def _download_url(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=hf_headers())
    try:
        with urllib.request.urlopen(req) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def first_venusrem2_repo(
    filename: str,
    repos: tuple[str, ...] = VENUSREM2_REPOS,
    available: Callable[[str, str], bool] = hf_file_available,
) -> Optional[str]:
    for repo in repos:
        if available(repo, filename):
            return repo
    return None


def download_from_venusrem2(
    filename: str,
    dest: Path,
    force: bool = False,
    repos: tuple[str, ...] = VENUSREM2_REPOS,
    log=print,
) -> Optional[Path]:
    """Download ``filename`` from the first working VenusREM2 mirror."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    errors: list[str] = []
    for repo in repos:
        if not hf_file_available(repo, filename):
            errors.append(f"{repo}: missing")
            continue
        try:
            log(f"Using {repo}/{filename}")
            return download_hf_file(repo, filename, dest, force=force)
        except Exception as exc:
            errors.append(f"{repo}: {exc}")
            log(f"{repo}/{filename} failed ({exc}); trying next mirror")
    if errors:
        log("VenusREM2 mirrors unavailable: " + "; ".join(errors))
    return None
