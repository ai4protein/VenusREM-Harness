"""Hugging Face mirrors for rem2 download.

Dataset repos: try ``AI4Protein/VenusREM2`` first, then ``tyang816/VenusREM2``.
Hub endpoints: ``huggingface.co`` first, then ``hf-mirror.com`` (or ``HF_MIRROR`` /
``HF_ENDPOINT``) for up to three network retries. Private repos need ``HF_TOKEN``.
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
HF_OFFICIAL_ENDPOINT = "https://huggingface.co"
HF_DEFAULT_MIRROR = "https://hf-mirror.com"
HF_DOWNLOAD_ATTEMPTS = 3


def hf_token_paths() -> tuple[Path, ...]:
    """Default files written by ``hf auth login`` / ``huggingface-cli login``."""
    hf_home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    return (
        hf_home / "token",
        Path.home() / ".huggingface" / "token",
    )


def hf_token() -> Optional[str]:
    """Env first (``HF_TOKEN``), then the Hugging Face CLI login token."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    try:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            return token
    except Exception:
        pass
    for path in hf_token_paths():
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text
    return None


def hf_headers() -> dict[str, str]:
    token = hf_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def hf_endpoints() -> tuple[str, ...]:
    """Official Hub first, then ``HF_ENDPOINT`` / ``HF_MIRROR`` / hf-mirror.com."""
    official = HF_OFFICIAL_ENDPOINT.rstrip("/")
    env = (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/")
    extra = (os.environ.get("HF_MIRROR") or HF_DEFAULT_MIRROR).strip().rstrip("/")
    ordered: list[str] = []
    for item in (official, env, extra):
        if item and item not in ordered:
            ordered.append(item)
    return tuple(ordered)


def rewrite_hf_url(url: str, endpoint: str) -> str:
    """Point a huggingface.co (or current-mirror) URL at ``endpoint``."""
    endpoint = endpoint.rstrip("/")
    prefixes = [item.rstrip("/") for item in hf_endpoints()]
    for prefix in ("https://huggingface.co", "http://huggingface.co", *prefixes):
        if url.startswith(prefix + "/"):
            return endpoint + url[len(prefix) :]
    return url


def hf_resolve_url(repo: str, filename: str, endpoint: Optional[str] = None) -> str:
    base = (endpoint or HF_OFFICIAL_ENDPOINT).rstrip("/")
    return f"{base}/datasets/{repo}/resolve/main/{filename}"


def is_hf_network_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError, ConnectionResetError)):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 409, 429, 500, 502, 503, 504}
    if isinstance(exc, urllib.error.URLError):
        return True
    name = type(exc).__name__
    if name in {
        "ConnectTimeout",
        "ReadTimeout",
        "ProxyError",
        "SSLError",
        "RemoteDisconnected",
        "IncompleteRead",
        "HfHubHTTPError",
    }:
        text = str(exc)
        if name == "HfHubHTTPError" and any(code in text for code in ("401", "403", "404")):
            return False
        return name != "HfHubHTTPError" or any(
            token in text for token in ("429", "500", "502", "503", "504", "timeout")
        )
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "network is unreachable",
            "name resolution",
            "temporary failure",
            "max retries",
            "502",
            "503",
            "504",
        )
    )


def call_with_hf_retry(
    fn: Callable,
    *,
    attempts: int = HF_DOWNLOAD_ATTEMPTS,
    log=None,
    what: str = "Hugging Face",
):
    """Call ``fn(endpoint)`` up to ``attempts`` times; switch mirrors on network errors."""
    endpoints = list(hf_endpoints())
    idx = 0
    last: Optional[BaseException] = None
    for attempt in range(1, int(attempts) + 1):
        endpoint = endpoints[idx]
        try:
            return fn(endpoint)
        except Exception as exc:
            last = exc
            if not is_hf_network_error(exc):
                raise
            nxt = idx + 1 if idx + 1 < len(endpoints) else idx
            if log:
                extra = f"; switching to {endpoints[nxt]}" if nxt != idx else ""
                log(
                    f"{what} via {endpoint} failed "
                    f"({type(exc).__name__}: {exc}) [{attempt}/{attempts}]{extra}"
                )
            if idx + 1 < len(endpoints):
                idx += 1
    assert last is not None
    raise last


def hf_file_available(repo: str, filename: str, timeout: float = 20) -> bool:
    for endpoint in hf_endpoints():
        req = urllib.request.Request(
            hf_resolve_url(repo, filename, endpoint=endpoint),
            method="HEAD",
            headers=hf_headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if 200 <= getattr(response, "status", 200) < 400:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code == 200:
                return True
            if exc.code in {401, 403, 404}:
                continue
        except Exception:
            continue
    return False


def download_hf_file(
    repo: str,
    filename: str,
    dest: Path,
    force: bool = False,
    *,
    progress: bool = False,
    desc: Optional[str] = None,
    log=None,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or filename
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        if progress:
            from rem2.download.progress import tqdm_bar

            with tqdm_bar(f"{label} (cached)", dest.stat().st_size) as bar:
                bar.update(dest.stat().st_size)
        return dest

    def _once(endpoint: str) -> Path:
        url = hf_resolve_url(repo, filename, endpoint=endpoint)
        if progress:
            from rem2.download.progress import download_url_with_progress

            try:
                return download_url_with_progress(
                    url, dest, desc=label, headers=hf_headers(), force=force
                )
            except Exception as exc:
                if not is_hf_network_error(exc):
                    raise
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return _download_url(url, dest, force=force)
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            local_dir=str(dest.parent),
            force_download=force,
            endpoint=endpoint,
        )
        downloaded = Path(path)
        if downloaded.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, dest)
        return dest

    return call_with_hf_retry(_once, log=log, what=f"{repo}/{filename}")


def _download_url(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    from rem2.download.progress import download_url_with_progress

    return download_url_with_progress(url, dest, desc=dest.name, headers=hf_headers(), force=force)


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
    *,
    progress: bool = False,
    desc: Optional[str] = None,
) -> Optional[Path]:
    """Download ``filename`` from the first working VenusREM2 mirror."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        if progress:
            from rem2.download.progress import tqdm_bar

            with tqdm_bar(f"{(desc or dest.name)} (cached)", dest.stat().st_size) as bar:
                bar.update(dest.stat().st_size)
        return dest
    errors: list[str] = []
    for repo in repos:
        if not hf_file_available(repo, filename):
            errors.append(f"{repo}: missing")
            continue
        try:
            log(f"Using {repo}/{filename}")
            return download_hf_file(
                repo,
                filename,
                dest,
                force=force,
                progress=progress,
                desc=desc or dest.name,
                log=log,
            )
        except Exception as exc:
            errors.append(f"{repo}: {exc}")
            log(f"{repo}/{filename} failed ({exc}); trying next mirror")
    if errors:
        log("VenusREM2 mirrors unavailable: " + "; ".join(errors))
    return None
