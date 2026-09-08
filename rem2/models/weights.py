"""Shared weight / binary cache helpers."""

from __future__ import annotations

import os
import shutil
import stat
import urllib.request
from pathlib import Path
from typing import Optional

FOLDSEEK_HF_REPO = "tyang816/Foldseek_bin"
FOLDSEEK_HF_FILE = "foldseek"
FOLDSEEK_URL = (
    "https://huggingface.co/tyang816/Foldseek_bin/resolve/main/foldseek?download=true"
)

S3F_HF_REPO = "tyang816/S3F_weights"
S3F_HF_FILE = "s3f.pth"
S3F_ZENODO_URL = "https://zenodo.org/records/14257708/files/s3f.pth?download=1"
PROSST_STATIC_REPO = "tyang816/ProSST"
PROSST_STATIC_NAMES = (
    "AE.pt",
    "20.joblib",
    "64.joblib",
    "128.joblib",
    "512.joblib",
    "1024.joblib",
    "2048.joblib",
    "4096.joblib",
)
ESM_IF_URL = (
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt"
)


def default_cache_dir(explicit: Optional[str] = None) -> str:
    if explicit:
        path = os.path.expanduser(explicit)
    else:
        env = (
            os.environ.get("REM2_CACHE")
            or os.environ.get("VENUSREM2_CACHE")
            or os.environ.get("VENUSREM_CACHE")
            or os.environ.get("VENUS_ORBIT_CACHE")
        )
        if env:
            path = os.path.expanduser(env)
        else:
            path = os.path.join(os.path.expanduser("~"), ".cache", "rem2", "weights")
    # If the user pointed at ~/.cache/rem2 or ~/.cache/venusrem2, prefer the weights/ child for writes.
    if os.path.isdir(os.path.join(path, "weights")) and not os.path.isdir(
        os.path.join(path, "protein_mpnn")
    ):
        path = os.path.join(path, "weights")
    os.makedirs(path, exist_ok=True)
    return path


def _unique_existing_dirs(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in paths:
        if not raw:
            continue
        key = os.path.abspath(os.path.expanduser(raw))
        if key in seen:
            continue
        seen.add(key)
        if os.path.isdir(key):
            out.append(key)
    return out


def legacy_weight_cache_dirs() -> list[str]:
    """Read-only fallback roots (old rem2 / VenusREM-Orbit caches). Do not write here."""
    roots = [
        os.environ.get("VENUSREM2_CACHE") or "",
        os.environ.get("VENUS_ORBIT_CACHE") or "",
        os.path.join(os.path.expanduser("~"), ".cache", "venusrem2", "weights"),
        os.path.join(os.path.expanduser("~"), ".cache", "venusrem2"),
        os.path.join(os.path.expanduser("~"), ".cache", "venus_orbit", "weights"),
        os.path.join(os.path.expanduser("~"), ".cache", "venus_orbit"),
    ]
    return _unique_existing_dirs(roots)


def cache_search_roots(cache_dir: Optional[str] = None) -> list[str]:
    """Directories to scan for an existing checkpoint (write cache first)."""
    primary = default_cache_dir(cache_dir)
    extras: list[str] = [primary]
    if cache_dir:
        extras.append(os.path.expanduser(cache_dir))
        extras.append(os.path.join(os.path.expanduser(cache_dir), "weights"))
    extras.extend(legacy_weight_cache_dirs())
    return _unique_existing_dirs(extras)


def weight_candidates(*relparts: str, cache_dir: Optional[str] = None) -> list[str]:
    return [os.path.join(root, *relparts) for root in cache_search_roots(cache_dir)]


def resolve_existing_weight(*relparts: str, cache_dir: Optional[str] = None) -> Optional[str]:
    """Return an existing weight file from rem2 cache, then Orbit cache."""
    for path in weight_candidates(*relparts, cache_dir=cache_dir):
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    return None


def resolve_existing_dir(*relparts: str, cache_dir: Optional[str] = None) -> Optional[str]:
    for path in weight_candidates(*relparts, cache_dir=cache_dir):
        if os.path.isdir(path):
            return path
    return None


def log_cache_hit(name: str, path: str, logger=None) -> str:
    msg = f"Using cached {name}: {path}"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
    return path


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def package_root() -> Path:
    """``venusrem2/`` package directory."""
    return Path(__file__).resolve().parents[1]


def repo_root() -> Optional[Path]:
    """Checkout root when developing from source (contains ``data/``)."""
    candidate = package_root().parent
    if (candidate / "data").is_dir() or (candidate / "pyproject.toml").is_file():
        return candidate
    return None


def bundled_s3f_config() -> str:
    path = package_root() / "baseline" / "s2f" / "config" / "evaluate" / "s3f.yaml"
    return str(path)


def bundled_s2f_config() -> str:
    path = package_root() / "baseline" / "s2f" / "config" / "evaluate" / "s2f.yaml"
    return str(path)


def ensure_url_file(
    url: str,
    dest: str,
    logger=None,
    *,
    name: Optional[str] = None,
    looked_in: Optional[list[str]] = None,
) -> str:
    """Download ``url`` to ``dest`` if missing (after cache check + consent)."""
    dest_path = Path(dest)
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        return log_cache_hit(name or dest_path.name, str(dest_path), logger)
    from rem2.models.download_policy import confirm_download

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    confirm_download(
        name=name or dest_path.name,
        dest=str(dest_path),
        source=url,
        looked_in=looked_in,
        logger=logger,
    )
    tmp = dest_path.with_suffix(dest_path.suffix + ".partial")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest_path)
    return str(dest_path)


def ensure_file(
    *,
    dest: str,
    url: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    logger=None,
) -> str:
    """Unified download helper for a URL or Hugging Face repo file.

    Prefer an existing ``dest``; otherwise download from ``url`` or
    ``hf_repo``/``hf_filename`` into ``cache_dir`` (or ``dest``'s parent).
    """
    dest_path = Path(os.path.expanduser(dest))
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        return log_cache_hit(dest_path.name, str(dest_path), logger)
    rel_name = dest_path.name
    parent_name = dest_path.parent.name
    fallback = resolve_existing_weight(parent_name, rel_name, cache_dir=cache_dir) if parent_name else None
    if fallback:
        return log_cache_hit(rel_name, fallback, logger)
    source = f"hf://{hf_repo}/{hf_filename}" if hf_repo and hf_filename else (url or "")
    looked = weight_candidates(parent_name, rel_name, cache_dir=cache_dir) if parent_name else [str(dest_path)]
    if hf_repo and hf_filename:
        local_dir = str(dest_path.parent) if dest_path.suffix else str(dest_path)
        if cache_dir and not dest_path.suffix:
            local_dir = ensure_dir(os.path.join(default_cache_dir(cache_dir), dest_path.name))
        return _hf_download(
            hf_repo, hf_filename, local_dir, logger=logger, name=rel_name, looked_in=looked
        )
    if url:
        return ensure_url_file(
            url, str(dest_path), logger=logger, name=rel_name, looked_in=looked
        )
    raise ValueError("ensure_file requires url= or hf_repo=+hf_filename=")


def _hf_download(
    repo_id: str,
    filename: str,
    local_dir: str,
    logger=None,
    *,
    name: Optional[str] = None,
    looked_in: Optional[list[str]] = None,
) -> str:
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        return log_cache_hit(name or filename, local_path, logger)
    from rem2.models.download_policy import confirm_download

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for auto-download. "
            "Install with: pip install huggingface-hub"
        ) from exc
    confirm_download(
        name=name or filename,
        dest=local_path,
        source=f"hf://{repo_id}/{filename}",
        looked_in=looked_in,
        logger=logger,
    )
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)


def ensure_foldseek_bin(cache_dir: Optional[str] = None, explicit: Optional[str] = None, logger=None) -> str:
    """Resolve foldseek binary: explicit path, PATH, or auto-download."""
    if explicit:
        path = os.path.expanduser(explicit)
        if os.path.isfile(path):
            _ensure_executable(path)
            return log_cache_hit("foldseek", path, logger)
        raise FileNotFoundError(f"foldseek binary not found: {path}")

    found = shutil.which("foldseek")
    if found:
        return log_cache_hit("foldseek (PATH)", found, logger)

    cache = default_cache_dir(cache_dir)
    dest_dir = ensure_dir(os.path.join(cache, "bin"))
    dest = os.path.join(dest_dir, "foldseek")
    existing = resolve_existing_weight("bin", "foldseek", cache_dir=cache_dir)
    if existing:
        _ensure_executable(existing)
        return log_cache_hit("foldseek", existing, logger)
    looked = weight_candidates("bin", "foldseek", cache_dir=cache_dir)
    if not (os.path.isfile(dest) and os.path.getsize(dest) > 0):
        try:
            _hf_download(
                FOLDSEEK_HF_REPO,
                FOLDSEEK_HF_FILE,
                dest_dir,
                logger=logger,
                name="foldseek",
                looked_in=looked,
            )
        except Exception as exc:
            from rem2.models.download_policy import DownloadRefused

            if isinstance(exc, DownloadRefused):
                raise
            ensure_url_file(
                FOLDSEEK_URL, dest, logger=logger, name="foldseek", looked_in=looked
            )
        # hf_hub_download may write to dest_dir/foldseek
        if not os.path.isfile(dest):
            # sometimes nested
            for root, _dirs, files in os.walk(dest_dir):
                if "foldseek" in files:
                    src = os.path.join(root, "foldseek")
                    if src != dest:
                        shutil.copy2(src, dest)
                    break
    if not os.path.isfile(dest):
        raise FileNotFoundError(
            "Failed to download foldseek binary. "
            f"Expected at {dest} from {FOLDSEEK_HF_REPO}"
        )
    _ensure_executable(dest)
    return dest


def _ensure_executable(path: str) -> None:
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def ensure_s3f_checkpoint(
    cache_dir: Optional[str] = None,
    explicit: Optional[str] = None,
    logger=None,
) -> str:
    """Resolve S3F checkpoint: explicit, local data/, cache, or HF download."""
    if explicit:
        path = os.path.expanduser(explicit)
        if os.path.isfile(path):
            return log_cache_hit("S3F checkpoint", path, logger)
        raise FileNotFoundError(f"S3F checkpoint not found: {path}")

    env_ckpt = os.environ.get("S3F_CHECKPOINT")
    if env_ckpt and os.path.isfile(os.path.expanduser(env_ckpt)):
        return log_cache_hit("S3F checkpoint", os.path.expanduser(env_ckpt), logger)

    cache = default_cache_dir(cache_dir)
    candidates = weight_candidates("s3f", "s3f.pth", cache_dir=cache_dir)
    root = repo_root()
    if root is not None:
        candidates.append(str(root / "data" / "s3f_weights" / "s3f.pth"))
    candidates.append(os.path.abspath(os.path.join("data", "s3f_weights", "s3f.pth")))

    for path in candidates:
        if path and os.path.isfile(path) and os.path.getsize(path) > 0:
            return log_cache_hit("S3F checkpoint", path, logger)

    dest_dir = ensure_dir(os.path.join(cache, "s3f"))
    dest = os.path.join(dest_dir, "s3f.pth")
    url = os.environ.get("S3F_CHECKPOINT_URL") or S3F_ZENODO_URL
    try:
        return _hf_download(
            S3F_HF_REPO,
            S3F_HF_FILE,
            dest_dir,
            logger=logger,
            name="S3F checkpoint",
            looked_in=candidates,
        )
    except Exception as hf_exc:
        from rem2.models.download_policy import DownloadRefused

        if isinstance(hf_exc, DownloadRefused):
            raise
        return ensure_url_file(
            url, dest, logger=logger, name="S3F checkpoint", looked_in=candidates
        )


def ensure_esm_if_checkpoint(cache_dir: Optional[str] = None, logger=None) -> str:
    existing = resolve_existing_weight(
        "esm_if", "esm_if1_gvp4_t16_142M_UR50.pt", cache_dir=cache_dir
    )
    if existing:
        return log_cache_hit("ESM-IF checkpoint", existing, logger)
    cache = default_cache_dir(cache_dir)
    dest = os.path.join(cache, "esm_if", "esm_if1_gvp4_t16_142M_UR50.pt")
    return ensure_url_file(
        ESM_IF_URL,
        dest,
        logger=logger,
        name="ESM-IF checkpoint",
        looked_in=weight_candidates(
            "esm_if", "esm_if1_gvp4_t16_142M_UR50.pt", cache_dir=cache_dir
        ),
    )


def ensure_fair_esm_source(cache_dir: Optional[str] = None, logger=None) -> str:
    """Return path to facebookresearch/esm sources (for ESM-IF, isolated from ESM3)."""
    import torch

    hub_dir = torch.hub.get_dir()
    repo_dir = os.path.join(hub_dir, "facebookresearch_esm_main")
    marker = os.path.join(repo_dir, "esm", "pretrained.py")
    if os.path.isfile(marker):
        return repo_dir

    # Fall back: download zip into cache and extract
    cache = default_cache_dir(cache_dir)
    zip_path = os.path.join(cache, "esm_if", "fair-esm-main.zip")
    extract_root = os.path.join(cache, "esm_if", "fair-esm-src")
    ensure_url_file(
        "https://github.com/facebookresearch/esm/archive/refs/heads/main.zip",
        zip_path,
        logger=logger,
    )
    if not os.path.isfile(os.path.join(extract_root, "esm", "pretrained.py")):
        import zipfile

        ensure_dir(extract_root)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(extract_root))
        # github archive extracts to esm-main/
        extracted = os.path.join(os.path.dirname(extract_root), "esm-main")
        if os.path.isdir(extracted) and not os.path.isdir(extract_root):
            os.rename(extracted, extract_root)
        elif os.path.isdir(extracted):
            # merge
            for name in os.listdir(extracted):
                src = os.path.join(extracted, name)
                dst = os.path.join(extract_root, name)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
    if os.path.isfile(os.path.join(extract_root, "esm", "pretrained.py")):
        return extract_root
    if os.path.isfile(marker):
        return repo_dir
    raise FileNotFoundError("Failed to obtain facebookresearch/esm sources for ESM-IF")


def bundled_prosst_static(name: str) -> Optional[str]:
    path = package_root() / "baseline" / "prosst" / "static" / name
    if path.is_file() and path.stat().st_size > 0:
        return str(path)
    return None


def resolve_prosst_static_file(
    name: str,
    cache_dir: Optional[str] = None,
    logger=None,
) -> str:
    """AE.pt / {K}.joblib: bundled copy, rem2 cache, then ``tyang816/ProSST``."""
    filename = Path(name).name
    bundled = bundled_prosst_static(filename)
    if bundled:
        return log_cache_hit(f"ProSST {filename}", bundled, logger)
    existing = resolve_existing_weight("prosst", "static", filename, cache_dir=cache_dir)
    if existing:
        return log_cache_hit(f"ProSST {filename}", existing, logger)
    cache = default_cache_dir(cache_dir)
    local_dir = ensure_dir(os.path.join(cache, "prosst"))
    return _hf_download(
        PROSST_STATIC_REPO,
        f"static/{filename}",
        local_dir,
        logger=logger,
        name=f"ProSST {filename}",
        looked_in=weight_candidates("prosst", "static", filename, cache_dir=cache_dir),
    )
