"""Hugging Face checkpoint presence + download consent."""

from __future__ import annotations

import os
from typing import Any, Optional


def _hub_cache_root() -> str:
    hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub:
        return os.path.expanduser(hub)
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    return os.path.join(os.path.expanduser(hf_home), "hub")


def hf_snapshot_dir(repo_id: str) -> str:
    slug = "models--" + str(repo_id).replace("/", "--")
    return os.path.join(_hub_cache_root(), slug)


def is_local_model_path(repo_id: str) -> bool:
    path = os.path.expanduser(str(repo_id))
    return os.path.isdir(path)


def hf_repo_cached(repo_id: str) -> bool:
    """True if ``repo_id`` is a local dir or already in the HF hub cache."""
    if not repo_id:
        return False
    if is_local_model_path(repo_id):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        for name in (
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
        ):
            path = try_to_load_from_cache(repo_id, name)
            if isinstance(path, str) and os.path.exists(path):
                return True
    except Exception:
        pass
    root = hf_snapshot_dir(repo_id)
    snaps = os.path.join(root, "snapshots")
    if not os.path.isdir(snaps):
        return False
    for dirpath, _dirs, files in os.walk(snaps):
        if files:
            return True
    return False


def confirm_hf_repo(
    repo_id: str,
    *,
    name: Optional[str] = None,
    logger: Any = None,
) -> None:
    """Ask before the first download of a missing Hugging Face repo."""
    if not repo_id or is_local_model_path(repo_id) or hf_repo_cached(repo_id):
        return
    from vrh.models.download_policy import confirm_download

    confirm_download(
        name=name or repo_id,
        dest=hf_snapshot_dir(repo_id),
        source=f"https://huggingface.co/{repo_id}",
        looked_in=[hf_snapshot_dir(repo_id)],
        logger=logger,
    )


def from_pretrained(loader: Any, repo_id: str, *args: Any, logger: Any = None, **kwargs: Any):
    """``loader.from_pretrained`` after an optional Y/n download prompt."""
    confirm_hf_repo(repo_id, logger=logger)
    return loader.from_pretrained(repo_id, *args, **kwargs)
