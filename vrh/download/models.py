"""Prefetch vrh backbone weights into the default caches.

Hugging Face checkpoints go to the hub cache (``~/.cache/huggingface/hub``),
which is what ``from_pretrained`` reads. ProteinMPNN / CARP / S3F / Foldseek /
ProSST tokenizer files go to ``~/.cache/vrh/weights``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from vrh.download.progress import (
    disable_hf_bars,
    download_url_with_progress,
    enable_hf_bars,
    format_hint,
    print_plan,
    tqdm_bar,
)
from vrh.models.hf import hf_repo_cached
from vrh.models.weights import (
    ESM_IF_URL,
    FOLDSEEK_HF_FILE,
    FOLDSEEK_HF_REPO,
    PROSST_STATIC_REPO,
    S3F_HF_FILE,
    S3F_HF_REPO,
    S3F_ZENODO_URL,
    default_cache_dir,
    resolve_existing_weight,
    resolve_prosst_static_file,
)
from vrh.naming import (
    PROSST_ENSEMBLE_IDS,
    PROSST_ENSEMBLE_SIZES,
    is_ensemble_model_key,
)

SKIP_MODELS = {"auto", "s2f"}

HF_ALLOW = (
    "*.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.py",
    "*.safetensors",
    "*.bin",
    "*.msgpack",
    "tokenizer*",
    "vocab*",
    "merges*",
    "special_tokens*",
    "config*",
    "generation*",
)
HF_IGNORE = (
    "*.onnx",
    "*.ot",
    "*.h5",
    "flax_model*",
    "tf_model*",
    "openvino*",
    "rust_model*",
    "*.tflite",
    "*.pb",
)

ESM3_REPOS = {
    "esm3_sm_open_v1": "EvolutionaryScale/esm3-sm-open-v1",
    "esm3-open": "EvolutionaryScale/esm3-sm-open-v1",
    "esm3-sm-open-v1": "EvolutionaryScale/esm3-sm-open-v1",
    "esmc_300m": "EvolutionaryScale/esmc-300m-2024-12",
    "esmc-300m": "EvolutionaryScale/esmc-300m-2024-12",
    "esmc_600m": "EvolutionaryScale/esmc-600m-2024-12",
    "esmc-600m": "EvolutionaryScale/esmc-600m-2024-12",
}

MPNN_URL = "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/{folder}/{remote}"
CARP_ZENODO = "https://zenodo.org/record/6564798/files/{name}.pt?download=1"
MIF_ZENODO = "https://zenodo.org/record/6573779/files/mifst.pt?download=1"

PROTSSN_CONFIGS = tuple((k, h) for k in (10, 20, 30) for h in (512, 768, 1280))

SIZE_HINTS = {
    "esm2": 2_500_000_000,
    "esm2-650m": 2_500_000_000,
    "esm2-8m": 30_000_000,
    "esm2-35m": 150_000_000,
    "esm2-150m": 600_000_000,
    "esm2-3b": 11_000_000_000,
    "esm1b": 2_500_000_000,
    "esm1v": 12_500_000_000,
    "saprot": 2_500_000_000,
    "saprot-35m-af2": 150_000_000,
    "saprot-650m-pdb": 2_500_000_000,
    "venusrem2": 8_000_000_000,
    "prosst": 1_400_000_000,
    "prosst-20": 1_400_000_000,
    "prosst-128": 1_400_000_000,
    "prosst-512": 1_400_000_000,
    "prosst-1024": 1_400_000_000,
    "prosst-2048": 1_400_000_000,
    "prosst-4096": 1_400_000_000,
    "progen2": 3_000_000_000,
    "progen2-s": 150_000_000,
    "progen2-m": 600_000_000,
    "progen2-b": 1_500_000_000,
    "progen2-xl": 6_000_000_000,
    "progen3": 4_000_000_000,
    "progen3-112m": 450_000_000,
    "progen3-219m": 900_000_000,
    "progen3-339m": 1_400_000_000,
    "progen3-762m": 3_000_000_000,
    "progen3-3b": 12_000_000_000,
    "protgpt2": 3_000_000_000,
    "rita": 3_000_000_000,
    "rita-s": 150_000_000,
    "rita-m": 600_000_000,
    "rita-l": 1_500_000_000,
    "esm3": 1_400_000_000,
    "esmc": 1_200_000_000,
    "esmc-600m": 2_400_000_000,
    "protssn": 3_000_000_000,
    "carp": 2_500_000_000,
    "mifst": 2_700_000_000,
    "s3f": 2_800_000_000,
    "esm_if": 140_000_000,
    "protein_mpnn": 20_000_000,
}


@dataclass(frozen=True)
class Artifact:
    kind: str
    label: str
    source: str
    dest_hint: str = ""
    extra: dict = field(default_factory=dict)


def _ensure_builtins() -> None:
    from vrh.models import builtins as _builtins  # noqa: F401


def registered_model_names() -> list[str]:
    _ensure_builtins()
    from vrh.models.registry import list_models

    return [spec.name for spec in list_models() if spec.name not in SKIP_MODELS]


def resolve_model_key(name: str) -> Optional[str]:
    _ensure_builtins()
    from vrh.models.registry import get_model
    from vrh.naming import is_ensemble_model_key

    key = str(name).strip().lower()
    if not key:
        return None
    if is_ensemble_model_key(key):
        return key
    try:
        return get_model(key).spec.name if key not in SKIP_MODELS else None
    except KeyError:
        return None


def _prosst_k(model_key: str, default_id: Optional[str]) -> int:
    for size in (4096, 2048, 1024, 512, 128, 64, 20):
        token = str(size)
        if token in model_key.replace("_", "-").split("-"):
            return size
        if default_id and default_id.endswith(f"-{size}"):
            return size
    return 2048


def _mpnn_spec(model_key: str, default_id: Optional[str]) -> tuple[str, str, str]:
    name = (default_id or "v_48_020.pt").split("/")[-1]
    if not name.endswith(".pt"):
        name = "v_48_020.pt"
    soluble = "soluble" in model_key.lower() or name.startswith("soluble_")
    remote = name[len("soluble_") :] if name.startswith("soluble_") else name
    folder = "soluble_model_weights" if soluble else "vanilla_model_weights"
    local = f"soluble_{remote}" if soluble else remote
    return folder, remote, local


def _esm3_repo(default_id: Optional[str]) -> Optional[str]:
    raw = (default_id or "").strip()
    if raw in ESM3_REPOS:
        return ESM3_REPOS[raw]
    folded = raw.replace("_", "-")
    return ESM3_REPOS.get(folded)


def model_artifacts(model_key: str) -> list[Artifact]:
    _ensure_builtins()
    from vrh.models.registry import get_model

    key = str(model_key).strip().lower()
    if is_ensemble_model_key(key):
        arts: list[Artifact] = [
            Artifact("prosst_static", "ProSST AE.pt", f"hf://{PROSST_STATIC_REPO}/static/AE.pt", extra={"name": "AE.pt"})
        ]
        for size, repo in zip(PROSST_ENSEMBLE_SIZES, PROSST_ENSEMBLE_IDS):
            arts.append(Artifact("hf_repo", f"ProSST-{size}", repo, extra={"repo": repo}))
            arts.append(
                Artifact(
                    "prosst_static",
                    f"ProSST {size}.joblib",
                    f"hf://{PROSST_STATIC_REPO}/static/{size}.joblib",
                    extra={"name": f"{size}.joblib"},
                )
            )
        return arts

    spec = get_model(key).spec
    baseline = spec.baseline_type or spec.name
    default_id = spec.default_model_id

    if baseline == "esm1v":
        return [
            Artifact(
                "hf_repo",
                f"ESM-1v seed {seed}",
                f"facebook/esm1v_t33_650M_UR90S_{seed}",
                extra={"repo": f"facebook/esm1v_t33_650M_UR90S_{seed}"},
            )
            for seed in (1, 2, 3, 4, 5)
        ]
    if baseline == "saprot":
        repo = default_id or "westlake-repl/SaProt_650M_AF2"
        return [
            Artifact("hf_repo", spec.name, repo, extra={"repo": repo}),
            Artifact("foldseek", "foldseek", f"hf://{FOLDSEEK_HF_REPO}/{FOLDSEEK_HF_FILE}"),
        ]
    if baseline == "prosst" or key.startswith("prosst"):
        size = _prosst_k(key, default_id)
        repo = default_id or f"AI4Protein/ProSST-{size}"
        return [
            Artifact("hf_repo", f"ProSST-{size}", repo, extra={"repo": repo}),
            Artifact("prosst_static", "ProSST AE.pt", f"hf://{PROSST_STATIC_REPO}/static/AE.pt", extra={"name": "AE.pt"}),
            Artifact(
                "prosst_static",
                f"ProSST {size}.joblib",
                f"hf://{PROSST_STATIC_REPO}/static/{size}.joblib",
                extra={"name": f"{size}.joblib"},
            ),
        ]
    if baseline == "protein_mpnn":
        folder, remote, local = _mpnn_spec(key, default_id)
        url = MPNN_URL.format(folder=folder, remote=remote)
        return [Artifact("url", local, url, dest_hint=f"protein_mpnn/{local}", extra={"rel": ("protein_mpnn", local)})]
    if baseline == "protssn":
        arts = [
            Artifact(
                "hf_file",
                f"protssn_k{k}_h{h}.pt",
                f"hf://tyang816/ProtSSN/protssn_k{k}_h{h}.pt",
                dest_hint=f"protssn/protssn_k{k}_h{h}.pt",
                extra={"repo": "tyang816/ProtSSN", "filename": f"protssn_k{k}_h{h}.pt", "rel": ("protssn", f"protssn_k{k}_h{h}.pt")},
            )
            for k, h in PROTSSN_CONFIGS
        ]
        arts.append(
            Artifact("hf_repo", "ESM-2 650M (ProtSSN encoder)", "facebook/esm2_t33_650M_UR50D", extra={"repo": "facebook/esm2_t33_650M_UR50D"})
        )
        return arts
    if baseline == "carp":
        name = default_id or "carp_640M"
        return [
            Artifact(
                "url",
                f"{name}.pt",
                CARP_ZENODO.format(name=name),
                dest_hint=f"carp/{name}.pt",
                extra={"rel": ("carp", f"{name}.pt")},
            )
        ]
    if baseline == "mifst":
        return [
            Artifact("url", "mifst.pt", MIF_ZENODO, dest_hint="mifst/mifst.pt", extra={"rel": ("mifst", "mifst.pt")}),
            Artifact(
                "url",
                "carp_640M.pt",
                CARP_ZENODO.format(name="carp_640M"),
                dest_hint="carp/carp_640M.pt",
                extra={"rel": ("carp", "carp_640M.pt")},
            ),
        ]
    if baseline == "s3f":
        return [
            Artifact(
                "url",
                S3F_HF_FILE,
                S3F_ZENODO_URL,
                dest_hint=f"s3f/{S3F_HF_FILE}",
                extra={"rel": ("s3f", S3F_HF_FILE), "hf_repo": S3F_HF_REPO},
            )
        ]
    if baseline == "esm_if":
        return [
            Artifact(
                "url",
                "esm_if1_gvp4_t16_142M_UR50.pt",
                ESM_IF_URL,
                dest_hint="esm_if/esm_if1_gvp4_t16_142M_UR50.pt",
                extra={"rel": ("esm_if", "esm_if1_gvp4_t16_142M_UR50.pt")},
            )
        ]
    if baseline == "esm3":
        repo = _esm3_repo(default_id)
        if repo:
            return [Artifact("hf_repo", spec.name, repo, extra={"repo": repo})]
        raise SystemExit(f"No public snapshot mapping for ESM-3 id {default_id!r}")

    if default_id and "/" in default_id:
        return [Artifact("hf_repo", spec.name, default_id, extra={"repo": default_id})]
    raise SystemExit(f"Don't know how to prefetch weights for {model_key!r}")


def model_size_hint(model_key: str) -> Optional[int]:
    key = model_key.lower()
    if key in SIZE_HINTS:
        return SIZE_HINTS[key]
    if key.startswith("protein_mpnn"):
        return SIZE_HINTS["protein_mpnn"]
    if key.startswith("prosst"):
        return SIZE_HINTS["prosst"]
    return None


def list_downloadable_models() -> list[str]:
    names = registered_model_names()
    # venusrem2 is an alias of prosst but downloads the official ensemble.
    extra = ["venusrem2"]
    out = []
    seen = set()
    for name in extra + names:
        if name in seen or name in SKIP_MODELS:
            continue
        seen.add(name)
        out.append(name)
    return out


def _hf_headers_for(url: str) -> dict[str, str]:
    from vrh.data.mirrors import hf_headers_for

    return hf_headers_for(url)


def _prefetch_hf_repo(repo_id: str, desc: str, force: bool) -> str:
    if hf_repo_cached(repo_id) and not force:
        with tqdm_bar(f"{desc} (cached)", 1, unit="file") as bar:
            bar.update(1)
        return "cached"
    from huggingface_hub import snapshot_download
    from vrh.data.mirrors import call_with_hf_retry

    def _once(endpoint: str) -> str:
        from vrh.data.mirrors import hf_hub_token

        snapshot_download(
            repo_id=repo_id,
            allow_patterns=list(HF_ALLOW),
            ignore_patterns=list(HF_IGNORE),
            force_download=force,
            endpoint=endpoint,
            token=hf_hub_token(endpoint),
        )
        return "ok"

    return call_with_hf_retry(_once, what=desc)


def _prefetch_hf_file(repo: str, filename: str, dest: Path, desc: str, force: bool) -> str:
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        with tqdm_bar(f"{desc} (cached)", dest.stat().st_size) as bar:
            bar.update(dest.stat().st_size)
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from vrh.data.mirrors import call_with_hf_retry

    def _once(endpoint: str) -> str:
        disable_hf_bars()
        try:
            from vrh.data.mirrors import hf_hub_token

            path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(dest.parent),
                force_download=force,
                endpoint=endpoint,
                token=hf_hub_token(endpoint),
            )
        finally:
            enable_hf_bars()
        downloaded = Path(path)
        if downloaded.resolve() != dest.resolve() and downloaded.is_file():
            import shutil

            shutil.copy2(downloaded, dest)
        size = dest.stat().st_size if dest.is_file() else 0
        with tqdm_bar(desc, size or 1) as bar:
            bar.update(size or 1)
        return "ok"

    return call_with_hf_retry(_once, what=desc)


def _prefetch_artifact(art: Artifact, cache: str, force: bool, log: Callable) -> str:
    if art.kind == "hf_repo":
        return _prefetch_hf_repo(art.extra["repo"], art.label, force)
    if art.kind == "prosst_static":
        path = resolve_prosst_static_file(art.extra["name"])
        size = Path(path).stat().st_size if path and os.path.isfile(path) else 0
        with tqdm_bar(art.label, size or 1) as bar:
            bar.update(size or 1)
        return "ok"
    if art.kind == "foldseek":
        from vrh.models.weights import ensure_foldseek_bin

        path = ensure_foldseek_bin(cache_dir=cache)
        size = Path(path).stat().st_size if path and os.path.isfile(path) else 0
        with tqdm_bar(art.label, size or 1) as bar:
            bar.update(size or 1)
        return "ok"
    if art.kind == "url":
        rel = art.extra.get("rel") or ()
        dest = Path(cache).joinpath(*rel) if rel else Path(cache) / Path(art.source).name
        existing = resolve_existing_weight(*rel, cache_dir=cache) if rel else None
        if existing and not force:
            dest = Path(existing)
        download_url_with_progress(art.source, dest, desc=art.label, force=force)
        return "ok"
    if art.kind == "hf_file":
        rel = art.extra.get("rel") or (art.extra["filename"],)
        dest = Path(cache).joinpath(*rel)
        existing = resolve_existing_weight(*rel, cache_dir=cache)
        if existing and not force:
            dest = Path(existing)
        from vrh.data.mirrors import call_with_hf_retry, rewrite_hf_url

        def _once(endpoint: str) -> str:
            url = rewrite_hf_url(
                f"https://huggingface.co/{art.extra['repo']}/resolve/main/{art.extra['filename']}",
                endpoint,
            )
            download_url_with_progress(
                url, dest, desc=art.label, headers=_hf_headers_for(url), force=force
            )
            return "ok"

        try:
            return call_with_hf_retry(_once, what=art.label)
        except Exception:
            return _prefetch_hf_file(
                art.extra["repo"], art.extra["filename"], dest, art.label, force
            )
    raise ValueError(f"unknown artifact kind {art.kind}")


def print_model_plan(keys: list[str], cache: str, log=print) -> None:
    from vrh.models.hf import _hub_cache_root

    rows = []
    total = 0
    skip_prosst_hint = "venusrem2" in {k.lower() for k in keys}
    for key in keys:
        hint = model_size_hint(key)
        if hint and not (skip_prosst_hint and key.startswith("prosst")):
            total += hint
        arts = model_artifacts(key)
        if key in {"venusrem2", "prosst_ensemble", "prosst-ensemble"}:
            detail = "6× ProSST + tokenizer"
        elif len(arts) == 1:
            detail = arts[0].source
        else:
            detail = f"{len(arts)} files"
        rows.append((key, format_hint(hint), detail))
    log(f"Models → HF hub {_hub_cache_root()}")
    log(f"         vrh cache {cache}")
    print_plan(f"Will download {len(keys)} model(s)  ({format_hint(total)})", rows, log=log)
    if len(keys) > 8:
        log("This can be tens of GB. Shared files are fetched once. Ctrl-C to abort.")


def download_models(
    keys: list[str],
    *,
    cache_dir: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    log=print,
) -> dict[str, str]:
    from vrh.models.download_policy import set_download_policy

    set_download_policy("yes")
    cache = default_cache_dir(cache_dir)
    print_model_plan(keys, cache, log=log)
    results: dict[str, str] = {}
    if dry_run:
        if len(keys) == 1:
            for art in model_artifacts(keys[0]):
                log(f"  {art.label}: {art.source}")
        return {key: "dry-run" for key in keys}
    for i, key in enumerate(keys, 1):
        arts = model_artifacts(key)
        log(f"[{i}/{len(keys)}] {key}  ({len(arts)} file(s))")
        try:
            states = [_prefetch_artifact(art, cache, force, log) for art in arts]
            results[key] = "cached" if states and all(s == "cached" for s in states) else "ok"
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            results[key] = f"error: {exc}"
            log(f"{key} failed: {exc}")
    log("Model download summary:")
    for key, state in results.items():
        log(f"  {key:<22} {state}")
    return results
