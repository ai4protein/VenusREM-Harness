"""Friendly dashboard dependency checks and first-run installs."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from typing import Optional

TORCH_CUDA_HINT = "pip install torch --index-url https://download.pytorch.org/whl/cu124"
TORCH_CPU_HINT = "pip install torch"

# Pip name, import name, why the dashboard needs it.
_CORE = (
    ("numpy", "numpy", "numeric arrays"),
    ("pandas", "pandas", "score tables"),
    ("tqdm", "tqdm", "progress logs"),
    ("pyyaml", "yaml", "run metadata"),
    ("scipy", "scipy", "ranking stats"),
    ("biopython", "Bio", "FASTA / PDB"),
    ("biotite", "biotite", "structure I/O"),
    ("fastapi", "fastapi", "the web app"),
    ("uvicorn", "uvicorn", "the web server"),
    ("python-multipart", "multipart", "file uploads"),
    ("transformers", "transformers", "ESM-2 and other HF backbones"),
)

_HINTS = {
    "torch": (
        "PyTorch is not installed, so this dashboard can open but cannot score mutants yet. "
        "Install a wheel that matches this machine, then restart `vrh dashboard`.\n"
        f"  GPU:  {TORCH_CUDA_HINT}\n"
        f"  CPU:  {TORCH_CPU_HINT}"
    ),
    "tqdm": "Progress logging needs tqdm. The dashboard will try to install it, or run: pip install tqdm",
    "pandas": "Score tables need pandas. The dashboard will try to install it, or run: pip install pandas",
    "fastapi": "The dashboard needs FastAPI. Run: pip install 'vrh[dashboard]'",
    "uvicorn": "The dashboard needs uvicorn. Run: pip install 'vrh[dashboard]'",
    "python-multipart": "File uploads need python-multipart. Run: pip install python-multipart",
    "biopython": "Sequence / PDB parsing needs biopython. Run: pip install biopython",
    "biotite": "Structure I/O needs biotite. Run: pip install biotite",
    "transformers": "ESM-2 scoring needs transformers. Run: pip install transformers",
    "torch-geometric": "VenusREM2 / ProSST need torch-geometric. Run: pip install 'vrh[prosst]'",
}


def _mod_ok(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def torch_ok() -> bool:
    return _mod_ok("torch")


def hint_for(pip_name: str) -> str:
    return _HINTS.get(
        pip_name,
        f"{pip_name} is not installed. Run: pip install {pip_name}",
    )


def missing_core() -> list[str]:
    missing = [pip for pip, mod, _note in _CORE if not _mod_ok(mod)]
    if not torch_ok():
        missing.insert(0, "torch")
    return missing


def setup_hints(names: Optional[list[str]] = None) -> list[str]:
    names = list(names if names is not None else missing_core())
    return [hint_for(name) for name in names]


def friendly_import_error(exc: BaseException) -> str:
    name = getattr(exc, "name", None) or ""
    text = str(exc)
    if not name:
        marker = "No module named "
        if marker in text:
            name = text.split(marker, 1)[-1].strip().strip("'\"")
    alias = {
        "torch": "torch",
        "tqdm": "tqdm",
        "pandas": "pandas",
        "yaml": "pyyaml",
        "Bio": "biopython",
        "biotite": "biotite",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "python_multipart": "python-multipart",
        "transformers": "transformers",
        "torch_geometric": "torch-geometric",
        "torch_scatter": "torch-geometric",
    }.get(name.split(".", 1)[0], name.replace("_", "-") if name else "")
    if alias:
        return hint_for(alias)
    return text or "A required package is missing. Run `vrh doctor` for details."


def scoring_blocked_message() -> Optional[str]:
    if torch_ok():
        return None
    return hint_for("torch")


def _pip_install(packages: list[str], *, index_url: Optional[str] = None) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    if index_url:
        cmd.extend(["--index-url", index_url])
    print("  $ " + " ".join(cmd), flush=True)
    try:
        completed = subprocess.run(cmd, check=False)
    except OSError as exc:
        print(f"  install failed: {exc}", flush=True)
        return False
    return completed.returncode == 0


def _torch_index_url() -> Optional[str]:
    if shutil.which("nvidia-smi"):
        return "https://download.pytorch.org/whl/cu124"
    return None


def ensure_dashboard_deps(*, install: bool = True) -> list[str]:
    """Install missing core packages when the dashboard starts. Returns leftovers."""
    env_off = os.environ.get("VRH_DASHBOARD_NO_INSTALL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if env_off:
        install = False

    leftover: list[str] = []
    for pip_name, mod, note in _CORE:
        if _mod_ok(mod):
            continue
        if not install:
            leftover.append(pip_name)
            continue
        print(f"vrh dashboard  {pip_name} is missing ({note}). Installing…", flush=True)
        if not _pip_install([pip_name]) or not _mod_ok(mod):
            leftover.append(pip_name)
            print(f"vrh dashboard  could not install {pip_name}.", flush=True)
            print("  " + hint_for(pip_name).replace("\n", "\n  "), flush=True)

    if not torch_ok():
        if install:
            index = _torch_index_url()
            kind = "CUDA 12.4" if index else "CPU"
            print(f"vrh dashboard  PyTorch is missing. Installing a {kind} wheel…", flush=True)
            ok = _pip_install(["torch"], index_url=index)
            if not ok or not torch_ok():
                leftover.insert(0, "torch")
                print("vrh dashboard  could not install PyTorch.", flush=True)
                print("  " + hint_for("torch").replace("\n", "\n  "), flush=True)
        else:
            leftover.insert(0, "torch")

    if leftover:
        print("vrh dashboard  still missing: " + ", ".join(leftover), flush=True)
        print("  Scoring will show a short install hint instead of a crash.", flush=True)
    return leftover
