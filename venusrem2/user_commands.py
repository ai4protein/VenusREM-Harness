"""Install-user commands: rem2 demo / rem2 doctor / bare rem2."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional


GETTING_STARTED = """\
rem2 — calibrate a protein language model for variant effect prediction

  rem2 doctor              check install (torch, extras, cache)
  rem2 demo                score the bundled trp-cage with ESM-2 8M
  rem2 --model esm2 --fasta prot.fasta
  rem2 --model esm2 --base_dir data/proteingym_v1
  rem2 --list-models
  rem2 --help

Python:  from venusrem2 import score
Default model is esm2. Official VenusREM2 is --model venusrem2 (ProSST ensemble).
"""

MODEL_SIZE_HINTS = {
    "esm2": "first download: ESM-2 650M, about 2.5 GB",
    "esm1b": "first download: ESM-1b 650M, about 2.5 GB",
    "esm1v": "first download: ESM-1v 5-seed ensemble, about 5 × 650M",
    "venusrem2": "first download: 6 ProSST checkpoints, several GB",
    "prosst": "first download: ProSST-2048 from Hugging Face",
    "esm2-8m": "first download: ESM-2 8M, about 30 MB",
}


def _has_flag(argv: list[str], *names: str) -> bool:
    for arg in argv:
        for name in names:
            if arg == name or arg.startswith(f"{name}="):
                return True
    return False


def demo_dataset_dir() -> Path:
    from venusrem2.examples import bundled_trp_cage_dir

    bundled = bundled_trp_cage_dir()
    if (bundled / "aa_seq").is_dir():
        return bundled
    repo = Path(__file__).resolve().parents[1] / "test" / "fixtures" / "trp_cage"
    return repo


def build_demo_argv(user_argv: Optional[list[str]] = None) -> list[str]:
    user_argv = list(user_argv or [])
    demo_dir = demo_dataset_dir()
    if not (demo_dir / "aa_seq").is_dir():
        raise SystemExit(
            "Bundled demo dataset is missing. Reinstall the package or pass --base_dir yourself."
        )
    injected: list[str] = []
    if not _has_flag(user_argv, "--model", "--baseline_type"):
        injected.extend(["--model", "esm2-8m"])
    if not _has_flag(user_argv, "--base_dir", "--fasta", "--aa_seq_dir"):
        injected.extend(["--base_dir", str(demo_dir)])
    if not _has_flag(user_argv, "--out_scores_dir"):
        injected.extend(["--out_scores_dir", "result/demo"])
    return injected + user_argv


def model_size_hint(model_key: str) -> Optional[str]:
    return MODEL_SIZE_HINTS.get((model_key or "").lower())


def run_doctor(argv: Optional[list[str]] = None) -> int:
    from venusrem2 import __version__
    from venusrem2.models.weights import default_cache_dir

    argv = list(argv or [])
    strict = "--strict" in argv
    problems: list[str] = []

    print(f"rem2 {__version__}")
    print(f"python {sys.version.split()[0]}  ({sys.executable})")

    try:
        import torch

        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "cpu"
        print(f"torch {torch.__version__}  cuda={cuda}  device={device}")
        if not cuda:
            print("note  no GPU visible; esm2 650M and venusrem2 will be slow on CPU")
    except ImportError:
        print("torch  MISSING  (install a CUDA wheel from pytorch.org first)")
        problems.append("torch")

    extras = [
        ("Bio", "biopython", "core: FASTA + RSA", True),
        ("biotite", "biotite", "recommended: PDB I/O  pip install 'venusrem2[recommended]'", False),
        ("torch_geometric", "torch-geometric", "ProSST / VenusREM2  pip install 'venusrem2[prosst]'", False),
        ("esm", "esm", "ESM-3  pip install 'venusrem2[esm3]'", False),
        ("sequence_models", "sequence-models", "CARP  pip install 'venusrem2[carp]'", False),
    ]
    for mod, pip_name, note, required in extras:
        found = importlib.util.find_spec(mod) is not None
        print(f"{pip_name:<20}  {'ok' if found else 'missing':<7}  {note}")
        if required and not found:
            problems.append(pip_name)

    cache = default_cache_dir()
    print(f"weight cache  {cache}  ({'exists' if Path(cache).is_dir() else 'not created yet'})")
    print(f"demo dataset  {demo_dataset_dir()}")
    print()
    print("Next: rem2 demo")
    print("      rem2 --model esm2 --fasta prot.fasta")
    if problems:
        print("problems: " + ", ".join(problems))
        if strict:
            return 1
    return 0
