"""Install-user commands: rem2 demo / rem2 doctor / bare rem2."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional


GETTING_STARTED = """\
rem2 — calibrate a protein language model for variant effect prediction

  rem2 doctor              check install (torch, extras, cache)
  rem2 demo                ProteinGym SDA_BACSU_Tsuboyama_2023_1PV0 + ESM-2 8M
  rem2 download                 ProteinGym 217 → data/proteingym_v1
  rem2 download ProteinGym      same
  rem2 download VenusMutHub     or muthub → data/VenusMutHub
  rem2 download VenusViroHub    or virohub → data/venusvirohub
  rem2 download benchmark-all   ProteinGym + MutHub + ViroHub
  rem2 download example         same ProteinGym assay as rem2 demo
  rem2 download esm2            prefetch ESM-2 650M into the default cache
  rem2 download venusrem2       official 6 ProSST checkpoints + tokenizer
  rem2 download model-all       every rem2 backbone (tens of GB)
  rem2 --model esm2 --fasta prot.fasta
  rem2 --model saprot --pdb prot.pdb
  rem2 --model prosst-2048 --pdb prot.pdb
  rem2 --model esm2 --base_dir data/proteingym_v1
  rem2 --model saprot --base_dir data/proteingym_v1
  rem2 --model venusrem2 --base_dir data/proteingym_v1
  rem2 --list-models
  rem2 --help

Python:  from rem2 import score
Default model is esm2. Official VenusREM2 is --model venusrem2 (ProSST ensemble).
No extra flags = full rem2. Sequence from FASTA or PDB. Dataset: substitutions/ plus aa_seq/ or pdbs/.
--scoring_strategy wt (default) | mask (esm2/saprot/…) | tf (proteinmpnn). ProSST is wt only.
Missing checkpoint: type y to download. Missing data: rem2 prints the expected folders.
"""

MODEL_SIZE_HINTS = {
    "esm2": "first download: ESM-2 650M, about 2.5 GB",
    "esm2-650m": "first download: ESM-2 650M, about 2.5 GB",
    "esm2-8m": "first download: ESM-2 8M, about 30 MB",
    "esm2-35m": "first download: ESM-2 35M",
    "esm2-150m": "first download: ESM-2 150M",
    "esm2-3b": "first download: ESM-2 3B, several GB",
    "esm1b": "first download: ESM-1b 650M, about 2.5 GB",
    "esm1v": "first download: ESM-1v 5-seed ensemble, about 5 × 650M",
    "venusrem2": "first download: 6 ProSST checkpoints, several GB",
    "prosst_ensemble": "first download: 6 ProSST checkpoints, several GB",
    "prosst": "first download: ProSST-2048 from Hugging Face",
    "prosst-2048": "first download: ProSST-2048 from Hugging Face (VenusREM v1 backbone)",
    "prosst-4096": "first download: ProSST-4096 from Hugging Face",
    "progen2-xl": "first download: ProGen2-xlarge from Hugging Face",
    "progen3-3b": "first download: ProGen3-3B from Hugging Face",
    "esmc-600m": "first download: ESM-C 600M (`pip install -e '.[esm3]'`)",
}


def _has_flag(argv: list[str], *names: str) -> bool:
    for arg in argv:
        for name in names:
            if arg == name or arg.startswith(f"{name}="):
                return True
    return False


def demo_dataset_dir() -> Path:
    """Local example dir without downloading (cache, then bundled / fixtures)."""
    from rem2.download.example import bundled_example_dir, default_example_dir

    cache = default_example_dir()
    if (cache / "aa_seq").is_dir():
        return cache
    bundled = bundled_example_dir()
    if bundled is not None:
        return bundled
    return cache


def build_demo_argv(user_argv: Optional[list[str]] = None) -> list[str]:
    user_argv = list(user_argv or [])
    if not _has_flag(user_argv, "--base_dir", "--fasta", "--aa_seq_dir", "--pdb"):
        from rem2.download.example import ensure_demo_dataset

        demo_dir = ensure_demo_dataset()
    else:
        demo_dir = demo_dataset_dir()
    if not (Path(demo_dir) / "aa_seq").is_dir() and not (Path(demo_dir) / "pdbs").is_dir():
        raise SystemExit(
            "Demo dataset is missing. rem2 demo downloads it from Hugging Face, "
            "or pass --base_dir yourself."
        )
    injected: list[str] = []
    if not _has_flag(user_argv, "--model", "--baseline_type"):
        injected.extend(["--model", "esm2-8m"])
    if not _has_flag(user_argv, "--base_dir", "--fasta", "--aa_seq_dir", "--pdb"):
        injected.extend(["--base_dir", str(demo_dir)])
    if not _has_flag(user_argv, "--out_scores_dir"):
        injected.extend(["--out_scores_dir", "result/demo"])
    return injected + user_argv


def model_size_hint(model_key: str) -> Optional[str]:
    raw = (model_key or "").lower()
    key = raw.replace("_", "-")
    if raw in MODEL_SIZE_HINTS:
        return MODEL_SIZE_HINTS[raw]
    if key in MODEL_SIZE_HINTS:
        return MODEL_SIZE_HINTS[key]
    if raw in {"venusrem2", "prosst_ensemble"} or key in {"venusrem2", "prosst-ensemble"}:
        return "first download: 6 ProSST checkpoints, several GB"
    if key.startswith("prosst"):
        return "first download: ProSST checkpoint from Hugging Face"
    if key.startswith("progen3"):
        return "first download: ProGen3 from Hugging Face (large)"
    if key.startswith("progen2"):
        return "first download: ProGen2 from Hugging Face"
    if key.startswith("esm2"):
        return "first download: ESM-2 checkpoint from Hugging Face"
    return None


def print_model_table() -> None:
    from rem2.models import list_models
    from rem2.models.scoring_strategy import forward_modes_label

    specs = list_models()
    name_w = max(len(s.name) for s in specs)
    print("rem2 backbones. FWD = allowed --scoring_strategy (wt / mask / tf).")
    print("venusrem2 = official ProSST ensemble (VenusREM2), wt only.")
    print("Aliases work as --model: saprot, esmif, protssn-ensemble,")
    print("  prosst-{k}, esm2-{size}m, proteinmpnn-{xx} (e.g. proteinmpnn-020).")
    print("CSV *_mask / *_wt is --scoring_strategy, not a second --model.")
    print()
    print(f"{'MODEL':<{name_w}}  PDB  FWD       AUTO  DEFAULT_ID / NOTES")
    print("-" * (name_w + 70))
    for spec in specs:
        pdb = "yes" if spec.needs_pdb else "no"
        fwd = forward_modes_label(spec)
        auto = "yes" if spec.auto_download else "no"
        extra = spec.default_model_id or ""
        if spec.notes:
            extra = f"{extra}  ({spec.notes})" if extra else spec.notes
        if spec.extras:
            extra = f"{extra}  [extras:{spec.extras}]"
        print(f"{spec.name:<{name_w}}  {pdb:<3}  {fwd:<9}  {auto:<4}  {extra}")


def run_doctor(argv: Optional[list[str]] = None) -> int:
    from rem2 import __version__
    from rem2.models.weights import default_cache_dir

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
        ("biotite", "biotite", "recommended: PDB I/O  pip install 'rem2[recommended]'", False),
        ("torch_geometric", "torch-geometric", "ProSST / VenusREM2  pip install 'rem2[prosst]'", False),
        ("esm", "esm", "ESM-3  pip install 'rem2[esm3]'", False),
        ("sequence_models", "sequence-models", "CARP  pip install 'rem2[carp]'", False),
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
    print("      rem2 download")
    print("      rem2 download benchmark-all")
    print("      rem2 download model-all")
    print("      rem2 --model esm2 --base_dir data/proteingym_v1")
    if problems:
        print("problems: " + ", ".join(problems))
        if strict:
            return 1
    return 0
