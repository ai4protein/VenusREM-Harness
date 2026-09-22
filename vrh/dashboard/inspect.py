"""Lightweight metadata for the dashboard (no torch import)."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from vrh import __version__
from vrh.user_commands import MODEL_SIZE_HINTS, demo_dataset_dir, model_size_hint


def doctor_report() -> dict[str, Any]:
    from vrh.models.weights import default_cache_dir

    extras = [
        ("biopython", "Bio", "core: FASTA + RSA"),
        ("biotite", "biotite", "core: PDB I/O / RSA"),
        ("fastapi", "fastapi", "core: vrh dashboard"),
        ("torch-geometric", "torch_geometric", "ProSST / VenusREM2"),
        ("esm", "esm", "ESM-3"),
        ("sequence-models", "sequence_models", "CARP"),
    ]
    extra_rows = []
    problems: list[str] = []
    for name, mod, note in extras:
        ok = importlib.util.find_spec(mod) is not None
        extra_rows.append({"name": name, "ok": ok, "note": note})
        if name in {"biopython", "biotite", "fastapi"} and not ok:
            problems.append(name)

    torch_ver = None
    cuda = False
    device = "cpu"
    try:
        import torch

        torch_ver = torch.__version__
        cuda = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if cuda else "cpu"
    except Exception:
        problems.append("torch")

    from vrh.dashboard.deps import setup_hints

    cache = default_cache_dir()
    return {
        "version": __version__,
        "python": sys.version.split()[0],
        "torch": torch_ver,
        "cuda": cuda,
        "device": device,
        "extras": extra_rows,
        "cache": str(cache),
        "demo": str(demo_dataset_dir()),
        "problems": problems,
        "hints": setup_hints(problems),
        "ready": not problems,
    }


_SERIES_LABELS = {
    "venusrem2": "VenusREM2",
    "prosst": "ProSST",
    "esm": "ESM",
    "saprot": "SaProt",
    "progen": "ProGen",
    "proteinmpnn": "ProteinMPNN",
    "rita": "RITA",
    "protssn": "ProtSSN",
    "carp": "CARP / MIF-ST",
    "s3f": "S2F / S3F",
    "other": "Other",
}

_SERIES_VARIANT_ORDER = {
    "venusrem2": ("venusrem2", "venusrem"),
    "prosst": (
        "prosst",
        "prosst-20",
        "prosst-128",
        "prosst-512",
        "prosst-1024",
        "prosst-2048",
        "prosst-4096",
    ),
    "esm": (
        "esm2-8m",
        "esm2-35m",
        "esm2-150m",
        "esm2",
        "esm2-3b",
        "esm1b",
        "esm1v",
        "esmc",
        "esmc-600m",
        "esm3",
        "esm_if",
    ),
    "saprot": ("saprot", "saprot-35m-af2", "saprot-650m-pdb"),
    "progen": (
        "progen2-s",
        "progen2-m",
        "progen2-b",
        "progen2",
        "progen2-xl",
        "progen3-112m",
        "progen3-219m",
        "progen3-339m",
        "progen3-762m",
        "progen3",
        "progen3-3b",
    ),
    "proteinmpnn": (
        "protein_mpnn",
        "protein_mpnn-v_48_002",
        "protein_mpnn-v_48_010",
        "protein_mpnn-v_48_030",
        "protein_mpnn-soluble-v_48_002",
        "protein_mpnn-soluble-v_48_010",
        "protein_mpnn-soluble-v_48_020",
        "protein_mpnn-soluble-v_48_030",
    ),
    "rita": ("rita-s", "rita-m", "rita-l", "rita"),
    "protssn": (
        "protssn",
        "protssn-k10-h512",
        "protssn-k10-h768",
        "protssn-k10-h1280",
        "protssn-k20-h512",
        "protssn-k20-h768",
        "protssn-k20-h1280",
        "protssn-k30-h512",
        "protssn-k30-h768",
        "protssn-k30-h1280",
    ),
    "carp": ("carp-600k", "carp-38m", "carp-76m", "carp", "mifst"),
    "s3f": ("s2f", "s3f"),
    "other": ("protgpt2", "auto"),
}

_VARIANT_LABELS = {
    "venusrem2": "VenusREM2",
    "venusrem": "VenusREM (ProSST-2048, fixed α=0.8)",
    "prosst": "Default (K=2048)",
    "prosst-20": "K=20",
    "prosst-128": "K=128",
    "prosst-512": "K=512",
    "prosst-1024": "K=1024",
    "prosst-2048": "K=2048",
    "prosst-4096": "K=4096",
    "esm2": "ESM-2 650M",
    "esm2-8m": "ESM-2 8M",
    "esm2-35m": "ESM-2 35M",
    "esm2-150m": "ESM-2 150M",
    "esm2-3b": "ESM-2 3B",
    "esm1b": "ESM-1b",
    "esm1v": "ESM-1v",
    "esm3": "ESM3",
    "esmc": "ESM-C 300M",
    "esmc-600m": "ESM-C 600M",
    "esm_if": "ESM-IF",
    "saprot": "650M AF2",
    "saprot-35m-af2": "35M AF2",
    "saprot-650m-pdb": "650M PDB",
    "progen2": "ProGen2 L",
    "progen2-s": "ProGen2 S",
    "progen2-m": "ProGen2 M",
    "progen2-b": "ProGen2 B",
    "progen2-xl": "ProGen2 XL",
    "progen3": "ProGen3 1B",
    "progen3-112m": "ProGen3 112M",
    "progen3-219m": "ProGen3 219M",
    "progen3-339m": "ProGen3 339M",
    "progen3-762m": "ProGen3 762M",
    "progen3-3b": "ProGen3 3B",
    "protein_mpnn": "v_48_020",
    "protein_mpnn-v_48_002": "v_48_002",
    "protein_mpnn-v_48_010": "v_48_010",
    "protein_mpnn-v_48_030": "v_48_030",
    "protein_mpnn-soluble-v_48_002": "soluble v_48_002",
    "protein_mpnn-soluble-v_48_010": "soluble v_48_010",
    "protein_mpnn-soluble-v_48_020": "soluble v_48_020",
    "protein_mpnn-soluble-v_48_030": "soluble v_48_030",
    "rita": "RITA XL",
    "rita-s": "RITA S",
    "rita-m": "RITA M",
    "rita-l": "RITA L",
    "protssn": "Ensemble",
    "protssn-k10-h512": "k=10 h=512",
    "protssn-k10-h768": "k=10 h=768",
    "protssn-k10-h1280": "k=10 h=1280",
    "protssn-k20-h512": "k=20 h=512",
    "protssn-k20-h768": "k=20 h=768",
    "protssn-k20-h1280": "k=20 h=1280",
    "protssn-k30-h512": "k=30 h=512",
    "protssn-k30-h768": "k=30 h=768",
    "protssn-k30-h1280": "k=30 h=1280",
    "carp": "640M",
    "carp-600k": "600k",
    "carp-38m": "38M",
    "carp-76m": "76M",
    "mifst": "MIF-ST",
    "s2f": "S2F",
    "s3f": "S3F",
    "protgpt2": "ProtGPT2",
    "auto": "HF AutoMLM",
}


def infer_model_series(name: str) -> str:
    raw = (name or "").lower()
    if raw == "venusrem2" or raw.startswith("venusrem2") or raw == "venusrem":
        return "venusrem2"
    if raw.startswith("prosst"):
        return "prosst"
    if raw.startswith("saprot"):
        return "saprot"
    if raw.startswith("progen"):
        return "progen"
    if raw.startswith("protein_mpnn") or raw.startswith("proteinmpnn"):
        return "proteinmpnn"
    if raw.startswith("rita"):
        return "rita"
    if raw.startswith("protssn"):
        return "protssn"
    if raw.startswith("carp") or raw in {"mifst", "mif_st", "mif-st"}:
        return "carp"
    if raw.startswith("s2f") or raw.startswith("s3f"):
        return "s3f"
    if raw.startswith("esm"):
        return "esm"
    return "other"


def model_variant_label(name: str, description: str = "") -> str:
    if name in _VARIANT_LABELS:
        return _VARIANT_LABELS[name]
    return description or name


def _variant_order(series: str, name: str) -> int:
    order = _SERIES_VARIANT_ORDER.get(series) or ()
    try:
        return order.index(name)
    except ValueError:
        return 1000 + len(name)


def _decorate_model_row(row: dict[str, Any]) -> dict[str, Any]:
    series = infer_model_series(row["name"])
    row["series"] = series
    row["series_label"] = _SERIES_LABELS.get(series, series)
    row["label"] = model_variant_label(row["name"], row.get("description") or "")
    row["variant_order"] = _variant_order(series, row["name"])
    return row


def list_model_payload() -> list[dict[str, Any]]:
    from vrh.models import list_models

    rows = []
    for spec in list_models():
        hint = model_size_hint(spec.name) or MODEL_SIZE_HINTS.get(spec.name)
        rows.append(
            _decorate_model_row(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "default_model_id": spec.default_model_id,
                    "needs_pdb": spec.needs_pdb,
                    "needs_msa": False,
                    "input_kind": "structure" if spec.needs_pdb else "sequence",
                    "extras": spec.extras,
                    "notes": spec.notes,
                    "aliases": list(spec.aliases),
                    "supports_mask": spec.supports_mask,
                    "supports_tf": spec.supports_tf,
                    "size_hint": hint,
                }
            )
        )
    if not any(row["name"] == "venusrem2" for row in rows):
        rows.insert(
            0,
            _decorate_model_row(
                {
                    "name": "venusrem2",
                    "description": "Official ProSST ensemble",
                    "default_model_id": None,
                    "needs_pdb": True,
                    "needs_msa": False,
                    "input_kind": "structure",
                    "extras": "prosst",
                    "notes": "wt only; six official ProSST checkpoints; MSA optional (none → α=0)",
                    "aliases": ["prosst_ensemble"],
                    "supports_mask": False,
                    "supports_tf": False,
                    "size_hint": MODEL_SIZE_HINTS.get("venusrem2"),
                }
            ),
        )
    rows.sort(
        key=lambda row: (
            row["name"] != "venusrem2",
            row.get("variant_order", 1000),
            row["name"],
        )
    )
    return rows
