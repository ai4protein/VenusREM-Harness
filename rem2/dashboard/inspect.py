"""Lightweight metadata for the dashboard (no torch import)."""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from rem2 import __version__
from rem2.user_commands import MODEL_SIZE_HINTS, demo_dataset_dir, model_size_hint


def doctor_report() -> dict[str, Any]:
    from rem2.models.weights import default_cache_dir

    extras = [
        ("biopython", "Bio", "core: FASTA + RSA"),
        ("biotite", "biotite", "core: PDB I/O / RSA"),
        ("fastapi", "fastapi", "core: rem2 dashboard"),
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
    }


def list_model_payload() -> list[dict[str, Any]]:
    from rem2.models import list_models

    rows = []
    for spec in list_models():
        hint = model_size_hint(spec.name) or MODEL_SIZE_HINTS.get(spec.name)
        rows.append(
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
    if not any(row["name"] == "venusrem2" for row in rows):
        rows.insert(
            0,
            {
                "name": "venusrem2",
                "description": "Official ProSST ensemble",
                "default_model_id": None,
                "needs_pdb": True,
                "needs_msa": False,
                "input_kind": "structure",
                "extras": "prosst",
                "notes": "wt only; six official ProSST checkpoints; MSA optional (none → α=0)",
                "aliases": ["venusrem", "prosst_ensemble"],
                "supports_mask": False,
                "supports_tf": False,
                "size_hint": MODEL_SIZE_HINTS.get("venusrem2"),
            },
        )
    rows.sort(key=lambda row: (row["name"] != "venusrem2", row["name"]))
    return rows
