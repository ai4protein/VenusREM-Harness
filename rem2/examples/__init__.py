"""Bundled demo datasets shipped in the rem2 wheel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

DEMO_ASSAY = "HCP_LAMBD_Tsuboyama_2023_2L6Q"
DEMO_PDB_ID = "2L6Q"
DEMO_LABEL = "2L6Q λ repressor"

DEMO_FILE_KINDS = {
    "fasta": (".fasta", ".fa"),
    "pdb": (".pdb",),
    "msa": (".a2m", ".a3m"),
    "mutants": (".csv",),
}

DEMO_PRESETS = (
    {"id": "sequence", "label": "Sequence", "files": ("fasta",)},
    {"id": "structure", "label": "+ Structure", "files": ("fasta", "pdb")},
    {"id": "full", "label": "Full rem2", "files": ("fasta", "pdb", "msa")},
)


def bundled_demo_dir() -> Path:
    return Path(__file__).resolve().parent / DEMO_ASSAY


def resolve_demo_dir() -> Optional[Path]:
    from rem2.download.example import bundled_example_dir, ensure_demo_dataset

    bundled = bundled_example_dir()
    if bundled is not None:
        return bundled
    try:
        path = ensure_demo_dataset(log=lambda *_args, **_kwargs: None)
        if path:
            return Path(path)
    except SystemExit:
        pass
    except Exception:
        pass
    return None


def demo_file_path(kind: str, root: Optional[Path] = None) -> Optional[Path]:
    suffixes = DEMO_FILE_KINDS.get(kind)
    if not suffixes:
        return None
    root = Path(root) if root is not None else resolve_demo_dir()
    if root is None or not root.exists():
        return None
    hits = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    return sorted(hits)[0] if hits else None


def demo_example_payload() -> dict:
    root = resolve_demo_dir()
    files = {}
    if root is not None:
        for kind in DEMO_FILE_KINDS:
            path = demo_file_path(kind, root)
            if path is not None:
                files[kind] = path.name
    return {
        "id": "demo",
        "label": DEMO_LABEL,
        "assay": DEMO_ASSAY,
        "pdb_id": DEMO_PDB_ID,
        "default_preset": "full",
        "presets": [
            {"id": item["id"], "label": item["label"], "files": list(item["files"])}
            for item in DEMO_PRESETS
        ],
        "files": files,
    }
