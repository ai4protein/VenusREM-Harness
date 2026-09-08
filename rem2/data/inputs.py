"""Dataset presence checks and expected-folder messages."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional


def _dir_has_suffix(path: Optional[str], suffixes: tuple[str, ...]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return any(n.endswith(suffixes) for n in names)


def _existing_for_proteins(directory: Optional[str], names: Iterable[str], suffixes: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    if not directory:
        return found
    for name in names:
        for suffix in suffixes:
            if os.path.exists(os.path.join(directory, f"{name}{suffix}")):
                found.append(name)
                break
    return found


def expected_layout_text(model_key: str = "esm2") -> str:
    return f"""\
Expected data layout (what you pass with --base_dir):

  aa_seq/<protein>.fasta              required (wild-type sequence)
  substitutions/<protein>.csv         required (mutant[,DMS_score]); or use --fasta

  aa_seq_aln_a2m*/<protein>.a2m       MSA for entropy-α  (missing → α=0)
  pdbs*/<protein>.pdb                 RSA / pLDDT and structure models
  struc_seq*/<protein>.fasta          ProSST / VenusREM2 structure tokens

Single protein:  rem2 --fasta prot.fasta [--pdb prot.pdb] [--mutants m.csv]
Dataset:         rem2 --model {model_key} --base_dir data/my_assay
"""


def format_missing_data(problems: list[str], model_key: str) -> str:
    body = "\n".join(f"  - {item}" for item in problems)
    return (
        "No data (or incomplete data) for this run.\n"
        f"{body}\n\n"
        f"{expected_layout_text(model_key)}"
    )


def require_run_inputs(args: Any, model_key: str, protein_names: list[str]) -> None:
    """Exit with a layout hint when required inputs are missing."""
    from rem2.models.registry import get_model
    from rem2.naming import is_prosst_key

    try:
        spec = get_model(model_key).spec
    except KeyError:
        spec = None
    problems: list[str] = []

    aa_dir = getattr(args, "aa_seq_dir", None)
    if not aa_dir or not os.path.isdir(aa_dir):
        problems.append(f"FASTA directory missing: {aa_dir or '(not set)'}  (need aa_seq/*.fasta)")
    elif not protein_names:
        problems.append(f"No FASTA files in {aa_dir} (need <protein>.fasta)")

    mut_dir = getattr(args, "mutant_dir", None)
    if not mut_dir or not os.path.isdir(mut_dir):
        problems.append(
            f"Mutant CSV directory missing: {mut_dir or '(not set)'}  "
            "(need substitutions/*.csv, or --fasta without --mutants)"
        )
    elif protein_names:
        have = _existing_for_proteins(mut_dir, protein_names, (".csv",))
        if not have:
            problems.append(
                f"No mutant CSVs in {mut_dir} for {len(protein_names)} protein(s) "
                f"(need <protein>.csv with a 'mutant' column)"
            )

    needs_pdb = bool(spec and spec.needs_pdb)
    pdb_dir = getattr(args, "pdb_dir", None)
    pdb_file = getattr(args, "pdb", None)
    has_explicit_pdb = bool(pdb_file and os.path.exists(pdb_file))
    if needs_pdb and not has_explicit_pdb:
        if protein_names:
            have = _existing_for_proteins(pdb_dir, protein_names, (".pdb",))
            missing = [n for n in protein_names if n not in have]
            if missing:
                preview = ", ".join(missing[:5])
                more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
                problems.append(
                    f"--model {model_key} needs PDB for: {preview}{more} "
                    f"(--pdb or {pdb_dir or '<pdb_dir>'}/<protein>.pdb)"
                )
        elif not _dir_has_suffix(pdb_dir, (".pdb",)):
            problems.append(
                f"--model {model_key} needs PDB files "
                f"(--pdb or --pdb_dir / <base_dir>/pdbs*/<protein>.pdb)"
            )

    if is_prosst_key(model_key):
        struc_dir = getattr(args, "struc_seq_dir", None)
        if protein_names:
            have = _existing_for_proteins(struc_dir, protein_names, (".fasta", ".fa"))
            missing = [n for n in protein_names if n not in have]
            if missing:
                preview = ", ".join(missing[:5])
                more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
                problems.append(
                    f"--model {model_key} needs structure-token FASTA for: {preview}{more} "
                    f"({struc_dir or '<struc_seq_dir>'}/<protein>.fasta)"
                )
        elif not _dir_has_suffix(struc_dir, (".fasta", ".fa")):
            problems.append(
                f"--model {model_key} needs ProSST structure-token FASTA "
                f"(--struc_seq_dir / <base_dir>/struc_seq*/<protein>.fasta)"
            )

    if problems:
        raise SystemExit(format_missing_data(problems, model_key))
