"""Dataset presence checks and expected-folder messages."""

from __future__ import annotations

import os
import shutil
from typing import Any, Iterable, Optional, Sequence


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
Give rem2 the least you have. Defaults are the full recipe.

Single protein:
  rem2 --fasta prot.fasta [--mutants m.csv]
  rem2 --model saprot --pdb prot.pdb [--mutants m.csv]
  rem2 --model venusrem2 --pdb prot.pdb [--mutants m.csv]

Dataset (--base_dir):
  substitutions/<protein>.csv         mutants (required)
  aa_seq/<protein>.fasta              optional if pdbs/ is present
  pdbs/<protein>.pdb                  enough for saprot / prosst / venusrem2
  aa_seq_aln_a2m*/<protein>.a2m       optional MSA (missing → α=0)
  struc_seq*/                         optional; built from pdbs/ if missing

  rem2 --model {model_key} --base_dir data/my_assay
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
    pdb_dir = getattr(args, "pdb_dir", None)
    pdb_file = getattr(args, "pdb", None)
    has_pdb = bool(
        (pdb_file and os.path.exists(pdb_file))
        or _dir_has_suffix(pdb_dir, (".pdb",))
    )
    if not protein_names:
        if has_pdb:
            problems.append(
                "No proteins to score after reading PDB. "
                "Need substitutions/<protein>.csv with the same stem as the PDB."
            )
        else:
            problems.append(
                "No wild-type sequence. Pass --fasta, --pdb, or --base_dir with aa_seq/ or pdbs/."
            )

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

    if is_prosst_key(model_key) and protein_names:
        from rem2.baseline.prosst.structure_tokens import needed_structure_vocab_sizes

        vocabs = needed_structure_vocab_sizes(model_key, args)
        struc_dir = getattr(args, "struc_seq_dir", None)
        missing = [n for n in protein_names if not _has_struc_tokens(struc_dir, n, vocabs)]
        if missing:
            have_pdb = set(_existing_for_proteins(pdb_dir, missing, (".pdb",)))
            if has_explicit_pdb:
                have_pdb.update(missing)
            still = [n for n in missing if n not in have_pdb]
            if still:
                preview = ", ".join(still[:5])
                more = f" (+{len(still) - 5} more)" if len(still) > 5 else ""
                problems.append(
                    f"--model {model_key} needs a PDB to build structure tokens for: {preview}{more}"
                )

    if problems:
        raise SystemExit(format_missing_data(problems, model_key))


def _has_struc_tokens(
    directory: Optional[str],
    name: str,
    vocab_sizes: Optional[Sequence[int]] = None,
) -> bool:
    if not directory:
        return False
    if vocab_sizes:
        return all(
            os.path.isfile(os.path.join(directory, str(vocab), f"{name}.fasta"))
            or (len(vocab_sizes) == 1 and os.path.isfile(os.path.join(directory, f"{name}.fasta")))
            for vocab in vocab_sizes
        )
    if os.path.exists(os.path.join(directory, f"{name}.fasta")):
        return True
    if not os.path.isdir(directory):
        return False
    try:
        children = os.listdir(directory)
    except OSError:
        return False
    return any(
        os.path.isfile(os.path.join(directory, child, f"{name}.fasta"))
        for child in children
    )


def _copy_struc_tokens(
    src: str,
    dest: str,
    name: str,
    vocab_sizes: Sequence[int],
) -> None:
    src = os.path.abspath(src)
    dest = os.path.abspath(dest)
    if src == dest or not os.path.isdir(src):
        return
    for vocab in vocab_sizes:
        nested_src = os.path.join(src, str(vocab), f"{name}.fasta")
        nested_dest = os.path.join(dest, str(vocab), f"{name}.fasta")
        if os.path.isfile(nested_src) and not os.path.isfile(nested_dest):
            os.makedirs(os.path.dirname(nested_dest), exist_ok=True)
            shutil.copy2(nested_src, nested_dest)
    flat_src = os.path.join(src, f"{name}.fasta")
    flat_dest = os.path.join(dest, f"{name}.fasta")
    if os.path.isfile(flat_src) and not os.path.isfile(flat_dest):
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(flat_src, flat_dest)


def list_pdb_stems(pdb_dir: Optional[str]) -> list[str]:
    if not pdb_dir or not os.path.isdir(pdb_dir):
        return []
    try:
        names = os.listdir(pdb_dir)
    except OSError:
        return []
    return sorted(os.path.splitext(name)[0] for name in names if name.endswith(".pdb"))


def fill_aa_seq_from_pdb(args: Any, logger: Any = None) -> int:
    """If no FASTA is present, write sequences from PDBs under out/_inputs/aa_seq."""
    from rem2.data.mutagenesis import write_fasta
    from rem2.data.pdb_sequence import extract_sequence_from_pdb
    from rem2.scoring.run_utils import read_names

    aa_dir = getattr(args, "aa_seq_dir", None)
    existing = sorted(read_names(aa_dir)) if aa_dir and os.path.isdir(aa_dir) else []
    if existing:
        return 0
    pdb_dir = getattr(args, "pdb_dir", None)
    stems = list_pdb_stems(pdb_dir)
    if not stems:
        return 0
    dest = os.path.join(getattr(args, "out_scores_dir", "result"), "_inputs", "aa_seq")
    os.makedirs(dest, exist_ok=True)
    chain = getattr(args, "pdb_chain", None)
    written = 0
    for name in stems:
        pdb_path = os.path.join(pdb_dir, f"{name}.pdb")
        _stem, sequence, used_chain = extract_sequence_from_pdb(pdb_path, chain=chain)
        write_fasta(os.path.join(dest, f"{name}.fasta"), name, sequence)
        written += 1
        if logger and written == 1:
            logger.info(f"Reading sequences from PDB (chain {used_chain})")
    args.aa_seq_dir = dest
    if logger:
        logger.info(f"Wrote {written} FASTA file(s) from PDB; no aa_seq/ given")
    return written


def fill_prosst_tokens_from_pdb(args: Any, model_key: str, protein_names: list[str], logger: Any = None) -> int:
    """Build missing ProSST tokens from PDBs. Returns how many proteins were tokenized."""
    from rem2.naming import is_prosst_key

    if not is_prosst_key(model_key) or not protein_names:
        return 0
    from rem2.baseline.prosst.structure_tokens import (
        generate_struc_seq_from_pdbs,
        needed_structure_vocab_sizes,
    )

    vocabs = needed_structure_vocab_sizes(model_key, args)
    struc_dir = getattr(args, "struc_seq_dir", None)
    missing = [n for n in protein_names if not _has_struc_tokens(struc_dir, n, vocabs)]
    if not missing:
        return 0
    pdb_dir = getattr(args, "pdb_dir", None)
    items = []
    for name in missing:
        pdb_path = os.path.join(pdb_dir, f"{name}.pdb") if pdb_dir else ""
        if pdb_path and os.path.isfile(pdb_path):
            items.append((pdb_path, name))
    if not items:
        return 0
    dest = os.path.join(getattr(args, "out_scores_dir", "result"), "_inputs", "struc_seq")
    for name in protein_names:
        if struc_dir:
            _copy_struc_tokens(struc_dir, dest, name, vocabs)
    if logger:
        logger.info(
            f"Building ProSST structure tokens from PDB for {len(items)} protein(s) "
            f"(K={','.join(str(v) for v in vocabs)})"
        )
    generate_struc_seq_from_pdbs(items, dest, vocabs)
    args.struc_seq_dir = dest
    return len(items)
