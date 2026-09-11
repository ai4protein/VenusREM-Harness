"""Generate n-point saturation mutagenesis libraries for VenusREM2 scoring."""

from __future__ import annotations

import csv
import math
import re
from itertools import combinations, product
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_MAX_MUTANTS = 1_000_000


def parse_orders(spec: str) -> List[int]:
    """Parse ``--mutant_sites`` like ``1``, ``1,2``, ``1,2,3`` into sorted unique orders."""
    if not spec or not str(spec).strip():
        raise ValueError("--mutant_sites must be a non-empty list of positive integers, e.g. 1 or 1,2,3")
    orders: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --mutant_sites entry '{part}': expected positive integers "
                "(1=single, 2=double, 3=triple, ...)"
            ) from exc
        if n < 1:
            raise ValueError(f"Invalid --mutant_sites entry {n}: must be >= 1")
        orders.append(n)
    if not orders:
        raise ValueError("--mutant_sites must contain at least one order")
    return sorted(set(orders))


def parse_positions_list(spec: Optional[str]) -> List[int]:
    """Parse ``--positions 10,11,12`` into 1-based positions."""
    if not spec:
        return []
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        pos = int(part)
        if pos < 1:
            raise ValueError(f"Invalid position {pos}: must be 1-based (>=1)")
        out.append(pos)
    return out


def parse_residue_range(spec: Optional[str]) -> List[int]:
    """Parse ``--residue_range 10-20`` (inclusive) into 1-based positions."""
    if not spec:
        return []
    text = str(spec).strip()
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
    if not m:
        raise ValueError(
            f"Invalid --residue_range '{spec}': expected START-END, e.g. 10-20"
        )
    start, end = int(m.group(1)), int(m.group(2))
    if start < 1 or end < start:
        raise ValueError(f"Invalid --residue_range '{spec}': require 1 <= START <= END")
    return list(range(start, end + 1))


def resolve_positions(
    sequence: str,
    positions: Optional[str] = None,
    residue_range: Optional[str] = None,
    *,
    orders: Optional[Sequence[int]] = None,
) -> List[int]:
    """Resolve residue set ``P`` (1-based) for mutagenesis.

    - If orders include 2+: ``positions`` and/or ``residue_range`` are required.
    - If orders are only {1} and neither is set: use the full sequence.
    """
    length = len(sequence)
    if length == 0:
        raise ValueError("Empty sequence")

    pos = parse_positions_list(positions)
    rng = parse_residue_range(residue_range)
    merged = sorted(set(pos) | set(rng))

    order_list = list(orders or [])
    needs_explicit = any(n >= 2 for n in order_list)

    if not merged:
        if needs_explicit:
            raise ValueError(
                "Orders >= 2 require --positions and/or --residue_range "
                "to limit combinatorial explosion"
            )
        return list(range(1, length + 1))

    for p in merged:
        if p > length:
            raise ValueError(
                f"Position {p} is out of range for sequence length {length}"
            )
    return merged


def estimate_mutant_count(n_positions: int, orders: Sequence[int]) -> int:
    """Exact library size without materializing mutants: sum_n C(k,n)*19^n."""
    total = 0
    k = n_positions
    for n in orders:
        if n > k:
            continue
        total += math.comb(k, n) * (19 ** n)
    return total


def _aa_choices(wt: str) -> List[str]:
    return [aa for aa in AMINO_ACIDS if aa != wt]


def generate_npoint_saturation(
    sequence: str,
    orders: Sequence[int],
    positions_1based: Sequence[int],
    *,
    max_mutants: int = DEFAULT_MAX_MUTANTS,
) -> List[str]:
    """Generate mutant strings for the requested orders over positions."""
    seq = sequence.strip().upper()
    for ch in seq:
        if ch not in AMINO_ACIDS:
            raise ValueError(f"Non-standard amino acid '{ch}' in sequence")

    positions = sorted(set(int(p) for p in positions_1based))
    for p in positions:
        if p < 1 or p > len(seq):
            raise ValueError(f"Position {p} out of range for L={len(seq)}")

    est = estimate_mutant_count(len(positions), orders)
    if est > max_mutants:
        raise ValueError(
            f"Requested library has {est:,} mutants (> --max_mutants={max_mutants:,}). "
            "Narrow --positions / --residue_range or raise --max_mutants."
        )

    mutants: List[str] = []
    for n in sorted(set(orders)):
        if n > len(positions):
            continue
        for site_tuple in combinations(positions, n):
            choice_lists = [_aa_choices(seq[p - 1]) for p in site_tuple]
            for aas in product(*choice_lists):
                parts = [
                    f"{seq[p - 1]}{p}{aa}" for p, aa in zip(site_tuple, aas)
                ]
                mutants.append(":".join(parts))
                if len(mutants) > max_mutants:
                    raise ValueError(
                        f"Generated more than --max_mutants={max_mutants:,} mutants"
                    )
    return mutants


def write_mutant_csv(
    mutants: Iterable[str],
    out_path: Union[str, Path],
    *,
    dms_score: float = 0.0,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mutant", "DMS_score"])
        for m in mutants:
            writer.writerow([m, dms_score])
    return path


def read_fasta_sequence(fasta_path: Union[str, Path]) -> tuple[str, str]:
    """Return ``(name, sequence)`` from a single-record FASTA."""
    path = Path(fasta_path)
    if not path.is_file():
        raise FileNotFoundError(f"FASTA not found: {path}")
    name = path.stem
    seq_parts: List[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:].split()[0] if line[1:] else path.stem
                name = header
                continue
            seq_parts.append(line)
    sequence = "".join(seq_parts).upper()
    if not sequence:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return name, sequence


def write_fasta(path: Union[str, Path], name: str, sequence: str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f">{name}\n{sequence}\n")
    return dest


def materialize_single_protein_inputs(
    *,
    fasta_path: Optional[Union[str, Path]] = None,
    out_root: Union[str, Path],
    pdb_path: Optional[Union[str, Path]] = None,
    mutants_path: Optional[Union[str, Path]] = None,
    mutant_sites: Optional[str] = None,
    positions: Optional[str] = None,
    residue_range: Optional[str] = None,
    max_mutants: int = DEFAULT_MAX_MUTANTS,
    pdb_chain: Optional[str] = None,
) -> dict:
    """Build a mini ``base_dir`` layout under ``out_root`` for single-protein scoring.

    ``fasta_path`` or ``pdb_path`` is required. PDB-only mode writes a FASTA
    from the structure sequence (chain A, or ``pdb_chain``).

    Returns dict with keys: name, sequence, aa_seq_dir, mutant_dir, pdb_dir,
    mutant_csv, n_mutants, pdb_chain.
    """
    import shutil

    from vrh.data.pdb_sequence import extract_sequence_from_pdb

    if not fasta_path and not pdb_path:
        raise ValueError("Need --fasta or --pdb")

    out_root = Path(out_root)
    pdb_seq = None
    used_chain = None
    if pdb_path:
        pdb_path = Path(pdb_path)
        if not pdb_path.is_file():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")
        _pdb_name, pdb_seq, used_chain = extract_sequence_from_pdb(pdb_path, chain=pdb_chain)

    if fasta_path:
        fasta_path = Path(fasta_path)
        name, sequence = read_fasta_sequence(fasta_path)
    else:
        name, sequence = Path(pdb_path).stem, pdb_seq

    aa_seq_dir = out_root / "aa_seq"
    mutant_dir = out_root / "substitutions"
    pdb_dir = out_root / "pdbs"
    gen_dir = out_root.parent / "generated_mutants" if out_root.name == "_inputs" else out_root / "generated_mutants"
    aa_seq_dir.mkdir(parents=True, exist_ok=True)
    mutant_dir.mkdir(parents=True, exist_ok=True)

    dest_fasta = aa_seq_dir / f"{name}.fasta"
    if fasta_path:
        if fasta_path.resolve() != dest_fasta.resolve():
            shutil.copy2(fasta_path, dest_fasta)
    else:
        write_fasta(dest_fasta, name, sequence)

    dest_pdb = None
    if pdb_path:
        pdb_dir.mkdir(parents=True, exist_ok=True)
        dest_pdb = pdb_dir / f"{name}.pdb"
        if pdb_path.resolve() != dest_pdb.resolve():
            shutil.copy2(pdb_path, dest_pdb)

    dest_csv = mutant_dir / f"{name}.csv"
    if mutants_path:
        mutants_path = Path(mutants_path)
        if not mutants_path.is_file():
            raise FileNotFoundError(f"Mutants CSV not found: {mutants_path}")
        if mutants_path.resolve() != dest_csv.resolve():
            shutil.copy2(mutants_path, dest_csv)
        # Count rows excluding header
        with dest_csv.open() as fh:
            n_mutants = max(sum(1 for _ in fh) - 1, 0)
    else:
        if not mutant_sites:
            raise ValueError(
                "Single-protein mode without --mutants requires --mutant_sites "
                "(e.g. 1 or 1,2,3)"
            )
        orders = parse_orders(mutant_sites)
        pos = resolve_positions(
            sequence,
            positions=positions,
            residue_range=residue_range,
            orders=orders,
        )
        mutants = generate_npoint_saturation(
            sequence, orders, pos, max_mutants=max_mutants
        )
        write_mutant_csv(mutants, dest_csv)
        gen_dir.mkdir(parents=True, exist_ok=True)
        write_mutant_csv(mutants, gen_dir / f"{name}.csv")
        n_mutants = len(mutants)

    return {
        "name": name,
        "sequence": sequence,
        "aa_seq_dir": str(aa_seq_dir),
        "mutant_dir": str(mutant_dir),
        "pdb_dir": str(pdb_dir) if dest_pdb else None,
        "mutant_csv": str(dest_csv),
        "n_mutants": n_mutants,
        "pdb_chain": used_chain,
        "pdb_sequence": pdb_seq,
    }
