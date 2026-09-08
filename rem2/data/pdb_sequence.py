"""Read a wild-type amino-acid sequence from a PDB file."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from rem2.scoring.structure_weights import THREE_TO_ONE


def extract_sequence_from_pdb(
    pdb_path: Union[str, Path],
    chain: Optional[str] = None,
) -> tuple[str, str, str]:
    """Return ``(name, sequence, chain_id)`` from CA / standard residues.

    Prefers chain ``A`` when ``chain`` is omitted, otherwise the first polymer
    chain that yields a standard amino-acid sequence. Residue order follows
    the PDB chain, and mutant numbering is 1-based along this sequence.
    """
    from Bio.PDB import PDBParser

    path = Path(pdb_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDB not found: {path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))
    model = next(structure.get_models())
    chains = list(model.get_chains())
    if not chains:
        raise ValueError(f"No chains in PDB: {path}")

    wanted = (chain or "").strip() or None
    if wanted:
        chosen = next((c for c in chains if c.id == wanted), None)
        if chosen is None:
            ids = ", ".join(c.id for c in chains) or "(none)"
            raise ValueError(f"Chain {wanted!r} not in {path} (have {ids})")
        sequence = _chain_sequence(chosen)
        if not sequence:
            raise ValueError(f"No standard amino acids in chain {wanted} of {path}")
        return path.stem, sequence, chosen.id

    preferred = next((c for c in chains if c.id == "A"), None)
    candidates = [preferred] if preferred is not None else []
    candidates.extend(c for c in chains if c is not preferred)
    for cand in candidates:
        sequence = _chain_sequence(cand)
        if sequence:
            return path.stem, sequence, cand.id
    raise ValueError(f"No standard amino acids in PDB: {path}")


def _chain_sequence(chain) -> str:
    letters: list[str] = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        aa = THREE_TO_ONE.get(residue.get_resname().upper())
        if aa is None:
            continue
        letters.append(aa)
    return "".join(letters)
