"""Unit tests for n-point saturation mutagenesis."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from vrh.data.mutagenesis import (
    estimate_mutant_count,
    generate_npoint_saturation,
    materialize_single_protein_inputs,
    parse_orders,
    resolve_positions,
    write_mutant_csv,
)


def test_parse_orders():
    assert parse_orders("1") == [1]
    assert parse_orders("1,2,3") == [1, 2, 3]
    assert parse_orders("3,1,2,1") == [1, 2, 3]
    with pytest.raises(ValueError):
        parse_orders("0")
    with pytest.raises(ValueError):
        parse_orders("a")


def test_resolve_positions_requires_explicit_for_double():
    seq = "ACDE"
    with pytest.raises(ValueError, match="Orders >= 2"):
        resolve_positions(seq, orders=[1, 2])
    assert resolve_positions(seq, orders=[1]) == [1, 2, 3, 4]
    assert resolve_positions(seq, positions="2,4", orders=[2]) == [2, 4]
    assert resolve_positions(seq, residue_range="2-3", positions="4", orders=[2]) == [2, 3, 4]


def test_generate_single_and_double_counts():
    seq = "ACD"  # L=3
    pos = [1, 2, 3]
    singles = generate_npoint_saturation(seq, [1], pos)
    assert len(singles) == 3 * 19
    assert "A1C" in singles
    assert "A1A" not in singles

    doubles = generate_npoint_saturation(seq, [2], pos)
    assert len(doubles) == math.comb(3, 2) * 19 * 19
    assert all(":" in m for m in doubles)
    # Site order ascending
    assert doubles[0].startswith("A1")

    both = generate_npoint_saturation(seq, [1, 2], pos)
    assert len(both) == len(singles) + len(doubles)
    assert estimate_mutant_count(3, [1, 2]) == len(both)


def test_max_mutants_guard():
    seq = "ACDEFGHIKL"
    with pytest.raises(ValueError, match="max_mutants"):
        generate_npoint_saturation(seq, [1, 2, 3], list(range(1, 11)), max_mutants=1000)


def test_materialize_single_protein(tmp_path: Path):
    fasta = tmp_path / "toy.fasta"
    fasta.write_text(">toy\nACD\n")
    out = tmp_path / "run" / "_inputs"
    meta = materialize_single_protein_inputs(
        fasta_path=fasta,
        out_root=out,
        mutant_sites="1,2",
        positions="1,2,3",
        max_mutants=1_000_000,
    )
    assert meta["name"] == "toy"
    assert meta["n_mutants"] == 3 * 19 + math.comb(3, 2) * 19 * 19
    csv_path = Path(meta["mutant_csv"])
    assert csv_path.is_file()
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "mutant,DMS_score"
    assert len(lines) - 1 == meta["n_mutants"]
    # also copied under generated_mutants
    gen = tmp_path / "run" / "generated_mutants" / "toy.csv"
    assert gen.is_file()


def test_materialize_from_pdb_only(tmp_path: Path):
    from vrh.data.pdb_sequence import extract_sequence_from_pdb

    pdb = Path(__file__).resolve().parent / "fixtures" / "trp_cage" / "pdbs" / "trp_cage.pdb"
    name, sequence, chain = extract_sequence_from_pdb(pdb)
    assert name == "trp_cage"
    assert sequence.startswith("NLYIQ")
    assert chain == "A"

    out = tmp_path / "_inputs"
    meta = materialize_single_protein_inputs(
        pdb_path=pdb,
        out_root=out,
        mutant_sites="1",
        max_mutants=1_000_000,
    )
    assert meta["name"] == "trp_cage"
    assert meta["sequence"] == sequence
    assert meta["pdb_chain"] == "A"
    assert Path(meta["aa_seq_dir"], "trp_cage.fasta").is_file()
    assert Path(meta["pdb_dir"], "trp_cage.pdb").is_file()
    assert meta["n_mutants"] == len(sequence) * 19


def test_write_mutant_csv(tmp_path: Path):
    path = write_mutant_csv(["A1C", "A1C:D2E"], tmp_path / "m.csv")
    text = path.read_text()
    assert "A1C:D2E" in text
