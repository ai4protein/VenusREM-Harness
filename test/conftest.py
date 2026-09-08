"""Shared fixtures for VenusREM2 integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from helpers import FIXTURE_ROOT, PROTEIN, SEQUENCE


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return FIXTURE_ROOT


@pytest.fixture(scope="session")
def protein_name() -> str:
    return PROTEIN


@pytest.fixture(scope="session")
def sequence() -> str:
    return SEQUENCE


@pytest.fixture(scope="session")
def fasta_path(fixture_root: Path) -> Path:
    path = fixture_root / "aa_seq" / f"{PROTEIN}.fasta"
    assert path.is_file(), path
    return path


@pytest.fixture(scope="session")
def pdb_path(fixture_root: Path) -> Path:
    path = fixture_root / "pdbs" / f"{PROTEIN}.pdb"
    assert path.is_file(), path
    return path


@pytest.fixture(scope="session")
def mutant_csv(fixture_root: Path) -> Path:
    path = fixture_root / "substitutions" / f"{PROTEIN}.csv"
    assert path.is_file(), path
    return path


@pytest.fixture(scope="session")
def msa_path(fixture_root: Path) -> Path:
    path = fixture_root / "aa_seq_aln_a2m" / f"{PROTEIN}.a2m"
    assert path.is_file(), path
    return path


@pytest.fixture(scope="session")
def struc_fasta_path(fixture_root: Path) -> Path:
    path = fixture_root / "struc_seq" / f"{PROTEIN}.fasta"
    assert path.is_file(), path
    return path
