"""Least-input dataset fill: FASTA / ProSST tokens from PDB."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from vrh.config import create_parser, postprocess_args
from vrh.data.inputs import (
    _has_struc_tokens,
    fill_aa_seq_from_pdb,
    fill_prosst_tokens_from_pdb,
    require_run_inputs,
)
from vrh.scoring.run_utils import read_names

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "trp_cage"


def test_fill_aa_seq_from_pdb(tmp_path):
    args = SimpleNamespace(
        aa_seq_dir=str(tmp_path / "missing_aa"),
        pdb_dir=str(FIXTURE / "pdbs"),
        out_scores_dir=str(tmp_path / "out"),
        pdb_chain=None,
    )
    assert fill_aa_seq_from_pdb(args) == 1
    fasta = Path(args.aa_seq_dir) / "trp_cage.fasta"
    assert fasta.is_file()
    assert "NLYIQ" in fasta.read_text()


def test_fill_aa_seq_skips_when_fasta_exists(tmp_path):
    aa = tmp_path / "aa_seq"
    aa.mkdir()
    (aa / "keep.fasta").write_text(">keep\nACDE\n")
    args = SimpleNamespace(
        aa_seq_dir=str(aa),
        pdb_dir=str(FIXTURE / "pdbs"),
        out_scores_dir=str(tmp_path / "out"),
        pdb_chain=None,
    )
    assert fill_aa_seq_from_pdb(args) == 0
    assert args.aa_seq_dir == str(aa)


def test_dataset_pdb_only_fills_fasta(tmp_path):
    base = tmp_path / "assay"
    (base / "pdbs").mkdir(parents=True)
    (base / "substitutions").mkdir()
    shutil.copy(FIXTURE / "pdbs" / "trp_cage.pdb", base / "pdbs" / "trp_cage.pdb")
    shutil.copy(
        FIXTURE / "substitutions" / "trp_cage.csv",
        base / "substitutions" / "trp_cage.csv",
    )
    args = postprocess_args(
        create_parser().parse_args(
            [
                "--model",
                "esm2",
                "--base_dir",
                str(base),
                "--out_scores_dir",
                str(tmp_path / "out"),
            ]
        )
    )
    assert fill_aa_seq_from_pdb(args) == 1
    names = sorted(read_names(args.aa_seq_dir))
    assert names == ["trp_cage"]
    require_run_inputs(args, "esm2", names)


def test_fill_prosst_skips_non_prosst():
    assert fill_prosst_tokens_from_pdb(SimpleNamespace(), "esm2", ["p"]) == 0


def test_has_struc_tokens_needs_all_ensemble_k(tmp_path):
    root = tmp_path / "struc_seq"
    (root / "2048").mkdir(parents=True)
    (root / "2048" / "p.fasta").write_text(">p\n1\n")
    assert _has_struc_tokens(str(root), "p", [2048])
    assert not _has_struc_tokens(str(root), "p", [20, 128, 512, 1024, 2048, 4096])


def test_fill_prosst_keeps_existing_tokens(tmp_path, monkeypatch):
    user = tmp_path / "struc_seq"
    (user / "2048").mkdir(parents=True)
    (user / "2048" / "keep.fasta").write_text(">keep\n1,2\n")
    pdbs = tmp_path / "pdbs"
    pdbs.mkdir()
    (pdbs / "keep.pdb").write_text("ATOM\n")
    (pdbs / "new.pdb").write_text("ATOM\n")
    generated = []

    def fake_gen(items, dest, vocabs):
        generated.append((list(items), dest, list(vocabs)))
        for _path, name in items:
            nested = Path(dest) / "2048"
            nested.mkdir(parents=True, exist_ok=True)
            (nested / f"{name}.fasta").write_text(f">{name}\n0\n")
            (Path(dest) / f"{name}.fasta").write_text(f">{name}\n0\n")
        return dest

    monkeypatch.setattr(
        "vrh.baseline.prosst.structure_tokens.generate_struc_seq_from_pdbs",
        fake_gen,
    )
    args = SimpleNamespace(
        struc_seq_dir=str(user),
        pdb_dir=str(pdbs),
        out_scores_dir=str(tmp_path / "out"),
        model_name=["AI4Protein/ProSST-2048"],
    )
    assert fill_prosst_tokens_from_pdb(args, "prosst-2048", ["keep", "new"]) == 1
    assert generated[0][0] == [(str(pdbs / "new.pdb"), "new")]
    dest = Path(args.struc_seq_dir)
    assert (dest / "2048" / "keep.fasta").read_text() == ">keep\n1,2\n"
    assert (dest / "2048" / "new.fasta").is_file()


def test_require_prosst_without_pdb_or_tokens(tmp_path):
    args = SimpleNamespace(
        aa_seq_dir=str(tmp_path / "aa_seq"),
        mutant_dir=str(tmp_path / "substitutions"),
        pdb_dir=str(tmp_path / "pdbs"),
        struc_seq_dir=str(tmp_path / "struc_seq"),
        pdb=None,
        model_name=["AI4Protein/ProSST-2048"],
    )
    (tmp_path / "aa_seq").mkdir()
    (tmp_path / "substitutions").mkdir()
    (tmp_path / "aa_seq" / "p.fasta").write_text(">p\nACDE\n")
    (tmp_path / "substitutions" / "p.csv").write_text("mutant\nA1C\n")
    with pytest.raises(SystemExit, match="needs a PDB to build structure tokens"):
        require_run_inputs(args, "prosst-2048", ["p"])
