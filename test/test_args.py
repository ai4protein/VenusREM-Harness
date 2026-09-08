"""CLI argument defaults and --base_dir auto-pick."""

from __future__ import annotations

import pytest

from rem2.config import create_parser, postprocess_args


def _parse(*argv):
    return postprocess_args(create_parser().parse_args(list(argv)))


def test_cli_prog_is_rem2():
    assert create_parser().prog == "rem2"


def test_help_shows_short_usage_and_examples():
    help_text = create_parser().format_help()
    assert "usage: rem2 [--model MODEL] (--base_dir DIR | --fasta FILE)" in help_text
    assert "rem2 --model esm2 --base_dir data/proteingym_v1" in help_text
    assert "rem2 --model venusrem2 --base_dir data/proteingym_v1" in help_text
    assert "rem2 --model prosst-4096 --base_dir data/proteingym_v1" in help_text
    assert "rem2 --model saprot --base_dir data/proteingym_v1" in help_text
    assert "rem2 --model proteinmpnn-020 --base_dir data/proteingym_v1" in help_text
    assert "--scoring_strategy  wt (default) | mask | tf" in help_text
    assert "Missing checkpoint" in help_text
    assert "Missing data" in help_text
    assert "rem2 --model esm2 --fasta prot.fasta" in help_text
    assert "rem2 --list-models" in help_text
    assert "rem2 demo" in help_text
    assert "rem2 doctor" in help_text
    assert "common:" in help_text
    assert "dataset (--base_dir):" in help_text
    assert "single protein (--fasta):" in help_text
    assert "rem2 scoring:" in help_text
    assert "per-protein entropy-α" in help_text
    assert "wt-marginals" in help_text
    assert "{backbone}__rem2" in help_text
    assert "other names are ablations" in help_text
    assert "coherence gate" in help_text


def test_invalid_alpha_exits():
    with pytest.raises(SystemExit, match="--alpha must be"):
        _parse("--model", "esm2", "--fasta", "prot.fasta", "--alpha", "hotdog")


def test_out_scores_dir_defaults_to_result():
    args = _parse("--model", "esm2", "--base_dir", "data/foo")
    assert args.out_scores_dir == "result"


def test_fasta_defaults_to_single_site_saturation():
    args = _parse("--model", "esm2", "--fasta", "prot.fasta")
    assert args.mutant_sites == "1"


def test_fasta_keeps_explicit_mutant_sites():
    args = _parse("--model", "esm2", "--fasta", "prot.fasta", "--mutant_sites", "1,2")
    assert args.mutant_sites == "1,2"


def test_base_dir_picks_existing_subdirs(tmp_path):
    (tmp_path / "aa_seq_aln_a2m_af2cf").mkdir()
    (tmp_path / "pdbs_af2_assay_resolved_full").mkdir()
    (tmp_path / "struc_seq_af2_assay_resolved_full").mkdir()
    args = _parse("--model", "esm2", "--base_dir", str(tmp_path))
    assert args.aa_seq_aln_dir == str(tmp_path / "aa_seq_aln_a2m_af2cf")
    assert args.pdb_dir == str(tmp_path / "pdbs_af2_assay_resolved_full")
    assert args.struc_seq_dir == str(tmp_path / "struc_seq_af2_assay_resolved_full")
    assert args.aa_seq_dir == str(tmp_path / "aa_seq")
    assert args.mutant_dir == str(tmp_path / "substitutions")


def test_base_dir_falls_back_to_short_names(tmp_path):
    args = _parse("--model", "esm2", "--base_dir", str(tmp_path))
    assert args.aa_seq_aln_dir == str(tmp_path / "aa_seq_aln_a2m")
    assert args.pdb_dir == str(tmp_path / "pdbs")
    assert args.struc_seq_dir == str(tmp_path / "struc_seq")


def test_base_dir_prefers_short_af2_over_missing_full(tmp_path):
    (tmp_path / "af2").mkdir()
    (tmp_path / "aa_seq_aln_a2m").mkdir()
    args = _parse("--model", "esm2", "--base_dir", str(tmp_path))
    assert args.pdb_dir == str(tmp_path / "af2")
    assert args.aa_seq_aln_dir == str(tmp_path / "aa_seq_aln_a2m")


def test_explicit_data_dirs_win(tmp_path):
    (tmp_path / "pdbs_af2_assay_resolved_full").mkdir()
    args = _parse(
        "--model",
        "esm2",
        "--base_dir",
        str(tmp_path),
        "--pdb_dir",
        "pdbs",
        "--aa_seq_aln_dir",
        "aa_seq_aln_a2m",
    )
    assert args.pdb_dir == str(tmp_path / "pdbs")
    assert args.aa_seq_aln_dir == str(tmp_path / "aa_seq_aln_a2m")


def test_fasta_rejects_mutants_and_mutant_sites_together():
    with pytest.raises(SystemExit, match="either --mutants"):
        _parse(
            "--model",
            "esm2",
            "--fasta",
            "prot.fasta",
            "--mutants",
            "mut.csv",
            "--mutant_sites",
            "1",
        )
