"""CLI argument defaults and --base_dir auto-pick."""

from __future__ import annotations

import pytest

from vrh.config import create_parser, postprocess_args


def _parse(*argv):
    return postprocess_args(create_parser().parse_args(list(argv)))


def test_cli_prog_is_vrh():
    assert create_parser().prog == "vrh"


def test_package_version_matches_pyproject():
    import vrh
    from pathlib import Path

    text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text()
    line = next(item for item in text.splitlines() if item.startswith("version = "))
    assert vrh.__version__ == line.split("=", 1)[1].strip().strip('"')


def test_help_shows_short_usage_and_examples():
    help_text = create_parser().format_help()
    assert "usage: vrh [--model MODEL] (--base_dir DIR | --fasta FILE | --pdb FILE)" in help_text
    assert "vrh --model esm2 --base_dir data/proteingym_v1" in help_text
    assert "vrh --model venusrem2 --base_dir data/proteingym_v1" in help_text
    assert "vrh --model prosst-4096 --base_dir data/proteingym_v1" in help_text
    assert "vrh --model saprot --base_dir data/proteingym_v1" in help_text
    assert "vrh --model proteinmpnn-020 --base_dir data/proteingym_v1" in help_text
    assert "--scoring_strategy  wt (default) | mask | tf" in help_text
    assert "Missing checkpoint" in help_text
    assert "Missing data" in help_text
    assert "vrh --model esm2 --fasta prot.fasta" in help_text
    assert "vrh --list-models" in help_text
    assert "vrh demo" in help_text
    assert "vrh doctor" in help_text
    assert "vrh dashboard" in help_text
    assert "vrh download" in help_text
    assert "common:" in help_text
    assert "dataset (--base_dir):" in help_text
    assert "single protein (--fasta / --pdb):" in help_text
    assert "vrh --model saprot --pdb prot.pdb" in help_text
    assert "vrh --model prosst-2048 --pdb prot.pdb" in help_text
    assert "vrh scoring:" in help_text
    assert "per-protein entropy-α" in help_text
    assert "wt-marginals" in help_text
    assert "{backbone}__vrh" in help_text
    assert "other names are ablations" in help_text
    assert "Give vrh the least you have" in help_text
    assert "aa_seq/ is optional" in help_text
    assert "coherence gate" in help_text


def test_invalid_alpha_exits():
    with pytest.raises(SystemExit, match="--alpha must be"):
        _parse("--model", "esm2", "--fasta", "prot.fasta", "--alpha", "hotdog")


def test_trust_remote_code_defaults_off():
    args = _parse("--model", "auto", "--model_id", "some/repo", "--fasta", "prot.fasta")
    assert args.trust_remote_code is False
    args = _parse(
        "--model",
        "auto",
        "--model_id",
        "some/repo",
        "--fasta",
        "prot.fasta",
        "--trust_remote_code",
    )
    assert args.trust_remote_code is True


def test_official_prosst_models_enable_their_required_remote_code():
    from types import SimpleNamespace

    from vrh.backbone.baseline_dispatch import should_trust_remote_code

    args = SimpleNamespace(trust_remote_code=False)
    assert should_trust_remote_code("AI4Protein/ProSST-20", args) is True
    assert should_trust_remote_code("AI4Protein/ProSST-4096", args) is True
    assert should_trust_remote_code("third-party/custom-model", args) is False


def test_out_scores_dir_defaults_to_result():
    args = _parse("--model", "esm2", "--base_dir", "data/foo")
    assert args.out_scores_dir == "result"


def test_fasta_defaults_to_single_site_saturation():
    args = _parse("--model", "esm2", "--fasta", "prot.fasta")
    assert args.mutant_sites == "1"


def test_pdb_defaults_to_single_site_saturation():
    args = _parse("--model", "saprot", "--pdb", "prot.pdb")
    assert args.mutant_sites == "1"
    assert args.fasta is None


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


def test_needed_structure_vocab_sizes():
    from vrh.baseline.prosst.structure_tokens import needed_structure_vocab_sizes

    assert needed_structure_vocab_sizes("prosst-4096") == [4096]
    assert needed_structure_vocab_sizes("prosst") == [2048]
    assert needed_structure_vocab_sizes("venusrem2") == [20, 128, 512, 1024, 2048, 4096]
