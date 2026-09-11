from __future__ import annotations

from vrh.api import build_score_argv
from vrh.scoring.logits_cache import load_cached_logits
from vrh.user_commands import build_demo_argv


def test_demo_argv_injects_small_model_and_fixture(tmp_path, monkeypatch):
    (tmp_path / "aa_seq").mkdir()
    monkeypatch.setattr("vrh.download.example.ensure_demo_dataset", lambda **k: tmp_path)
    argv = build_demo_argv([])
    assert "--model" in argv
    assert "esm2-8m" in argv
    assert "--base_dir" in argv
    base = argv[argv.index("--base_dir") + 1]
    assert base == str(tmp_path)


def test_demo_argv_keeps_user_model(tmp_path, monkeypatch):
    (tmp_path / "aa_seq").mkdir()
    monkeypatch.setattr("vrh.download.example.ensure_demo_dataset", lambda **k: tmp_path)
    argv = build_demo_argv(["--model", "esm2"])
    assert argv.count("--model") == 1
    assert argv[argv.index("--model") + 1] == "esm2"


def test_build_score_argv_fasta_and_dataset():
    argv = build_score_argv(fasta="prot.fasta", mutants="m.csv", model="esm2-8m", out_dir="out")
    assert argv[:4] == ["--model", "esm2-8m", "--out_scores_dir", "out"]
    assert "--fasta" in argv and "prot.fasta" in argv
    pdb_only = build_score_argv(pdb="prot.pdb", mutants="m.csv", model="saprot", out_dir="out")
    assert "--pdb" in pdb_only and "prot.pdb" in pdb_only
    assert "--fasta" not in pdb_only
    assert "--model" in pdb_only and "saprot" in pdb_only
    dataset = build_score_argv(base_dir="data/foo", extra_argv=["--alpha", "0"])
    assert "--base_dir" in dataset and "data/foo" in dataset
    assert dataset[-2:] == ["--alpha", "0"]


def test_cache_miss_policy_error_ignored_without_reuse():
    logits, final = load_cached_logits(
        logits_cache_path=None,
        sequence="ACDE",
        reuse_logits_cache=False,
        cache_miss_policy="error",
        logits_cache_stage="raw",
        device="cpu",
        log_local=lambda m: None,
        log_warn_local=lambda m: None,
    )
    assert logits is None
    assert final is False
