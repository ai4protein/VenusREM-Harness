"""Unzipped HF example used by rem2 demo (no network)."""

from pathlib import Path

from rem2.cli import main
from rem2.download.example import (
    EXAMPLE_FILES,
    EXAMPLE_NAME,
    ensure_demo_dataset,
)


def test_example_dry_run(capsys):
    main(["download", "example", "--dry-run"])
    out = capsys.readouterr().out
    assay = EXAMPLE_NAME
    assert f"example/{assay}/aa_seq/{assay}.fasta" in out
    assert f"example/{assay}/substitutions/{assay}.csv" in out
    assert f"example/{assay}/pdbs/{assay}.pdb" in out
    assert f"example/{assay}/aa_seq_aln_a2m/{assay}.a2m" in out
    assert "tyang816/VenusREM2" in out


def test_ensure_demo_uses_cache_then_hf(tmp_path, monkeypatch):
    dest = tmp_path / EXAMPLE_NAME
    calls = []

    def fake_download(filename, target, force=False, log=print, progress=False, desc=None):
        calls.append(filename)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("ok")
        return Path(target)

    monkeypatch.setattr("rem2.download.example.download_from_venusrem2", fake_download)
    got = ensure_demo_dataset(dest=str(dest), log=lambda *_: None)
    assert got == dest
    assert calls == list(EXAMPLE_FILES)
    assert (dest / "pdbs" / f"{EXAMPLE_NAME}.pdb").is_file()

    calls.clear()
    again = ensure_demo_dataset(dest=str(dest), log=lambda *_: None)
    assert again == dest
    assert calls == []


def test_ensure_demo_fails_without_huggingface(tmp_path, monkeypatch):
    monkeypatch.setattr("rem2.download.example.download_from_venusrem2", lambda *a, **k: None)
    try:
        ensure_demo_dataset(dest=str(tmp_path / "empty"), log=lambda *_: None)
    except SystemExit as exc:
        assert "Hugging Face" in str(exc) or "VenusREM2" in str(exc)
    else:
        raise AssertionError("expected SystemExit when Hugging Face download fails")
