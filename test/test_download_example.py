"""Unzipped HF example used by rem2 demo (no network)."""

from pathlib import Path

from rem2.cli import main
from rem2.download.example import (
    EXAMPLE_FILES,
    bundled_example_dir,
    ensure_demo_dataset,
)


def test_example_dry_run(capsys):
    main(["download", "example", "--dry-run"])
    out = capsys.readouterr().out
    assert "example/trp_cage/aa_seq/trp_cage.fasta" in out
    assert "example/trp_cage/substitutions/trp_cage.csv" in out
    assert "example/trp_cage/pdbs/trp_cage.pdb" in out
    assert "example/trp_cage/aa_seq_aln_a2m/trp_cage.a2m" in out
    assert "tyang816/VenusREM2" in out


def test_ensure_demo_uses_cache_then_hf(tmp_path, monkeypatch):
    dest = tmp_path / "trp_cage"
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
    assert (dest / "pdbs" / "trp_cage.pdb").is_file()

    calls.clear()
    again = ensure_demo_dataset(dest=str(dest), log=lambda *_: None)
    assert again == dest
    assert calls == []


def test_ensure_demo_falls_back_to_bundled(tmp_path, monkeypatch):
    bundled = bundled_example_dir()
    assert bundled is not None
    monkeypatch.setattr("rem2.download.example.download_from_venusrem2", lambda *a, **k: None)
    got = ensure_demo_dataset(dest=str(tmp_path / "empty"), log=lambda *_: None)
    assert got == bundled
