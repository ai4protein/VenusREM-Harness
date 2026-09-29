"""Unzipped HF example used by vrh demo (no network)."""

from pathlib import Path

from vrh.cli import main
from vrh.download.example import (
    EXAMPLE_FILES,
    EXAMPLE_NAME,
    bundled_example_dir,
    ensure_demo_dataset,
)
from vrh.examples import demo_example_payload, demo_file_path


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

    monkeypatch.setattr("vrh.download.example.download_from_venusrem2", fake_download)
    got = ensure_demo_dataset(dest=str(dest), log=lambda *_: None)
    assert got == dest
    assert calls == list(EXAMPLE_FILES)
    assert (dest / "pdbs" / f"{EXAMPLE_NAME}.pdb").is_file()

    calls.clear()
    again = ensure_demo_dataset(dest=str(dest), log=lambda *_: None)
    assert again == dest
    assert calls == []


def test_ensure_demo_uses_bundled_without_dest(tmp_path, monkeypatch):
    bundled = bundled_example_dir()
    assert bundled is not None
    monkeypatch.setattr(
        "vrh.download.example.default_example_dir", lambda explicit=None: tmp_path / "cache"
    )
    got = ensure_demo_dataset(log=lambda *_: None)
    assert got == bundled


def test_demo_example_payload_lists_bundled_files():
    payload = demo_example_payload()
    assert payload["id"] == "demo"
    assert payload["default_preset"] == "full"
    assert payload["files"]["fasta"].endswith(".fasta")
    assert demo_file_path("fasta").is_file()
    assert demo_file_path("pdb").is_file()
    assert demo_file_path("msa").is_file()


def test_ensure_demo_falls_back_to_bundled(tmp_path, monkeypatch):
    bundled = bundled_example_dir()
    assert bundled is not None
    monkeypatch.setattr("vrh.download.example.download_from_venusrem2", lambda *a, **k: None)
    got = ensure_demo_dataset(dest=str(tmp_path / "empty"), log=lambda *_: None)
    assert got == bundled
