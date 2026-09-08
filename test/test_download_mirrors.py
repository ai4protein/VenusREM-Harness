"""VenusREM2 mirror order (no network)."""

from pathlib import Path

from rem2.data.download import normalize_dataset
from rem2.data.mirrors import first_venusrem2_repo, download_from_venusrem2


def test_first_repo_prefers_ai4protein():
    hits = []

    def available(repo, filename):
        hits.append((repo, filename))
        return repo.startswith("AI4Protein")

    repo = first_venusrem2_repo("VenusMutHub/pdbs.tar.gz", available=available)
    assert repo == "AI4Protein/VenusREM2"
    assert hits[0][0] == "AI4Protein/VenusREM2"


def test_first_repo_falls_back_to_tyang816():
    def available(repo, filename):
        return repo.startswith("tyang816")

    repo = first_venusrem2_repo("VenusViroHub/aa_seq.tar.gz", available=available)
    assert repo == "tyang816/VenusREM2"


def test_download_from_venusrem2_uses_first_working(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        "rem2.data.mirrors.hf_file_available",
        lambda repo, filename: repo.startswith("tyang816"),
    )

    def fake_download(repo, filename, dest, force=False):
        calls.append(repo)
        Path(dest).write_text("ok")
        return Path(dest)

    monkeypatch.setattr("rem2.data.mirrors.download_hf_file", fake_download)
    dest = tmp_path / "aa_seq.tar.gz"
    got = download_from_venusrem2("ProteinGym/aa_seq.tar.gz", dest, log=lambda *_: None)
    assert got == dest
    assert calls == ["tyang816/VenusREM2"]
    assert dest.read_text() == "ok"


def test_dataset_aliases():
    assert normalize_dataset("pg") == "proteingym"
    assert normalize_dataset("ProteinGym") == "proteingym"
    assert normalize_dataset("proteingym") == "proteingym"
    assert normalize_dataset("protein-gym") == "proteingym"
    assert normalize_dataset("VenusMutHub") == "muthub"
    assert normalize_dataset("muthub") == "muthub"
    assert normalize_dataset("MutHub") == "muthub"
    assert normalize_dataset("venus_mut_hub") == "muthub"
    assert normalize_dataset("VenusViroHub") == "virohub"
    assert normalize_dataset("virohub") == "virohub"
    assert normalize_dataset("vvh") == "virohub"
    assert normalize_dataset("ALL") == "all"
