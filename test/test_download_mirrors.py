"""VenusREM2 mirror order (no network)."""

import urllib.error
from pathlib import Path

import pytest

from vrh.data.download import normalize_dataset
from vrh.data.mirrors import (
    call_with_hf_retry,
    download_from_venusrem2,
    first_venusrem2_repo,
    hf_endpoint_trusted,
    hf_endpoints,
    hf_headers_for,
    hf_hub_token,
    hf_token,
    hf_token_paths,
    is_hf_network_error,
    rewrite_hf_url,
)


def test_hf_token_prefers_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_other")
    assert hf_token() == "hf_from_env"


def test_hf_token_reads_cli_file(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("hf_from_file\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    assert hf_token_paths()[0] == token_file
    assert hf_token() == "hf_from_file"


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
        "vrh.data.mirrors.hf_file_available",
        lambda repo, filename: repo.startswith("tyang816"),
    )

    def fake_download(repo, filename, dest, force=False, **_kwargs):
        calls.append(repo)
        Path(dest).write_text("ok")
        return Path(dest)

    monkeypatch.setattr("vrh.data.mirrors.download_hf_file", fake_download)
    dest = tmp_path / "aa_seq.tar.gz"
    got = download_from_venusrem2("ProteinGym/aa_seq.tar.gz", dest, log=lambda *_: None)
    assert got == dest
    assert calls == ["tyang816/VenusREM2"]
    assert dest.read_text() == "ok"


def test_hf_headers_only_on_official_hub(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    official = "https://huggingface.co/datasets/x/resolve/main/a.txt"
    mirror = "https://hf-mirror.com/datasets/x/resolve/main/a.txt"
    assert hf_endpoint_trusted(official)
    assert not hf_endpoint_trusted(mirror)
    assert not hf_endpoint_trusted("https://hf-mirror.com")
    assert hf_headers_for(official)["Authorization"] == "Bearer hf_secret_token"
    assert hf_headers_for(mirror) == {}
    assert hf_headers_for("https://hf-mirror.com") == {}
    assert hf_hub_token("https://huggingface.co") == "hf_secret_token"
    assert hf_hub_token("https://hf-mirror.com") is False


def test_hf_token_not_sent_if_endpoint_is_public_mirror(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    assert not hf_endpoint_trusted("https://hf-mirror.com")
    assert hf_hub_token("https://hf-mirror.com") is False


def test_hf_endpoints_official_then_mirror(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_MIRROR", raising=False)
    assert hf_endpoints()[0] == "https://huggingface.co"
    assert "https://hf-mirror.com" in hf_endpoints()


def test_rewrite_hf_url_to_mirror():
    url = "https://huggingface.co/datasets/x/resolve/main/a.txt"
    assert (
        rewrite_hf_url(url, "https://hf-mirror.com")
        == "https://hf-mirror.com/datasets/x/resolve/main/a.txt"
    )


def test_call_with_hf_retry_switches_mirror(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_MIRROR", raising=False)
    calls = []

    def fn(endpoint):
        calls.append(endpoint)
        if endpoint == "https://huggingface.co":
            raise urllib.error.URLError("timed out")
        return "ok"

    assert call_with_hf_retry(fn) == "ok"
    assert calls[0] == "https://huggingface.co"
    assert calls[1] == "https://hf-mirror.com"


def test_call_with_hf_retry_stops_after_three(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_MIRROR", raising=False)
    calls = []

    def fn(endpoint):
        calls.append(endpoint)
        raise TimeoutError("down")

    with pytest.raises(TimeoutError):
        call_with_hf_retry(fn, attempts=3)
    assert len(calls) == 3


def test_non_network_error_does_not_retry(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    n = 0

    def fn(endpoint):
        nonlocal n
        n += 1
        raise FileNotFoundError("missing")

    with pytest.raises(FileNotFoundError):
        call_with_hf_retry(fn)
    assert n == 1
    assert is_hf_network_error(urllib.error.URLError("timed out"))
    assert not is_hf_network_error(FileNotFoundError("missing"))


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
    assert normalize_dataset("benchmark-all") == "all"
    assert normalize_dataset("benchmarks-all") == "all"
    assert normalize_dataset("benchmarks") == "all"
