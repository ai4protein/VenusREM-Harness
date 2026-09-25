"""Checkpoint cache lookup + download consent (no network)."""

from __future__ import annotations

import pytest

from vrh.config import create_parser
from vrh.models.download_policy import (
    DownloadRefused,
    apply_download_policy_from_args,
    confirm_download,
    set_download_policy,
)
from vrh.models.weights import (
    cache_search_roots,
    resolve_existing_dir,
    resolve_existing_weight,
    resolve_prosst_static_file,
)


@pytest.fixture(autouse=True)
def _reset_download_policy():
    yield
    set_download_policy("ask")


def test_parser_auto_download_flags():
    parser = create_parser()
    assert parser.parse_args([]).auto_download is None
    assert parser.parse_args(["--auto_download"]).auto_download is True
    assert parser.parse_args(["--no_auto_download"]).auto_download is False


def test_policy_from_args():
    parser = create_parser()
    apply_download_policy_from_args(parser.parse_args([]))
    from vrh.models.download_policy import get_download_policy

    assert get_download_policy() == "ask"
    apply_download_policy_from_args(parser.parse_args(["--auto_download"]))
    assert get_download_policy() == "yes"
    apply_download_policy_from_args(parser.parse_args(["--no_auto_download"]))
    assert get_download_policy() == "no"


def test_confirm_refuses_when_disabled():
    set_download_policy("no")
    with pytest.raises(DownloadRefused, match="Download disabled"):
        confirm_download(
            name="ProteinMPNN",
            dest="/tmp/missing.pt",
            source="https://example.invalid/ckpt.pt",
        )


def test_confirm_allows_when_enabled(capsys):
    set_download_policy("yes")
    confirm_download(
        name="ProteinMPNN",
        dest="/tmp/missing.pt",
        source="https://example.invalid/ckpt.pt",
    )
    assert "Downloading ProteinMPNN" in capsys.readouterr().out


def test_confirm_ask_nontty_downloads(monkeypatch, capsys):
    set_download_policy("ask")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    confirm_download(
        name="ProteinMPNN",
        dest="/tmp/missing.pt",
        source="https://example.invalid/ckpt.pt",
    )
    assert "Downloading ProteinMPNN" in capsys.readouterr().out


def test_confirm_ask_tty_enter_downloads(monkeypatch):
    set_download_policy("ask")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "")
    confirm_download(
        name="ProteinMPNN",
        dest="/tmp/missing.pt",
        source="https://example.invalid/ckpt.pt",
    )


def test_confirm_ask_tty_no_refuses(monkeypatch):
    set_download_policy("ask")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(DownloadRefused, match="Download declined"):
        confirm_download(
            name="ProteinMPNN",
            dest="/tmp/missing.pt",
            source="https://example.invalid/ckpt.pt",
        )


def test_resolve_weight_from_manual_cache(tmp_path):
    cache = tmp_path / "my_cache"
    (cache / "protein_mpnn").mkdir(parents=True)
    ckpt = cache / "protein_mpnn" / "v_48_020.pt"
    ckpt.write_bytes(b"fake-ckpt")
    found = resolve_existing_weight(
        "protein_mpnn", "v_48_020.pt", cache_dir=str(cache)
    )
    assert found == str(ckpt)
    assert str(cache) in cache_search_roots(str(cache))
    assert resolve_existing_dir("protein_mpnn", cache_dir=str(cache)) == str(
        cache / "protein_mpnn"
    )


def test_resolve_weight_from_parent_cache_layout(tmp_path):
    root = tmp_path / "venusrem2"
    weights = root / "weights" / "s3f"
    weights.mkdir(parents=True)
    ckpt = weights / "s3f.pth"
    ckpt.write_bytes(b"s3f")
    found = resolve_existing_weight("s3f", "s3f.pth", cache_dir=str(root))
    assert found == str(ckpt)


def test_resolve_prosst_static_uses_cache_then_hf(tmp_path, monkeypatch):
    monkeypatch.setattr("vrh.models.weights.bundled_prosst_static", lambda name: None)
    cache = tmp_path / "cache"
    static = cache / "prosst" / "static"
    static.mkdir(parents=True)
    (static / "2048.joblib").write_bytes(b"joblib")
    found = resolve_prosst_static_file("2048.joblib", cache_dir=str(cache))
    assert found == str(static / "2048.joblib")

    called = {}

    def fake_hf(repo, filename, local_dir, logger=None, name=None, looked_in=None):
        called["repo"] = repo
        called["filename"] = filename
        dest = tmp_path / "dl" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ae")
        return str(dest)

    monkeypatch.setenv("VRH_PROSST_STATIC_REPO", "review/prosst-static")
    monkeypatch.setattr("vrh.models.weights.resolve_existing_weight", lambda *a, **k: None)
    monkeypatch.setattr("vrh.models.weights.default_cache_dir", lambda explicit=None: str(tmp_path / "empty"))
    monkeypatch.setattr("vrh.models.weights._hf_download", fake_hf)
    path = resolve_prosst_static_file("AE.pt", cache_dir=str(tmp_path / "empty"))
    assert called["repo"] == "review/prosst-static"
    assert called["filename"] == "static/AE.pt"
    assert path.endswith("AE.pt")
