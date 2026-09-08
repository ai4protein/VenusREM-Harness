"""Missing HF checkpoint asks before download."""

from __future__ import annotations

import pytest

from rem2.models.download_policy import DownloadRefused, set_download_policy
from rem2.models.hf import confirm_hf_repo, hf_repo_cached


@pytest.fixture(autouse=True)
def _reset_policy():
    yield
    set_download_policy("ask")


def test_confirm_skips_when_cached(monkeypatch):
    monkeypatch.setattr("rem2.models.hf.hf_repo_cached", lambda repo: True)
    set_download_policy("no")
    confirm_hf_repo("facebook/esm2_t6_8M_UR50D")


def test_confirm_skips_local_dir(tmp_path):
    set_download_policy("no")
    confirm_hf_repo(str(tmp_path))


def test_confirm_refuses_when_missing_and_disabled(monkeypatch):
    monkeypatch.setattr("rem2.models.hf.hf_repo_cached", lambda repo: False)
    set_download_policy("no")
    with pytest.raises(DownloadRefused, match="not found"):
        confirm_hf_repo("org/missing-model-for-rem2-test")


def test_hf_repo_cached_false_for_garbage():
    assert hf_repo_cached("org/definitely-not-cached-rem2-xyz") is False
