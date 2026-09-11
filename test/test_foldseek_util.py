"""Foldseek wrapper: argv list + tempdir, no shell."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vrh.baseline.saprot.foldseek_util import get_struc_seq


def test_get_struc_seq_uses_argv_and_tempdir(tmp_path, monkeypatch):
    pdb = tmp_path / "prot.pdb"
    pdb.write_text("ATOM\n", encoding="utf-8")
    foldseek = tmp_path / "foldseek"
    foldseek.write_text("", encoding="utf-8")
    cwd_before = set(Path.cwd().iterdir())
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"prot.pdb_A\tACDE\tpynw\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("vrh.baseline.saprot.foldseek_util.subprocess.run", fake_run)
    result = get_struc_seq(str(foldseek), str(pdb), chains=["A"])
    assert "A" in result
    assert result["A"][0] == "ACDE"
    assert calls
    cmd, kwargs = calls[0]
    assert cmd[0] == str(foldseek)
    assert cmd[1] == "structureto3didescriptor"
    assert cmd[-2] == str(pdb)
    assert kwargs.get("check") is True
    assert not isinstance(cmd, str)
    leftover = set(Path.cwd().iterdir()) - cwd_before
    assert not any(path.name.startswith("get_struc_seq_") for path in leftover)


def test_get_struc_seq_raises_on_foldseek_error(tmp_path, monkeypatch):
    pdb = tmp_path / "prot.pdb"
    pdb.write_text("ATOM\n", encoding="utf-8")
    foldseek = tmp_path / "foldseek"
    foldseek.write_text("", encoding="utf-8")

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="boom")

    monkeypatch.setattr("vrh.baseline.saprot.foldseek_util.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="Foldseek failed"):
        get_struc_seq(str(foldseek), str(pdb))
