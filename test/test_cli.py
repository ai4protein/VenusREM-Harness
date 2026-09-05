"""CLI smoke tests."""

from __future__ import annotations

from venus_orbit.cli import main


def test_list_models_cli(capsys):
    main(["--list-models"])
    out = capsys.readouterr().out
    assert "esm2" in out
    assert "prosst" in out
    assert "protein_mpnn" in out
