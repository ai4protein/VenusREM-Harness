"""CLI smoke tests."""

from __future__ import annotations

import pytest

from venusrem2.cli import main


def test_bare_rem2_prints_getting_started(capsys):
    main([])
    out = capsys.readouterr().out
    assert "rem2 doctor" in out
    assert "rem2 demo" in out
    assert "from venusrem2 import score" in out


def test_list_models_cli(capsys):
    main(["--list-models"])
    out = capsys.readouterr().out
    assert "MASK" in out
    assert "esm2" in out
    assert "prosst" in out
    assert "protein_mpnn" in out
    assert "esm2-8m" in out
    assert "rem2 backbones" in out
    assert "venusrem2 = rem2 on the official ProSST ensemble" in out


def test_unknown_model_is_systemexit():
    with pytest.raises(SystemExit, match="Unknown model") as exc:
        main(["--model", "not-a-model", "--base_dir", "/tmp/rem2_nope"])
    assert "esm2" in str(exc.value)


def test_auto_without_model_id_is_systemexit():
    with pytest.raises(SystemExit, match="--model auto requires --model_id"):
        main(["--model", "auto", "--base_dir", "/tmp/rem2_nope"])


def test_doctor_cli(capsys):
    main(["doctor"])
    out = capsys.readouterr().out
    assert "rem2 " in out
    assert "torch" in out
    assert "demo" in out.lower()


def test_cli_refuses_masked_marginals_on_causal_lm():
    with pytest.raises(SystemExit, match="masked-marginals is not supported") as exc:
        main(
            [
                "--model",
                "progen2",
                "--scoring_strategy",
                "masked-marginals",
                "--out_scores_dir",
                "/tmp/venusrem2_should_not_score",
            ]
        )
    assert "wt-marginals" in str(exc.value)
