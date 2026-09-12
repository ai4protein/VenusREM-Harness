"""CLI smoke tests."""

from __future__ import annotations

import subprocess
import sys

import pytest

from vrh.cli import main


def test_help_does_not_import_torch():
    script = (
        "import sys\n"
        "from vrh.cli import main\n"
        "try:\n"
        "    main(['--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'transformers' not in sys.modules\n"
        "assert 'vrh.cli_run' not in sys.modules\n"
    )
    subprocess.check_call([sys.executable, "-c", script])


def test_download_help_does_not_import_torch():
    script = (
        "import sys\n"
        "from vrh.cli import main\n"
        "try:\n"
        "    main(['download', '--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'transformers' not in sys.modules\n"
    )
    subprocess.check_call([sys.executable, "-c", script])


def test_download_model_dry_run_does_not_import_torch():
    script = (
        "import sys\n"
        "from vrh.cli import main\n"
        "try:\n"
        "    main(['download', 'esm2', '--dry-run'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'transformers' not in sys.modules\n"
    )
    subprocess.check_call([sys.executable, "-c", script])


def test_bare_vrh_prints_getting_started(capsys):
    main([])
    out = capsys.readouterr().out
    assert "vrh doctor" in out
    assert "vrh demo" in out
    assert "vrh dashboard" in out
    assert "127.0.0.1:8765" in out
    assert "vrh download" in out
    assert "from vrh import score" in out
    assert "--model venusrem2" in out
    assert "--model saprot --pdb prot.pdb" in out
    assert "wt (default)" in out
    assert "type y to download" in out


def test_list_models_cli(capsys):
    main(["--list-models"])
    out = capsys.readouterr().out
    assert "FWD" in out
    assert "esm2" in out
    assert "prosst" in out
    assert "protein_mpnn" in out
    assert "esm2-8m" in out
    assert "prosst-4096" in out
    assert "esmc-600m" in out
    assert "progen2-xl" in out
    assert "vrh backbones" in out
    assert "venusrem2 = official ProSST ensemble" in out
    assert "proteinmpnn" in out
    assert "*_mask / *_wt" in out


def test_unknown_model_is_systemexit():
    with pytest.raises(SystemExit, match="Unknown model") as exc:
        main(["--model", "not-a-model", "--base_dir", "/tmp/vrh_nope"])
    assert "esm2" in str(exc.value)


def test_auto_without_model_id_is_systemexit():
    with pytest.raises(SystemExit, match="--model auto requires --model_id"):
        main(["--model", "auto", "--base_dir", "/tmp/vrh_nope"])


def test_dashboard_help_does_not_import_torch():
    script = (
        "import sys\n"
        "from vrh.cli import main\n"
        "try:\n"
        "    main(['dashboard', '--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'transformers' not in sys.modules\n"
        "assert 'vrh.cli_run' not in sys.modules\n"
    )
    subprocess.check_call([sys.executable, "-c", script])


def test_doctor_cli(capsys):
    main(["doctor"])
    out = capsys.readouterr().out
    assert "vrh " in out
    assert "torch" in out
    assert "demo" in out.lower()
    assert "download" in out.lower()
    assert "core: vrh dashboard" in out
    assert "core: PDB I/O" in out


def test_optional_cli_and_dashboard_extras():
    from pathlib import Path

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    core = " ".join(data["project"]["dependencies"]).lower()
    extras = data["project"]["optional-dependencies"]
    assert "torch" not in core
    assert "fastapi" in core
    assert "biotite" in core
    assert extras["recommended"] == extras["cli"]
    assert extras["all"] == extras["dashboard"]
    all_extra = " ".join(extras["all"]).lower()
    assert "fastapi" in all_extra
    assert "torchdrug" not in all_extra
    assert "pykeops" not in all_extra
    assert "sequence-models" not in all_extra
    assert "torch-scatter" not in all_extra
    assert "esm>=3" not in all_extra


def test_cli_refuses_masked_marginals_on_causal_lm():
    with pytest.raises(SystemExit, match="mask is not supported") as exc:
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
    assert "wt" in str(exc.value).lower()


def test_cli_refuses_mask_on_prosst():
    with pytest.raises(SystemExit, match="mask is not supported"):
        main(
            [
                "--model",
                "prosst-4096",
                "--scoring_strategy",
                "mask",
                "--out_scores_dir",
                "/tmp/vrh_should_not_score",
            ]
        )


def test_cli_refuses_tf_on_esm2():
    with pytest.raises(SystemExit, match="tf is not supported"):
        main(
            [
                "--model",
                "esm2",
                "--scoring_strategy",
                "tf",
                "--out_scores_dir",
                "/tmp/vrh_should_not_score",
            ]
        )


def test_missing_base_dir_data_is_systemexit(tmp_path, capsys):
    empty = tmp_path / "empty_assay"
    empty.mkdir()
    with pytest.raises(SystemExit, match="No data") as exc:
        main(["--model", "esm2", "--base_dir", str(empty), "--out_scores_dir", str(tmp_path / "out")])
    assert "aa_seq" in str(exc.value)
    out = capsys.readouterr().out
    assert "Scoring proteins" not in out
    assert "vrh scoring run" not in out


def test_structure_model_requires_pdb(tmp_path):
    base = tmp_path / "assay"
    (base / "aa_seq").mkdir(parents=True)
    (base / "substitutions").mkdir()
    (base / "aa_seq" / "p.fasta").write_text(">p\nACDE\n")
    (base / "substitutions" / "p.csv").write_text("mutant\nA1C\n")
    with pytest.raises(SystemExit, match="needs PDB") as exc:
        main(
            [
                "--model",
                "esmif",
                "--base_dir",
                str(base),
                "--out_scores_dir",
                str(tmp_path / "out"),
                "--no_auto_download",
            ]
        )
    assert "pdb" in str(exc.value).lower()


def test_prosst_requires_struc_seq(tmp_path, capsys):
    base = tmp_path / "assay"
    (base / "aa_seq").mkdir(parents=True)
    (base / "substitutions").mkdir()
    (base / "aa_seq" / "p.fasta").write_text(">p\nACDE\n")
    (base / "substitutions" / "p.csv").write_text("mutant\nA1C\n")
    with pytest.raises(SystemExit, match="structure tokens") as exc:
        main(
            [
                "--model",
                "prosst-2048",
                "--base_dir",
                str(base),
                "--out_scores_dir",
                str(tmp_path / "out"),
                "--no_auto_download",
            ]
        )
    assert "struc" in str(exc.value).lower()
    assert "Scoring proteins" not in capsys.readouterr().out
