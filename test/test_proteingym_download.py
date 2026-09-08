"""ProteinGym download helpers (no network)."""

from __future__ import annotations

from pathlib import Path

from rem2.cli import main
from rem2.data.proteingym import (
    copy_named_files,
    map_pdbs_to_assays,
    safe_extract_tar,
    write_aa_seq_from_reference,
)


def _reference(path: Path) -> Path:
    path.write_text(
        "DMS_id,target_seq,pdb_file\n"
        "PROT_A_Assay,ACDE,PROT_A.pdb\n"
        "PROT_A_Other,ACDE,PROT_A.pdb\n",
        encoding="utf-8",
    )
    return path


def test_write_aa_seq_from_reference(tmp_path):
    ref = _reference(tmp_path / "DMS_substitutions.csv")
    n = write_aa_seq_from_reference(ref, tmp_path / "aa_seq")
    assert n == 2
    text = (tmp_path / "aa_seq" / "PROT_A_Assay.fasta").read_text()
    assert text == ">PROT_A_Assay\nACDE\n"


def test_map_pdbs_to_assays(tmp_path):
    extracted = tmp_path / "af2"
    extracted.mkdir()
    (extracted / "PROT_A.pdb").write_text("ATOM\n")
    n = map_pdbs_to_assays(
        extracted,
        _reference(tmp_path / "ref.csv"),
        tmp_path / "pdbs",
    )
    assert n == 2
    assert (tmp_path / "pdbs" / "PROT_A_Assay.pdb").read_text() == "ATOM\n"
    assert (tmp_path / "pdbs" / "PROT_A_Other.pdb").read_text() == "ATOM\n"


def test_copy_named_files(tmp_path):
    src = tmp_path / "zip"
    (src / "nested").mkdir(parents=True)
    (src / "nested" / "P.csv").write_text("mutant\nA1C\n")
    n = copy_named_files(src, tmp_path / "substitutions", (".csv",))
    assert n == 1
    assert (tmp_path / "substitutions" / "P.csv").is_file()


def test_safe_extract_tar(tmp_path):
    import tarfile

    inner = tmp_path / "aa_seq"
    inner.mkdir()
    (inner / "p.fasta").write_text(">p\nA\n")
    archive = tmp_path / "aa_seq.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(inner, arcname="aa_seq")
    dest = tmp_path / "out"
    safe_extract_tar(archive, dest)
    assert (dest / "aa_seq" / "p.fasta").read_text() == ">p\nA\n"


def test_cli_download_dry_run(tmp_path, capsys):
    main(["download", "--dry-run", "--dest", str(tmp_path / "pg")])
    out = capsys.readouterr().out
    assert "AI4Protein/VenusREM2" in out
    assert "tyang816/VenusREM2" in out
    assert "ProteinGym/aa_seq_aln_a2m.tar.gz" in out
    assert "AI4Protein/VenusREM" in out
    assert "DMS_ProteinGym_substitutions.zip" in out
    assert "ProteinGym_AF2_structures.zip" in out


def test_cli_download_muthub_dry_run(tmp_path, capsys):
    main(["download", "VenusMutHub", "--dry-run", "--dest", str(tmp_path / "vmh")])
    out = capsys.readouterr().out
    assert "VenusMutHub/substitutions.tar.gz" in out
    assert "AI4Protein/VenusREM2" in out
    assert "tyang816/VenusREM2" in out


def test_cli_download_virohub_dry_run(tmp_path, capsys):
    main(["download", "virohub", "--dry-run", "--dest", str(tmp_path / "vvh")])
    out = capsys.readouterr().out
    assert "VenusViroHub/aa_seq.tar.gz" in out
    assert "VenusViroHub/DMS_substitutions.csv" in out


def test_cli_download_unknown_dataset():
    import pytest

    with pytest.raises(SystemExit, match="Unknown download target"):
        main(["download", "not-a-hub"])


def test_cli_download_benchmark_all_dry_run(tmp_path, capsys):
    main(["download", "benchmark-all", "--dry-run"])
    out = capsys.readouterr().out
    assert "Will download 3 benchmark" in out
    assert "proteingym" in out
    assert "muthub" in out
    assert "virohub" in out
    assert "ProteinGym/aa_seq.tar.gz" in out
    assert "VenusMutHub/substitutions.tar.gz" in out
    assert "VenusViroHub/aa_seq.tar.gz" in out


def test_cli_download_all_still_means_benchmarks(capsys):
    main(["download", "all", "--dry-run"])
    out = capsys.readouterr().out
    assert "Will download 3 benchmark" in out


def test_cli_download_esm2_dry_run(capsys):
    main(["download", "esm2", "--dry-run"])
    out = capsys.readouterr().out
    assert "Will download 1 model" in out
    assert "facebook/esm2_t33_650M_UR50D" in out
    assert "huggingface" in out.lower()


def test_cli_download_venusrem2_dry_run(capsys):
    main(["download", "venusrem2", "--dry-run"])
    out = capsys.readouterr().out
    assert "venusrem2" in out
    assert "AI4Protein/ProSST-2048" in out
    assert "AI4Protein/ProSST-20" in out


def test_cli_download_model_all_dry_run(capsys):
    main(["download", "model-all", "--dry-run"])
    out = capsys.readouterr().out
    assert "Will download" in out
    assert "model" in out
    assert "esm2" in out
    assert "venusrem2" in out
    assert "saprot" in out
    assert "tens of GB" in out


def test_cli_download_help_lists_model_and_benchmark():
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["download", "--help"])
    assert exc.value.code == 0


def test_download_help_text():
    from rem2.data.download import build_download_parser

    text = build_download_parser().format_help()
    assert "model-all" in text
    assert "benchmark-all" in text
    assert "esm2" in text
