from __future__ import annotations

from pathlib import Path

from vrh.data.fetch_structure import (
    describe_pdb,
    fetch_structures,
    guess_accessions,
    install_structures,
    normalize_pdb_id,
    normalize_uniprot,
)

_ATOM = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 91.00           C\n"
    "ATOM      2  CA  GLY A   2       1.000   0.000   0.000  1.00 72.00           C\n"
)
_CRYSTAL = "HEADER    HYDROLASE\nEXPDTA    X-RAY DIFFRACTION\n" + _ATOM
_AF = "TITLE     ALPHAFOLD MONOMER V2.0\nREMARK 99 pLDDT\n" + _ATOM


def test_guess_proteingym_and_uniprot():
    guessed = guess_accessions("HCP_LAMBD_Tsuboyama_2023_2L6Q", ">sp|P0A6Y8|DNAK_ECOLI")
    assert guessed["pdb_ids"] == ["2L6Q"]
    assert "P0A6Y8" in guessed["uniprot_ids"]
    assert normalize_pdb_id("2l6q.pdb") == "2L6Q"
    assert normalize_pdb_id("2023") is None
    assert normalize_uniprot("AF-P0A6Y8-F1") == "P0A6Y8"


def test_fetch_both_with_fake_opener(tmp_path):
    def opener(url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "alphafold" in url:
            dest.write_text(_AF)
        else:
            dest.write_text(_CRYSTAL)
        return dest

    result = fetch_structures(
        pdb_id="2L6Q",
        uniprot_id="P0A6Y8",
        dest_dir=tmp_path / "raw",
        source="both",
        opener=opener,
        map_uniprot=lambda _pid: "P0A6Y8",
    )
    assert result["pdb_id"] == "2L6Q"
    assert result["uniprot_id"] == "P0A6Y8"
    assert result["sources"]["rcsb"]["origin"] == "experimental"
    assert result["sources"]["rcsb"]["has_plddt"] is False
    assert result["sources"]["afdb"]["origin"] == "predicted"
    assert result["sources"]["afdb"]["has_plddt"] is True
    assert result["preferred"]["kind"] == "afdb"

    installed = install_structures(result, tmp_path / "inputs", as_query=True)
    assert (tmp_path / "inputs" / "rcsb.pdb").is_file()
    assert (tmp_path / "inputs" / "afdb.pdb").is_file()
    assert (tmp_path / "inputs" / "query.pdb").is_file()
    assert installed["installed"]["query"]["has_plddt"] is True


def test_describe_crystal_has_no_plddt(tmp_path):
    path = tmp_path / "xtal.pdb"
    path.write_text(_CRYSTAL)
    meta = describe_pdb(path)
    assert meta["origin"] == "experimental"
    assert meta["has_plddt"] is False
