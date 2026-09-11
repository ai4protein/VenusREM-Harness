from pathlib import Path

import pytest

from vrh.data.fetch_sequence import RCSB_FASTA, UNIPROT_FASTA, fetch_query_fasta


def test_fetch_query_fasta_uniprot(tmp_path):
    seen = []

    def opener(url: str, dest: Path) -> Path:
        seen.append(url)
        dest.write_text(">P0A6Y8\nACDE\n", encoding="utf-8")
        return dest

    path = fetch_query_fasta("P0A6Y8", tmp_path / "query.fasta", opener=opener)
    assert seen == [UNIPROT_FASTA.format(acc="P0A6Y8")]
    assert Path(path).read_text(encoding="utf-8").startswith(">P0A6Y8")


def test_fetch_query_fasta_pdb(tmp_path):
    seen = []

    def opener(url: str, dest: Path) -> Path:
        seen.append(url)
        dest.write_text(">2L6Q_1|Chain A\nACDE\n", encoding="utf-8")
        return dest

    path = fetch_query_fasta("2L6Q", tmp_path / "query.fasta", opener=opener)
    assert seen == [RCSB_FASTA.format(pdb="2L6Q")]
    assert "ACDE" in Path(path).read_text(encoding="utf-8")


def test_fetch_query_fasta_rejects_bad_id(tmp_path):
    with pytest.raises(ValueError):
        fetch_query_fasta("not-an-id", tmp_path / "query.fasta")


def test_fetch_query_fasta_rejects_non_fasta(tmp_path):
    def opener(url: str, dest: Path) -> Path:
        dest.write_text("<html>nope</html>", encoding="utf-8")
        return dest

    with pytest.raises(OSError, match="not a FASTA"):
        fetch_query_fasta("P0A6Y8", tmp_path / "query.fasta", opener=opener)
