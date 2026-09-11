from __future__ import annotations

import pytest

from vrh.scoring.mutant_parse import MutantParseError, parse_substitution

VOCAB = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
SEQ = "NLYIQWLKDGGPSSGRPPPS"


def test_parse_ok_and_lowercase():
    assert parse_substitution("N1A", SEQ, VOCAB) == ("N", 0, "A")
    assert parse_substitution("n1a", SEQ, VOCAB) == ("N", 0, "A")


def test_parse_rejects_bad_format():
    with pytest.raises(MutantParseError, match="Bad mutant"):
        parse_substitution("N1", SEQ, VOCAB)


def test_parse_rejects_wt_mismatch():
    with pytest.raises(MutantParseError, match="Wild-type mismatch"):
        parse_substitution("A1G", SEQ, VOCAB)


def test_parse_rejects_oob():
    with pytest.raises(MutantParseError, match="outside sequence"):
        parse_substitution("N99A", SEQ, VOCAB)
