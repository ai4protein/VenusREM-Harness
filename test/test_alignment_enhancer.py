from pathlib import Path

from rem2.scoring.alignment_enhancer import _fast_count_matrix, read_multi_fasta


class _Tokenizer:
    pad_token_id = 0
    vocab_size = 21

    def get_vocab(self):
        return {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def test_a3m_comments_and_lowercase_insertions_are_not_alignment_columns(tmp_path: Path):
    msa = tmp_path / "query.a3m"
    msa.write_text("#5 1\n>query\nACDEF\n>hit\nACdDEF\n", encoding="utf-8")

    parsed = read_multi_fasta(msa)
    assert list(parsed.values()) == ["ACDEF", "ACDEF"]

    matrix, start, end = _fast_count_matrix(msa, _Tokenizer())
    assert matrix.shape == (5, 21)
    assert (start, end) == (0, 5)
