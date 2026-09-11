"""wt / mask / tf aliases and per-model capability."""

from __future__ import annotations

import pytest

from vrh.models.scoring_strategy import (
    MASKED_MARGINALS,
    TEACHER_FORCE,
    WT_MARGINALS,
    UnsupportedScoringStrategy,
    normalize_scoring_strategy,
    require_scoring_strategy,
)


@pytest.mark.parametrize(
    "raw, canonical",
    [
        ("wt", WT_MARGINALS),
        ("wt-marginals", WT_MARGINALS),
        ("wt_marginals", WT_MARGINALS),
        ("mask", MASKED_MARGINALS),
        ("masked", MASKED_MARGINALS),
        ("masked-marginals", MASKED_MARGINALS),
        ("tf", TEACHER_FORCE),
        ("teacher-force", TEACHER_FORCE),
        ("teacher_force", TEACHER_FORCE),
        (None, WT_MARGINALS),
    ],
)
def test_normalize_scoring_strategy(raw, canonical):
    assert normalize_scoring_strategy(raw) == canonical


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown --scoring_strategy"):
        normalize_scoring_strategy("beam")


def test_parser_accepts_short_aliases():
    from vrh.config import create_parser, postprocess_args

    args = postprocess_args(create_parser().parse_args(["--model", "esm2", "--scoring_strategy", "mask", "--fasta", "p.fa"]))
    assert args.scoring_strategy == MASKED_MARGINALS
    args = postprocess_args(create_parser().parse_args(["--model", "protein_mpnn", "--scoring_strategy", "tf", "--fasta", "p.fa"]))
    assert args.scoring_strategy == TEACHER_FORCE


def test_require_matches_product_contract():
    require_scoring_strategy("esm2", "mask")
    require_scoring_strategy("saprot", "wt")
    require_scoring_strategy("prosst-2048", "wt")
    require_scoring_strategy("proteinmpnn-020", "tf")
    with pytest.raises(UnsupportedScoringStrategy):
        require_scoring_strategy("prosst", "mask")
    with pytest.raises(UnsupportedScoringStrategy):
        require_scoring_strategy("protein_mpnn", "mask")
    with pytest.raises(UnsupportedScoringStrategy):
        require_scoring_strategy("esmif", "mask")
