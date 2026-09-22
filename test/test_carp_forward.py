"""CARP wt is one unmasked forward; mask is one forward per site."""

from __future__ import annotations

import torch

from vrh.baseline.carp.carp import forward_carp
from vrh.models.scoring_strategy import MASKED_MARGINALS, WT_MARGINALS

ALPHABET = "ACDEFGHIKLMNPQRSTVWY#"


class _CountModel:
    def __init__(self):
        self.n = 0

    def __call__(self, x, logits=True):
        self.n += 1
        _, length = x.shape
        return {"logits": torch.zeros(1, length, len(ALPHABET))}


class _Collater:
    def __call__(self, batch):
        seq = batch[0][0]
        return [torch.zeros(1, len(seq), dtype=torch.long)]


class _Tok:
    vocab_size = 33

    def get_vocab(self):
        return {aa: i + 4 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def _run(strategy):
    model = _CountModel()
    seq = "ACDEFGHIKL"
    out = forward_carp(
        model=model,
        collater=_Collater(),
        esm_tokenizer=_Tok(),
        protein_alphabet=ALPHABET,
        sequence=seq,
        device=torch.device("cpu"),
        scoring_strategy=strategy,
    )
    return model.n, out


def test_carp_wt_is_one_forward():
    n, out = _run("wt")
    assert n == 1
    assert out.shape == (10, 33)


def test_carp_default_is_wt():
    n, _ = _run(None)
    assert n == 1


def test_carp_mask_is_one_forward_per_site():
    n, out = _run("mask")
    assert n == 10
    assert out.shape == (10, 33)


def test_carp_wt_alias_matches_canonical():
    n_wt, _ = _run(WT_MARGINALS)
    n_mask, _ = _run(MASKED_MARGINALS)
    assert n_wt == 1
    assert n_mask == 10
