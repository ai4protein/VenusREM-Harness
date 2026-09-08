"""rem2 entropy-α / β=1-α helpers (no model weights)."""

from __future__ import annotations

import math

import torch

from venusrem2.scoring.entropy_alpha import (
    AA20,
    beta_to_raw_alpha,
    entropy_weighted_rho,
    native_preference_features,
    parse_alpha_arg,
    parse_background_weight_arg,
    resolve_mix_weights,
    site_mean_entropy,
)


def test_parse_alpha_and_beta():
    assert parse_alpha_arg(None) == "entropy"
    assert parse_alpha_arg("entropy") == "entropy"
    assert parse_alpha_arg("0.8") == 0.8
    assert parse_alpha_arg(0) == 0.0
    assert parse_background_weight_arg(None) == "one_minus_alpha"
    assert parse_background_weight_arg("1-alpha") == "one_minus_alpha"
    assert parse_background_weight_arg("0.2") == 0.2


def test_entropy_weighted_rho_and_scale_map():
    assert entropy_weighted_rho(0.2, 0.0, 0.8) == 0.2
    assert entropy_weighted_rho(0.2, 1.0, 0.8) == 0.8
    assert abs(entropy_weighted_rho(0.2, 0.5, 0.8) - 0.5) < 1e-12
    # equal scales → α = ρ
    assert abs(beta_to_raw_alpha(0.4, 1.0, 1.0) - 0.4) < 1e-12
    # larger MSA scale pulls α down
    assert beta_to_raw_alpha(0.5, 1.0, 3.0) < 0.5


def test_uniform_logits_entropy_is_one():
    logits = torch.zeros(8, 20, dtype=torch.float64)
    assert abs(site_mean_entropy(logits) - 1.0) < 1e-6


def test_resolve_mix_weights_entropy_and_dynamic_beta():
    vocab = {aa: i for i, aa in enumerate(AA20)}
    vocab.update({"<pad>": 20, "<cls>": 21})
    raw = torch.zeros(6, 22)
    raw[:, 0] = 4.0  # peak on A
    counts = torch.full((6, 22), -4.0)
    counts[:, 1] = 2.0  # MSA prefers C
    sequence = "AAAAAA"
    alpha, beta, feats = resolve_mix_weights(
        raw, counts, 0, 6, sequence, vocab, "entropy", "one_minus_alpha"
    )
    assert 0.0 <= alpha <= 1.0
    assert abs(beta - (1.0 - alpha)) < 1e-12
    assert "rho" in feats
    assert feats["n"] == 6


def test_cli_zmean_columns():
    import numpy as np
    import pandas as pd

    from venusrem2.cli import _zmean_columns

    frame = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
    })
    out = _zmean_columns(frame, ["a", "b"])
    assert out.shape == (3,)
    assert abs(out[1]) < 1e-12
    assert out[0] < 0 < out[2]


def test_native_preference_features_keys():
    vocab = {aa: i for i, aa in enumerate(AA20)}
    raw = torch.randn(10, 20)
    counts = torch.log_softmax(torch.randn(10, 20), dim=-1)
    feats = native_preference_features(raw, counts, 0, 10, "A" * 10, vocab)
    assert set(feats) >= {
        "alpha",
        "rho",
        "hbar",
        "rho_pi",
        "rho_rot",
        "scale_plm",
        "scale_msa",
        "alpha_pi",
    }
    assert math.log(20) > 0
    assert 0 <= feats["alpha"] <= 1
