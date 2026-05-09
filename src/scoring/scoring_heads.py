import math

import torch


def build_calibration_terms(logits, sequence, vocab, scoring_mode, raw_logits=None, rsa_weights=None, plddt_weights=None):
    terms = {}
    if scoring_mode not in {
        "calibrated_margin",
        "ccd_exact",
        "temp_scaled_log_odds",
        "entropy_adaptive_margin",
        "orbit_confidence_gate",
        "rsa_modulated_ccd",
    }:
        return terms

    source = raw_logits if raw_logits is not None else logits
    probs = source.exp()
    terms["background"] = torch.logsumexp(source, dim=0) - math.log(max(source.size(0), 1))
    terms["entropy"] = -(probs * source).sum(dim=-1)
    terms["wt_confidence"] = torch.zeros(source.size(0), device=source.device, dtype=source.dtype)
    terms["wt_log_probability"] = torch.zeros(
        source.size(0), device=source.device, dtype=source.dtype
    )
    wt_vocab_ids = torch.tensor([vocab.get(aa, -1) for aa in sequence], device=source.device, dtype=torch.long)
    valid_mask = wt_vocab_ids >= 0
    if valid_mask.any():
        valid_positions = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        terms["wt_log_probability"][valid_positions] = source[
            valid_positions, wt_vocab_ids[valid_positions]
        ]
        terms["wt_confidence"][valid_positions] = torch.sigmoid(
            source[valid_positions, wt_vocab_ids[valid_positions]]
        )
    terms["calibrated_on_raw"] = raw_logits is not None
    if rsa_weights is not None:
        rsa = rsa_weights.squeeze(-1).clamp(0.0, 1.0)
        terms["rsa"] = rsa
        valid = rsa[rsa > 0]
        terms["rsa_mean"] = valid.mean().item() if len(valid) > 0 else 0.0
        terms["rsa_std"] = valid.std().item() if len(valid) > 1 else 0.0
    if plddt_weights is not None:
        plddt = plddt_weights.squeeze(-1).clamp(0.0, 1.0)
        terms["plddt"] = plddt
        valid_p = plddt[plddt > 0]
        terms["plddt_mean"] = valid_p.mean().item() if len(valid_p) > 0 else 0.0
        disorder = 1.0 - plddt[plddt > 0] if len(plddt[plddt > 0]) > 0 else plddt
        terms["plddt_disorder_std"] = disorder.std().item() if len(disorder) > 1 else 0.0
    return terms


def score_sub_mutation(
    sub_mutant,
    logits,
    sequence,
    vocab,
    scoring_mode,
    background_weight,
    uncertainty_weight,
    wt_confidence_weight,
    calibration_terms,
    score_temperature=1.0,
    entropy_adaptive_power=1.0,
    gate_center=0.5,
    gate_sharpness=8.0,
    use_rsa_decay=False,
    rsa_decay_mode="raw",
    plddt_mode="off",
    task_type="default",
):
    wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
    assert sequence[idx] == wt, f"Wild type mismatch: {sequence[idx]} != {wt}, idx {idx}"
    mt_id = vocab[mt]
    wt_id = vocab[wt]
    delta_log_odds = logits[idx, mt_id] - logits[idx, wt_id]

    safe_temperature = max(float(score_temperature), 1e-6)
    plddt_gate = 1.0
    if plddt_mode in ("gate_ccd", "gate_all") and "plddt" in calibration_terms:
        plddt_gate = calibration_terms["plddt"][idx].clamp(0.0, 1.0)

    if scoring_mode == "calibrated_margin":
        background_penalty = background_weight * (
            calibration_terms["background"][mt_id] - calibration_terms["background"][wt_id]
        )
        uncertainty_penalty = uncertainty_weight * calibration_terms["entropy"][idx]
        wt_confidence_bonus = wt_confidence_weight * calibration_terms["wt_confidence"][idx]
        score = delta_log_odds - plddt_gate * (background_penalty + uncertainty_penalty - wt_confidence_bonus)
    elif scoring_mode == "ccd_exact":
        score = (
            logits[idx, mt_id]
            - background_weight * calibration_terms["background"][mt_id]
            - uncertainty_weight * calibration_terms["entropy"][idx]
            + wt_confidence_weight * calibration_terms["wt_log_probability"][idx]
        )
    elif scoring_mode == "temp_scaled_log_odds":
        score = delta_log_odds / safe_temperature
    elif scoring_mode == "entropy_adaptive_margin":
        # Increase entropy penalty on uncertain wild-type positions.
        wt_confidence = calibration_terms["wt_confidence"][idx]
        adaptive_uncertainty = (
            calibration_terms["entropy"][idx]
            * torch.pow(1.0 - wt_confidence, float(entropy_adaptive_power))
        )
        background_penalty = background_weight * (
            calibration_terms["background"][mt_id] - calibration_terms["background"][wt_id]
        )
        score = (
            delta_log_odds / safe_temperature
            - background_penalty
            - uncertainty_weight * adaptive_uncertainty
            + wt_confidence_weight * wt_confidence
        )
    elif scoring_mode == "rsa_modulated_ccd":
        # RSA-modulated CCD: scale background and entropy penalties by structural context.
        # Buried residues (low RSA) get stronger background penalty;
        # exposed residues (high RSA) get stronger entropy penalty.
        background_penalty_raw = background_weight * (
            calibration_terms["background"][mt_id] - calibration_terms["background"][wt_id]
        )
        uncertainty_penalty_raw = uncertainty_weight * calibration_terms["entropy"][idx]
        wt_confidence_bonus = wt_confidence_weight * calibration_terms["wt_confidence"][idx]
        if "rsa" in calibration_terms:
            rsa_i = calibration_terms["rsa"][idx].clamp(0.0, 1.0)
            background_penalty = (1.0 - rsa_i) * background_penalty_raw
            uncertainty_penalty = rsa_i * uncertainty_penalty_raw
        else:
            # Fallback: unmodulated (same as calibrated_margin)
            background_penalty = background_penalty_raw
            uncertainty_penalty = uncertainty_penalty_raw
        score = delta_log_odds - background_penalty - uncertainty_penalty + wt_confidence_bonus
    elif scoring_mode == "orbit_confidence_gate":
        # Gate score contribution by wild-type confidence as a proxy for Orbit evidence reliability.
        wt_confidence = calibration_terms["wt_confidence"][idx]
        gate = torch.sigmoid((wt_confidence - float(gate_center)) * float(gate_sharpness))
        score = gate * (delta_log_odds / safe_temperature)
    else:
        score = delta_log_odds

    if use_rsa_decay and "rsa" in calibration_terms:
        rsa_val = calibration_terms["rsa"][idx].clamp(0.0, 1.0)
        if rsa_decay_mode == "above_mean":
            mean_rsa = calibration_terms.get("rsa_mean", 0.0)
            decay = max(0.0, float(rsa_val) - mean_rsa)
        elif rsa_decay_mode == "std_scaled":
            mean_rsa = calibration_terms.get("rsa_mean", 0.0)
            std_rsa = calibration_terms.get("rsa_std", 0.0)
            decay = std_rsa * max(0.0, float(rsa_val) - mean_rsa)
        elif rsa_decay_mode == "adaptive":
            mean_rsa = calibration_terms.get("rsa_mean", 0.0)
            strength = max(0.0, 1.0 - 2.0 * mean_rsa)
            decay = strength * float(rsa_val)
        else:
            decay = float(rsa_val)
        if plddt_mode in ("gate_rsa", "gate_all") and "plddt" in calibration_terms:
            decay = decay * float(calibration_terms["plddt"][idx].clamp(0.0, 1.0))
        score = score * (1.0 - decay)

    if plddt_mode in ("above_mean_decay", "std_scaled_decay") and "plddt" in calibration_terms:
        p_i = float(calibration_terms["plddt"][idx].clamp(0.0, 1.0))
        disorder_i = 1.0 - p_i
        mean_disorder = 1.0 - calibration_terms.get("plddt_mean", 0.0)
        q_i = max(0.0, disorder_i - mean_disorder)
        if plddt_mode == "std_scaled_decay":
            q_i = q_i * calibration_terms.get("plddt_disorder_std", 0.0)
        score = score * (1.0 - q_i)

    return score.item()
