import hashlib
import json
import os

import torch
from Bio import SeqIO

from venus_orbit.backbone import (
    backbone_supports_structure_tokens,
    forward_sequence_logits,
    resolve_structure_fasta_path,
)
from venus_orbit.scoring import (
    apply_alignment_prior,
    build_calibration_terms,
    load_cached_logits,
    load_residue_plddt_from_pdb,
    load_residue_rsa_weights_from_pdb,
    save_cached_logits,
    score_mutations_batch,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def _native_cache_path(logits_cache_path):
    if not logits_cache_path:
        return None
    # Keep the full cache key (incl. hash from logits_cache_tag) so wt/mask
    # native scores do not collide after stripping only the trailing hash.
    if logits_cache_path.endswith(".pt"):
        return logits_cache_path[:-3] + "__native_scores.pt"
    return logits_cache_path + "__native_scores.pt"


def _load_native_cache(cache_path, mutant_df):
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu")
        cached_mutants = payload.get("mutants", [])
        current_mutants = mutant_df["mutant"].tolist()
        if cached_mutants == current_mutants:
            return payload["scores"]
    except Exception:
        pass
    return None


def _save_native_cache(cache_path, mutant_df, scores):
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({
        "mutants": mutant_df["mutant"].tolist(),
        "scores": scores,
    }, cache_path)


def read_seq(fasta):
    for record in SeqIO.parse(fasta, "fasta"):
        return str(record.seq)


@torch.no_grad()
def score_protein(model, tokenizer, residue_fasta, structure_fasta, mutant_df,
                  alpha=0.7, aa_seq_aln_file=None, struc_seq_aln_file=None,
                  sample_size=None, sample_ratio=1.0, sample_times=1,
                  protein_name=None, backbone_mode="auto", pdb_file=None,
                  max_residue_len=None, long_seq_mode="auto_window", long_seq_overlap=256,
                  quiet=False, show_progress=True, logger=None,
                  scoring_mode="log_odds", background_weight=0.2,
                  wt_confidence_weight=0.05,
                  logits_cache_path=None, reuse_logits_cache=False, write_logits_cache=False,
                  cache_miss_policy="forward", logits_cache_stage="raw",
                  calibrate_on_raw=False, precomputed_logits_dir=None,
                  use_rsa_decay=False, pdb_dir=None, rsa_decay_mode="raw",
                  use_plddt_decay=False, plddt_decay_mode="above_mean",
                  task_type="default",
                  disable_adaptive_ccd=False,
                  baseline_forward_fn=None,
                  aln_count_cache_dir=None,
                  native_scorer_fn=None):
    def log_local(msg):
        if not quiet:
            if logger is not None:
                logger.info(msg, protein=protein_name)
            elif protein_name:
                print(f"[{protein_name}] {msg}")
            else:
                print(msg)
    def log_warn_local(msg):
        if not quiet:
            if logger is not None:
                logger.warn(msg, protein=protein_name)
            elif protein_name:
                print(f"[WARN][{protein_name}] {msg}")
            else:
                print(f"[WARN] {msg}")

    sequence = read_seq(residue_fasta)

    if native_scorer_fn is not None and alpha == 0 and scoring_mode == "log_odds":
        native_cache_path = _native_cache_path(logits_cache_path)
        native_scores = None
        if native_cache_path and reuse_logits_cache:
            native_scores = _load_native_cache(native_cache_path, mutant_df)
            if native_scores is not None:
                log_local(f"Loaded cached native scores ({len(native_scores)} mutants)")
        if native_scores is None:
            native_scores = native_scorer_fn(sequence=sequence, mutant_df=mutant_df)
            if native_cache_path and write_logits_cache:
                _save_native_cache(native_cache_path, mutant_df, native_scores)
        if not (use_rsa_decay or use_plddt_decay):
            log_local("Using native per-mutant baseline scores")
            return native_scores

    supports_structure = backbone_supports_structure_tokens(model)
    if backbone_mode == "prosst":
        use_structure = True
    elif backbone_mode == "plain_mlm":
        use_structure = False
    else:
        use_structure = supports_structure and structure_fasta is not None

    if use_structure and not supports_structure:
        raise ValueError(
            "Current backbone does not support structure tokens (ss_input_ids). "
            "Use --backbone_mode plain_mlm or auto for this model."
        )

    if use_structure:
        if structure_fasta is None or not os.path.exists(structure_fasta):
            raise FileNotFoundError(
                f"Structure sequence is required for backbone_mode={backbone_mode}, missing: {structure_fasta}"
            )
        structure_sequence = read_seq(structure_fasta)
        structure_sequence = [int(i) for i in structure_sequence.split(",")]
    else:
        structure_sequence = None

    precomputed_path = None
    if precomputed_logits_dir and protein_name:
        precomputed_path = os.path.join(precomputed_logits_dir, f"{protein_name}.pt")

    logits = None
    loaded_final_logits = False

    logits, loaded_final_logits = load_cached_logits(
        logits_cache_path=logits_cache_path,
        sequence=sequence,
        reuse_logits_cache=reuse_logits_cache,
        cache_miss_policy=cache_miss_policy,
        logits_cache_stage=logits_cache_stage,
        device=device,
        log_local=log_local,
        log_warn_local=log_warn_local,
    )

    if logits is None and precomputed_path and os.path.exists(precomputed_path):
        try:
            payload = torch.load(precomputed_path, map_location="cpu")
            cached_seq = payload.get("sequence", "")
            if cached_seq == sequence:
                logits = payload["logits"].to(device)
                log_local(f"Loaded precomputed backbone logits: {precomputed_path}")
            else:
                log_warn_local(f"Sequence mismatch in precomputed logits, skipping: {precomputed_path}")
        except Exception as e:
            log_warn_local(f"Failed to load precomputed logits ({e}), falling back")

    if logits is None:
        if baseline_forward_fn is not None:
            logits = baseline_forward_fn(sequence=sequence)
        else:
            logits = forward_sequence_logits(
                model=model,
                tokenizer=tokenizer,
                sequence=sequence,
                device=device,
                use_structure=use_structure,
                structure_sequence=structure_sequence,
                max_residue_len=max_residue_len,
                long_seq_mode=long_seq_mode,
                long_seq_overlap=long_seq_overlap,
                logger=logger,
                protein_name=protein_name,
            )
        if logits_cache_path and write_logits_cache and logits_cache_stage == "raw":
            save_cached_logits(
                logits_cache_path=logits_cache_path,
                sequence=sequence,
                logits=logits,
                cache_stage="raw",
                log_local=log_local,
            )

    if not loaded_final_logits:
        backbone_logits = logits.clone()

        if alpha != 0:
            logits = apply_alignment_prior(
                logits=logits,
                tokenizer=tokenizer,
                alpha=alpha,
                aa_seq_aln_file=aa_seq_aln_file,
                struc_seq_aln_file=struc_seq_aln_file,
                sample_ratio=sample_ratio,
                sample_times=sample_times,
                show_progress=show_progress,
                quiet=quiet,
                logger=logger,
                protein_name=protein_name,
                count_matrix_cache_dir=aln_count_cache_dir,
            )
        else:
            log_local("No alignment matrix used")
        if logits_cache_path and write_logits_cache and logits_cache_stage == "final":
            save_cached_logits(
                logits_cache_path=logits_cache_path,
                sequence=sequence,
                logits=logits,
                cache_stage="final",
                log_local=log_local,
            )
    else:
        backbone_logits = None
        log_local("Skip logits fusion: using final-stage cache")

    raw_for_calib = None
    if calibrate_on_raw:
        if backbone_logits is not None:
            raw_for_calib = backbone_logits
            log_local("Calibration terms will be computed from raw backbone logits")
        elif precomputed_logits_dir and protein_name:
            raw_path = os.path.join(precomputed_logits_dir, f"{protein_name}.pt")
            if os.path.exists(raw_path):
                try:
                    raw_payload = torch.load(raw_path, map_location="cpu")
                    if raw_payload.get("sequence", "") == sequence:
                        raw_for_calib = raw_payload["logits"].to(device)
                        log_local("Loaded raw backbone logits for calibration from precomputed dir")
                except Exception:
                    pass
        if raw_for_calib is None and calibrate_on_raw:
            log_warn_local(
                "--calibrate_on_raw requested but raw backbone logits unavailable "
                "(final-stage cache in use, no precomputed dir); falling back to fused calibration"
            )

    vocab = tokenizer.get_vocab()
    mutants = mutant_df["mutant"].tolist()

    rsa_weights = None
    if use_rsa_decay or scoring_mode == "rsa_modulated_ccd":
        rsa_weights = load_residue_rsa_weights_from_pdb(
            seq_len=len(sequence),
            protein_name=protein_name,
            pdb_file=pdb_file,
            pdb_dir=pdb_dir,
        )
        if rsa_weights is None:
            log_warn_local("RSA weights requested but RSA computation failed; ignoring RSA")

    plddt_weights = None
    if use_plddt_decay:
        plddt_weights = load_residue_plddt_from_pdb(
            seq_len=len(sequence),
            protein_name=protein_name,
            pdb_file=pdb_file,
            pdb_dir=pdb_dir,
        )
        if plddt_weights is None:
            log_warn_local("pLDDT weights requested but loading failed; ignoring pLDDT")

    calibration_terms = build_calibration_terms(
        logits=logits,
        sequence=sequence,
        vocab=vocab,
        scoring_mode=scoring_mode,
        raw_logits=raw_for_calib,
        rsa_weights=rsa_weights,
        plddt_weights=plddt_weights,
    )
    # disable_adaptive_ccd kept for CLI compatibility; CCD v1 no longer emits bg_consistency.
    _ = disable_adaptive_ccd

    log_local(f"Scoring mode: {scoring_mode}")

    native_scores = None
    if native_scorer_fn is not None:
        native_cache_path = _native_cache_path(logits_cache_path)
        if native_cache_path and reuse_logits_cache:
            native_scores = _load_native_cache(native_cache_path, mutant_df)
            if native_scores is not None:
                log_local(f"Loaded cached native scores ({len(native_scores)} mutants)")
        if native_scores is None:
            native_scores = native_scorer_fn(sequence=sequence, mutant_df=mutant_df)
            if native_cache_path and write_logits_cache:
                _save_native_cache(native_cache_path, mutant_df, native_scores)

    log_local("Scoring mutants")

    scores = score_mutations_batch(
        mutants=mutants,
        logits=logits,
        sequence=sequence,
        vocab=vocab,
        scoring_mode=scoring_mode,
        background_weight=background_weight,
        wt_confidence_weight=wt_confidence_weight,
        calibration_terms=calibration_terms,
        use_rsa_decay=use_rsa_decay,
        rsa_decay_mode=rsa_decay_mode,
        use_plddt_decay=use_plddt_decay,
        plddt_decay_mode=plddt_decay_mode,
        task_type=task_type,
        native_deltas=native_scores,
        alpha=alpha,
        raw_logits=backbone_logits,
    )

    return scores


def build_logits_cache_path(args, protein_name, model_name, variant_label):
    if not args.logits_cache_dir:
        return None
    cache_payload = {
        "model_name": model_name,
        "variant_label": variant_label,
        "backbone_mode": args.backbone_mode,
        "max_residue_len": args.max_residue_len,
        "long_seq_mode": args.long_seq_mode,
        "long_seq_overlap": args.long_seq_overlap,
        "cache_tag": args.logits_cache_tag,
        "logits_cache_stage": args.logits_cache_stage,
    }
    if args.logits_cache_stage != "raw":
        cache_payload.update({
            "alpha": args.alpha,
            "logit_mode": args.logit_mode,
            "sample_size": args.sample_size,
            "sample_ratio": args.sample_ratio,
            "sample_times": args.sample_times,
        })
    cache_key = hashlib.sha1(
        json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    safe_model_name = model_name.split("/")[-1].replace(" ", "_")
    return os.path.join(
        args.logits_cache_dir,
        f"{protein_name}__{safe_model_name}__{variant_label}__{cache_key}.pt",
    )
