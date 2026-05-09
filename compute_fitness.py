
import torch
import os
import json
import hashlib
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForMaskedLM
from argparse import ArgumentParser
from src.backbone import (
    backbone_supports_structure_tokens,
    force_config_max_residue_len,
    forward_sequence_logits,
    infer_model_max_residue_len,
    resolve_structure_fasta_path,
)
from src.orbit.pipeline import maybe_fuse_logits_with_orbit
from src.orbit.retrievers.psalor_common import load_residue_rsa_weights_from_pdb, load_residue_plddt_from_pdb
from src.scoring import (
    CliLogger,
    apply_alignment_prior,
    build_calibration_terms,
    clone_args_with_overrides,
    format_name_preview,
    load_cached_logits,
    print_compare_table_header,
    print_compare_table_row,
    print_compare_table_row_extended,
    read_names,
    save_cached_logits,
    score_sub_mutation,
    set_deterministic_inference,
    should_use_color,
)

amino_acid_properties = {
    'A': {'hydrophobicity': 1.8,  'charge':  0, 'polarity':  0, 'molecular_weight':  89.09, 'volume':  88.6},
    'R': {'hydrophobicity': -4.5, 'charge': +1, 'polarity':  1, 'molecular_weight': 174.20, 'volume': 173.4},
    'N': {'hydrophobicity': -3.5, 'charge':  0, 'polarity':  1, 'molecular_weight': 132.12, 'volume': 114.1},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'polarity':  1, 'molecular_weight': 133.10, 'volume': 111.1},
    'C': {'hydrophobicity': 2.5,  'charge':  0, 'polarity':  0, 'molecular_weight': 121.15, 'volume': 108.5},
    'Q': {'hydrophobicity': -3.5, 'charge':  0, 'polarity':  1, 'molecular_weight': 146.15, 'volume': 143.8},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'polarity':  1, 'molecular_weight': 147.13, 'volume': 138.4},
    'G': {'hydrophobicity': -0.4, 'charge':  0, 'polarity':  0, 'molecular_weight':  75.07, 'volume':  60.1},
    'H': {'hydrophobicity': -3.2, 'charge':  0, 'polarity':  1, 'molecular_weight': 155.16, 'volume': 153.2},
    'I': {'hydrophobicity': 4.5,  'charge':  0, 'polarity':  0, 'molecular_weight': 131.17, 'volume': 166.7},
    'L': {'hydrophobicity': 3.8,  'charge':  0, 'polarity':  0, 'molecular_weight': 131.17, 'volume': 166.7},
    'K': {'hydrophobicity': -3.9, 'charge': +1, 'polarity':  1, 'molecular_weight': 146.19, 'volume': 168.6},
    'M': {'hydrophobicity': 1.9,  'charge':  0, 'polarity':  0, 'molecular_weight': 149.21, 'volume': 162.9},
    'F': {'hydrophobicity': 2.8,  'charge':  0, 'polarity':  0, 'molecular_weight': 165.19, 'volume': 189.9},
    'P': {'hydrophobicity': -1.6, 'charge':  0, 'polarity':  0, 'molecular_weight': 115.13, 'volume': 112.7},
    'S': {'hydrophobicity': -0.8, 'charge':  0, 'polarity':  1, 'molecular_weight': 105.09, 'volume':  89.0},
    'T': {'hydrophobicity': -0.7, 'charge':  0, 'polarity':  1, 'molecular_weight': 119.12, 'volume': 116.1},
    'W': {'hydrophobicity': -0.9, 'charge':  0, 'polarity':  0, 'molecular_weight': 204.23, 'volume': 227.8},
    'Y': {'hydrophobicity': -1.3, 'charge':  0, 'polarity':  1, 'molecular_weight': 181.19, 'volume': 193.6},
    'V': {'hydrophobicity': 4.2,  'charge':  0, 'polarity':  0, 'molecular_weight': 117.15, 'volume': 140.0},
}
device = "cuda" if torch.cuda.is_available() else "cpu"
V1_BASELINE_ALPHA = 0.8
V1_BASELINE_LOGIT_MODE = "aa_seq_aln"
V1_BASELINE_COL_SUFFIX = "__venusrem_v1_fixed_a08_aa"


def read_seq(fasta):
    for record in SeqIO.parse(fasta, "fasta"):
        return str(record.seq)
    


def calculate_property_difference(wild_aa, mutant_aa, weights=None):
    properties = amino_acid_properties[wild_aa].keys()
    if weights is None:
        weights = {prop: 1 for prop in properties}
    differences = []
    for prop in properties:
        wild_value = amino_acid_properties[wild_aa][prop]
        mutant_value = amino_acid_properties[mutant_aa][prop]
        difference = abs(mutant_value - wild_value)
        weighted_diff = weights.get(prop, 1) * difference
        differences.append(weighted_diff)
    return differences


@torch.no_grad()
def score_protein(model, tokenizer, residue_fasta, structure_fasta, mutant_df, 
                  alpha=0.7, aa_seq_aln_file=None, struc_seq_aln_file=None,
                  sample_size=None, sample_ratio=1.0, sample_times=1,
                  orbit_args=None, protein_name=None, backbone_mode="auto", pdb_file=None,
                  max_residue_len=None, long_seq_mode="auto_window", long_seq_overlap=256,
                  quiet=False, show_progress=True, logger=None,
                  scoring_mode="log_odds", background_weight=0.25,
                  uncertainty_weight=0.15, wt_confidence_weight=0.1,
                  score_temperature=1.0, entropy_adaptive_power=1.0,
                  gate_center=0.5, gate_sharpness=8.0,
                  logits_cache_path=None, reuse_logits_cache=False, write_logits_cache=False,
                  cache_miss_policy="forward", logits_cache_stage="raw",
                  calibrate_on_raw=False, precomputed_logits_dir=None,
                  use_rsa_decay=False, pdb_dir=None, rsa_decay_mode="raw", plddt_mode="off",
                  task_type="default"):
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

        if orbit_args is not None and orbit_args.orbit_enable and alpha != 0:
            orbit_logits = maybe_fuse_logits_with_orbit(
                args=orbit_args,
                tokenizer=tokenizer,
                plm_logits=backbone_logits.clone(),
                aa_seq_aln_file=aa_seq_aln_file,
                struc_seq_aln_file=struc_seq_aln_file,
                protein_name=protein_name,
                residue_fasta=residue_fasta,
                structure_fasta=structure_fasta,
                pdb_file=pdb_file,
            )
            if getattr(orbit_args, "enhance_on_v1", False):
                enhance_lambda = float(getattr(orbit_args, "enhance_lambda", 0.3))
                enhance_lambda = min(max(enhance_lambda, 0.0), 1.0)
                v1_logits = apply_alignment_prior(
                    logits=backbone_logits.clone(),
                    tokenizer=tokenizer,
                    alpha=V1_BASELINE_ALPHA,
                    aa_seq_aln_file=aa_seq_aln_file,
                    struc_seq_aln_file=None,
                    sample_ratio=sample_ratio,
                    sample_times=sample_times,
                    show_progress=show_progress,
                    quiet=quiet,
                    logger=logger,
                    protein_name=protein_name,
                )
                logits = (1.0 - enhance_lambda) * v1_logits + enhance_lambda * orbit_logits
                log_local(
                    f"Enhancing from fixed v1 anchor (alpha={V1_BASELINE_ALPHA:g}, mode={V1_BASELINE_LOGIT_MODE}, lambda={enhance_lambda:g})"
                )
            else:
                logits = orbit_logits
        elif alpha != 0:
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
    scores = []

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
    if plddt_mode != "off":
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

    log_local(f"Scoring mode: {scoring_mode}")
    log_local("Scoring mutants")
    mutant_desc = f"Mutants[{protein_name}]" if protein_name else "Mutants"
    for mutant in tqdm(
        mutants,
        desc=mutant_desc,
        disable=(not show_progress) or quiet,
        leave=True,
        dynamic_ncols=True,
    ):
        pred_score = 0
        for sub_mutant in mutant.split(":"):
            pred_score += score_sub_mutation(
                sub_mutant=sub_mutant,
                logits=logits,
                sequence=sequence,
                vocab=vocab,
                scoring_mode=scoring_mode,
                background_weight=background_weight,
                uncertainty_weight=uncertainty_weight,
                wt_confidence_weight=wt_confidence_weight,
                calibration_terms=calibration_terms,
                score_temperature=score_temperature,
                entropy_adaptive_power=entropy_adaptive_power,
                gate_center=gate_center,
                gate_sharpness=gate_sharpness,
                use_rsa_decay=use_rsa_decay,
                rsa_decay_mode=rsa_decay_mode,
                plddt_mode=plddt_mode,
                task_type=task_type,
            )
        scores.append(pred_score)

    return scores
    


def build_logits_cache_path(args, protein_name, model_name, variant_label):
    if not args.logits_cache_dir:
        return None
    cache_payload = {
        "model_name": model_name,
        "variant_label": variant_label,
        "alpha": args.alpha,
        "logit_mode": args.logit_mode,
        "orbit_enable": args.orbit_enable,
        "enhance_on_v1": args.enhance_on_v1,
        "enhance_lambda": args.enhance_lambda,
        "sample_size": args.sample_size,
        "sample_ratio": args.sample_ratio,
        "sample_times": args.sample_times,
        "backbone_mode": args.backbone_mode,
        "max_residue_len": args.max_residue_len,
        "long_seq_mode": args.long_seq_mode,
        "long_seq_overlap": args.long_seq_overlap,
        "cache_tag": args.logits_cache_tag,
        "logits_cache_stage": args.logits_cache_stage,
    }
    cache_key = hashlib.sha1(
        json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    safe_model_name = model_name.split("/")[-1].replace(" ", "_")
    return os.path.join(
        args.logits_cache_dir,
        f"{protein_name}__{safe_model_name}__{variant_label}__{cache_key}.pt",
    )
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default=["AI4Protein/ProSST-2048"], nargs="+", help="Model name",)
    parser.add_argument("--model_out_name", type=str, default=["VenusREM"], nargs="+", help="Output model name",)
    
    # data directories
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory containing all data",)
    parser.add_argument("--aa_seq_dir", type=str, default=None, help="Directory containing FASTA files of residue sequences",)
    parser.add_argument("--struc_seq_dir", type=str, default=None, help="Directory containing FASTA files of structure sequences",)
    parser.add_argument("--mutant_dir", type=str, default=None, help="Directory containing CSV files with mutants",)
    
    # retrieval and logits mode
    parser.add_argument("--logit_mode", type=str, default="aa_seq_aln", choices=["aa_seq_aln", "struc_seq_aln", "aa_seq_aln+struc_seq_aln", "struc_seq_aln+aa_seq_aln"], help="Mode to retrieve data",)
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha value for Orbit/Orbit-cal logits fusion",)
    parser.add_argument("--enhance_on_v1", action="store_true", help="Use fixed v1 baseline logits (alpha=0.8, aa_seq_aln) as anchor, then blend Orbit logits as residual enhancement")
    parser.add_argument("--enhance_lambda", type=float, default=0.3, help="Blend weight for Orbit logits in v1-anchor enhancement: final=(1-lambda)*v1+lambda*orbit")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use",)
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Ratio of samples to use",)
    parser.add_argument("--sample_times", type=int, default=1, help="Number of times to sample",)
    parser.add_argument("--aa_seq_aln_dir", type=str, default=None, help="Directory containing a2m files of residue alignments",)
    parser.add_argument("--struc_seq_aln_dir", type=str, default=None, help="Directory containing fasta files of foldseek structure alignments",)
    
    # orbit plugin options
    parser.add_argument("--orbit_enable", action="store_true", help="Enable VenusREM-Orbit plugin pipeline")
    parser.add_argument("--retriever", type=str, default="msa", choices=["msa", "hits"], help="Retriever plugin used in Orbit mode")
    parser.add_argument("--retriever2", type=str, default="none", choices=["none", "msa", "hits"], help="Second-layer retriever plugin")
    parser.add_argument("--fusion", type=str, default="linear_alpha", choices=["linear_alpha", "adaptive_gate", "two_stage_learnable_gate"], help="Primary fusion plugin used in Orbit mode")
    parser.add_argument("--fusion2", type=str, default="none", choices=["none", "two_stage_learnable_gate"], help="Second-stage fusion plugin")
    parser.add_argument("--hits_dir", type=str, default=None, help="Directory containing homolog hits metadata files")
    parser.add_argument("--hits_file_suffix", type=str, default="_tblout.txt", help="Suffix of hits metadata files")
    parser.add_argument("--layer2_weight", type=float, default=1.0, help="Layer2 scalar weight in two-stage fusion")
    parser.add_argument("--gate_temperature", type=float, default=1.0, help="Gate temperature in two-stage fusion")
    parser.add_argument("--alpha_family", type=float, default=1.0, help="Family-level scaling factor for layer2 gate")
    parser.add_argument("--adaptive_min_gate", type=float, default=0.0, help="Minimum gate for adaptive fusion")
    parser.add_argument("--adaptive_max_gate", type=float, default=0.95, help="Maximum gate for adaptive fusion")
    parser.add_argument("--enable_gate_diagnostics", action="store_true", help="Print gate diagnostics for two-stage fusion")
    parser.add_argument("--print_compare_spearman", action="store_true", help="Print raw backbone vs VenusREM vs VenusREM-Orbit Spearman per protein")
    parser.add_argument("--backbone_mode", type=str, default="auto", choices=["auto", "prosst", "plain_mlm"], help="Backbone forward mode: auto/prosst/plain_mlm")
    parser.add_argument("--structure_vocab_subdir", type=str, default=None, help="Optional structure-seq subdir under struc_seq_dir (e.g. 2048)")
    parser.add_argument("--pdb_dir", type=str, default=None, help="Directory containing pdb files for RSA computation")
    parser.add_argument("--max_residue_len", type=int, default=None, help="Override max residue length for backbone forward; defaults to model limit")
    parser.add_argument("--long_seq_mode", type=str, default="auto_window", choices=["auto_window", "error"], help="How to handle proteins longer than model limit")
    parser.add_argument("--long_seq_overlap", type=int, default=256, help="Overlap size for long-sequence sliding-window inference")
    parser.add_argument("--scoring_mode", type=str, default="log_odds", choices=["log_odds", "calibrated_margin", "ccd_exact", "temp_scaled_log_odds", "entropy_adaptive_margin", "orbit_confidence_gate", "rsa_modulated_ccd"], help="Mutant scoring formula: standard log-odds, calibrated margin, ccd_exact, rsa_modulated_ccd, or extended Orbit scoring heads")
    parser.add_argument("--background_weight", type=float, default=0.25, help="Weight of amino-acid background correction in calibrated margin mode")
    parser.add_argument("--uncertainty_weight", type=float, default=0.15, help="Weight of per-position entropy penalty in calibrated margin mode")
    parser.add_argument("--wt_confidence_weight", type=float, default=0.1, help="Weight of wild-type confidence bonus in calibrated margin mode")
    parser.add_argument("--score_temperature", type=float, default=1.0, help="Temperature used by temp-scaled scoring heads")
    parser.add_argument("--entropy_adaptive_power", type=float, default=1.0, help="Power for confidence-adaptive entropy penalty in entropy_adaptive_margin")
    parser.add_argument("--gate_center", type=float, default=0.5, help="Confidence center used by orbit_confidence_gate")
    parser.add_argument("--gate_sharpness", type=float, default=8.0, help="Sigmoid sharpness used by orbit_confidence_gate")
    parser.add_argument("--use_rsa_decay", action="store_true", help="Apply RSA-based position decay: score *= (1 - RSA_i), suppressing exposed-residue mutations")
    parser.add_argument("--rsa_decay_mode", type=str, default="raw", choices=["raw", "above_mean", "adaptive", "std_scaled"], help="RSA decay mode: raw=(1-RSA), above_mean=penalize only above protein mean, adaptive=strength scaled by 1-2*mean_RSA, std_scaled=above_mean weighted by protein std")
    parser.add_argument("--plddt_mode", type=str, default="off", choices=["off", "gate_ccd", "gate_rsa", "gate_all", "above_mean_decay", "std_scaled_decay"], help="pLDDT modulation: off, gate_ccd (A), gate_rsa (B), gate_all (C), above_mean_decay (D), std_scaled_decay (D+std)")
    parser.add_argument("--task_type", type=str, default="default", choices=["default", "surface"], help="Task type: default penalizes surface mutations (stability/activity), surface boosts them (binding)")
    parser.add_argument("--calibrate_on_raw", action="store_true", help="Compute CCD/calibrated_margin calibration terms from raw backbone logits instead of fused logits, avoiding double-penalty with LOR")
    parser.add_argument("--precomputed_logits_dir", type=str, default=None, help="Directory of precomputed backbone logits ({protein}.pt files); bypasses model forward entirely when available")
    parser.add_argument("--logits_cache_dir", type=str, default=None, help="Optional directory to persist per-protein logits for scoring-head reuse")
    parser.add_argument("--reuse_logits_cache", action="store_true", help="Load logits from cache when available")
    parser.add_argument("--write_logits_cache", action="store_true", help="Write computed logits to cache")
    parser.add_argument("--require_logits_cache", action="store_true", help="Deprecated: equivalent to --cache_miss_policy error")
    parser.add_argument("--cache_miss_policy", type=str, default="forward", choices=["forward", "error"], help="On missing/invalid cache: forward (warn+recompute) or error")
    parser.add_argument("--logits_cache_tag", type=str, default="", help="Extra cache namespace tag to avoid accidental cross-run reuse")
    parser.add_argument("--logits_cache_stage", type=str, default="raw", choices=["raw", "final"], help="Cache stage: raw backbone logits or final fused logits (after Orbit/alignment)")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bars for cleaner logs")
    parser.add_argument("--max_proteins", type=int, default=None, help="Only score first N proteins (debug/subset validation)")
    parser.add_argument("--protein_list", type=str, default=None, help="Comma-separated list of protein names to score (filters to only these)")
    parser.add_argument("--no_color", action="store_true", help="Disable ANSI colors in logs")
    parser.add_argument("--log_level", type=str, default="info", choices=["debug", "info", "warn", "error"], help="Console log verbosity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic inference")
    
    # output directory
    parser.add_argument("--out_scores_dir", default=None, help="Directory to save scores")
    args = parser.parse_args()
    args.show_progress = not args.disable_tqdm
    if args.logits_cache_dir:
        os.makedirs(args.logits_cache_dir, exist_ok=True)
    if args.require_logits_cache:
        args.cache_miss_policy = "error"
    set_deterministic_inference(args.seed)
    logger = CliLogger(level=args.log_level, use_color=should_use_color(args.no_color))

    logger.section("VenusREM / Orbit scoring run")
    logger.info("Scoring proteins")
    os.makedirs(args.out_scores_dir, exist_ok=True)
    os.makedirs(f"{args.out_scores_dir}/scores", exist_ok=True)
    if args.base_dir:
        if args.aa_seq_dir is None:
            args.aa_seq_dir = f"{args.base_dir}/aa_seq"
        else:
            args.aa_seq_dir = f"{args.base_dir}/{args.aa_seq_dir}"
            
        if args.struc_seq_dir is None:
            args.struc_seq_dir = f"{args.base_dir}/struc_seq"
        else:
            args.struc_seq_dir = f"{args.base_dir}/{args.struc_seq_dir}"
        
        if args.mutant_dir is None:
            args.mutant_dir = f"{args.base_dir}/substitutions"
        else:
            args.mutant_dir = f"{args.base_dir}/{args.mutant_dir}"
        if args.pdb_dir is None:
            args.pdb_dir = f"{args.base_dir}/pdbs"
        
        if args.aa_seq_aln_dir is None:
            args.aa_seq_aln_dir = f"{args.base_dir}/aa_seq_aln_a2m"
        else:
            args.aa_seq_aln_dir = f"{args.base_dir}/{args.aa_seq_aln_dir}"
            
        if args.struc_seq_aln_dir is None:
            args.struc_seq_aln_dir = f"{args.base_dir}/struc_seq_aln_foldseek"
        else:
            args.struc_seq_aln_dir = f"{args.base_dir}/{args.struc_seq_aln_dir}"

    protein_names = sorted(read_names(args.aa_seq_dir))
    if args.protein_list:
        allowed = set(args.protein_list.split(","))
        protein_names = [p for p in protein_names if p in allowed]
    if args.max_proteins is not None and args.max_proteins > 0:
        protein_names = protein_names[: args.max_proteins]
    logger.info(f"Total proteins: {len(protein_names)}")
    logger.debug(f"Protein preview: {format_name_preview(protein_names)}")
    
    for model_idx, model_name in enumerate(args.model_name):
        corrs = []
        compare_table_printed = False
        logger.section(f"Model {model_idx+1}/{len(args.model_name)}: {model_name}")
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # User-requested override: force config max residue length to 4096.
        force_config_max_residue_len(model, residue_len=4096)
        model_max_residue_len = args.max_residue_len
        if model_max_residue_len is None:
            model_max_residue_len = infer_model_max_residue_len(model, tokenizer)
        logger.info(f"Max residue length for forward: {model_max_residue_len}")
        
        protein_progress = tqdm(
            protein_names,
            desc=f"Proteins[{model_name.split('/')[-1]}]",
            disable=not args.show_progress,
            leave=True,
            dynamic_ncols=True,
        )
        for idx, protein_name in enumerate(protein_progress):
            logger.info(f"Scoring protein {idx+1}/{len(protein_names)}", protein=protein_name)
            # load data
            residue_fasta = f"{args.aa_seq_dir}/{protein_name}.fasta"
            structure_fasta = resolve_structure_fasta_path(args, protein_name, model_name)
            pdb_file = (
                f"{args.pdb_dir}/{protein_name}.pdb"
                if args.pdb_dir and os.path.exists(f"{args.pdb_dir}/{protein_name}.pdb")
                else None
            )
            mutant_file = f"{args.mutant_dir}/{protein_name}.csv"
            aa_seq_aln_file = None
            struc_seq_aln_file = None
            if args.logit_mode is not None:
                if "aa_seq_aln" in args.logit_mode:
                    if os.path.exists(f"{args.aa_seq_aln_dir}/{protein_name}.a2m"):
                        aa_seq_aln_file = f"{args.aa_seq_aln_dir}/{protein_name}.a2m"
                    elif os.path.exists(f"{args.aa_seq_aln_dir}/{protein_name}.a3m"):
                        aa_seq_aln_file = f"{args.aa_seq_aln_dir}/{protein_name}.a3m"
                    elif os.path.exists(f"{args.aa_seq_aln_dir}/{protein_name}.fasta"):
                        aa_seq_aln_file = f"{args.aa_seq_aln_dir}/{protein_name}.fasta"
                else:
                    aa_seq_aln_file = None
                
                if "struc_seq_aln" in args.logit_mode:
                    struc_seq_aln_file = f"{args.struc_seq_aln_dir}/{protein_name}.fasta"
                else:
                    struc_seq_aln_file = None
            else:
                aa_seq_aln_file = None
                struc_seq_aln_file = None
                    
            if os.path.exists(f"{args.out_scores_dir}/scores/{protein_name}.csv"):
                mutant_file = f"{args.out_scores_dir}/scores/{protein_name}.csv"
            mutant_df = pd.read_csv(mutant_file)
            
            if args.model_out_name:
                model_out_name = args.model_out_name[model_idx]
            else:
                model_out_name = model_name.split("/")[-1]

            backbone_name = model_name.split("/")[-1]
            raw_col = f"{backbone_name}__raw_backbone"
            venus_col = f"{backbone_name}{V1_BASELINE_COL_SUFFIX}"
            orbit_col = model_out_name
            raw_logits_cache_path = build_logits_cache_path(
                args=args,
                protein_name=protein_name,
                model_name=model_name,
                variant_label="raw_backbone",
            )
            venus_logits_cache_path = build_logits_cache_path(
                args=args,
                protein_name=protein_name,
                model_name=model_name,
                variant_label="venus_v1",
            )
            orbit_logits_cache_path = build_logits_cache_path(
                args=args,
                protein_name=protein_name,
                model_name=model_name,
                variant_label="orbit_main",
            )
            orbit_log_odds_cache_path = build_logits_cache_path(
                args=args,
                protein_name=protein_name,
                model_name=model_name,
                variant_label="orbit_log_odds",
            )
                
            if args.print_compare_spearman:
                if raw_col not in mutant_df.columns:
                    raw_scores = score_protein(
                        model=model,
                        tokenizer=tokenizer,
                        residue_fasta=residue_fasta,
                        structure_fasta=structure_fasta,
                        mutant_df=mutant_df,
                        alpha=0.0,
                        aa_seq_aln_file=None,
                        struc_seq_aln_file=None,
                        sample_size=args.sample_size,
                        sample_ratio=args.sample_ratio,
                        sample_times=args.sample_times,
                        orbit_args=None,
                        protein_name=protein_name,
                        backbone_mode=args.backbone_mode,
                        pdb_file=pdb_file,
                        max_residue_len=model_max_residue_len,
                        long_seq_mode=args.long_seq_mode,
                        long_seq_overlap=args.long_seq_overlap,
                        quiet=True,
                        show_progress=args.show_progress,
                        logger=logger,
                        scoring_mode=args.scoring_mode,
                        background_weight=args.background_weight,
                        uncertainty_weight=args.uncertainty_weight,
                        wt_confidence_weight=args.wt_confidence_weight,
                        score_temperature=args.score_temperature,
                        entropy_adaptive_power=args.entropy_adaptive_power,
                        gate_center=args.gate_center,
                        gate_sharpness=args.gate_sharpness,
                        logits_cache_path=raw_logits_cache_path,
                        reuse_logits_cache=args.reuse_logits_cache,
                        write_logits_cache=args.write_logits_cache,
                        cache_miss_policy=args.cache_miss_policy,
                        logits_cache_stage=args.logits_cache_stage,
                        calibrate_on_raw=args.calibrate_on_raw,
                        precomputed_logits_dir=args.precomputed_logits_dir,
                        use_rsa_decay=args.use_rsa_decay,
                        rsa_decay_mode=args.rsa_decay_mode,
                        pdb_dir=args.pdb_dir,
                        plddt_mode=args.plddt_mode,
                        task_type=args.task_type,
                    )
                    mutant_df[raw_col] = raw_scores
                raw_corr = spearmanr(mutant_df["DMS_score"], mutant_df[raw_col]).correlation

                if args.orbit_enable:
                    if venus_col not in mutant_df.columns:
                        venus_args = clone_args_with_overrides(args, orbit_enable=False)
                        venus_scores = score_protein(
                            model=model,
                            tokenizer=tokenizer,
                            residue_fasta=residue_fasta,
                            structure_fasta=structure_fasta,
                            mutant_df=mutant_df,
                            alpha=V1_BASELINE_ALPHA,
                            aa_seq_aln_file=aa_seq_aln_file,
                            struc_seq_aln_file=None,
                            sample_size=args.sample_size,
                            sample_ratio=args.sample_ratio,
                            sample_times=args.sample_times,
                            orbit_args=venus_args,
                            protein_name=protein_name,
                            backbone_mode=args.backbone_mode,
                            pdb_file=pdb_file,
                            max_residue_len=model_max_residue_len,
                            long_seq_mode=args.long_seq_mode,
                            long_seq_overlap=args.long_seq_overlap,
                            quiet=True,
                            show_progress=args.show_progress,
                            logger=logger,
                            scoring_mode="log_odds",
                            background_weight=args.background_weight,
                            uncertainty_weight=args.uncertainty_weight,
                            wt_confidence_weight=args.wt_confidence_weight,
                            score_temperature=args.score_temperature,
                            entropy_adaptive_power=args.entropy_adaptive_power,
                            gate_center=args.gate_center,
                            gate_sharpness=args.gate_sharpness,
                            logits_cache_path=venus_logits_cache_path,
                            reuse_logits_cache=args.reuse_logits_cache,
                            write_logits_cache=args.write_logits_cache,
                            cache_miss_policy=args.cache_miss_policy,
                            logits_cache_stage=args.logits_cache_stage,
                            calibrate_on_raw=args.calibrate_on_raw,
                        precomputed_logits_dir=args.precomputed_logits_dir,
                        use_rsa_decay=args.use_rsa_decay,
                        rsa_decay_mode=args.rsa_decay_mode,
                        pdb_dir=args.pdb_dir,
                        plddt_mode=args.plddt_mode,
                        task_type=args.task_type,
                        )
                        mutant_df[venus_col] = venus_scores
                    venus_corr = spearmanr(mutant_df["DMS_score"], mutant_df[venus_col]).correlation

            if orbit_col not in mutant_df.columns:
                scores = score_protein(
                        model=model,
                        tokenizer=tokenizer,
                        residue_fasta=residue_fasta,
                        structure_fasta=structure_fasta,
                        mutant_df=mutant_df,
                        alpha=args.alpha,
                        aa_seq_aln_file=aa_seq_aln_file,
                        struc_seq_aln_file=struc_seq_aln_file,
                        sample_size=args.sample_size,
                        sample_ratio=args.sample_ratio,
                        sample_times=args.sample_times,
                        orbit_args=args,
                        protein_name=protein_name,
                        backbone_mode=args.backbone_mode,
                        pdb_file=pdb_file,
                        max_residue_len=model_max_residue_len,
                        long_seq_mode=args.long_seq_mode,
                        long_seq_overlap=args.long_seq_overlap,
                        quiet=False,
                        show_progress=args.show_progress,
                        logger=logger,
                        scoring_mode=args.scoring_mode,
                        background_weight=args.background_weight,
                        uncertainty_weight=args.uncertainty_weight,
                        wt_confidence_weight=args.wt_confidence_weight,
                        score_temperature=args.score_temperature,
                        entropy_adaptive_power=args.entropy_adaptive_power,
                        gate_center=args.gate_center,
                        gate_sharpness=args.gate_sharpness,
                        logits_cache_path=orbit_logits_cache_path,
                        reuse_logits_cache=args.reuse_logits_cache,
                        write_logits_cache=args.write_logits_cache,
                        cache_miss_policy=args.cache_miss_policy,
                        logits_cache_stage=args.logits_cache_stage,
                        calibrate_on_raw=args.calibrate_on_raw,
                        precomputed_logits_dir=args.precomputed_logits_dir,
                        use_rsa_decay=args.use_rsa_decay,
                        rsa_decay_mode=args.rsa_decay_mode,
                        pdb_dir=args.pdb_dir,
                        plddt_mode=args.plddt_mode,
                        task_type=args.task_type,
                    )
                mutant_df[orbit_col] = scores
        
            corr = spearmanr(mutant_df["DMS_score"], mutant_df[orbit_col]).correlation
            corrs.append(corr)
            if args.print_compare_spearman and args.orbit_enable:
                orbit_cal_mode = args.scoring_mode != "log_odds"
                orbit_base_col = f"{backbone_name}__orbit_log_odds"
                orbit_base_corr = None
                orbit_cal_corr = None
                if orbit_cal_mode:
                    if orbit_base_col not in mutant_df.columns:
                        orbit_base_args = clone_args_with_overrides(args, scoring_mode="log_odds")
                        orbit_base_scores = score_protein(
                            model=model,
                            tokenizer=tokenizer,
                            residue_fasta=residue_fasta,
                            structure_fasta=structure_fasta,
                            mutant_df=mutant_df,
                            alpha=args.alpha,
                            aa_seq_aln_file=aa_seq_aln_file,
                            struc_seq_aln_file=struc_seq_aln_file,
                            sample_size=args.sample_size,
                            sample_ratio=args.sample_ratio,
                            sample_times=args.sample_times,
                            orbit_args=orbit_base_args,
                            protein_name=protein_name,
                            backbone_mode=args.backbone_mode,
                            pdb_file=pdb_file,
                            max_residue_len=model_max_residue_len,
                            long_seq_mode=args.long_seq_mode,
                            long_seq_overlap=args.long_seq_overlap,
                            quiet=True,
                            show_progress=args.show_progress,
                            logger=logger,
                            scoring_mode="log_odds",
                            background_weight=args.background_weight,
                            uncertainty_weight=args.uncertainty_weight,
                            wt_confidence_weight=args.wt_confidence_weight,
                            score_temperature=args.score_temperature,
                            entropy_adaptive_power=args.entropy_adaptive_power,
                            gate_center=args.gate_center,
                            gate_sharpness=args.gate_sharpness,
                            logits_cache_path=orbit_log_odds_cache_path,
                            reuse_logits_cache=args.reuse_logits_cache,
                            write_logits_cache=args.write_logits_cache,
                            cache_miss_policy=args.cache_miss_policy,
                            logits_cache_stage=args.logits_cache_stage,
                            calibrate_on_raw=args.calibrate_on_raw,
                        precomputed_logits_dir=args.precomputed_logits_dir,
                        use_rsa_decay=args.use_rsa_decay,
                        rsa_decay_mode=args.rsa_decay_mode,
                        pdb_dir=args.pdb_dir,
                        plddt_mode=args.plddt_mode,
                        task_type=args.task_type,
                        )
                        mutant_df[orbit_base_col] = orbit_base_scores
                    orbit_base_corr = spearmanr(
                        mutant_df["DMS_score"], mutant_df[orbit_base_col]
                    ).correlation
                    orbit_cal_corr = corr

                if not compare_table_printed:
                    if orbit_cal_mode:
                        logger.info(
                            f"Compare Spearman table (v1 fixed baseline: alpha={V1_BASELINE_ALPHA:g}, mode={V1_BASELINE_LOGIT_MODE}; Delta columns: orbit-v1 and orbit-cal-v1)"
                        )
                        print_compare_table_header(
                            logger,
                            include_venus=True,
                            raw_label=backbone_name,
                            venus_label="v1",
                            orbit_label="orbit",
                            orbit_cal_label="orbit-cal",
                        )
                    else:
                        logger.info(
                            f"Compare Spearman table (v1 fixed baseline: alpha={V1_BASELINE_ALPHA:g}, mode={V1_BASELINE_LOGIT_MODE}; Orbit - v1 as Delta)"
                        )
                        print_compare_table_header(
                            logger,
                            include_venus=True,
                            raw_label=backbone_name,
                            venus_label="v1",
                            orbit_label="orbit",
                        )
                    compare_table_printed = True
                if orbit_cal_mode:
                    print_compare_table_row_extended(
                        logger=logger,
                        protein_name=protein_name,
                        raw_corr=raw_corr,
                        venus_corr=venus_corr,
                        orbit_corr=orbit_base_corr,
                        orbit_cal_corr=orbit_cal_corr,
                    )
                else:
                    print_compare_table_row(
                        logger=logger,
                        protein_name=protein_name,
                        raw_corr=raw_corr,
                        venus_corr=venus_corr,
                        orbit_corr=corr,
                    )
            elif args.print_compare_spearman:
                if not compare_table_printed:
                    logger.info("Compare Spearman table (Current - Raw as Delta)")
                    print_compare_table_header(
                        logger,
                        include_venus=False,
                        raw_label=backbone_name,
                        current_label=orbit_col,
                    )
                    compare_table_printed = True
                print_compare_table_row(
                    logger=logger,
                    protein_name=protein_name,
                    raw_corr=raw_corr,
                    orbit_corr=corr,
                    venus_corr=None,
                )
            else:
                logger.success(f"{model_out_name} Spearman={corr:.4f}", protein=protein_name)
            mutant_df.to_csv(f"{args.out_scores_dir}/scores/{protein_name}.csv", index=False)
        
        if compare_table_printed:
            if args.orbit_enable and args.scoring_mode in {"calibrated_margin", "ccd_exact", "rsa_modulated_ccd"}:
                logger.info(
                    "+"
                    + "-" * 42
                    + "+"
                    + "-" * 11
                    + "+"
                    + "-" * 11
                    + "+"
                    + "-" * 11
                    + "+"
                    + "-" * 11
                    + "+"
                    + "-" * 13
                    + "+"
                    + "-" * 13
                    + "+"
                )
            elif args.orbit_enable:
                logger.info("+" + "-" * 42 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 10 + "+")
            else:
                logger.info("+" + "-" * 42 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 10 + "+")
        logger.section(f"{model_out_name} average Spearman: {sum(corrs)/len(corrs):.4f}")
        summary_df_path = f"{args.out_scores_dir}/summary_performance.csv"
        if os.path.exists(summary_df_path):
            summary_df = pd.read_csv(summary_df_path)
            summary_df[model_out_name] = corrs
        else:
            summary_df = pd.DataFrame({'protein': protein_names, model_out_name: corrs})
        summary_df.to_csv(f"{args.out_scores_dir}/summary_performance.csv", index=False)