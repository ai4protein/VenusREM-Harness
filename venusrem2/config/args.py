import os
from argparse import ArgumentParser


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="venus-orbit", description="Venus-Orbit: PLM variant-effect scoring with Orbit calibration")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Backbone model key (preferred). Use --list-models to see options. Falls back to --baseline_type.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional weight id / HF repo / local path override for the selected --model",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Weight cache directory (default: ~/.cache/venus_orbit/weights or $VENUS_ORBIT_CACHE)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List registered backbone models and exit",
    )
    parser.add_argument("--model_name", type=str, default=["AI4Protein/ProSST-2048"], nargs="+", help="HF model id(s) (legacy; prefer --model / --model_id)")
    parser.add_argument("--model_out_name", type=str, default=["VenusREM"], nargs="+", help="Output model name")

    # data directories
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory containing all data")
    parser.add_argument("--aa_seq_dir", type=str, default=None, help="Directory containing FASTA files of residue sequences")
    parser.add_argument("--struc_seq_dir", type=str, default=None, help="Directory containing FASTA files of structure sequences")
    parser.add_argument("--mutant_dir", type=str, default=None, help="Directory containing CSV files with mutants")

    # single-protein mode (no base_dir)
    parser.add_argument("--fasta", type=str, default=None, help="Single wild-type FASTA (mutually exclusive with --base_dir / --aa_seq_dir)")
    parser.add_argument("--pdb", type=str, default=None, help="Optional single PDB for RSA / structure models")
    parser.add_argument("--mutants", type=str, default=None, help="Optional existing mutants CSV (mutant[,DMS_score]); skips auto-generation")
    parser.add_argument(
        "--mutant_sites",
        type=str,
        default=None,
        help="n-point saturation orders to generate when --mutants is omitted, e.g. 1 or 1,2,3 (1=single, 2=double, 3=triple)",
    )
    parser.add_argument(
        "--positions",
        type=str,
        default=None,
        help="1-based residue list for mutagenesis, e.g. 10,11,12 (required when --mutant_sites includes 2+)",
    )
    parser.add_argument(
        "--residue_range",
        type=str,
        default=None,
        help="1-based inclusive residue range for mutagenesis, e.g. 10-20 (union with --positions)",
    )
    parser.add_argument(
        "--max_mutants",
        type=int,
        default=1_000_000,
        help="Refuse auto-generated libraries larger than this (default: 1000000)",
    )

    # retrieval and logits mode
    parser.add_argument("--logit_mode", type=str, default="aa_seq_aln", choices=["aa_seq_aln", "struc_seq_aln", "aa_seq_aln+struc_seq_aln", "struc_seq_aln+aa_seq_aln"], help="Mode to retrieve data")
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha value for MSA alignment prior fusion: (1-alpha)*logits + alpha*msa_counts")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Ratio of samples to use")
    parser.add_argument("--sample_times", type=int, default=1, help="Number of times to sample")
    parser.add_argument("--aa_seq_aln_dir", type=str, default=None, help="Directory containing a2m files of residue alignments")
    parser.add_argument("--struc_seq_aln_dir", type=str, default=None, help="Directory containing fasta files of foldseek structure alignments")

    parser.add_argument("--print_compare_spearman", action="store_true", help="Print raw backbone vs VenusREM vs VenusREM-Orbit Spearman per protein")
    parser.add_argument("--backbone_mode", type=str, default="auto", choices=["auto", "prosst", "plain_mlm"], help="Backbone forward mode: auto/prosst/plain_mlm")
    parser.add_argument("--structure_vocab_subdir", type=str, default=None, help="Optional structure-seq subdir under struc_seq_dir (e.g. 2048)")
    parser.add_argument("--pdb_dir", type=str, default=None, help="Directory containing pdb files for RSA computation")
    parser.add_argument("--max_residue_len", type=int, default=None, help="Override max residue length for backbone forward; defaults to model limit")
    parser.add_argument("--long_seq_mode", type=str, default="auto_window", choices=["auto_window", "error"], help="How to handle proteins longer than model limit")
    parser.add_argument("--long_seq_overlap", type=int, default=256, help="Overlap size for long-sequence sliding-window inference")
    parser.add_argument("--scoring_mode", type=str, default="log_odds", choices=["log_odds", "calibrated_margin", "ccd_exact", "rsa_modulated_ccd"], help="Mutant scoring formula: standard log-odds, calibrated margin, ccd_exact, or rsa_modulated_ccd")
    parser.add_argument("--background_weight", type=float, default=0.2, help="Weight of amino-acid background correction in calibrated margin mode")
    parser.add_argument("--wt_confidence_weight", type=float, default=0.05, help="Weight of wild-type confidence bonus in calibrated margin mode")
    parser.add_argument("--use_rsa_decay", action="store_true", help="Apply RSA-based position decay: score *= (1 - RSA_i), suppressing exposed-residue mutations")
    parser.add_argument("--rsa_decay_mode", type=str, default="raw", choices=["raw", "above_mean", "adaptive", "std_scaled"], help="RSA decay mode: raw=(1-RSA), above_mean=penalize only above protein mean, adaptive=strength scaled by 1-2*mean_RSA, std_scaled=above_mean weighted by protein std")
    parser.add_argument("--use_plddt_decay", action="store_true", help="Apply pLDDT-based disorder decay: score *= (1 - disorder_excess), suppressing disordered-region mutations")
    parser.add_argument("--plddt_decay_mode", type=str, default="above_mean", choices=["above_mean", "std_scaled"], help="pLDDT decay mode: above_mean=penalize only above protein mean disorder, std_scaled=above_mean weighted by protein disorder std")
    parser.add_argument("--task_type", type=str, default="default", choices=["default", "surface"], help="Task type: default penalizes surface mutations (stability/activity), surface boosts them (binding)")
    parser.add_argument("--calibrate_on_raw", action="store_true", help="Compute CCD/calibrated_margin calibration terms from raw backbone logits instead of fused logits, avoiding double-penalty with LOR")
    parser.add_argument("--scoring_strategy", type=str, default="wt-marginals", choices=["wt-marginals", "masked-marginals"], help="Forward strategy: wt-marginals (single WT pass) or masked-marginals (per-position masking)")
    parser.add_argument(
        "--disable_adaptive_ccd",
        action="store_true",
        help="Deprecated no-op: CCD is always v1 (docs formula; no bg_scale/bg_consistency). Kept for old scripts.",
    )
    parser.add_argument("--precomputed_logits_dir", type=str, default=None, help="Directory of precomputed backbone logits ({protein}.pt files); bypasses model forward entirely when available")
    parser.add_argument("--logits_cache_dir", type=str, default=None, help="Optional directory to persist per-protein logits for scoring-head reuse")
    parser.add_argument("--reuse_logits_cache", action="store_true", help="Load logits from cache when available")
    parser.add_argument("--write_logits_cache", action="store_true", help="Write computed logits to cache")
    parser.add_argument("--cache_miss_policy", type=str, default="forward", choices=["forward", "error"], help="On missing/invalid cache: forward (warn+recompute) or error")
    parser.add_argument("--logits_cache_tag", type=str, default="", help="Extra cache namespace tag to avoid accidental cross-run reuse")
    parser.add_argument("--logits_cache_stage", type=str, default="raw", choices=["raw", "final"], help="Cache stage: raw backbone logits or final fused logits (after MSA alignment prior)")
    parser.add_argument("--aln_count_cache_dir", type=str, default=None, help="Directory to cache MSA count matrices (.pt); avoids re-parsing a2m files across variants")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bars for cleaner logs")
    parser.add_argument("--max_proteins", type=int, default=None, help="Only score first N proteins (debug/subset validation)")
    parser.add_argument("--protein_list", type=str, default=None, help="Comma-separated list of protein names to score (filters to only these)")
    parser.add_argument("--no_color", action="store_true", help="Disable ANSI colors in logs")
    parser.add_argument("--log_level", type=str, default="info", choices=["debug", "info", "warn", "error"], help="Console log verbosity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic inference")

    # baseline integration
    parser.add_argument("--baseline_type", type=str, default="auto", choices=["auto", "esm2", "esm1b", "esm1v", "saprot", "protssn", "esm_if", "protein_mpnn", "progen2", "progen3", "protgpt2", "rita", "esm3", "tranception", "carp", "s2f", "s3f"], help="Legacy alias of --model (auto = HF MLM / ProSST)")
    parser.add_argument("--foldseek_bin", type=str, default=None, help="Path to foldseek binary (saprot; auto-download if missing)")
    parser.add_argument("--esm1v_seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="ESM-1v seed indices for ensemble")
    parser.add_argument("--protssn_model_dir", type=str, default=None, help="ProtSSN GNN weights dir (default: cache/protssn, auto-download)")
    parser.add_argument("--protssn_norm_dir", type=str, default=None, help="Directory containing ProtSSN norm files (defaults to bundled)")
    parser.add_argument("--protssn_no_ensemble", action="store_true", help="Use single ProtSSN model instead of 9-model ensemble")
    parser.add_argument("--esm_if_chain", type=str, default="A", help="PDB chain ID for ESM-IF inverse folding (default: A)")
    parser.add_argument("--protein_mpnn_checkpoint", type=str, default=None, help="ProteinMPNN .pt or dir (default: cache/protein_mpnn, auto-download)")
    parser.add_argument("--protein_mpnn_chain", type=str, default="A", help="PDB chain ID for ProteinMPNN (default: A)")
    parser.add_argument("--protein_mpnn_scoring_mode", type=str, default="teacher_force", choices=["teacher_force", "random_order"], help="ProteinMPNN scoring mode: deterministic teacher-forced order or random decoding-order scoring")
    parser.add_argument("--protein_mpnn_random_orders", type=int, default=1, help="Number of random decoding orders to average when --protein_mpnn_scoring_mode=random_order")
    parser.add_argument("--progen2_model_name_or_path", type=str, default="hugohrban/progen2-large", choices=["hugohrban/progen2-small", "hugohrban/progen2-medium", "hugohrban/progen2-base", "hugohrban/progen2-large", "hugohrban/progen2-xlarge"], help="HF model name for Progen2")
    parser.add_argument("--progen2_fp16", action="store_true", help="Use fp16 for Progen2 inference")
    parser.add_argument("--tranception_checkpoint", type=str, default="OATML-Markslab/Tranception_Large", choices=["OATML-Markslab/Tranception_Small", "OATML-Markslab/Tranception_Medium", "OATML-Markslab/Tranception_Large"], help="HF model name or local path for Tranception")
    parser.add_argument("--tranception_no_mirror", action="store_true", help="Disable R->L scoring for Tranception (only use L->R)")
    parser.add_argument("--s2f_config", type=str, default=None, help="Path to S2F/S3F YAML config (default: bundled)")
    parser.add_argument("--s2f_checkpoint", type=str, default=None, help="S2F/S3F checkpoint (.pt); S3F defaults to data/s3f_weights/s3f.pth or cache")
    parser.add_argument(
        "--s3f_surface_dir",
        type=str,
        default=None,
        help="Directory of S3F surface .pkl files (default: <base_dir>/s3f_surfaces_af2_assay_resolved_full)",
    )
    parser.add_argument("--progen3_model_name_or_path", type=str, default="Profluent-Bio/progen3-1b", choices=["Profluent-Bio/progen3-112m", "Profluent-Bio/progen3-219m", "Profluent-Bio/progen3-339m", "Profluent-Bio/progen3-762m", "Profluent-Bio/progen3-1b", "Profluent-Bio/progen3-3b"], help="HF model name for Progen3")
    parser.add_argument("--progen3_fp16", action="store_true", help="Use fp16 for Progen3 inference")
    parser.add_argument("--protgpt2_model_name_or_path", type=str, default="nferruz/ProtGPT2", help="HF model name or path for ProtGPT2")
    parser.add_argument("--rita_model_name_or_path", type=str, default="lightonai/RITA_xl", choices=["lightonai/RITA_s", "lightonai/RITA_m", "lightonai/RITA_l", "lightonai/RITA_xl"], help="HF model name for RITA")
    parser.add_argument("--esm3_model_name", type=str, default="esmc_300m", choices=["esmc_300m", "esmc_600m", "esm3_sm_open_v1"], help="ESM3/ESM-C model name")
    parser.add_argument("--carp_model_name", type=str, default="carp_640M", choices=["carp_600k", "carp_38M", "carp_76M", "carp_640M"], help="CARP model name (auto-downloaded from Zenodo)")

    # output directory
    parser.add_argument("--out_scores_dir", default=None, help="Directory to save scores")
    return parser


def postprocess_args(args):
    args.show_progress = not args.disable_tqdm
    if args.logits_cache_dir:
        os.makedirs(args.logits_cache_dir, exist_ok=True)

    fasta = getattr(args, "fasta", None)
    if fasta:
        if args.base_dir or args.aa_seq_dir or args.mutant_dir:
            raise SystemExit(
                "--fasta is mutually exclusive with --base_dir / --aa_seq_dir / --mutant_dir"
            )
        if getattr(args, "mutants", None) and getattr(args, "mutant_sites", None):
            raise SystemExit("Use either --mutants (existing CSV) or --mutant_sites (auto-generate), not both")
        if not getattr(args, "mutants", None) and not getattr(args, "mutant_sites", None):
            raise SystemExit(
                "Single-protein mode (--fasta) requires --mutants or --mutant_sites "
                "(e.g. --mutant_sites 1 or --mutant_sites 1,2,3)"
            )
    elif getattr(args, "mutant_sites", None) or getattr(args, "mutants", None) or getattr(args, "pdb", None):
        if not args.base_dir and not args.aa_seq_dir:
            raise SystemExit("--mutant_sites / --mutants / --pdb require --fasta or --base_dir / --aa_seq_dir")

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
        else:
            args.pdb_dir = f"{args.base_dir}/{args.pdb_dir}"

        if args.aa_seq_aln_dir is None:
            args.aa_seq_aln_dir = f"{args.base_dir}/aa_seq_aln_a2m"
        else:
            args.aa_seq_aln_dir = f"{args.base_dir}/{args.aa_seq_aln_dir}"

        if args.struc_seq_aln_dir is None:
            args.struc_seq_aln_dir = f"{args.base_dir}/struc_seq_aln_foldseek"
        else:
            args.struc_seq_aln_dir = f"{args.base_dir}/{args.struc_seq_aln_dir}"

    return args
