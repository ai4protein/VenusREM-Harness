import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter

_HELP_EPILOG = """\
examples:
  rem2
  rem2 demo
  rem2 doctor
  rem2 --model esm2 --base_dir data/proteingym_v1
  rem2 --model venusrem2 --base_dir data/proteingym_v1
  rem2 --model esm2 --fasta prot.fasta
  rem2 --list-models

notes:
  rem2 is the generic calibration recipe (any backbone). Default --model is esm2.
  VenusREM2 is rem2 on a ProSST ensemble only (--model venusrem2).
  rem2 demo scores the bundled trp-cage fixture with ESM-2 8M.
  --base_dir auto-picks MSA / PDB / structure-token subdirs if present.
  --fasta without --mutants does full-length single-site saturation.
  scores write to result/ unless --out_scores_dir is set.
"""


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="rem2",
        usage=(
            "%(prog)s --model MODEL (--base_dir DIR | --fasta FILE) [options]\n"
            "       %(prog)s demo\n"
            "       %(prog)s doctor\n"
            "       %(prog)s --list-models"
        ),
        description=(
            "Calibrate any protein language model for variant effect prediction.\n"
            "Default recipe: entropy-α, β=1-α, CCD, RSA / pLDDT decay."
        ),
        epilog=_HELP_EPILOG,
        formatter_class=RawDescriptionHelpFormatter,
    )

    common = parser.add_argument_group("common")
    common.add_argument(
        "--model",
        type=str,
        default=None,
        help="backbone key (default: esm2). see rem2 --list-models",
    )
    common.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="override weights: HF repo, local path, or checkpoint id",
    )
    common.add_argument(
        "--list-models",
        action="store_true",
        help="print registered backbones and exit",
    )
    common.add_argument(
        "--out_scores_dir",
        default="result",
        help="output directory (default: result)",
    )
    common.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "weight cache (default: ~/.cache/venusrem2/weights or $VENUSREM2_CACHE). "
            "also searches $VENUS_ORBIT_CACHE and ~/.cache/venus_orbit/weights"
        ),
    )
    common.add_argument(
        "--auto_download",
        dest="auto_download",
        action="store_true",
        help="download missing checkpoints without asking",
    )
    common.add_argument(
        "--no_auto_download",
        dest="auto_download",
        action="store_false",
        help="never download; use --cache_dir or explicit checkpoint paths only",
    )
    parser.set_defaults(auto_download=None)

    data = parser.add_argument_group("dataset (--base_dir)")
    data.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="dataset root with aa_seq/ and substitutions/; auto-picks MSA / PDB / struc_seq",
    )
    data.add_argument("--aa_seq_dir", type=str, default=None, help="wild-type FASTA directory (default: <base_dir>/aa_seq)")
    data.add_argument("--mutant_dir", type=str, default=None, help="mutant CSV directory (default: <base_dir>/substitutions)")
    data.add_argument(
        "--aa_seq_aln_dir",
        type=str,
        default=None,
        help="MSA a2m directory (auto: aa_seq_aln_a2m_af2cf, then aa_seq_aln_a2m)",
    )
    data.add_argument(
        "--pdb_dir",
        type=str,
        default=None,
        help="PDB directory for RSA / structure models (auto: pdbs_af2_*, then pdbs)",
    )
    data.add_argument(
        "--struc_seq_dir",
        type=str,
        default=None,
        help="ProSST structure-token FASTA directory (auto: struc_seq_af2_*, then struc_seq)",
    )
    data.add_argument(
        "--struc_seq_aln_dir",
        type=str,
        default=None,
        help="Foldseek structure-alignment FASTA directory",
    )
    data.add_argument(
        "--structure_vocab_subdir",
        type=str,
        default=None,
        help="optional subdir under --struc_seq_dir (e.g. 2048)",
    )
    data.add_argument(
        "--protein_list",
        type=str,
        default=None,
        help="comma-separated protein names to score",
    )
    data.add_argument(
        "--max_proteins",
        type=int,
        default=None,
        help="score only the first N proteins",
    )

    single = parser.add_argument_group("single protein (--fasta)")
    single.add_argument(
        "--fasta",
        type=str,
        default=None,
        help="one wild-type FASTA (cannot combine with --base_dir)",
    )
    single.add_argument("--pdb", type=str, default=None, help="optional PDB for RSA / structure models")
    single.add_argument(
        "--mutants",
        type=str,
        default=None,
        help="existing mutants CSV (mutant[,DMS_score]); skips auto-generation",
    )
    single.add_argument(
        "--mutant_sites",
        type=str,
        default=None,
        help="n-point saturation if --mutants is omitted, e.g. 1 or 1,2,3 (--fasta defaults to 1)",
    )
    single.add_argument(
        "--positions",
        type=str,
        default=None,
        help="1-based residue list, e.g. 10,11,12 (required when --mutant_sites includes 2+)",
    )
    single.add_argument(
        "--residue_range",
        type=str,
        default=None,
        help="1-based inclusive range, e.g. 10-20 (union with --positions)",
    )
    single.add_argument(
        "--max_mutants",
        type=int,
        default=1_000_000,
        help="refuse auto-generated libraries larger than this (default: 1000000)",
    )

    score = parser.add_argument_group("rem2 scoring")
    score.add_argument(
        "--alpha",
        type=str,
        default="entropy",
        help="MSA mix weight. entropy (default) or a float (0=off, 0.8=fixed ablation)",
    )
    score.add_argument(
        "--background_weight",
        type=str,
        default="one_minus_alpha",
        help="CCD β. one_minus_alpha (default, β=1-α) or a float",
    )
    score.add_argument(
        "--scoring_mode",
        type=str,
        default="calibrated_margin",
        choices=[
            "log_odds",
            "calibrated_margin",
            "ccd_exact",
            "rsa_modulated_ccd",
            "temp_scaled_log_odds",
            "entropy_adaptive_margin",
            "venusrem2_confidence_gate",
        ],
        help="mutant formula (default: calibrated_margin). use log_odds for raw Δ",
    )
    score.add_argument(
        "--scoring_strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "masked-marginals"],
        help="wt-marginals (one WT pass) or masked-marginals (MASK=yes models only)",
    )
    score.add_argument(
        "--wt_confidence_weight",
        type=float,
        default=0.0,
        help="wild-type confidence bonus (default: 0)",
    )
    score.add_argument(
        "--use_rsa_decay",
        dest="use_rsa_decay",
        action="store_true",
        help="RSA above-mean decay (on by default)",
    )
    score.add_argument(
        "--no_rsa_decay",
        dest="use_rsa_decay",
        action="store_false",
        help="disable RSA decay",
    )
    score.add_argument(
        "--rsa_decay_mode",
        type=str,
        default="above_mean",
        choices=["raw", "above_mean", "adaptive", "std_scaled"],
        help="RSA decay mode (default: above_mean)",
    )
    score.add_argument(
        "--use_plddt_decay",
        dest="use_plddt_decay",
        action="store_true",
        help="pLDDT above-mean decay (on by default; skipped for crystal/experimental PDBs)",
    )
    score.add_argument(
        "--no_plddt_decay",
        dest="use_plddt_decay",
        action="store_false",
        help="disable pLDDT decay",
    )
    score.add_argument(
        "--plddt_decay_mode",
        type=str,
        default="above_mean",
        choices=["above_mean", "std_scaled"],
        help="pLDDT decay mode (default: above_mean)",
    )
    score.add_argument(
        "--task_type",
        type=str,
        default="default",
        choices=["default", "surface"],
        help="default penalizes surface mutations; surface boosts them (binding)",
    )
    score.add_argument(
        "--calibrate_on_raw",
        dest="calibrate_on_raw",
        action="store_true",
        help="CCD terms from raw logits (default)",
    )
    score.add_argument(
        "--no_calibrate_on_raw",
        dest="calibrate_on_raw",
        action="store_false",
        help="CCD terms from fused logits",
    )
    score.add_argument(
        "--disable_adaptive_ccd",
        action="store_true",
        help="force CCD bg_scale=1 (default is adaptive)",
    )
    score.add_argument(
        "--logit_mode",
        type=str,
        default="aa_seq_aln",
        choices=["aa_seq_aln", "struc_seq_aln", "aa_seq_aln+struc_seq_aln", "struc_seq_aln+aa_seq_aln"],
        help="retrieval source for MSA / structure alignments (default: aa_seq_aln)",
    )
    score.add_argument("--sample_size", type=int, default=None, help="MSA subsample size")
    score.add_argument("--sample_ratio", type=float, default=1.0, help="MSA subsample ratio (default: 1.0)")
    score.add_argument("--sample_times", type=int, default=1, help="MSA subsample repeats (default: 1)")
    score.add_argument(
        "--model_out_name",
        type=str,
        default=None,
        nargs="+",
        help="score column name(s). default {backbone}__rem2, or VenusREM2__{K} for a ProSST ensemble",
    )
    score.add_argument(
        "--print_compare_spearman",
        action="store_true",
        help="print raw backbone vs rem2 Spearman per protein",
    )

    forward = parser.add_argument_group("forward")
    forward.add_argument(
        "--backbone_mode",
        type=str,
        default="auto",
        choices=["auto", "prosst", "plain_mlm"],
        help="backbone forward path (default: auto)",
    )
    forward.add_argument(
        "--max_residue_len",
        type=int,
        default=None,
        help="override max residue length (default: model limit)",
    )
    forward.add_argument(
        "--long_seq_mode",
        type=str,
        default="auto_window",
        choices=["auto_window", "error"],
        help="proteins longer than the model limit (default: auto_window)",
    )
    forward.add_argument(
        "--long_seq_overlap",
        type=int,
        default=256,
        help="sliding-window overlap for long sequences (default: 256)",
    )
    forward.add_argument(
        "--disable_native_scorer",
        action="store_true",
        help="disable native per-mutant scorers (e.g. Tranception)",
    )

    cache = parser.add_argument_group("logits cache")
    cache.add_argument(
        "--precomputed_logits_dir",
        type=str,
        default=None,
        help="directory of {protein}.pt logits; skips model forward when present",
    )
    cache.add_argument("--logits_cache_dir", type=str, default=None, help="persist per-protein logits for reuse")
    cache.add_argument("--reuse_logits_cache", action="store_true", help="load logits from --logits_cache_dir when present")
    cache.add_argument("--write_logits_cache", action="store_true", help="write computed logits to --logits_cache_dir")
    cache.add_argument(
        "--skip_mutant_scoring",
        action="store_true",
        help="after writing logits, skip per-mutant scoring",
    )
    cache.add_argument(
        "--cache_miss_policy",
        type=str,
        default="forward",
        choices=["forward", "error"],
        help="missing cache: recompute (default) or error",
    )
    cache.add_argument("--logits_cache_tag", type=str, default="", help="extra cache namespace to avoid cross-run reuse")
    cache.add_argument(
        "--logits_cache_stage",
        type=str,
        default="raw",
        choices=["raw", "final"],
        help="cache raw backbone logits (default) or fused logits",
    )
    cache.add_argument(
        "--aln_count_cache_dir",
        type=str,
        default=None,
        help="cache MSA count matrices (.pt) across scoring variants",
    )

    run = parser.add_argument_group("logging")
    run.add_argument("--disable_tqdm", action="store_true", help="disable progress bars")
    run.add_argument("--no_color", action="store_true", help="disable ANSI colors")
    run.add_argument(
        "--log_level",
        type=str,
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="console verbosity (default: info)",
    )
    run.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")

    extras = parser.add_argument_group("backbone extras")
    extras.add_argument("--foldseek_bin", type=str, default=None, help="Foldseek binary (else cache, then download)")
    extras.add_argument("--esm1v_seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="ESM-1v ensemble seeds (default: 1 2 3 4 5)")
    extras.add_argument("--protssn_model_dir", type=str, default=None, help="ProtSSN GNN weights directory")
    extras.add_argument("--protssn_norm_dir", type=str, default=None, help="ProtSSN norm-file directory (default: bundled)")
    extras.add_argument("--protssn_no_ensemble", action="store_true", help="use one ProtSSN model instead of the 9-model ensemble")
    extras.add_argument("--esm_if_chain", type=str, default="A", help="PDB chain for ESM-IF (default: A)")
    extras.add_argument("--protein_mpnn_checkpoint", type=str, default=None, help="ProteinMPNN .pt or directory")
    extras.add_argument("--protein_mpnn_chain", type=str, default="A", help="PDB chain for ProteinMPNN (default: A)")
    extras.add_argument(
        "--protein_mpnn_scoring_mode",
        type=str,
        default="teacher_force",
        choices=["teacher_force", "random_order"],
        help="ProteinMPNN scoring (default: teacher_force)",
    )
    extras.add_argument(
        "--protein_mpnn_random_orders",
        type=int,
        default=1,
        help="random decoding orders when --protein_mpnn_scoring_mode=random_order",
    )
    extras.add_argument(
        "--progen2_model_name_or_path",
        type=str,
        default="hugohrban/progen2-large",
        choices=["hugohrban/progen2-small", "hugohrban/progen2-medium", "hugohrban/progen2-base", "hugohrban/progen2-large", "hugohrban/progen2-xlarge"],
        help="ProGen2 HF id (default: hugohrban/progen2-large)",
    )
    extras.add_argument("--progen2_fp16", action="store_true", help="ProGen2 fp16")
    extras.add_argument(
        "--progen3_model_name_or_path",
        type=str,
        default="Profluent-Bio/progen3-1b",
        choices=["Profluent-Bio/progen3-112m", "Profluent-Bio/progen3-219m", "Profluent-Bio/progen3-339m", "Profluent-Bio/progen3-762m", "Profluent-Bio/progen3-1b", "Profluent-Bio/progen3-3b"],
        help="ProGen3 HF id (default: Profluent-Bio/progen3-1b)",
    )
    extras.add_argument("--progen3_fp16", action="store_true", help="ProGen3 fp16")
    extras.add_argument("--protgpt2_model_name_or_path", type=str, default="nferruz/ProtGPT2", help="ProtGPT2 HF id or path")
    extras.add_argument(
        "--rita_model_name_or_path",
        type=str,
        default="lightonai/RITA_xl",
        choices=["lightonai/RITA_s", "lightonai/RITA_m", "lightonai/RITA_l", "lightonai/RITA_xl"],
        help="RITA HF id (default: lightonai/RITA_xl)",
    )
    extras.add_argument(
        "--esm3_model_name",
        type=str,
        default="esmc_300m",
        choices=["esmc_300m", "esmc_600m", "esm3_sm_open_v1"],
        help="ESM3 / ESM-C name (default: esmc_300m)",
    )
    extras.add_argument(
        "--carp_model_name",
        type=str,
        default="carp_640M",
        choices=["carp_600k", "carp_38M", "carp_76M", "carp_640M"],
        help="CARP name (default: carp_640M)",
    )
    extras.add_argument(
        "--tranception_checkpoint",
        type=str,
        default="OATML-Markslab/Tranception_Large",
        choices=["OATML-Markslab/Tranception_Small", "OATML-Markslab/Tranception_Medium", "OATML-Markslab/Tranception_Large"],
        help="Tranception HF id or local path",
    )
    extras.add_argument("--tranception_no_mirror", action="store_true", help="Tranception L→R only (disable R→L)")
    extras.add_argument("--s2f_config", type=str, default=None, help="S2F / S3F YAML config (default: bundled)")
    extras.add_argument("--s2f_checkpoint", type=str, default=None, help="S2F / S3F checkpoint (.pt)")
    extras.add_argument(
        "--s3f_surface_dir",
        type=str,
        default=None,
        help="S3F surface .pkl directory (default: <base_dir>/s3f_surfaces_af2_assay_resolved_full)",
    )

    legacy = parser.add_argument_group("legacy")
    legacy.add_argument(
        "--model_name",
        type=str,
        default=["AI4Protein/ProSST-2048"],
        nargs="+",
        help="legacy HF id(s); prefer --model / --model_id",
    )
    legacy.add_argument(
        "--baseline_type",
        type=str,
        default="auto",
        choices=["auto", "esm2", "esm1b", "esm1v", "saprot", "protssn", "esm_if", "protein_mpnn", "progen2", "progen3", "protgpt2", "rita", "esm3", "tranception", "carp", "mifst", "mif_st", "s2f", "s3f"],
        help="legacy alias of --model",
    )

    from venusrem2 import __version__

    parser.add_argument("--version", action="version", version=f"rem2 {__version__}")
    parser.set_defaults(use_rsa_decay=True, use_plddt_decay=True, calibrate_on_raw=True)
    return parser


_PDB_DIR_CANDIDATES = (
    "pdbs_af2_assay_resolved_full",
    "pdbs_af2",
    "af2",
    "pdbs",
)
_MSA_DIR_CANDIDATES = (
    "aa_seq_aln_a2m_af2cf",
    "aa_seq_aln",
    "aa_seq_aln_a2m",
)
_STRUC_DIR_CANDIDATES = (
    "struc_seq_af2_assay_resolved_full",
    "struc_seq",
)


def _join_under_base(base: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.join(base, value)


def _first_existing_subdir(base: str, names: tuple[str, ...]) -> str:
    for name in names:
        path = os.path.join(base, name)
        if os.path.isdir(path):
            return path
    return os.path.join(base, names[-1])


def postprocess_args(args):
    args.show_progress = not args.disable_tqdm
    if args.logits_cache_dir:
        os.makedirs(args.logits_cache_dir, exist_ok=True)

    from venusrem2.scoring.entropy_alpha import parse_alpha_arg, parse_background_weight_arg

    try:
        parse_alpha_arg(args.alpha)
    except (TypeError, ValueError):
        raise SystemExit("--alpha must be 'entropy' or a float (e.g. 0, 0.8)")
    try:
        parse_background_weight_arg(args.background_weight)
    except (TypeError, ValueError):
        raise SystemExit("--background_weight must be 'one_minus_alpha' or a float")

    fasta = getattr(args, "fasta", None)
    if fasta:
        if args.base_dir or args.aa_seq_dir or args.mutant_dir:
            raise SystemExit(
                "--fasta is mutually exclusive with --base_dir / --aa_seq_dir / --mutant_dir"
            )
        if getattr(args, "mutants", None) and getattr(args, "mutant_sites", None):
            raise SystemExit("Use either --mutants (existing CSV) or --mutant_sites (auto-generate), not both")
        if not getattr(args, "mutants", None) and not getattr(args, "mutant_sites", None):
            args.mutant_sites = "1"
    elif getattr(args, "mutant_sites", None) or getattr(args, "mutants", None) or getattr(args, "pdb", None):
        if not args.base_dir and not args.aa_seq_dir:
            raise SystemExit("--mutant_sites / --mutants / --pdb require --fasta or --base_dir / --aa_seq_dir")

    if args.base_dir:
        if args.aa_seq_dir is None:
            args.aa_seq_dir = os.path.join(args.base_dir, "aa_seq")
        else:
            args.aa_seq_dir = _join_under_base(args.base_dir, args.aa_seq_dir)

        if args.mutant_dir is None:
            args.mutant_dir = os.path.join(args.base_dir, "substitutions")
        else:
            args.mutant_dir = _join_under_base(args.base_dir, args.mutant_dir)

        if args.struc_seq_dir is None:
            args.struc_seq_dir = _first_existing_subdir(args.base_dir, _STRUC_DIR_CANDIDATES)
        else:
            args.struc_seq_dir = _join_under_base(args.base_dir, args.struc_seq_dir)

        if args.pdb_dir is None:
            args.pdb_dir = _first_existing_subdir(args.base_dir, _PDB_DIR_CANDIDATES)
        else:
            args.pdb_dir = _join_under_base(args.base_dir, args.pdb_dir)

        if args.aa_seq_aln_dir is None:
            args.aa_seq_aln_dir = _first_existing_subdir(args.base_dir, _MSA_DIR_CANDIDATES)
        else:
            args.aa_seq_aln_dir = _join_under_base(args.base_dir, args.aa_seq_aln_dir)

        if args.struc_seq_aln_dir is None:
            args.struc_seq_aln_dir = os.path.join(args.base_dir, "struc_seq_aln_foldseek")
        else:
            args.struc_seq_aln_dir = _join_under_base(args.base_dir, args.struc_seq_aln_dir)

    return args
