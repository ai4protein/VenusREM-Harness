"""CLI entrypoint for VenusREM / Orbit scoring."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

from venus_orbit.backbone import (
    BaselineState,
    resolve_structure_fasta_path,
)
from venus_orbit.config import create_parser, postprocess_args
from venus_orbit.models import apply_model_defaults, get_model, list_models, resolve_model_name
from venus_orbit.scoring import (
    CliLogger,
    build_logits_cache_path,
    format_name_preview,
    print_compare_table_header,
    print_compare_table_row,
    read_names,
    score_protein,
    set_deterministic_inference,
    should_use_color,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_ESM2_TOKENIZER = "facebook/esm2_t33_650M_UR50D"


def finite_spearman(df, target_col, score_col, logger, protein_name, label):
    # Legacy path used by VenusMutHub AF2 Orbit baselines (2026-06): call scipy
    # spearmanr on the raw columns. Do NOT pre-filter with pd.to_numeric(...,
    # errors="coerce") — that silently drops valid DMS values containing
    # unicode spaces such as NBSP ("100\xa0"), thin/narrow spaces, etc., while
    # Python/scipy can still coerce those strings to floats.
    try:
        corr = spearmanr(df[target_col], df[score_col]).correlation
    except Exception as exc:
        logger.warn(
            f"{label}: Spearman failed ({type(exc).__name__}: {exc})",
            protein=protein_name,
        )
        return float("nan")
    if corr is None or not np.isfinite(corr):
        logger.warn(
            f"{label}: Spearman undefined",
            protein=protein_name,
        )
        return float("nan")
    return float(corr)


def _print_model_table():
    specs = list_models()
    name_w = max(len(s.name) for s in specs)
    print(f"{'MODEL':<{name_w}}  PDB  AUTO  DEFAULT_ID / NOTES")
    print("-" * (name_w + 60))
    for s in specs:
        pdb = "yes" if s.needs_pdb else "no"
        auto = "yes" if s.auto_download else "no"
        extra = s.default_model_id or ""
        if s.notes:
            extra = f"{extra}  ({s.notes})" if extra else s.notes
        if s.extras:
            extra = f"{extra}  [extras:{s.extras}]"
        print(f"{s.name:<{name_w}}  {pdb:<3}  {auto:<4}  {extra}")


def _prepare_single_protein(args, logger):
    """Materialize --fasta/--pdb/--mutants|--mutant_sites into a mini dataset layout."""
    from venus_orbit.data.mutagenesis import materialize_single_protein_inputs

    if not args.out_scores_dir:
        raise SystemExit("--out_scores_dir is required with --fasta")
    inputs_root = os.path.join(args.out_scores_dir, "_inputs")
    try:
        meta = materialize_single_protein_inputs(
            fasta_path=args.fasta,
            out_root=inputs_root,
            pdb_path=getattr(args, "pdb", None),
            mutants_path=getattr(args, "mutants", None),
            mutant_sites=getattr(args, "mutant_sites", None),
            positions=getattr(args, "positions", None),
            residue_range=getattr(args, "residue_range", None),
            max_mutants=getattr(args, "max_mutants", 1_000_000),
        )
    except Exception as exc:
        raise SystemExit(f"Single-protein setup failed: {exc}") from exc

    args.aa_seq_dir = meta["aa_seq_dir"]
    args.mutant_dir = meta["mutant_dir"]
    if meta["pdb_dir"]:
        args.pdb_dir = meta["pdb_dir"]
    args.protein_list = meta["name"]
    logger.info(
        f"Single-protein mode: {meta['name']} (L={len(meta['sequence'])}, "
        f"mutants={meta['n_mutants']})"
    )
    logger.info(f"Materialized inputs under {inputs_root}")
    if getattr(args, "mutant_sites", None) and not getattr(args, "mutants", None):
        logger.info(
            f"Generated saturation library (--mutant_sites {args.mutant_sites}) "
            f"-> {meta['mutant_csv']}"
        )
    return meta


def main(argv=None):
    raw_args = create_parser().parse_args(argv)
    if getattr(raw_args, "list_models", False):
        _print_model_table()
        return

    args = postprocess_args(raw_args)
    set_deterministic_inference(args.seed)
    logger = CliLogger(level=args.log_level, use_color=should_use_color(args.no_color))

    model_key = resolve_model_name(args)
    apply_model_defaults(args, model_key)
    adapter_cls = get_model(model_key)

    logger.section("Venus-Orbit scoring run")
    logger.info(f"Backbone model: {model_key} (baseline_type={args.baseline_type})")
    logger.info(f"Weight cache: {args.cache_dir}")
    logger.info("Scoring proteins")
    if not args.out_scores_dir:
        raise SystemExit("--out_scores_dir is required (or use --list-models)")
    os.makedirs(args.out_scores_dir, exist_ok=True)
    os.makedirs(f"{args.out_scores_dir}/scores", exist_ok=True)

    if getattr(args, "fasta", None):
        _prepare_single_protein(args, logger)

    if not args.aa_seq_dir:
        raise SystemExit("Provide --base_dir / --aa_seq_dir, or --fasta for single-protein mode")
    if not args.mutant_dir:
        raise SystemExit("Provide --mutant_dir (via --base_dir) or use --fasta with --mutants/--mutant_sites")

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
        if args.model_out_name:
            display_name = args.model_out_name[model_idx]
        else:
            display_name = model_name.split("/")[-1]
        baseline_type = getattr(args, "baseline_type", "auto")
        logger.section(f"Run {model_idx+1}/{len(args.model_name)}: {display_name}")
        if baseline_type == "auto":
            logger.info(f"Loading model: {display_name}")
        else:
            logger.info(f"Loading baseline ({baseline_type}): {display_name}")
        logger.debug(f"Underlying model id: {model_name}")

        precomputed_only = (
            args.precomputed_logits_dir is not None
            and args.backbone_mode == "plain_mlm"
            and baseline_type == "auto"
        )
        if precomputed_only:
            tokenizer_path = os.environ.get("ESM2_TOKENIZER_PATH", DEFAULT_ESM2_TOKENIZER)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            state = BaselineState(
                model=None,
                tokenizer=tokenizer,
                model_max_residue_len=args.max_residue_len,
                baseline_type=baseline_type,
                extra={"precomputed_only": True, "tokenizer_path": tokenizer_path},
            )
            adapter = None
            logger.info(f"Precomputed logits mode: tokenizer only ({tokenizer_path})")
        else:
            adapter = adapter_cls.load(
                model_id=getattr(args, "model_id", None) or model_name,
                device=device,
                cache_dir=args.cache_dir,
                args=args,
                logger=logger,
            )
            state = adapter.state
        model = state.model
        tokenizer = state.tokenizer
        model_max_residue_len = state.model_max_residue_len

        protein_progress = tqdm(
            protein_names,
            desc=f"Proteins[{display_name}]",
            disable=not args.show_progress,
            leave=True,
            dynamic_ncols=True,
        )
        for idx, protein_name in enumerate(protein_progress):
            logger.info(f"Scoring protein {idx+1}/{len(protein_names)}", protein=protein_name)

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
                if "struc_seq_aln" in args.logit_mode:
                    struc_seq_aln_file = f"{args.struc_seq_aln_dir}/{protein_name}.fasta"

            if os.path.exists(f"{args.out_scores_dir}/scores/{protein_name}.csv"):
                mutant_file = f"{args.out_scores_dir}/scores/{protein_name}.csv"
            mutant_df = pd.read_csv(mutant_file)

            if args.model_out_name:
                model_out_name = args.model_out_name[model_idx]
            else:
                model_out_name = model_name.split("/")[-1]

            backbone_name = model_name.split("/")[-1]
            raw_col = f"{backbone_name}__raw_backbone"
            orbit_col = model_out_name
            raw_logits_cache_path = build_logits_cache_path(
                args=args, protein_name=protein_name,
                model_name=model_name, variant_label="raw_backbone",
            )
            orbit_logits_cache_path = build_logits_cache_path(
                args=args, protein_name=protein_name,
                model_name=model_name, variant_label="orbit_main",
            )

            if adapter is not None:
                baseline_fwd_fn = adapter.create_forward_fn(
                    protein_name, pdb_file, structure_fasta, idx, logger
                )
                native_scorer = adapter.create_native_scorer_fn(protein_name, logger)
            else:
                from venus_orbit.backbone.baseline_dispatch import (
                    create_baseline_forward_fn,
                    create_native_scorer_fn,
                )
                baseline_fwd_fn = create_baseline_forward_fn(
                    state, args, protein_name, pdb_file, structure_fasta, idx, device, logger
                )
                native_scorer = create_native_scorer_fn(
                    state, args, protein_name, device, logger
                )

            score_kwargs = dict(
                model=model,
                tokenizer=tokenizer,
                residue_fasta=residue_fasta,
                structure_fasta=structure_fasta,
                mutant_df=mutant_df,
                sample_size=args.sample_size,
                sample_ratio=args.sample_ratio,
                sample_times=args.sample_times,
                protein_name=protein_name,
                backbone_mode=args.backbone_mode,
                pdb_file=pdb_file,
                max_residue_len=model_max_residue_len,
                long_seq_mode=args.long_seq_mode,
                long_seq_overlap=args.long_seq_overlap,
                show_progress=args.show_progress,
                logger=logger,
                scoring_mode=args.scoring_mode,
                background_weight=args.background_weight,
                wt_confidence_weight=args.wt_confidence_weight,
                reuse_logits_cache=args.reuse_logits_cache,
                write_logits_cache=args.write_logits_cache,
                cache_miss_policy=args.cache_miss_policy,
                logits_cache_stage=args.logits_cache_stage,
                calibrate_on_raw=args.calibrate_on_raw,
                precomputed_logits_dir=args.precomputed_logits_dir,
                use_rsa_decay=args.use_rsa_decay,
                rsa_decay_mode=args.rsa_decay_mode,
                pdb_dir=args.pdb_dir,
                use_plddt_decay=args.use_plddt_decay,
                plddt_decay_mode=args.plddt_decay_mode,
                task_type=args.task_type,
                disable_adaptive_ccd=getattr(args, "disable_adaptive_ccd", False),
                baseline_forward_fn=baseline_fwd_fn,
                aln_count_cache_dir=args.aln_count_cache_dir,
                native_scorer_fn=native_scorer,
            )

            if args.print_compare_spearman:
                if raw_col not in mutant_df.columns:
                    raw_scores = score_protein(
                        **score_kwargs,
                        alpha=0.0,
                        aa_seq_aln_file=None,
                        struc_seq_aln_file=None,
                        quiet=True,
                        logits_cache_path=raw_logits_cache_path,
                    )
                    mutant_df[raw_col] = raw_scores
                raw_corr = finite_spearman(
                    mutant_df, "DMS_score", raw_col, logger, protein_name, raw_col
                )

            if orbit_col not in mutant_df.columns:
                scores = score_protein(
                    **score_kwargs,
                    alpha=args.alpha,
                    aa_seq_aln_file=aa_seq_aln_file,
                    struc_seq_aln_file=struc_seq_aln_file,
                    quiet=False,
                    logits_cache_path=orbit_logits_cache_path,
                )
                mutant_df[orbit_col] = scores

            corr = finite_spearman(
                mutant_df, "DMS_score", orbit_col, logger, protein_name, orbit_col
            )
            corrs.append(corr)
            if args.print_compare_spearman:
                if not compare_table_printed:
                    logger.info("Compare Spearman table (Current - Raw as Delta)")
                    print_compare_table_header(
                        logger, include_venus=False,
                        raw_label=backbone_name, current_label=orbit_col,
                    )
                    compare_table_printed = True
                print_compare_table_row(
                    logger=logger, protein_name=protein_name,
                    raw_corr=raw_corr, orbit_corr=corr, venus_corr=None,
                )
            else:
                logger.success(f"{model_out_name} Spearman={corr:.4f}", protein=protein_name)
            mutant_df.to_csv(f"{args.out_scores_dir}/scores/{protein_name}.csv", index=False)

        if compare_table_printed:
            logger.info("+" + "-" * 42 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 10 + "+")
        mean_corr = pd.Series(corrs, dtype="float64").mean(skipna=True)
        logger.section(f"{model_out_name} average Spearman: {mean_corr:.4f}")
        summary_df_path = f"{args.out_scores_dir}/summary_performance.csv"
        if os.path.exists(summary_df_path):
            summary_df = pd.read_csv(summary_df_path)
            summary_df[model_out_name] = corrs
        else:
            summary_df = pd.DataFrame({"protein": protein_names, model_out_name: corrs})
        summary_df.to_csv(f"{args.out_scores_dir}/summary_performance.csv", index=False)


if __name__ == "__main__":
    main()
