"""CLI entrypoint for rem2 scoring (VenusREM2 = ProSST ensemble only)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

from rem2.backbone import (
    BaselineState,
    resolve_structure_fasta_path,
)
from rem2.config import create_parser, postprocess_args
from rem2.models import (
    apply_model_defaults,
    get_model,
    list_models,
    resolve_model_name,
)
from rem2.models.download_policy import (
    DownloadRefused,
    apply_download_policy_from_args,
    get_download_policy,
)
from rem2.scoring.mutant_parse import MutantParseError
from rem2.user_commands import (
    GETTING_STARTED,
    build_demo_argv,
    model_size_hint,
    run_doctor,
)
from rem2.data.inputs import expected_layout_text, require_run_inputs
from rem2.models.scoring_strategy import (
    MASKED_MARGINALS,
    TEACHER_FORCE,
    UnsupportedScoringStrategy,
    forward_modes_label,
    require_scoring_strategy,
    require_tokenizer_mask,
    strategy_short_name,
)
from rem2.naming import (
    OFFICIAL_SYSTEM_NAME,
    is_official_venusrem2,
    is_prosst_key,
    run_banner,
)


def _zmean_columns(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    mats = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        scale = float(np.nanstd(values))
        if scale > 0:
            mats.append((values - np.nanmean(values)) / scale)
        else:
            mats.append(np.zeros_like(values, dtype=float))
    return np.nanmean(np.vstack(mats), axis=0)
from rem2.scoring import (
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


def _numeric_series(series: pd.Series) -> pd.Series:
    # Strip all whitespace incl. NBSP (U+00A0). pandas StringDtype
    # ``str.replace(r"\s+")`` does not strip NBSP — use Python re.
    import re

    cleaned = series.map(
        lambda x: re.sub(r"\s+", "", str(x)) if pd.notna(x) else x
    )
    return pd.to_numeric(cleaned, errors="coerce")


def finite_spearman(df, target_col, score_col, logger, protein_name, label):
    # Clean then correlate — matches VMH official-metrics hygiene.
    try:
        x = _numeric_series(df[target_col])
        y = _numeric_series(df[score_col])
        ok = x.notna() & y.notna()
        if int(ok.sum()) < 2:
            raise ValueError(f"need >=2 finite pairs, got {int(ok.sum())}")
        corr = spearmanr(x[ok], y[ok]).correlation
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
    print("rem2 backbones. FWD = allowed --scoring_strategy (wt / mask / tf).")
    print("venusrem2 = official ProSST ensemble (VenusREM2), wt only.")
    print("Aliases work as --model: saprot, esmif, protssn-ensemble,")
    print("  prosst-{k}, esm2-{size}m, proteinmpnn-{xx} (e.g. proteinmpnn-020).")
    print("CSV *_mask / *_wt is --scoring_strategy, not a second --model.")
    print()
    print(f"{'MODEL':<{name_w}}  PDB  FWD       AUTO  DEFAULT_ID / NOTES")
    print("-" * (name_w + 70))
    for s in specs:
        pdb = "yes" if s.needs_pdb else "no"
        fwd = forward_modes_label(s)
        auto = "yes" if s.auto_download else "no"
        extra = s.default_model_id or ""
        if s.notes:
            extra = f"{extra}  ({s.notes})" if extra else s.notes
        if s.extras:
            extra = f"{extra}  [extras:{s.extras}]"
        print(f"{s.name:<{name_w}}  {pdb:<3}  {fwd:<9}  {auto:<4}  {extra}")


def _prepare_single_protein(args, logger):
    """Materialize --fasta/--pdb/--mutants|--mutant_sites into a mini dataset layout."""
    from rem2.data.mutagenesis import materialize_single_protein_inputs

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


def _log_available_inputs(args, logger):
    msa_dir = getattr(args, "aa_seq_aln_dir", None)
    pdb_dir = getattr(args, "pdb_dir", None)
    pdb_file = getattr(args, "pdb", None)
    has_msa = bool(msa_dir and os.path.isdir(msa_dir) and os.listdir(msa_dir))
    has_pdb = bool(
        (pdb_file and os.path.exists(pdb_file))
        or (pdb_dir and os.path.isdir(pdb_dir) and any(n.endswith(".pdb") for n in os.listdir(pdb_dir)))
    )
    if str(getattr(args, "alpha", "entropy")).strip().lower() in {"entropy", "auto", "adaptive"} and not has_msa:
        logger.warn("No MSA files found; entropy-α will use α=0")
    if (getattr(args, "use_rsa_decay", True) or getattr(args, "use_plddt_decay", True)) and not has_pdb:
        logger.warn("No PDB files found; RSA / pLDDT decay will be skipped")


def _write_run_meta(args, model_key, protein_names):
    import json
    from rem2 import __version__

    payload = {
        "rem2_version": __version__,
        "model": model_key,
        "model_ids": list(getattr(args, "model_name", None) or []),
        "alpha": getattr(args, "alpha", None),
        "scoring_mode": getattr(args, "scoring_mode", None),
        "scoring_strategy": getattr(args, "scoring_strategy", None),
        "out_scores_dir": args.out_scores_dir,
        "n_proteins": len(protein_names),
        "proteins": list(protein_names),
    }
    path = os.path.join(args.out_scores_dir, "run_meta.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(GETTING_STARTED, end="")
        return
    if argv and argv[0] == "demo":
        argv = build_demo_argv(argv[1:])
    elif argv and argv[0] == "doctor":
        code = run_doctor(argv[1:])
        if code:
            raise SystemExit(code)
        return

    raw_args = create_parser().parse_args(argv)
    if getattr(raw_args, "list_models", False):
        _print_model_table()
        return

    args = postprocess_args(raw_args)
    set_deterministic_inference(args.seed)
    logger = CliLogger(level=args.log_level, use_color=should_use_color(args.no_color))

    apply_download_policy_from_args(args)
    model_key = resolve_model_name(args)
    try:
        apply_model_defaults(args, model_key, logger=logger)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    except DownloadRefused as exc:
        raise SystemExit(str(exc)) from exc
    except SystemExit:
        raise
    adapter_cls = get_model(model_key)
    try:
        require_scoring_strategy(model_key, getattr(args, "scoring_strategy", None))
    except UnsupportedScoringStrategy as exc:
        raise SystemExit(str(exc)) from exc

    if getattr(args, "fasta", None):
        os.makedirs(args.out_scores_dir, exist_ok=True)
        _prepare_single_protein(args, logger)

    if not args.aa_seq_dir:
        raise SystemExit(
            "No data: provide --base_dir / --aa_seq_dir, or --fasta for single-protein mode.\n"
            + expected_layout_text(model_key)
        )
    if not args.mutant_dir:
        raise SystemExit(
            "No data: provide --mutant_dir (via --base_dir) or use --fasta.\n"
            + expected_layout_text(model_key)
        )

    protein_names = sorted(read_names(args.aa_seq_dir))
    if args.protein_list:
        allowed = set(args.protein_list.split(","))
        protein_names = [p for p in protein_names if p in allowed]
    if args.max_proteins is not None and args.max_proteins > 0:
        protein_names = protein_names[: args.max_proteins]
    if not protein_names and args.protein_list:
        raise SystemExit(
            f"No data: --protein_list matched no FASTA files in {args.aa_seq_dir}."
        )
    require_run_inputs(args, model_key, protein_names)
    if not protein_names:
        raise SystemExit(
            "No data: no proteins to score. Need FASTA files in aa_seq/.\n"
            + expected_layout_text(model_key)
        )

    os.makedirs(args.out_scores_dir, exist_ok=True)
    os.makedirs(f"{args.out_scores_dir}/scores", exist_ok=True)

    logger.section(run_banner(model_key, args))
    logger.info(f"Backbone model: {model_key} (baseline_type={args.baseline_type})")
    hint = model_size_hint(model_key)
    if hint:
        logger.info(hint)
    logger.info(
        f"Forward strategy: {strategy_short_name(args.scoring_strategy)} "
        f"({args.scoring_strategy})"
    )
    logger.info(f"Weight cache: {args.cache_dir}")
    policy = get_download_policy()
    if policy == "yes":
        logger.info("Missing checkpoints: auto-download")
    elif policy == "no":
        logger.info("Missing checkpoints: download disabled (cache / explicit path only)")
    else:
        logger.info("Missing checkpoints: download if missing (TTY: confirm [Y/n])")
    if is_official_venusrem2(model_key, args):
        logger.info(
            f"Official {OFFICIAL_SYSTEM_NAME}: rem2 on a ProSST ensemble "
            f"({len(args.model_name)} checkpoints)"
        )
    elif is_prosst_key(model_key) and not is_official_venusrem2(model_key, args):
        logger.warn(
            f"{OFFICIAL_SYSTEM_NAME} is rem2 on a ProSST ensemble only. "
            "This single-ProSST run is rem2 — pass 2+ --model_name AI4Protein/ProSST-* "
            f"to label the run {OFFICIAL_SYSTEM_NAME}."
        )
    out_names = getattr(args, "model_out_name", None) or []
    if any(n == OFFICIAL_SYSTEM_NAME for n in out_names) and not is_official_venusrem2(
        model_key, args
    ):
        logger.warn(
            f"--model_out_name {OFFICIAL_SYSTEM_NAME} on a non-ensemble run. "
            "Prefer {backbone}__rem2 unless this is a ProSST ensemble."
        )
    rsa = "RSA" if getattr(args, "use_rsa_decay", True) else "no RSA"
    plddt = "pLDDT" if getattr(args, "use_plddt_decay", True) else "no pLDDT"
    logger.info(
        f"rem2 recipe: α={args.alpha}, β={args.background_weight}, "
        f"{args.scoring_mode}, {rsa}, {plddt}"
        + (", CCD on raw logits" if getattr(args, "calibrate_on_raw", True) else "")
    )
    logger.info("Scoring proteins")
    if args.base_dir:
        logger.info(
            f"Data dirs: seq={args.aa_seq_dir}  msa={args.aa_seq_aln_dir}  "
            f"pdb={args.pdb_dir}  struc={args.struc_seq_dir}"
        )
    logger.info(f"Total proteins: {len(protein_names)}")
    _log_available_inputs(args, logger)
    logger.debug(f"Protein preview: {format_name_preview(protein_names)}")

    official = is_official_venusrem2(model_key, args)
    if official:
        args.structure_vocab_subdir = None

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
            try:
                adapter = adapter_cls.load(
                    model_id=getattr(args, "model_id", None) or model_name,
                    device=device,
                    cache_dir=args.cache_dir,
                    args=args,
                    logger=logger,
                )
            except DownloadRefused as exc:
                raise SystemExit(str(exc)) from exc
            except ImportError as exc:
                extra = getattr(adapter_cls.spec, "extras", "") or ""
                hint = f' Install with: pip install "rem2[{extra}]"' if extra else ""
                raise SystemExit(f"Missing dependency for --model {model_key}: {exc}.{hint}") from exc
            state = adapter.state
        model = state.model
        tokenizer = state.tokenizer
        model_max_residue_len = state.model_max_residue_len
        if getattr(args, "scoring_strategy", None) == MASKED_MARGINALS:
            try:
                require_tokenizer_mask(model_key, tokenizer)
            except UnsupportedScoringStrategy as exc:
                raise SystemExit(str(exc)) from exc

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
                    cand = f"{args.struc_seq_aln_dir}/{protein_name}.fasta"
                    if os.path.exists(cand):
                        struc_seq_aln_file = cand

            if not os.path.exists(residue_fasta):
                raise SystemExit(f"Missing FASTA: {residue_fasta}")
            if os.path.exists(f"{args.out_scores_dir}/scores/{protein_name}.csv"):
                mutant_file = f"{args.out_scores_dir}/scores/{protein_name}.csv"
            if not os.path.exists(mutant_file):
                raise SystemExit(f"Missing mutants CSV: {mutant_file}")
            mutant_df = pd.read_csv(mutant_file)
            if "mutant" not in mutant_df.columns:
                raise SystemExit(f"{mutant_file} needs a 'mutant' column (e.g. A42G)")

            if args.model_out_name:
                model_out_name = args.model_out_name[model_idx]
            else:
                model_out_name = model_name.split("/")[-1]

            backbone_name = model_name.split("/")[-1]
            raw_col = f"{backbone_name}__raw_backbone"
            venusrem2_col = model_out_name
            raw_logits_cache_path = build_logits_cache_path(
                args=args, protein_name=protein_name,
                model_name=model_name, variant_label="raw_backbone",
            )
            venusrem2_logits_cache_path = build_logits_cache_path(
                args=args, protein_name=protein_name,
                model_name=model_name, variant_label="venusrem2_main",
            )

            if adapter is not None:
                baseline_fwd_fn = adapter.create_forward_fn(
                    protein_name, pdb_file, structure_fasta, idx, logger
                )
                native_scorer = (
                    None
                    if getattr(args, "disable_native_scorer", False)
                    else adapter.create_native_scorer_fn(protein_name, logger)
                )
            else:
                from rem2.backbone.baseline_dispatch import (
                    create_baseline_forward_fn,
                    create_native_scorer_fn,
                )
                baseline_fwd_fn = create_baseline_forward_fn(
                    state, args, protein_name, pdb_file, structure_fasta, idx, device, logger
                )
                native_scorer = (
                    None
                    if getattr(args, "disable_native_scorer", False)
                    else create_native_scorer_fn(
                        state, args, protein_name, device, logger
                    )
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
                skip_mutant_scoring=getattr(args, "skip_mutant_scoring", False),
            )

            if args.print_compare_spearman:
                if raw_col not in mutant_df.columns:
                    raw_kwargs = dict(score_kwargs)
                    raw_kwargs.update(
                        alpha=0.0,
                        scoring_mode="log_odds",
                        background_weight=0.0,
                        use_rsa_decay=False,
                        use_plddt_decay=False,
                        calibrate_on_raw=False,
                        aa_seq_aln_file=None,
                        struc_seq_aln_file=None,
                        quiet=True,
                        logits_cache_path=raw_logits_cache_path,
                    )
                    raw_scores = score_protein(**raw_kwargs)
                    mutant_df[raw_col] = raw_scores
                raw_corr = finite_spearman(
                    mutant_df, "DMS_score", raw_col, logger, protein_name, raw_col
                )

            need_forward = venusrem2_col not in mutant_df.columns or (
                getattr(args, "write_logits_cache", False)
                and venusrem2_logits_cache_path
                and not os.path.exists(venusrem2_logits_cache_path)
            )
            if need_forward:
                try:
                    scores = score_protein(
                        **score_kwargs,
                        alpha=args.alpha,
                        aa_seq_aln_file=aa_seq_aln_file,
                        struc_seq_aln_file=struc_seq_aln_file,
                        quiet=False,
                        logits_cache_path=venusrem2_logits_cache_path,
                    )
                except (MutantParseError, FileNotFoundError, ValueError) as exc:
                    raise SystemExit(f"{protein_name}: {exc}") from exc
                mutant_df[venusrem2_col] = scores

            corr = finite_spearman(
                mutant_df, "DMS_score", venusrem2_col, logger, protein_name, venusrem2_col
            )
            corrs.append(corr)
            if args.print_compare_spearman:
                if not compare_table_printed:
                    logger.info("Compare Spearman table (Current - Raw as Delta)")
                    print_compare_table_header(
                        logger, include_venus=False,
                        raw_label=backbone_name, current_label=venusrem2_col,
                    )
                    compare_table_printed = True
                print_compare_table_row(
                    logger=logger, protein_name=protein_name,
                    raw_corr=raw_corr, venusrem2_corr=corr, venus_corr=None,
                )
            else:
                logger.success(f"{model_out_name} Spearman={corr:.4f}", protein=protein_name)
            mutant_df.to_csv(f"{args.out_scores_dir}/scores/{protein_name}.csv", index=False)

        if compare_table_printed:
            logger.info("+" + "-" * 42 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 10 + "+")
        mean_corr = pd.Series(corrs, dtype="float64").mean(skipna=True)
        logger.section(f"{model_out_name} average Spearman: {mean_corr:.4f}")
        summary_df_path = f"{args.out_scores_dir}/summary_performance.csv"
        new_rows = pd.DataFrame({"protein": protein_names, model_out_name: corrs})
        if os.path.exists(summary_df_path):
            summary_df = pd.read_csv(summary_df_path).set_index("protein")
            incoming = new_rows.set_index("protein")
            for name, row in incoming.iterrows():
                summary_df.loc[name, model_out_name] = row[model_out_name]
            summary_df = summary_df.reset_index()
        else:
            summary_df = new_rows
        summary_df.to_csv(summary_df_path, index=False)

    if official:
        ens_col = OFFICIAL_SYSTEM_NAME
        ens_corrs = []
        for protein_name in protein_names:
            path = f"{args.out_scores_dir}/scores/{protein_name}.csv"
            if not os.path.exists(path):
                ens_corrs.append(float("nan"))
                continue
            frame = pd.read_csv(path)
            member_cols = [c for c in frame.columns if c.startswith(f"{OFFICIAL_SYSTEM_NAME}__")]
            if len(member_cols) < 2:
                ens_corrs.append(float("nan"))
                continue
            frame[ens_col] = _zmean_columns(frame, member_cols)
            frame.to_csv(path, index=False)
            ens_corrs.append(
                finite_spearman(frame, "DMS_score", ens_col, logger, protein_name, ens_col)
            )
        mean_ens = pd.Series(ens_corrs, dtype="float64").mean(skipna=True)
        logger.section(f"{ens_col} z-mean ensemble Spearman: {mean_ens:.4f}")
        summary_df_path = f"{args.out_scores_dir}/summary_performance.csv"
        new_rows = pd.DataFrame({"protein": protein_names, ens_col: ens_corrs})
        if os.path.exists(summary_df_path):
            summary_df = pd.read_csv(summary_df_path).set_index("protein")
            incoming = new_rows.set_index("protein")
            for name, row in incoming.iterrows():
                summary_df.loc[name, ens_col] = row[ens_col]
            summary_df = summary_df.reset_index()
        else:
            summary_df = new_rows
        summary_df.to_csv(summary_df_path, index=False)

    _write_run_meta(args, model_key, protein_names)
    logger.info(f"Wrote {args.out_scores_dir}/run_meta.json")


if __name__ == "__main__":
    main()
