"""CLI entrypoint for vrh scoring (VenusREM2 = ProSST ensemble only)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

from vrh.backbone import (
    BaselineState,
    resolve_structure_fasta_path,
)
from vrh.config import create_parser, postprocess_args
from vrh.models import (
    apply_model_defaults,
    get_model,
    list_models,
    resolve_model_name,
)
from vrh.models.download_policy import (
    DownloadRefused,
    apply_download_policy_from_args,
    get_download_policy,
)
from vrh.scoring.mutant_parse import MutantParseError
from vrh.user_commands import (
    GETTING_STARTED,
    build_demo_argv,
    model_size_hint,
    run_doctor,
)
from vrh.data.inputs import expected_layout_text, require_run_inputs
from vrh.models.scoring_strategy import (
    MASKED_MARGINALS,
    TEACHER_FORCE,
    UnsupportedScoringStrategy,
    forward_modes_label,
    require_scoring_strategy,
    require_tokenizer_mask,
    strategy_short_name,
)
from vrh.naming import (
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
from vrh.scoring import (
    CliLogger,
    build_logits_cache_path,
    format_name_preview,
    has_experimental_dms,
    print_compare_table_header,
    print_compare_table_row,
    print_score_preview,
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
    from vrh.user_commands import print_model_table

    print_model_table()


def _is_single_protein(args) -> bool:
    return bool(
        getattr(args, "fasta", None)
        or (getattr(args, "pdb", None) and not getattr(args, "base_dir", None))
    )


def _prepare_single_protein(args, logger, model_key=None):
    """Materialize --fasta/--pdb/--mutants|--mutant_sites into a mini dataset layout."""
    from vrh.data.mutagenesis import materialize_single_protein_inputs
    from vrh.naming import is_prosst_key

    inputs_root = os.path.join(args.out_scores_dir, "_inputs")
    try:
        meta = materialize_single_protein_inputs(
            fasta_path=getattr(args, "fasta", None),
            out_root=inputs_root,
            pdb_path=getattr(args, "pdb", None),
            mutants_path=getattr(args, "mutants", None),
            mutant_sites=getattr(args, "mutant_sites", None),
            positions=getattr(args, "positions", None),
            residue_range=getattr(args, "residue_range", None),
            max_mutants=getattr(args, "max_mutants", 1_000_000),
            pdb_chain=getattr(args, "pdb_chain", None),
        )
    except Exception as exc:
        raise SystemExit(f"Single-protein setup failed: {exc}") from exc

    args.aa_seq_dir = meta["aa_seq_dir"]
    args.mutant_dir = meta["mutant_dir"]
    if meta["pdb_dir"]:
        args.pdb_dir = meta["pdb_dir"]
    args.protein_list = meta["name"]
    src = "PDB" if not getattr(args, "fasta", None) else "FASTA"
    extra = ""
    if meta.get("pdb_chain"):
        extra = f", chain={meta['pdb_chain']}"
    logger.info(
        f"Single-protein mode ({src}): {meta['name']} (L={len(meta['sequence'])}, "
        f"mutants={meta['n_mutants']}{extra})"
    )
    if (
        getattr(args, "fasta", None)
        and meta.get("pdb_sequence")
        and meta["pdb_sequence"] != meta["sequence"]
    ):
        logger.warn(
            f"FASTA and PDB sequences differ (L={len(meta['sequence'])} vs "
            f"{len(meta['pdb_sequence'])}); using FASTA for mutants"
        )
    logger.info(f"Materialized inputs under {inputs_root}")
    if getattr(args, "mutant_sites", None) and not getattr(args, "mutants", None):
        logger.info(
            f"Generated saturation library (--mutant_sites {args.mutant_sites}) "
            f"-> {meta['mutant_csv']}"
        )
    if is_prosst_key(model_key) and meta.get("pdb_dir"):
        from vrh.baseline.prosst.structure_tokens import (
            generate_struc_seq_from_pdb,
            needed_structure_vocab_sizes,
            resolve_structure_fasta_path,
        )

        vocabs = needed_structure_vocab_sizes(model_key, args)
        already = None
        if getattr(args, "struc_seq_dir", None):
            already = resolve_structure_fasta_path(
                args, meta["name"], f"AI4Protein/ProSST-{vocabs[0]}"
            )
        if already:
            logger.info(f"Using existing ProSST structure tokens: {already}")
        else:
            pdb_file = os.path.join(meta["pdb_dir"], f"{meta['name']}.pdb")
            struc_dir = os.path.join(inputs_root, "struc_seq")
            logger.info(
                f"Building ProSST structure tokens from PDB (K={','.join(str(v) for v in vocabs)})"
            )
            try:
                generate_struc_seq_from_pdb(pdb_file, struc_dir, meta["name"], vocabs)
            except Exception as exc:
                raise SystemExit(
                    f"Could not build ProSST structure tokens from {pdb_file}: {exc}\n"
                    "Install extras with pip install -e '.[prosst]', or pass --struc_seq_dir."
                ) from exc
            args.struc_seq_dir = struc_dir
    return meta


def _pdb_paths_for_run(args, protein_names=None):
    paths = []
    pdb_file = getattr(args, "pdb", None)
    if pdb_file and os.path.isfile(pdb_file):
        paths.append(pdb_file)
    pdb_dir = getattr(args, "pdb_dir", None)
    names = list(protein_names or [])
    if pdb_dir and names:
        for name in names:
            cand = os.path.join(pdb_dir, f"{name}.pdb")
            if os.path.isfile(cand):
                paths.append(cand)
    elif pdb_dir and os.path.isdir(pdb_dir):
        try:
            listed = sorted(os.listdir(pdb_dir))
        except OSError:
            listed = []
        paths.extend(os.path.join(pdb_dir, name) for name in listed if name.endswith(".pdb"))
    seen = set()
    unique = []
    for path in paths:
        key = os.path.realpath(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _crystal_plddt_skip_paths(args, protein_names=None):
    if not getattr(args, "use_plddt_decay", True):
        return []
    from vrh.scoring.structure_weights import plddt_skip_reason

    skipped = []
    for path in _pdb_paths_for_run(args, protein_names):
        reason = plddt_skip_reason(path)
        if reason:
            skipped.append((path, reason))
    return skipped


def _warn_crystal_no_plddt(args, logger, protein_names=None):
    skipped = _crystal_plddt_skip_paths(args, protein_names)
    if not skipped:
        return False
    why = (
        "You enabled pLDDT"
        if getattr(args, "plddt_explicit", False)
        else "Full vrh includes pLDDT"
    )
    preview = ", ".join(os.path.basename(path) for path, _ in skipped[:5])
    more = f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""
    logger.warn(
        f"{why}, but {preview}{more} is a crystal/experimental structure with no pLDDT "
        "(B-factor is a temperature factor). Skipping pLDDT decay."
    )
    return True


def _log_available_inputs(args, logger, protein_names=None):
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
    _warn_crystal_no_plddt(args, logger, protein_names)


def _write_run_meta(args, model_key, protein_names):
    import json
    from vrh import __version__

    payload = {
        "vrh_version": __version__,
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


def run_score(argv=None):
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
    elif argv and argv[0] in {"download", "download-proteingym"}:
        from vrh.data.download import run_download

        code = run_download(argv[1:])
        if code:
            raise SystemExit(code)
        return

    raw_args = create_parser().parse_args(argv)
    if getattr(raw_args, "list_models", False):
        _print_model_table()
        return

    args = postprocess_args(raw_args)
    args.plddt_explicit = any(
        item == "--use_plddt_decay" or item.startswith("--plddt_decay_mode")
        for item in argv
    )
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

    if _is_single_protein(args):
        os.makedirs(args.out_scores_dir, exist_ok=True)
        _prepare_single_protein(args, logger, model_key=model_key)
    else:
        os.makedirs(args.out_scores_dir, exist_ok=True)
        from vrh.data.inputs import fill_aa_seq_from_pdb

        fill_aa_seq_from_pdb(args, logger)

    if not args.aa_seq_dir:
        raise SystemExit(
            "No data: provide --base_dir, --fasta, or --pdb.\n"
            + expected_layout_text(model_key)
        )
    if not args.mutant_dir:
        raise SystemExit(
            "No data: provide substitutions/ via --base_dir, or use --fasta / --pdb.\n"
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
            f"No data: --protein_list matched no proteins in {args.aa_seq_dir}."
        )
    if not _is_single_protein(args):
        from vrh.data.inputs import fill_prosst_tokens_from_pdb

        try:
            fill_prosst_tokens_from_pdb(args, model_key, protein_names, logger)
        except Exception as exc:
            raise SystemExit(
                f"Could not build ProSST structure tokens from PDB: {exc}\n"
                "Install extras with pip install -e '.[prosst]', or pass --struc_seq_dir."
            ) from exc
    require_run_inputs(args, model_key, protein_names)
    if not protein_names:
        raise SystemExit(
            "No data: no proteins to score. Pass --fasta, --pdb, or aa_seq/ / pdbs/ under --base_dir.\n"
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
            f"Official {OFFICIAL_SYSTEM_NAME}: vrh on a ProSST ensemble "
            f"({len(args.model_name)} checkpoints)"
        )
    elif is_prosst_key(model_key) and not is_official_venusrem2(model_key, args):
        logger.warn(
            f"{OFFICIAL_SYSTEM_NAME} is vrh on a ProSST ensemble only. "
            "This single-ProSST run is vrh — pass 2+ --model_name AI4Protein/ProSST-* "
            f"to label the run {OFFICIAL_SYSTEM_NAME}."
        )
    out_names = getattr(args, "model_out_name", None) or []
    if any(n == OFFICIAL_SYSTEM_NAME for n in out_names) and not is_official_venusrem2(
        model_key, args
    ):
        logger.warn(
            f"--model_out_name {OFFICIAL_SYSTEM_NAME} on a non-ensemble run. "
            "Prefer {backbone}__vrh unless this is a ProSST ensemble."
        )
    rsa = "RSA" if getattr(args, "use_rsa_decay", True) else "no RSA"
    crystal_skip = bool(_crystal_plddt_skip_paths(args, protein_names))
    if not getattr(args, "use_plddt_decay", True):
        plddt = "no pLDDT"
    elif crystal_skip:
        plddt = "pLDDT skipped (crystal, no pLDDT)"
    else:
        plddt = "pLDDT"
    logger.info(
        f"vrh recipe: α={args.alpha}, β={args.background_weight}, "
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
    _log_available_inputs(args, logger, protein_names)
    logger.debug(f"Protein preview: {format_name_preview(protein_names)}")

    official = is_official_venusrem2(model_key, args)
    if official:
        args.structure_vocab_subdir = None

    for model_idx, model_name in enumerate(args.model_name):
        corrs = []
        raw_corrs = []
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
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=False
            )
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
                from vrh.status import working

                with working(f"loading {display_name}"):
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
                hint = f' Install with: pip install "vrh[{extra}]"' if extra else ""
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
                aln_dir = getattr(args, "aa_seq_aln_dir", None)
                if "aa_seq_aln" in args.logit_mode and aln_dir:
                    for suffix in (".a2m", ".a3m", ".fasta"):
                        cand = os.path.join(aln_dir, f"{protein_name}{suffix}")
                        if os.path.exists(cand):
                            aa_seq_aln_file = cand
                            break
                struc_aln_dir = getattr(args, "struc_seq_aln_dir", None)
                if "struc_seq_aln" in args.logit_mode and struc_aln_dir:
                    cand = os.path.join(struc_aln_dir, f"{protein_name}.fasta")
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
                from vrh.backbone.baseline_dispatch import (
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

            has_dms = has_experimental_dms(mutant_df)
            compare_spearman = has_dms and not getattr(
                args, "no_print_compare_spearman", False
            )
            raw_corr = float("nan")
            if compare_spearman:
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

            if has_dms:
                corr = finite_spearman(
                    mutant_df, "DMS_score", venusrem2_col, logger, protein_name, venusrem2_col
                )
            else:
                corr = float("nan")
            corrs.append(corr)
            raw_corrs.append(raw_corr if compare_spearman else float("nan"))
            if compare_spearman:
                if not compare_table_printed:
                    logger.info("Spearman: raw backbone vs vrh  (Δ = vrh − raw)")
                    print_compare_table_header(
                        logger, include_venus=False,
                        raw_label="raw", current_label="vrh",
                    )
                    compare_table_printed = True
                print_compare_table_row(
                    logger=logger, protein_name=protein_name,
                    raw_corr=raw_corr, venusrem2_corr=corr, venus_corr=None,
                )
                delta = corr - raw_corr if np.isfinite(corr) and np.isfinite(raw_corr) else float("nan")
                logger.success(
                    f"Spearman raw={raw_corr:.4f}  vrh={corr:.4f}  Δ={delta:+.4f}",
                    protein=protein_name,
                )
            elif has_dms:
                logger.success(f"{model_out_name} Spearman={corr:.4f}", protein=protein_name)
            else:
                logger.success(
                    f"Wrote {len(mutant_df)} mutant scores (no experimental DMS; Spearman skipped)",
                    protein=protein_name,
                )
            score_path = f"{args.out_scores_dir}/scores/{protein_name}.csv"
            mutant_df.to_csv(score_path, index=False)
            if idx == 0:
                preview_cols = [c for c in (raw_col, venusrem2_col) if c in mutant_df.columns]
                print_score_preview(
                    logger,
                    mutant_df,
                    preview_cols or venusrem2_col,
                    protein_name,
                    n=5,
                    path=score_path,
                )
                if len(protein_names) > 1:
                    logger.info(
                        f"Remaining {len(protein_names) - 1} assays write the same columns "
                        f"under {args.out_scores_dir}/scores/"
                    )

        if compare_table_printed:
            logger.info("+" + "-" * 42 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 10 + "+")
        mean_corr = pd.Series(corrs, dtype="float64").mean(skipna=True)
        mean_raw = pd.Series(raw_corrs, dtype="float64").mean(skipna=True)
        if np.isfinite(mean_raw):
            mean_delta = mean_corr - mean_raw if np.isfinite(mean_corr) else float("nan")
            logger.section(
                f"average Spearman  raw={mean_raw:.4f}  vrh={mean_corr:.4f}  Δ={mean_delta:+.4f}"
            )
        elif np.isfinite(mean_corr):
            logger.section(f"{model_out_name} average Spearman: {mean_corr:.4f}")
        else:
            logger.section("Scores written (no experimental DMS; Spearman skipped)")
        if any(np.isfinite(v) for v in corrs):
            summary_df_path = f"{args.out_scores_dir}/summary_performance.csv"
            payload = {"protein": protein_names, model_out_name: corrs}
            if any(np.isfinite(v) for v in raw_corrs):
                payload[f"{model_out_name}__raw"] = raw_corrs
            new_rows = pd.DataFrame(payload)
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
            if has_experimental_dms(frame):
                ens_corrs.append(
                    finite_spearman(frame, "DMS_score", ens_col, logger, protein_name, ens_col)
                )
            else:
                ens_corrs.append(float("nan"))
        mean_ens = pd.Series(ens_corrs, dtype="float64").mean(skipna=True)
        if np.isfinite(mean_ens):
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
