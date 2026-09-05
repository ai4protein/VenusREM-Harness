import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from venus_orbit.backbone.forward_utils import (
    force_config_max_residue_len,
    infer_model_max_residue_len,
)


@dataclass
class BaselineState:
    model: Any = None
    tokenizer: Any = None
    model_max_residue_len: Optional[int] = None
    baseline_type: str = "auto"
    extra: dict = field(default_factory=dict)


def load_baseline(baseline_type, model_name, args, device, logger) -> BaselineState:
    state = BaselineState(baseline_type=baseline_type)

    if baseline_type == "esm2":
        from venus_orbit.baseline.esm2 import load_esm2_model
        state.model, state.tokenizer, state.model_max_residue_len = load_esm2_model(model_name, device)
        if args.max_residue_len is not None:
            state.model_max_residue_len = args.max_residue_len
        logger.info(f"Baseline: ESM-2 ({model_name})")

    elif baseline_type == "esm1b":
        from venus_orbit.baseline.esm1b import load_esm1b_model
        state.model, state.tokenizer, state.model_max_residue_len = load_esm1b_model(model_name, device)
        if args.max_residue_len is not None:
            state.model_max_residue_len = args.max_residue_len
        logger.info(f"Baseline: ESM-1b ({model_name})")

    elif baseline_type == "esm1v":
        from venus_orbit.baseline.esm1v import load_esm1v_tokenizer
        state.tokenizer = load_esm1v_tokenizer()
        state.model_max_residue_len = args.max_residue_len
        logger.info("Baseline: ESM-1v 5-seed ensemble (cached per process)")

    elif baseline_type == "saprot":
        from venus_orbit.baseline.saprot import load_saprot_model_and_tokenizers
        from venus_orbit.models.weights import ensure_foldseek_bin

        args.foldseek_bin = ensure_foldseek_bin(
            cache_dir=getattr(args, "cache_dir", None),
            explicit=getattr(args, "foldseek_bin", None),
            logger=logger,
        )
        saprot_model, saprot_tokenizer, esm_tokenizer = load_saprot_model_and_tokenizers(
            model_name, device
        )
        state.tokenizer = esm_tokenizer
        state.model_max_residue_len = args.max_residue_len
        state.extra["saprot_model"] = saprot_model
        state.extra["saprot_tokenizer"] = saprot_tokenizer
        saprot_strategy = getattr(args, "scoring_strategy", "masked-marginals")
        logger.info(f"Baseline: SaProt with {saprot_strategy} scoring")
        logger.info(f"Foldseek binary: {args.foldseek_bin}")

    elif baseline_type == "protssn":
        from venus_orbit.baseline.protssn import load_protssn_models
        from venus_orbit.models.weights import default_cache_dir, ensure_dir
        if args.protssn_model_dir is None:
            args.protssn_model_dir = ensure_dir(
                os.path.join(default_cache_dir(getattr(args, "cache_dir", None)), "protssn")
            )
            logger.info(f"ProtSSN weights dir (auto): {args.protssn_model_dir}")
        protssn_pack, tokenizer = load_protssn_models(
            model_dir=args.protssn_model_dir,
            device=device,
            norm_dir=args.protssn_norm_dir,
            use_ensemble=not args.protssn_no_ensemble,
            logger=logger,
        )
        state.tokenizer = tokenizer
        state.model_max_residue_len = args.max_residue_len
        state.extra["protssn_pack"] = protssn_pack
        n_models = len(protssn_pack["models"])
        logger.info(f"Baseline: ProtSSN ({n_models} model{'s' if n_models > 1 else ''})")

    elif baseline_type == "esm_if":
        from venus_orbit.baseline.esm_if import load_esm_if_model
        esm_if_model, esm_if_alphabet, tokenizer, esm_if_helpers = load_esm_if_model(
            device,
            cache_dir=getattr(args, "cache_dir", None),
            logger=logger,
        )
        state.tokenizer = tokenizer
        state.model_max_residue_len = args.max_residue_len
        state.extra["esm_if_model"] = esm_if_model
        state.extra["esm_if_alphabet"] = esm_if_alphabet
        state.extra["esm_if_helpers"] = esm_if_helpers
        logger.info("Baseline: ESM-IF1 (142M, inverse folding)")

    elif baseline_type == "protein_mpnn":
        from venus_orbit.baseline.protein_mpnn import load_protein_mpnn_model
        from venus_orbit.models.weights import default_cache_dir, ensure_dir
        if args.protein_mpnn_checkpoint is None:
            args.protein_mpnn_checkpoint = ensure_dir(
                os.path.join(default_cache_dir(getattr(args, "cache_dir", None)), "protein_mpnn")
            )
            logger.info(f"ProteinMPNN checkpoint dir (auto): {args.protein_mpnn_checkpoint}")
        mpnn_model, tokenizer = load_protein_mpnn_model(
            args.protein_mpnn_checkpoint, device
        )
        state.tokenizer = tokenizer
        state.model_max_residue_len = args.max_residue_len
        state.extra["mpnn_model"] = mpnn_model
        logger.info("Baseline: ProteinMPNN (128-dim, 3-layer)")

    elif baseline_type == "progen2":
        from venus_orbit.baseline.progen2 import load_progen2_model
        progen2_model, progen2_tokenizer, esm_tokenizer, progen2_context_len = load_progen2_model(
            args.progen2_model_name_or_path, device, fp16=args.progen2_fp16
        )
        state.tokenizer = esm_tokenizer
        state.model_max_residue_len = args.max_residue_len
        state.extra["progen2_model"] = progen2_model
        state.extra["progen2_tokenizer"] = progen2_tokenizer
        state.extra["progen2_context_len"] = progen2_context_len
        logger.info(f"Baseline: Progen2 (context_len={progen2_context_len})")

    elif baseline_type == "progen3":
        from venus_orbit.baseline.progen3 import load_progen3_model
        progen3_model, progen3_tokenizer, esm_tokenizer, progen3_context_len = load_progen3_model(
            args.progen3_model_name_or_path, device, fp16=args.progen3_fp16
        )
        state.tokenizer = esm_tokenizer
        state.model_max_residue_len = args.max_residue_len
        state.extra["progen3_model"] = progen3_model
        state.extra["progen3_tokenizer"] = progen3_tokenizer
        state.extra["progen3_context_len"] = progen3_context_len
        logger.info(f"Baseline: Progen3 (context_len={progen3_context_len})")

    elif baseline_type == "protgpt2":
        from venus_orbit.baseline.protgpt2 import load_protgpt2_model
        protgpt2_model, protgpt2_tokenizer, esm_tokenizer, protgpt2_context_len = load_protgpt2_model(
            args.protgpt2_model_name_or_path, device
        )
        state.tokenizer = esm_tokenizer
        state.model_max_residue_len = args.max_residue_len
        state.extra["protgpt2_model"] = protgpt2_model
        state.extra["protgpt2_tokenizer"] = protgpt2_tokenizer
        state.extra["protgpt2_context_len"] = protgpt2_context_len
        logger.info(f"Baseline: ProtGPT2 (context_len={protgpt2_context_len})")

    elif baseline_type == "rita":
        from venus_orbit.baseline.rita import load_rita_model
        rita_model, rita_tokenizer, esm_tokenizer, rita_context_len = load_rita_model(
            args.rita_model_name_or_path, device
        )
        state.tokenizer = esm_tokenizer
        state.model_max_residue_len = args.max_residue_len
        state.extra["rita_model"] = rita_model
        state.extra["rita_tokenizer"] = rita_tokenizer
        state.extra["rita_context_len"] = rita_context_len
        logger.info(f"Baseline: RITA (context_len={rita_context_len})")

    elif baseline_type == "esm3":
        from venus_orbit.baseline.esm3 import load_esm3_model
        esm3_model, esm3_tokenizer, _, esm3_max_residue_len = load_esm3_model(
            args.esm3_model_name, device
        )
        from transformers import AutoTokenizer as _AT
        state.tokenizer = _AT.from_pretrained("facebook/esm2_t6_8M_UR50D")
        state.model_max_residue_len = esm3_max_residue_len
        if args.max_residue_len is not None:
            state.model_max_residue_len = args.max_residue_len
        state.extra["esm3_model"] = esm3_model
        state.extra["esm3_tokenizer"] = esm3_tokenizer
        from venus_orbit.baseline.esm3.esm3 import _resolve_model_type
        state.extra["esm3_model_type"] = _resolve_model_type(args.esm3_model_name)
        logger.info(f"Baseline: ESM3/ESM-C ({args.esm3_model_name})")

    elif baseline_type == "tranception":
        from venus_orbit.baseline.tranception import load_tranception_model
        tranception_model, tranception_tokenizer, esm_tokenizer, tranception_context_len = load_tranception_model(
            args.tranception_checkpoint, device
        )
        state.tokenizer = esm_tokenizer
        state.model_max_residue_len = args.max_residue_len
        state.extra["tranception_model"] = tranception_model
        state.extra["tranception_tokenizer"] = tranception_tokenizer
        state.extra["tranception_context_len"] = tranception_context_len
        logger.info(f"Baseline: Tranception (context_len={tranception_context_len})")

    elif baseline_type == "carp":
        from venus_orbit.baseline.carp import load_carp_model
        carp_model, carp_collater, esm_tokenizer, carp_alphabet = load_carp_model(
            args.carp_model_name, device
        )
        state.tokenizer = esm_tokenizer
        state.model_max_residue_len = args.max_residue_len
        state.extra["carp_model"] = carp_model
        state.extra["carp_collater"] = carp_collater
        state.extra["carp_alphabet"] = carp_alphabet
        logger.info(f"Baseline: CARP ({args.carp_model_name})")

    elif baseline_type == "s2f":
        from venus_orbit.models.weights import bundled_s2f_config
        if args.s2f_config is None and args.s2f_checkpoint is not None:
            args.s2f_config = bundled_s2f_config()
        if args.s2f_config is not None and args.s2f_checkpoint is not None:
            from venus_orbit.baseline.s2f import load_s2f_model
            s2f_task, tokenizer, s2f_config = load_s2f_model(
                args.s2f_config, args.s2f_checkpoint, device
            )
            state.tokenizer = tokenizer
            state.extra["s2f_task"] = s2f_task
            state.extra["s2f_config"] = s2f_config
            state.extra["s2f_lightweight"] = False
            logger.info("Baseline: S2F (full TorchDrug, sequence-only)")
        else:
            from venus_orbit.baseline.s2f.s2f import load_s2f_model_lightweight
            model, tokenizer, model_max_residue_len = load_s2f_model_lightweight(device=device)
            state.model = model
            state.tokenizer = tokenizer
            state.model_max_residue_len = model_max_residue_len
            if args.max_residue_len is not None:
                state.model_max_residue_len = args.max_residue_len
            state.extra["s2f_lightweight"] = True
            logger.info("Baseline: S2F (lightweight/ESM2)")

    elif baseline_type == "s3f":
        from venus_orbit.models.weights import bundled_s3f_config, ensure_s3f_checkpoint
        if args.s2f_config is None:
            args.s2f_config = bundled_s3f_config()
        args.s2f_checkpoint = ensure_s3f_checkpoint(
            cache_dir=getattr(args, "cache_dir", None),
            explicit=args.s2f_checkpoint,
            logger=logger,
        )
        surface_path = getattr(args, "s3f_surface_dir", None)
        if not surface_path and getattr(args, "base_dir", None):
            cand = os.path.join(args.base_dir, "s3f_surfaces_af2_assay_resolved_full")
            if os.path.isdir(cand):
                surface_path = cand
                args.s3f_surface_dir = cand
        from venus_orbit.baseline.s2f import load_s2f_model
        s2f_task, tokenizer, s2f_config = load_s2f_model(
            args.s2f_config,
            args.s2f_checkpoint,
            device,
            surface_path=surface_path,
            structure_path=getattr(args, "pdb_dir", None),
        )
        state.tokenizer = tokenizer
        state.extra["s2f_task"] = s2f_task
        state.extra["s2f_config"] = s2f_config
        logger.info(
            f"Baseline: S3F (TorchDrug, sequence+structure) ckpt={args.s2f_checkpoint} "
            f"surface={surface_path}"
        )

    else:
        # auto: standard HuggingFace model
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        oe = model.get_output_embeddings()
        ie = model.get_input_embeddings()
        if oe is not None and ie is not None and oe.weight.data_ptr() != ie.weight.data_ptr():
            oe.weight = ie.weight
        model = model.to(device)
        model.eval()
        state.model = model
        state.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if "prosst" in model_name.lower():
            force_config_max_residue_len(model, residue_len=4096)
        state.model_max_residue_len = args.max_residue_len
        if state.model_max_residue_len is None:
            state.model_max_residue_len = infer_model_max_residue_len(state.model, state.tokenizer)

    logger.info(f"Max residue length for forward: {state.model_max_residue_len}")
    return state


def create_baseline_forward_fn(
    state: BaselineState,
    args,
    protein_name: str,
    pdb_file: Optional[str],
    structure_fasta: Optional[str],
    idx: int,
    device,
    logger,
) -> Optional[Callable]:
    baseline_type = state.baseline_type
    scoring_strategy = getattr(args, "scoring_strategy", "wt-marginals")
    model = state.model
    tokenizer = state.tokenizer
    model_max_residue_len = state.model_max_residue_len

    if baseline_type == "esm2":
        if scoring_strategy == "masked-marginals":
            from venus_orbit.baseline.esm2 import forward_esm2_masked_marginal
            return partial(
                forward_esm2_masked_marginal,
                model=model, tokenizer=tokenizer, device=device,
                max_residue_len=model_max_residue_len,
                long_seq_mode=args.long_seq_mode, long_seq_overlap=args.long_seq_overlap,
                logger=logger, protein_name=protein_name,
            )
        else:
            from venus_orbit.baseline.esm2 import forward_esm2
            return partial(
                forward_esm2,
                model=model, tokenizer=tokenizer, device=device,
                max_residue_len=model_max_residue_len,
                long_seq_mode=args.long_seq_mode, long_seq_overlap=args.long_seq_overlap,
                logger=logger, protein_name=protein_name,
            )

    elif baseline_type == "esm1b":
        if scoring_strategy == "masked-marginals":
            from venus_orbit.baseline.esm1b import forward_esm1b_masked_marginal
            return partial(
                forward_esm1b_masked_marginal,
                model=model, tokenizer=tokenizer, device=device,
                max_residue_len=model_max_residue_len,
                long_seq_mode=args.long_seq_mode, long_seq_overlap=args.long_seq_overlap,
                logger=logger, protein_name=protein_name,
            )
        else:
            from venus_orbit.baseline.esm1b import forward_esm1b
            return partial(
                forward_esm1b,
                model=model, tokenizer=tokenizer, device=device,
                max_residue_len=model_max_residue_len,
                long_seq_mode=args.long_seq_mode, long_seq_overlap=args.long_seq_overlap,
                logger=logger, protein_name=protein_name,
            )

    elif baseline_type == "esm1v":
        if scoring_strategy == "masked-marginals":
            from venus_orbit.baseline.esm1v import forward_esm1v_masked_marginal
            return partial(
                forward_esm1v_masked_marginal,
                device=device, seeds=args.esm1v_seeds,
                max_residue_len=model_max_residue_len,
                long_seq_mode=args.long_seq_mode, long_seq_overlap=args.long_seq_overlap,
                logger=logger, protein_name=protein_name,
            )
        else:
            from venus_orbit.baseline.esm1v import forward_esm1v_ensemble
            return partial(
                forward_esm1v_ensemble,
                device=device, seeds=args.esm1v_seeds,
                max_residue_len=model_max_residue_len,
                long_seq_mode=args.long_seq_mode, long_seq_overlap=args.long_seq_overlap,
                logger=logger, protein_name=protein_name,
            )

    elif baseline_type == "saprot":
        from venus_orbit.models.weights import ensure_foldseek_bin
        args.foldseek_bin = ensure_foldseek_bin(
            cache_dir=getattr(args, "cache_dir", None),
            explicit=getattr(args, "foldseek_bin", None),
            logger=logger,
        )
        saprot_model = state.extra["saprot_model"]
        saprot_tokenizer = state.extra["saprot_tokenizer"]
        if scoring_strategy == "wt-marginals":
            from venus_orbit.baseline.saprot import forward_saprot_wt_marginal
            return partial(
                forward_saprot_wt_marginal,
                model=saprot_model, saprot_tokenizer=saprot_tokenizer,
                pdb_file=pdb_file, foldseek_bin=args.foldseek_bin,
                device=device, esm_tokenizer=tokenizer, process_id=idx,
                logger=logger, protein_name=protein_name,
            )
        else:
            from venus_orbit.baseline.saprot import forward_saprot_masked_marginal
            return partial(
                forward_saprot_masked_marginal,
                model=saprot_model, saprot_tokenizer=saprot_tokenizer,
                pdb_file=pdb_file, foldseek_bin=args.foldseek_bin,
                device=device, esm_tokenizer=tokenizer, process_id=idx,
                logger=logger, protein_name=protein_name,
            )

    elif baseline_type == "protssn":
        protssn_pack = state.extra["protssn_pack"]
        if scoring_strategy == "masked-marginals":
            from venus_orbit.baseline.protssn import forward_protssn_masked_marginal
            return partial(
                forward_protssn_masked_marginal,
                pdb_file=pdb_file, device=device, esm_tokenizer=tokenizer,
                protssn_pack=protssn_pack, logger=logger, protein_name=protein_name,
            )
        else:
            from venus_orbit.baseline.protssn import forward_protssn
            return partial(
                forward_protssn,
                pdb_file=pdb_file, device=device, esm_tokenizer=tokenizer,
                protssn_pack=protssn_pack, logger=logger, protein_name=protein_name,
            )

    elif baseline_type == "esm_if":
        from venus_orbit.baseline.esm_if import forward_esm_if
        helpers = state.extra.get("esm_if_helpers") or {}
        return partial(
            forward_esm_if,
            pdb_file=pdb_file, device=device, esm_tokenizer=tokenizer,
            esm_if_model=state.extra["esm_if_model"],
            esm_if_alphabet=state.extra["esm_if_alphabet"],
            chain_id=args.esm_if_chain, logger=logger, protein_name=protein_name,
            load_coords=helpers.get("load_coords"),
            CoordBatchConverter=helpers.get("CoordBatchConverter"),
        )

    elif baseline_type == "protein_mpnn":
        from venus_orbit.baseline.protein_mpnn import forward_protein_mpnn
        return partial(
            forward_protein_mpnn,
            pdb_file=pdb_file, device=device, esm_tokenizer=tokenizer,
            mpnn_model=state.extra["mpnn_model"],
            chain_id=args.protein_mpnn_chain,
            scoring_mode=args.protein_mpnn_scoring_mode,
            random_orders=args.protein_mpnn_random_orders,
            logger=logger, protein_name=protein_name,
        )

    elif baseline_type == "progen2":
        from venus_orbit.baseline.progen2 import forward_progen2
        return partial(
            forward_progen2,
            model=state.extra["progen2_model"],
            tokenizer=state.extra["progen2_tokenizer"],
            esm_tokenizer=tokenizer, device=device,
            model_context_len=state.extra["progen2_context_len"],
            fp16=args.progen2_fp16, logger=logger, protein_name=protein_name,
        )

    elif baseline_type == "progen3":
        from venus_orbit.baseline.progen3 import forward_progen3
        return partial(
            forward_progen3,
            model=state.extra["progen3_model"],
            tokenizer=state.extra["progen3_tokenizer"],
            esm_tokenizer=tokenizer, device=device,
            model_context_len=state.extra["progen3_context_len"],
            fp16=args.progen3_fp16, logger=logger, protein_name=protein_name,
        )

    elif baseline_type == "protgpt2":
        from venus_orbit.baseline.protgpt2 import forward_protgpt2
        return partial(
            forward_protgpt2,
            model=state.extra["protgpt2_model"],
            tokenizer=state.extra["protgpt2_tokenizer"],
            esm_tokenizer=tokenizer, device=device,
            model_context_len=state.extra["protgpt2_context_len"],
            logger=logger, protein_name=protein_name,
        )

    elif baseline_type == "rita":
        from venus_orbit.baseline.rita import forward_rita
        return partial(
            forward_rita,
            model=state.extra["rita_model"],
            tokenizer=state.extra["rita_tokenizer"],
            esm_tokenizer=tokenizer, device=device,
            model_context_len=state.extra["rita_context_len"],
            logger=logger, protein_name=protein_name,
        )

    elif baseline_type == "esm3":
        from venus_orbit.baseline.esm3 import forward_esm3
        return partial(
            forward_esm3,
            model=state.extra["esm3_model"],
            tokenizer=state.extra["esm3_tokenizer"],
            esm_tokenizer=None,
            device=device, max_residue_len=model_max_residue_len,
            model_type=state.extra.get("esm3_model_type", "esmc"),
            logger=logger, protein_name=protein_name,
        )

    elif baseline_type == "tranception":
        from venus_orbit.baseline.tranception import forward_tranception
        return partial(
            forward_tranception,
            model=state.extra["tranception_model"],
            tokenizer=state.extra["tranception_tokenizer"],
            esm_tokenizer=tokenizer, device=device,
            model_context_len=state.extra["tranception_context_len"],
            scoring_mirror=not args.tranception_no_mirror,
            logger=logger, protein_name=protein_name,
        )

    elif baseline_type == "carp":
        from venus_orbit.baseline.carp import forward_carp
        return partial(
            forward_carp,
            model=state.extra["carp_model"],
            collater=state.extra["carp_collater"],
            esm_tokenizer=tokenizer,
            protein_alphabet=state.extra["carp_alphabet"],
            device=device,
            logger=logger, protein_name=protein_name,
        )

    elif baseline_type == "s2f":
        if not state.extra.get("s2f_lightweight", True):
            from venus_orbit.baseline.s2f import forward_s2f
            return partial(
                forward_s2f,
                task=state.extra["s2f_task"], esm_tokenizer=tokenizer,
                device=device, pdb_file=None,
                config=state.extra["s2f_config"],
                logger=logger, protein_name=protein_name,
            )
        else:
            from venus_orbit.baseline.s2f.s2f import forward_s2f_lightweight
            return partial(
                forward_s2f_lightweight,
                model=model, tokenizer=tokenizer, device=device,
                max_residue_len=model_max_residue_len,
                long_seq_mode=args.long_seq_mode, long_seq_overlap=args.long_seq_overlap,
                logger=logger, protein_name=protein_name,
            )

    elif baseline_type == "s3f":
        if scoring_strategy == "masked-marginals":
            from venus_orbit.baseline.s2f import forward_s3f_masked_marginal
            return partial(
                forward_s3f_masked_marginal,
                task=state.extra["s2f_task"], esm_tokenizer=tokenizer,
                device=device, pdb_file=pdb_file,
                config=state.extra["s2f_config"],
                logger=logger, protein_name=protein_name,
            )
        else:
            from venus_orbit.baseline.s2f import forward_s2f
            return partial(
                forward_s2f,
                task=state.extra["s2f_task"], esm_tokenizer=tokenizer,
                device=device, pdb_file=pdb_file,
                config=state.extra["s2f_config"],
                logger=logger, protein_name=protein_name,
                long_seq_overlap=getattr(args, "long_seq_overlap", 256),
            )

    elif baseline_type == "auto" and scoring_strategy == "masked-marginals":
        from venus_orbit.backbone.forward_utils import forward_masked_marginal
        from venus_orbit.scoring.score_protein import read_seq
        mask_use_structure = (args.backbone_mode == "prosst" and structure_fasta is not None)
        mask_struc_seq = None
        if mask_use_structure:
            raw_struc = read_seq(structure_fasta)
            mask_struc_seq = [int(i) for i in raw_struc.split(",")]
        return partial(
            forward_masked_marginal,
            model=model, tokenizer=tokenizer, device=device,
            max_residue_len=model_max_residue_len,
            long_seq_mode=args.long_seq_mode, long_seq_overlap=args.long_seq_overlap,
            logger=logger, protein_name=protein_name,
            use_structure=mask_use_structure, structure_sequence=mask_struc_seq,
        )

    return None


def create_native_scorer_fn(
    state: BaselineState,
    args,
    protein_name: str,
    device,
    logger,
) -> Optional[Callable]:
    """Return a native per-mutation scorer for baselines that support it."""
    if state.baseline_type == "s3f":
        from venus_orbit.baseline.s2f.s2f import score_s3f_official_native
        pdb_file = (
            f"{args.pdb_dir}/{protein_name}.pdb"
            if args.pdb_dir and os.path.exists(f"{args.pdb_dir}/{protein_name}.pdb")
            else None
        )
        return partial(
            score_s3f_official_native,
            task=state.extra["s2f_task"],
            device=device,
            pdb_file=pdb_file,
            config=state.extra["s2f_config"],
            logger=logger,
            protein_name=protein_name,
        )

    if state.baseline_type != "tranception":
        return None

    from venus_orbit.baseline.tranception import score_tranception_native
    return partial(
        score_tranception_native,
        model=state.extra["tranception_model"],
        tokenizer=state.extra["tranception_tokenizer"],
        device=device,
        model_context_len=state.extra["tranception_context_len"],
        scoring_mirror=not getattr(args, "tranception_no_mirror", False),
        batch_size_inference=getattr(args, "tranception_batch_size", 10),
        logger=logger,
        protein_name=protein_name,
    )
