from argparse import Namespace
from typing import Optional

import torch
from transformers import AutoTokenizer

from src.orbit.fusions import (
    AdaptiveGateFusion,
    BaseFusion,
    LinearAlphaFusion,
    TwoStageLearnableGateFusion,
)
from src.orbit.retrievers import (
    BaseRetriever,
    HomologHitsRetriever,
    MSARetriever,
    PSALORExactRetriever,
    PSALORVariantRetriever,
)
from src.orbit.types import RetrievalOutput, merge_retrieval_outputs


def _safe_weight(value: float) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _resolve_layer2_weights(name_a: Optional[str], name_b: Optional[str], args: Namespace):
    default_exact = _safe_weight(getattr(args, "psalor_exact_weight", 0.5))
    default_variant = _safe_weight(getattr(args, "psalor_variant_weight", 0.5))
    name_a = (name_a or "").lower()
    name_b = (name_b or "").lower()

    w_a = 1.0
    w_b = 1.0
    if "psalor_exact" in name_a:
        w_a = default_exact
    elif "psalor_variant" in name_a:
        w_a = default_variant
    if "psalor_exact" in name_b:
        w_b = default_exact
    elif "psalor_variant" in name_b:
        w_b = default_variant

    total = w_a + w_b
    if total <= 0:
        return 0.5, 0.5
    return w_a / total, w_b / total


def _mix_layer2_outputs(
    primary: RetrievalOutput,
    secondary: RetrievalOutput,
    args: Namespace,
) -> Optional[RetrievalOutput]:
    if primary.layer2_logits is None or secondary.layer2_logits is None:
        return None

    w_primary, w_secondary = _resolve_layer2_weights(
        primary.layer2_name,
        secondary.layer2_name,
        args,
    )
    mixed_logits = w_primary * primary.layer2_logits + w_secondary * secondary.layer2_logits

    pos_w = None
    if primary.layer2_position_weights is not None and secondary.layer2_position_weights is not None:
        pos_w = w_primary * primary.layer2_position_weights + w_secondary * secondary.layer2_position_weights
    elif primary.layer2_position_weights is not None:
        pos_w = primary.layer2_position_weights
    elif secondary.layer2_position_weights is not None:
        pos_w = secondary.layer2_position_weights

    mask = None
    if primary.layer2_mask is not None and secondary.layer2_mask is not None:
        mask = torch.maximum(primary.layer2_mask, secondary.layer2_mask)
    elif primary.layer2_mask is not None:
        mask = primary.layer2_mask
    elif secondary.layer2_mask is not None:
        mask = secondary.layer2_mask

    mixed_name = f"mix({primary.layer2_name or 'layer2a'}+{secondary.layer2_name or 'layer2b'})"
    metadata = {
        **primary.metadata,
        **secondary.metadata,
        "layer2_mix_weight_primary": w_primary,
        "layer2_mix_weight_secondary": w_secondary,
        "layer2_mix_name_primary": primary.layer2_name,
        "layer2_mix_name_secondary": secondary.layer2_name,
    }
    return RetrievalOutput(
        layer2_logits=mixed_logits,
        layer2_position_weights=pos_w,
        layer2_mask=mask,
        layer2_name=mixed_name,
        metadata=metadata,
    )


def build_retriever(args: Namespace) -> BaseRetriever:
    retriever_name = (args.retriever or "msa").lower()
    if retriever_name == "msa":
        return MSARetriever()
    if retriever_name == "hits":
        return HomologHitsRetriever(
            hits_dir=args.hits_dir,
            hits_file_suffix=args.hits_file_suffix,
        )
    if retriever_name == "psalor_exact":
        return PSALORExactRetriever(
            plddt_dir=getattr(args, "plddt_dir", None),
            protein_info_file=getattr(args, "protein_info_file", None),
            rsa_mode=getattr(args, "rsa_mode", "auto"),
            pdb_dir=getattr(args, "pdb_dir", None),
            enable_sequence_dedup=getattr(args, "enable_sequence_dedup", True),
            dedup_identity_threshold=getattr(args, "dedup_identity_threshold", 0.8),
            max_sequences_for_clustering=getattr(args, "max_sequences_for_clustering", 2000),
        )
    if retriever_name == "psalor_variant":
        return PSALORVariantRetriever(
            hits_dir=args.hits_dir,
            hits_file_suffix=args.hits_file_suffix,
            plddt_dir=getattr(args, "plddt_dir", None),
            protein_info_file=getattr(args, "protein_info_file", None),
            psalor_mix=getattr(args, "psalor_mix", 0.5),
            rsa_mode=getattr(args, "rsa_mode", "auto"),
            pdb_dir=getattr(args, "pdb_dir", None),
            center_wt_lor=getattr(args, "psalor_variant_center_wt_lor", True),
        )
    raise ValueError(f"Unsupported retriever: {args.retriever}")


def build_retriever2(args: Namespace) -> Optional[BaseRetriever]:
    retriever2_name = getattr(args, "retriever2", None)
    if retriever2_name is None:
        return None
    retriever2_name = retriever2_name.lower()
    if retriever2_name in ("none", ""):
        return None
    if retriever2_name == "psalor_exact":
        return PSALORExactRetriever(
            plddt_dir=getattr(args, "plddt_dir", None),
            protein_info_file=getattr(args, "protein_info_file", None),
            rsa_mode=getattr(args, "rsa_mode", "auto"),
            pdb_dir=getattr(args, "pdb_dir", None),
            enable_sequence_dedup=getattr(args, "enable_sequence_dedup", True),
            dedup_identity_threshold=getattr(args, "dedup_identity_threshold", 0.8),
            max_sequences_for_clustering=getattr(args, "max_sequences_for_clustering", 2000),
        )
    if retriever2_name == "psalor_variant":
        return PSALORVariantRetriever(
            hits_dir=args.hits_dir,
            hits_file_suffix=args.hits_file_suffix,
            plddt_dir=getattr(args, "plddt_dir", None),
            protein_info_file=getattr(args, "protein_info_file", None),
            psalor_mix=getattr(args, "psalor_mix", 0.5),
            rsa_mode=getattr(args, "rsa_mode", "auto"),
            pdb_dir=getattr(args, "pdb_dir", None),
            center_wt_lor=getattr(args, "psalor_variant_center_wt_lor", True),
        )
    if retriever2_name == "msa":
        return MSARetriever()
    if retriever2_name == "hits":
        return HomologHitsRetriever(
            hits_dir=args.hits_dir,
            hits_file_suffix=args.hits_file_suffix,
        )
    raise ValueError(f"Unsupported retriever2: {retriever2_name}")


def build_stage1_fusion(args: Namespace) -> BaseFusion:
    fusion_name = (args.fusion or "linear_alpha").lower()
    if fusion_name == "linear_alpha":
        return LinearAlphaFusion(alpha=args.alpha)
    if fusion_name == "adaptive_gate":
        return AdaptiveGateFusion(
            alpha=args.alpha,
            min_gate=getattr(args, "adaptive_min_gate", 0.0),
            max_gate=getattr(args, "adaptive_max_gate", 0.95),
        )
    if fusion_name == "two_stage_learnable_gate":
        # Default stage-1 inside two-stage fusion.
        return LinearAlphaFusion(alpha=args.alpha)
    raise ValueError(f"Unsupported fusion: {args.fusion}")


def build_fusion(args: Namespace, stage1_fusion: BaseFusion) -> BaseFusion:
    fusion2_name = getattr(args, "fusion2", "none") or "none"
    if fusion2_name.lower() == "two_stage_learnable_gate" or (
        (args.fusion or "").lower() == "two_stage_learnable_gate"
    ):
        return TwoStageLearnableGateFusion(
            stage1_fusion=stage1_fusion,
            layer2_weight=getattr(args, "layer2_weight", 1.0),
            gate_temperature=getattr(args, "gate_temperature", 1.0),
            alpha_family=getattr(args, "alpha_family", 1.0),
            enable_gate_diagnostics=getattr(args, "enable_gate_diagnostics", False),
        )
    return stage1_fusion


def maybe_fuse_logits_with_orbit(
    args: Namespace,
    tokenizer: AutoTokenizer,
    plm_logits: torch.Tensor,
    aa_seq_aln_file: Optional[str] = None,
    struc_seq_aln_file: Optional[str] = None,
    protein_name: Optional[str] = None,
    residue_fasta: Optional[str] = None,
    structure_fasta: Optional[str] = None,
    pdb_file: Optional[str] = None,
) -> torch.Tensor:
    if not args.orbit_enable or args.alpha == 0:
        return plm_logits

    retriever = build_retriever(args)
    retriever2 = build_retriever2(args)
    stage1_fusion = build_stage1_fusion(args)
    fusion = build_fusion(args, stage1_fusion)
    retrieval_output: RetrievalOutput = retriever.retrieve(
        tokenizer=tokenizer,
        aa_seq_aln_file=aa_seq_aln_file,
        struc_seq_aln_file=struc_seq_aln_file,
        protein_name=protein_name,
        residue_fasta=residue_fasta,
        structure_fasta=structure_fasta,
        pdb_file=pdb_file,
    )
    if retriever2 is not None:
        retrieval_output2 = retriever2.retrieve(
            tokenizer=tokenizer,
            aa_seq_aln_file=aa_seq_aln_file,
            struc_seq_aln_file=struc_seq_aln_file,
            protein_name=protein_name,
            residue_fasta=residue_fasta,
            structure_fasta=structure_fasta,
            pdb_file=pdb_file,
        )
        mixed_layer2 = _mix_layer2_outputs(retrieval_output, retrieval_output2, args)
        if mixed_layer2 is not None:
            retrieval_output2 = merge_retrieval_outputs(retrieval_output2, mixed_layer2)
        retrieval_output = merge_retrieval_outputs(retrieval_output, retrieval_output2)
    elif retrieval_output.layer2_logits is None:
        # Allow single retriever to operate as layer2 retriever.
        psalor_mode = getattr(args, "psalor_mode", "none")
        if psalor_mode == "exact":
            retrieval_output2 = PSALORExactRetriever(
                plddt_dir=getattr(args, "plddt_dir", None),
                protein_info_file=getattr(args, "protein_info_file", None),
                rsa_mode=getattr(args, "rsa_mode", "auto"),
                pdb_dir=getattr(args, "pdb_dir", None),
                enable_sequence_dedup=getattr(args, "enable_sequence_dedup", True),
                dedup_identity_threshold=getattr(args, "dedup_identity_threshold", 0.8),
                max_sequences_for_clustering=getattr(args, "max_sequences_for_clustering", 2000),
            ).retrieve(
                tokenizer=tokenizer,
                aa_seq_aln_file=aa_seq_aln_file,
                struc_seq_aln_file=struc_seq_aln_file,
                protein_name=protein_name,
                residue_fasta=residue_fasta,
                structure_fasta=structure_fasta,
                pdb_file=pdb_file,
            )
            retrieval_output = merge_retrieval_outputs(retrieval_output, retrieval_output2)
        elif psalor_mode == "variant":
            retrieval_output2 = PSALORVariantRetriever(
                hits_dir=args.hits_dir,
                hits_file_suffix=args.hits_file_suffix,
                plddt_dir=getattr(args, "plddt_dir", None),
                protein_info_file=getattr(args, "protein_info_file", None),
                psalor_mix=getattr(args, "psalor_mix", 0.5),
                rsa_mode=getattr(args, "rsa_mode", "auto"),
                pdb_dir=getattr(args, "pdb_dir", None),
                center_wt_lor=getattr(args, "psalor_variant_center_wt_lor", True),
            ).retrieve(
                tokenizer=tokenizer,
                aa_seq_aln_file=aa_seq_aln_file,
                struc_seq_aln_file=struc_seq_aln_file,
                protein_name=protein_name,
                residue_fasta=residue_fasta,
                structure_fasta=structure_fasta,
                pdb_file=pdb_file,
            )
            retrieval_output = merge_retrieval_outputs(retrieval_output, retrieval_output2)
        elif psalor_mode == "both":
            exact_output = PSALORExactRetriever(
                plddt_dir=getattr(args, "plddt_dir", None),
                protein_info_file=getattr(args, "protein_info_file", None),
                rsa_mode=getattr(args, "rsa_mode", "auto"),
                pdb_dir=getattr(args, "pdb_dir", None),
                enable_sequence_dedup=getattr(args, "enable_sequence_dedup", True),
                dedup_identity_threshold=getattr(args, "dedup_identity_threshold", 0.8),
                max_sequences_for_clustering=getattr(args, "max_sequences_for_clustering", 2000),
            ).retrieve(
                tokenizer=tokenizer,
                aa_seq_aln_file=aa_seq_aln_file,
                struc_seq_aln_file=struc_seq_aln_file,
                protein_name=protein_name,
                residue_fasta=residue_fasta,
                structure_fasta=structure_fasta,
                pdb_file=pdb_file,
            )
            variant_output = PSALORVariantRetriever(
                hits_dir=args.hits_dir,
                hits_file_suffix=args.hits_file_suffix,
                plddt_dir=getattr(args, "plddt_dir", None),
                protein_info_file=getattr(args, "protein_info_file", None),
                psalor_mix=getattr(args, "psalor_mix", 0.5),
                rsa_mode=getattr(args, "rsa_mode", "auto"),
                pdb_dir=getattr(args, "pdb_dir", None),
                center_wt_lor=getattr(args, "psalor_variant_center_wt_lor", True),
            ).retrieve(
                tokenizer=tokenizer,
                aa_seq_aln_file=aa_seq_aln_file,
                struc_seq_aln_file=struc_seq_aln_file,
                protein_name=protein_name,
                residue_fasta=residue_fasta,
                structure_fasta=structure_fasta,
                pdb_file=pdb_file,
            )
            mixed = _mix_layer2_outputs(exact_output, variant_output, args)
            if mixed is not None:
                retrieval_output = merge_retrieval_outputs(retrieval_output, mixed)
    return fusion.fuse(plm_logits=plm_logits, retrieval_output=retrieval_output, mode=args.logit_mode)
