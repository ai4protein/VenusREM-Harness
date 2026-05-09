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
)
from src.orbit.types import RetrievalOutput, merge_retrieval_outputs


def build_retriever(args: Namespace) -> BaseRetriever:
    retriever_name = (args.retriever or "msa").lower()
    if retriever_name == "msa":
        return MSARetriever()
    if retriever_name == "hits":
        return HomologHitsRetriever(
            hits_dir=args.hits_dir,
            hits_file_suffix=args.hits_file_suffix,
        )
    raise ValueError(f"Unsupported retriever: {args.retriever}")


def build_retriever2(args: Namespace) -> Optional[BaseRetriever]:
    retriever2_name = getattr(args, "retriever2", None)
    if retriever2_name is None:
        return None
    retriever2_name = retriever2_name.lower()
    if retriever2_name in ("none", ""):
        return None
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
        retrieval_output = merge_retrieval_outputs(retrieval_output, retrieval_output2)
    return fusion.fuse(plm_logits=plm_logits, retrieval_output=retrieval_output, mode=args.logit_mode)
