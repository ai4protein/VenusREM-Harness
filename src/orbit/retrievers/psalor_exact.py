from typing import List, Optional

import torch
from transformers import AutoTokenizer

from src.orbit.retrievers.base import BaseRetriever
from src.orbit.retrievers.psalor_common import (
    build_position_mask,
    compute_sequence_dedup_weights,
    compute_weighted_counts,
    expand_aln_logits_to_full,
    get_wildtype_token_ids,
    load_alignment_ids_and_span,
    load_residue_accessibility_weights,
    read_first_fasta_sequence,
)
from src.orbit.types import RetrievalOutput


class PSALORExactRetriever(BaseRetriever):
    """
    RSALOR-inspired second-layer retriever:
    - Use weighted MSA frequencies.
    - Build LOR-style correction log(P(token)/P(wt)).
    - Apply accessibility proxy from pLDDT as position weights.
    """

    def __init__(
        self,
        plddt_dir: Optional[str] = None,
        protein_info_file: Optional[str] = None,
        pseudocount: float = 1e-4,
        sequence_weights: Optional[List[float]] = None,
        rsa_mode: str = "auto",
        pdb_dir: Optional[str] = None,
        enable_sequence_dedup: bool = True,
        dedup_identity_threshold: float = 0.8,
        max_sequences_for_clustering: int = 2000,
    ) -> None:
        self.plddt_dir = plddt_dir
        self.protein_info_file = protein_info_file
        self.pseudocount = pseudocount
        self.sequence_weights = sequence_weights
        self.rsa_mode = rsa_mode
        self.pdb_dir = pdb_dir
        self.enable_sequence_dedup = enable_sequence_dedup
        self.dedup_identity_threshold = dedup_identity_threshold
        self.max_sequences_for_clustering = max_sequences_for_clustering

    def retrieve(
        self,
        tokenizer: AutoTokenizer,
        aa_seq_aln_file: Optional[str] = None,
        struc_seq_aln_file: Optional[str] = None,
        protein_name: Optional[str] = None,
        **kwargs,
    ) -> RetrievalOutput:
        residue_fasta = kwargs.get("residue_fasta")
        pdb_file = kwargs.get("pdb_file")
        if aa_seq_aln_file is None or residue_fasta is None:
            return RetrievalOutput(
                layer2_name="psalor_exact",
                metadata={"psalor_exact_skipped": True, "protein_name": protein_name},
            )

        sequence = read_first_fasta_sequence(residue_fasta)
        seq_len = len(sequence)
        wt_token_ids = get_wildtype_token_ids(tokenizer, sequence)
        aln_ids, aln_start, aln_end = load_alignment_ids_and_span(tokenizer, aa_seq_aln_file)
        cluster_weights = None
        if self.enable_sequence_dedup:
            cluster_weights = compute_sequence_dedup_weights(
                alignment_ids=aln_ids,
                pad_token_id=tokenizer.pad_token_id,
                identity_threshold=self.dedup_identity_threshold,
                max_sequences_for_clustering=self.max_sequences_for_clustering,
            )

        merged_weights = None
        if cluster_weights is not None:
            merged_weights = cluster_weights
        if self.sequence_weights is not None:
            if merged_weights is None:
                merged_weights = list(self.sequence_weights)
            else:
                merged_weights = [a * b for a, b in zip(merged_weights, self.sequence_weights)]

        counts = compute_weighted_counts(aln_ids, tokenizer.vocab_size, merged_weights)
        probs = counts / counts.sum(dim=1, keepdim=True).clamp_min(1e-12)

        lor_logits = torch.zeros_like(probs)
        for col_idx in range(probs.shape[0]):
            seq_idx = aln_start + col_idx
            if seq_idx >= seq_len:
                break
            wt_token_id = int(wt_token_ids[seq_idx].item())
            wt_prob = probs[col_idx, wt_token_id]
            lor_logits[col_idx] = torch.log(probs[col_idx] + self.pseudocount) - torch.log(
                wt_prob + self.pseudocount
            )

        full_lor_logits = expand_aln_logits_to_full(
            lor_logits, seq_len, tokenizer.vocab_size, aln_start, aln_end
        )
        position_weights = load_residue_accessibility_weights(
            seq_len=seq_len,
            protein_name=protein_name,
            rsa_mode=self.rsa_mode,
            pdb_file=pdb_file,
            pdb_dir=self.pdb_dir,
            plddt_dir=self.plddt_dir,
            protein_info_file=self.protein_info_file,
        )
        layer2_mask = build_position_mask(seq_len=seq_len, start=aln_start, end=aln_end)
        position_weights = position_weights * layer2_mask

        return RetrievalOutput(
            layer2_logits=full_lor_logits,
            layer2_position_weights=position_weights,
            layer2_mask=layer2_mask,
            layer2_name="psalor_exact",
            metadata={
                "protein_name": protein_name,
                "psalor_exact_aln_start": aln_start,
                "psalor_exact_aln_end": aln_end,
                "psalor_exact_rsa_mode": self.rsa_mode,
                "psalor_exact_dedup": self.enable_sequence_dedup,
            },
        )
