from typing import List, Optional

from transformers import AutoTokenizer

from src.orbit.retrievers.base import BaseRetriever
from src.orbit.retrievers.utils import (
    build_alignment_ids,
    build_logit_matrix_from_alignment_ids,
    infer_alignment_span,
    read_multi_fasta,
)
from src.orbit.types import RetrievalOutput


class MSARetriever(BaseRetriever):
    """Retrieve residue/structure alignment evidence and convert into logits."""

    def __init__(
        self,
        aa_sequence_weights: Optional[List[float]] = None,
        struc_sequence_weights: Optional[List[float]] = None,
    ) -> None:
        self.aa_sequence_weights = aa_sequence_weights
        self.struc_sequence_weights = struc_sequence_weights

    def retrieve(
        self,
        tokenizer: AutoTokenizer,
        aa_seq_aln_file: Optional[str] = None,
        struc_seq_aln_file: Optional[str] = None,
        protein_name: Optional[str] = None,
        **kwargs,
    ) -> RetrievalOutput:
        output = RetrievalOutput(metadata={"protein_name": protein_name})

        if aa_seq_aln_file is not None:
            alignment_dict = read_multi_fasta(aa_seq_aln_file)
            aln_ids = build_alignment_ids(tokenizer, alignment_dict)
            aln_start, aln_end = infer_alignment_span(alignment_dict)
            output.aa_seq_aln_logits = build_logit_matrix_from_alignment_ids(
                aln_ids, tokenizer.vocab_size, self.aa_sequence_weights
            )
            output.aa_aln_start = aln_start
            output.aa_aln_end = aln_end

        if struc_seq_aln_file is not None:
            alignment_dict = read_multi_fasta(struc_seq_aln_file)
            if len(alignment_dict) > 0:
                aln_ids = build_alignment_ids(tokenizer, alignment_dict)
                output.struc_seq_aln_logits = build_logit_matrix_from_alignment_ids(
                    aln_ids, tokenizer.vocab_size, self.struc_sequence_weights
                )

        return output
