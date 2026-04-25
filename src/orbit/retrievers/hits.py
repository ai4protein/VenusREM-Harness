import os
from typing import Dict, List, Optional

from transformers import AutoTokenizer

from src.orbit.retrievers.msa import MSARetriever
from src.orbit.retrievers.utils import read_multi_fasta
from src.orbit.types import RetrievalOutput


class HomologHitsRetriever(MSARetriever):
    """Use jackhmmer/evcouplings hits metadata to re-weight MSA rows."""

    def __init__(
        self,
        hits_dir: Optional[str] = None,
        hits_file_suffix: str = "_tblout.txt",
    ) -> None:
        super().__init__()
        self.hits_dir = hits_dir
        self.hits_file_suffix = hits_file_suffix

    @staticmethod
    def _normalize_seq_id(raw_id: str) -> str:
        seq_id = raw_id.strip().lstrip(">")
        if "/" in seq_id:
            seq_id = seq_id.split("/", 1)[0]
        return seq_id

    def _parse_hits_weights(self, hits_file: str) -> Dict[str, float]:
        seq_score_map: Dict[str, float] = {}
        if not os.path.exists(hits_file):
            return seq_score_map

        with open(hits_file, "r") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                cols = line.split()
                if len(cols) < 6:
                    continue
                seq_id = self._normalize_seq_id(cols[0])
                try:
                    score = float(cols[5])
                except ValueError:
                    continue
                # keep best score for duplicated sequence ids
                seq_score_map[seq_id] = max(score, seq_score_map.get(seq_id, float("-inf")))

        if not seq_score_map:
            return seq_score_map

        max_score = max(seq_score_map.values())
        min_score = min(seq_score_map.values())
        span = max(max_score - min_score, 1e-8)
        for seq_id, score in list(seq_score_map.items()):
            # normalized weight in [0.1, 1.0] to avoid zeroing long-tail homologs
            seq_score_map[seq_id] = 0.1 + 0.9 * ((score - min_score) / span)
        return seq_score_map

    def _build_alignment_weights(
        self, aa_seq_aln_file: str, seq_score_map: Dict[str, float]
    ) -> List[float]:
        alignment_dict = read_multi_fasta(aa_seq_aln_file)
        weights: List[float] = []
        for header in alignment_dict.keys():
            seq_id = self._normalize_seq_id(header)
            weights.append(seq_score_map.get(seq_id, 1.0))
        return weights

    def retrieve(
        self,
        tokenizer: AutoTokenizer,
        aa_seq_aln_file: Optional[str] = None,
        struc_seq_aln_file: Optional[str] = None,
        protein_name: Optional[str] = None,
        **kwargs,
    ) -> RetrievalOutput:
        if aa_seq_aln_file and protein_name and self.hits_dir:
            hits_file = os.path.join(self.hits_dir, f"{protein_name}{self.hits_file_suffix}")
            seq_score_map = self._parse_hits_weights(hits_file)
            if seq_score_map:
                self.aa_sequence_weights = self._build_alignment_weights(
                    aa_seq_aln_file, seq_score_map
                )

        return super().retrieve(
            tokenizer=tokenizer,
            aa_seq_aln_file=aa_seq_aln_file,
            struc_seq_aln_file=struc_seq_aln_file,
            protein_name=protein_name,
        )
