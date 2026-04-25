import os
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer

from src.orbit.retrievers.base import BaseRetriever
from src.orbit.retrievers.msa import MSARetriever
from src.orbit.retrievers.psalor_common import (
    build_position_mask,
    expand_aln_logits_to_full,
    get_wildtype_token_ids,
    load_residue_accessibility_weights,
    read_first_fasta_sequence,
)
from src.orbit.types import RetrievalOutput


class PSALORVariantRetriever(BaseRetriever):
    """
    VenusREM-adapted second layer:
    - Start from MSA logits.
    - Reweight by homolog hit quality.
    - Apply structure-confidence gate (pLDDT proxy).
    """

    def __init__(
        self,
        hits_dir: Optional[str] = None,
        hits_file_suffix: str = "_tblout.txt",
        plddt_dir: Optional[str] = None,
        protein_info_file: Optional[str] = None,
        psalor_mix: float = 0.5,
        rsa_mode: str = "auto",
        pdb_dir: Optional[str] = None,
        center_wt_lor: bool = True,
    ) -> None:
        self.hits_dir = hits_dir
        self.hits_file_suffix = hits_file_suffix
        self.plddt_dir = plddt_dir
        self.protein_info_file = protein_info_file
        self.psalor_mix = psalor_mix
        self.rsa_mode = rsa_mode
        self.pdb_dir = pdb_dir
        self.center_wt_lor = center_wt_lor
        self.base_msa = MSARetriever()

    @staticmethod
    def _parse_hits_quality(hits_file: str) -> float:
        if not os.path.exists(hits_file):
            return 1.0
        values = []
        with open(hits_file, "r") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                cols = line.split()
                if len(cols) < 6:
                    continue
                try:
                    values.append(float(cols[5]))
                except ValueError:
                    continue
        if not values:
            return 1.0
        vmin, vmax = min(values), max(values)
        if abs(vmax - vmin) < 1e-8:
            return 1.0
        normalized = [(v - vmin) / (vmax - vmin) for v in values]
        # keep in [0.5, 1.5] to avoid destabilizing logits scale
        return 0.5 + float(sum(normalized) / len(normalized))

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
                layer2_name="psalor_variant",
                metadata={"psalor_variant_skipped": True, "protein_name": protein_name},
            )

        primary_output = self.base_msa.retrieve(
            tokenizer=tokenizer,
            aa_seq_aln_file=aa_seq_aln_file,
            struc_seq_aln_file=struc_seq_aln_file,
            protein_name=protein_name,
        )
        if primary_output.aa_seq_aln_logits is None:
            return RetrievalOutput(
                layer2_name="psalor_variant",
                metadata={"psalor_variant_skipped": True, "protein_name": protein_name},
            )

        sequence = read_first_fasta_sequence(residue_fasta)
        seq_len = len(sequence)
        aa_start = primary_output.aa_aln_start or 0
        aa_end = primary_output.aa_aln_end or seq_len
        aa_full = expand_aln_logits_to_full(
            primary_output.aa_seq_aln_logits,
            seq_len,
            tokenizer.vocab_size,
            aa_start,
            aa_end,
        )

        struc_full = torch.zeros_like(aa_full)
        struc_mask = torch.zeros(seq_len, 1, dtype=torch.float32)
        structure_len = 0
        if primary_output.struc_seq_aln_logits is not None:
            n = min(seq_len, primary_output.struc_seq_aln_logits.shape[0])
            struc_full[:n] = primary_output.struc_seq_aln_logits[:n]
            struc_mask[:n] = 1.0
            structure_len = n

        # Use structure-aware mixing only where structure logits exist.
        # Outside structure coverage, keep AA-only signal to avoid attenuation.
        mix = float(min(max(self.psalor_mix, 0.0), 1.0))
        base_layer2 = aa_full.clone()
        if structure_len > 0:
            base_layer2[:structure_len] = (
                (1.0 - mix) * aa_full[:structure_len] + mix * struc_full[:structure_len]
            )

        hits_quality = 1.0
        if protein_name and self.hits_dir:
            hits_file = os.path.join(self.hits_dir, f"{protein_name}{self.hits_file_suffix}")
            hits_quality = self._parse_hits_quality(hits_file)
        base_layer2 = base_layer2 * hits_quality

        if self.center_wt_lor:
            wt_token_ids = get_wildtype_token_ids(tokenizer, sequence)
            idx = torch.arange(seq_len, dtype=torch.long)
            wt_col = base_layer2[idx, wt_token_ids].unsqueeze(-1)
            base_layer2 = base_layer2 - wt_col

        position_weights = load_residue_accessibility_weights(
            seq_len=seq_len,
            protein_name=protein_name,
            rsa_mode=self.rsa_mode,
            pdb_file=pdb_file,
            pdb_dir=self.pdb_dir,
            plddt_dir=self.plddt_dir,
            protein_info_file=self.protein_info_file,
        )
        aa_mask = build_position_mask(seq_len=seq_len, start=aa_start, end=aa_end)
        layer2_mask = torch.maximum(aa_mask, struc_mask)
        position_weights = position_weights * layer2_mask

        return RetrievalOutput(
            layer2_logits=base_layer2,
            layer2_position_weights=position_weights,
            layer2_mask=layer2_mask,
            layer2_name="psalor_variant",
            metadata={
                "protein_name": protein_name,
                "psalor_variant_hits_quality": hits_quality,
                "psalor_mix": self.psalor_mix,
                "psalor_variant_rsa_mode": self.rsa_mode,
                "psalor_variant_structure_len": structure_len,
                "psalor_variant_center_wt_lor": self.center_wt_lor,
            },
        )
