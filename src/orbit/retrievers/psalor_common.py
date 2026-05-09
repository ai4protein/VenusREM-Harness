import os
from typing import Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.orbit.retrievers.utils import build_alignment_ids, infer_alignment_span, read_multi_fasta


def read_first_fasta_sequence(fasta_path: str) -> str:
    seq = ""
    with open(fasta_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    break
                continue
            seq += line
    return seq.upper()


def get_wildtype_token_ids(tokenizer: AutoTokenizer, sequence: str) -> torch.Tensor:
    tokenized = tokenizer([sequence], return_tensors="pt")
    return tokenized["input_ids"][0, 1:-1]


def compute_weighted_counts(
    alignment_ids: torch.Tensor, vocab_size: int, sequence_weights: Optional[List[float]] = None
) -> torch.Tensor:
    num_rows, num_cols = alignment_ids.shape
    if sequence_weights is None:
        sequence_weights = [1.0] * num_rows
    weights = torch.tensor(sequence_weights, dtype=torch.float32).reshape(num_rows)
    counts = torch.zeros(num_cols, vocab_size, dtype=torch.float32)
    for col in range(num_cols):
        counts[col].scatter_add_(0, alignment_ids[:, col], weights)
    return counts


def compute_sequence_dedup_weights(
    alignment_ids: torch.Tensor,
    pad_token_id: Optional[int],
    identity_threshold: float = 0.8,
    max_sequences_for_clustering: int = 2000,
) -> List[float]:
    """
    RSALOR-style redundancy down-weighting:
    weight_i = 1 / |cluster_i|, cluster defined by sequence identity threshold.
    """
    num_rows = alignment_ids.shape[0]
    if num_rows == 0:
        return []

    # Safety cap to avoid O(N^2) blowups on very large MSAs.
    if num_rows > max_sequences_for_clustering:
        return [1.0] * num_rows

    if pad_token_id is None:
        pad_mask = torch.zeros_like(alignment_ids, dtype=torch.bool)
    else:
        pad_mask = alignment_ids.eq(pad_token_id)

    valid_mask = ~pad_mask
    cluster_sizes = torch.ones(num_rows, dtype=torch.float32)
    for i in range(num_rows):
        row = alignment_ids[i].unsqueeze(0)
        pair_valid = valid_mask & valid_mask[i].unsqueeze(0)
        matches = alignment_ids.eq(row) & pair_valid
        denom = pair_valid.sum(dim=1).clamp_min(1)
        identity = matches.sum(dim=1).float() / denom.float()
        cluster_sizes[i] = torch.sum(identity >= identity_threshold).float().clamp_min(1.0)

    weights = 1.0 / cluster_sizes
    return weights.tolist()


def load_alignment_ids_and_span(
    tokenizer: AutoTokenizer, aa_seq_aln_file: str
) -> Tuple[torch.Tensor, int, int]:
    alignment_dict = read_multi_fasta(aa_seq_aln_file)
    aln_ids = build_alignment_ids(tokenizer, alignment_dict)
    aln_start, aln_end = infer_alignment_span(alignment_dict)
    return aln_ids, aln_start, aln_end


def load_global_plddt(protein_info_file: Optional[str], protein_name: Optional[str]) -> Optional[float]:
    if protein_info_file is None or protein_name is None or not os.path.exists(protein_info_file):
        return None
    info_df = pd.read_csv(protein_info_file)
    if "name" not in info_df.columns or "plddt" not in info_df.columns:
        return None
    match = info_df[info_df["name"] == protein_name]
    if len(match) == 0:
        return None
    try:
        return float(match["plddt"].iloc[0])
    except Exception:
        return None


def load_residue_plddt_weights(
    seq_len: int,
    protein_name: Optional[str],
    plddt_dir: Optional[str],
    protein_info_file: Optional[str],
) -> torch.Tensor:
    # default = no accessibility prior
    weights = torch.ones(seq_len, 1, dtype=torch.float32)

    if plddt_dir and protein_name:
        csv_path = os.path.join(plddt_dir, f"{protein_name}.csv")
        if os.path.exists(csv_path):
            residue_df = pd.read_csv(csv_path)
            numeric_cols = residue_df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                values = residue_df[numeric_cols[-1]].values[:seq_len]
                if len(values) > 0:
                    scaled = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
                    # Approximate exposure proxy: lower pLDDT -> larger retrieval weight.
                    scaled = (1.0 - scaled / 100.0).clamp(0.0, 1.0)
                    weights[: len(scaled)] = scaled
                    return weights

    global_plddt = load_global_plddt(protein_info_file, protein_name)
    if global_plddt is not None:
        scalar = max(0.0, min(1.0, 1.0 - global_plddt / 100.0))
        weights[:] = scalar

    return weights


MAX_ASA = {
    "A": 121.0,
    "R": 265.0,
    "N": 187.0,
    "D": 187.0,
    "C": 148.0,
    "Q": 214.0,
    "E": 214.0,
    "G": 97.0,
    "H": 216.0,
    "I": 195.0,
    "L": 191.0,
    "K": 230.0,
    "M": 203.0,
    "F": 228.0,
    "P": 154.0,
    "S": 143.0,
    "T": 163.0,
    "W": 264.0,
    "Y": 255.0,
    "V": 165.0,
}

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def resolve_pdb_file(
    protein_name: Optional[str], pdb_file: Optional[str] = None, pdb_dir: Optional[str] = None
) -> Optional[str]:
    if pdb_file and os.path.exists(pdb_file):
        return pdb_file
    if protein_name and pdb_dir:
        candidate = os.path.join(pdb_dir, f"{protein_name}.pdb")
        if os.path.exists(candidate):
            return candidate
    return None


def load_residue_rsa_weights_from_pdb(
    seq_len: int,
    protein_name: Optional[str],
    pdb_file: Optional[str] = None,
    pdb_dir: Optional[str] = None,
) -> Optional[torch.Tensor]:
    resolved = resolve_pdb_file(protein_name=protein_name, pdb_file=pdb_file, pdb_dir=pdb_dir)
    if resolved is None:
        return None
    try:
        traj = md.load(resolved)
        sasa = md.shrake_rupley(traj, mode="residue")[0]
        residues = list(traj.topology.residues)
        rsa_vals: List[float] = []
        for res, sasa_val in zip(residues, sasa):
            aa = THREE_TO_ONE.get(res.name.upper(), None)
            if aa is None or aa not in MAX_ASA:
                rsa_vals.append(0.0)
                continue
            # mdtraj outputs nm^2, convert to A^2 with *100.
            asa = float(sasa_val) * 100.0
            rsa = asa / MAX_ASA[aa]
            rsa_vals.append(float(np.clip(rsa, 0.0, 1.5)))

        weights = torch.zeros(seq_len, 1, dtype=torch.float32)
        n = min(seq_len, len(rsa_vals))
        if n > 0:
            weights[:n, 0] = torch.tensor(rsa_vals[:n], dtype=torch.float32)
        return weights
    except Exception:
        return None


def load_residue_plddt_from_pdb(
    seq_len: int,
    protein_name: Optional[str],
    pdb_file: Optional[str] = None,
    pdb_dir: Optional[str] = None,
) -> Optional[torch.Tensor]:
    resolved = resolve_pdb_file(protein_name=protein_name, pdb_file=pdb_file, pdb_dir=pdb_dir)
    if resolved is None:
        return None
    try:
        from Bio.PDB import PDBParser
        parser_pdb = PDBParser(QUIET=True)
        structure = parser_pdb.get_structure("prot", resolved)
        model = list(structure.get_models())[0]
        plddt_vals: List[float] = []
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                ca = residue["CA"] if "CA" in residue else list(residue.get_atoms())[0]
                plddt_vals.append(float(ca.get_bfactor()))
        weights = torch.zeros(seq_len, 1, dtype=torch.float32)
        n = min(seq_len, len(plddt_vals))
        if n > 0:
            weights[:n, 0] = torch.tensor(plddt_vals[:n], dtype=torch.float32) / 100.0
        return weights.clamp(0.0, 1.0)
    except Exception:
        return None


def load_residue_accessibility_weights(
    seq_len: int,
    protein_name: Optional[str],
    rsa_mode: str = "auto",
    pdb_file: Optional[str] = None,
    pdb_dir: Optional[str] = None,
    plddt_dir: Optional[str] = None,
    protein_info_file: Optional[str] = None,
) -> torch.Tensor:
    if rsa_mode in ("auto", "rsa"):
        rsa_weights = load_residue_rsa_weights_from_pdb(
            seq_len=seq_len, protein_name=protein_name, pdb_file=pdb_file, pdb_dir=pdb_dir
        )
        if rsa_weights is not None:
            return rsa_weights
        if rsa_mode == "rsa":
            return torch.ones(seq_len, 1, dtype=torch.float32)

    return load_residue_plddt_weights(
        seq_len=seq_len,
        protein_name=protein_name,
        plddt_dir=plddt_dir,
        protein_info_file=protein_info_file,
    )


def expand_aln_logits_to_full(
    aln_logits: torch.Tensor, seq_len: int, vocab_size: int, aln_start: int, aln_end: int
) -> torch.Tensor:
    full_logits = torch.zeros(seq_len, vocab_size, dtype=torch.float32)
    upper = min(aln_end, seq_len)
    block_len = max(0, upper - aln_start)
    if block_len > 0:
        full_logits[aln_start:upper] = aln_logits[:block_len]
    return full_logits


def build_position_mask(seq_len: int, start: int, end: int) -> torch.Tensor:
    mask = torch.zeros(seq_len, 1, dtype=torch.float32)
    lower = max(0, start)
    upper = min(seq_len, end)
    if upper > lower:
        mask[lower:upper] = 1.0
    return mask
