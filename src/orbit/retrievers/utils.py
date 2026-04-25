from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer


def read_multi_fasta(file_path: str) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current_sequence = ""
    header = None
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences[header] = (
                        current_sequence.upper().replace("-", "<pad>").replace(".", "<pad>")
                    )
                header = line
                current_sequence = ""
            else:
                current_sequence += line

    if header is not None:
        sequences[header] = (
            current_sequence.upper().replace("-", "<pad>").replace(".", "<pad>")
        )
    return sequences


def infer_alignment_span(alignment_dict: Dict[str, str]) -> Tuple[int, int]:
    try:
        aln_start, aln_end = list(alignment_dict.keys())[0].split("/")[-1].split("-")
        return int(aln_start) - 1, int(aln_end)
    except Exception:
        first_seq = list(alignment_dict.values())[0]
        return 0, len(first_seq)


def build_alignment_ids(
    tokenizer: AutoTokenizer, alignment_dict: Dict[str, str]
) -> torch.Tensor:
    alignment_seqs = list(alignment_dict.values())
    tokenized_results = tokenizer(alignment_seqs, return_tensors="pt", padding=True)
    return tokenized_results["input_ids"][:, 1:-1]


def build_logit_matrix_from_alignment_ids(
    alignment_ids: torch.Tensor,
    vocab_size: int,
    sequence_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    num_rows, num_cols = alignment_ids.shape
    if num_rows == 0:
        raise ValueError("Alignment is empty")

    count_matrix = torch.zeros(num_cols, vocab_size, dtype=torch.float32)
    if sequence_weights is None:
        sequence_weights = [1.0] * num_rows
    if len(sequence_weights) != num_rows:
        raise ValueError(
            f"sequence_weights length mismatch: {len(sequence_weights)} != {num_rows}"
        )

    weights = torch.tensor(sequence_weights, dtype=torch.float32).reshape(num_rows, 1)
    for col in range(num_cols):
        column_tokens = alignment_ids[:, col]
        count_matrix[col].scatter_add_(0, column_tokens, weights[:, 0])

    count_matrix = count_matrix / count_matrix.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.log_softmax(count_matrix, dim=-1)
