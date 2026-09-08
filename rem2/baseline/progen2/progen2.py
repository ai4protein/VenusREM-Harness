"""
Progen2 baseline adapter for VenusREM2.

Progen2 is an autoregressive protein language model. This adapter scores
single protein sequences using forward + reverse log-probability averaging,
then projects the per-residue log-probs to ESM2 vocabulary for VenusREM2
compatibility.

Reference: Nijkamp et al., "ProGen2: Exploring the Boundaries of Protein
Language Models", Cell Systems 2023.
"""

import json
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from rem2.baseline.progen2.modeling_progen import ProGenForCausalLM

# Progen2 tokenizer vocab layout (from tokenizer.json):
#   0: <|pad|>, 1: <|bos|>, 2: <|eos|>, 3: "1" (BOS marker), 4: "2" (EOS marker)
#   5-29: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
PROGEN2_FIRST_AA_TOKEN = 5
PROGEN2_LAST_AA_TOKEN = 29
PROGEN2_NUM_AA = PROGEN2_LAST_AA_TOKEN - PROGEN2_FIRST_AA_TOKEN + 1  # 25

# The amino acid letters in order of Progen2 token indices 5..29
PROGEN2_AA_ORDER = "ABCDEFGHIKLMNOPQRSTUVWXYZ"

# Standard amino acids for projection
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def load_progen2_model(
    model_name_or_path: str,
    device: torch.device,
    fp16: bool = False,
) -> Tuple:
    """
    Load Progen2 model + custom tokenizer + ESM2 tokenizer for vocab projection.

    Returns: (model, tokenizer, esm_tokenizer, model_context_len)
    - model: ProGenForCausalLM on device
    - tokenizer: tokenizers.Tokenizer (ByteLevel BPE from tokenizer.json in this dir)
    - esm_tokenizer: AutoTokenizer from "facebook/esm2_t6_8M_UR50D" for vocab projection
    - model_context_len: int from config['n_positions']
    """
    # Load model. Do not use revision="float16": that is not a real HF git
    # revision for hugohrban/progen2-{large,xlarge} (404 on hf-mirror).
    if fp16:
        model = ProGenForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        model = ProGenForCausalLM.from_pretrained(model_name_or_path)

    model = model.to(device).eval()

    # Get context length from model config
    model_context_len = getattr(model.config, "n_positions", 2048)

    # Load custom ByteLevel BPE tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    with open(tokenizer_path, "r") as f:
        tokenizer = Tokenizer.from_str(f.read())

    # Load ESM2 tokenizer for vocab projection
    esm_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t6_8M_UR50D", trust_remote_code=True
    )

    return model, tokenizer, esm_tokenizer, model_context_len


def _build_esm_projection_map(esm_tokenizer: AutoTokenizer):
    """
    Build mapping from Progen2 AA slice indices to ESM2 token IDs.

    Returns:
        progen2_idx_to_esm_id: dict mapping index within [5:30] slice (0-24)
            to ESM2 token ID, for standard amino acids only.
    """
    esm_vocab = esm_tokenizer.get_vocab()
    progen2_idx_to_esm_id = {}

    for aa in STANDARD_AA:
        # Index within the [5:30] slice
        progen2_idx = PROGEN2_AA_ORDER.index(aa)
        # ESM2 token ID
        if aa in esm_vocab:
            progen2_idx_to_esm_id[progen2_idx] = esm_vocab[aa]

    return progen2_idx_to_esm_id


def _score_direction(model, ids, device, fp16):
    """
    Score a single direction (forward or reverse).

    Args:
        model: ProGenForCausalLM
        ids: token IDs tensor [seq_len] (already on device)
        device: torch device
        fp16: whether to use fp16

    Returns:
        log_probs: [num_positions, 25] log-probabilities for AA tokens only
        num_positions: number of scored positions
    """
    input_ids = ids[:-1]
    targets = ids[1:]

    with torch.cuda.amp.autocast(enabled=fp16):
        logits = model(input_ids).logits  # [seq_len-1, vocab_size]

    # Remove terminal tokens (BOS="1"=3, EOS="2"=4) from targets
    bos_token, eos_token = 3, 4
    if targets[-1] in [bos_token, eos_token]:
        logits = logits[:-1, ...]
        targets = targets[:-1]

    # Slice to AA-only logits [5:30]
    logits = logits[:, PROGEN2_FIRST_AA_TOKEN:(PROGEN2_LAST_AA_TOKEN + 1)]

    # Compute log softmax
    log_probs = F.log_softmax(logits, dim=-1)

    return log_probs


@torch.no_grad()
def forward_progen2(
    model,
    tokenizer: Tokenizer,
    esm_tokenizer: AutoTokenizer,
    sequence: str,
    device: torch.device,
    model_context_len: int = 1024,
    fp16: bool = False,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Score a single protein sequence using Progen2 (autoregressive, forward+reverse averaged).

    Returns: torch.Tensor of shape [L, esm_vocab_size] -- log-probability logits
             projected to ESM2 vocab.

    Implementation:
    1. Add boundary tokens: "1" + sequence + "2"
    2. Tokenize with the ByteLevel tokenizer
    3. For BOTH forward and reverse directions:
       a. input_ids = ids[:-1], targets = ids[1:]
       b. Forward pass -> logits [seq_len, vocab_size]
       c. Take log_softmax of logits
       d. Slice to [:, 5:30] (AA indices only)
       e. Extract per-position log-probs for each residue
    4. Average forward + reverse log-probs -> [L, 25]
    5. Project 25 AA channels -> ESM2 vocab
    6. Return [L, esm_vocab_size] tensor

    For sequences longer than model_context_len:
    - Split into chunks (like ProteinGym does)
    - Average log-probs across overlapping positions
    """
    L = len(sequence)
    esm_vocab_size = esm_tokenizer.vocab_size

    if logger is not None:
        logger.debug(
            f"Progen2 forward: seq_len={L}, context_len={model_context_len}",
            protein=protein_name,
        )

    # Prepare the full protein string with boundary tokens
    prot = "1" + sequence + "2"

    # Split into chunks if necessary (matching ProteinGym logic)
    sequence_chunks = []
    if len(prot) < model_context_len:
        sequence_chunks = [prot]
    else:
        len_target_seq = len(prot)
        num_windows = 1 + int(len_target_seq / model_context_len)
        start = 0
        for window_index in range(1, num_windows + 1):
            sequence_chunks.append(prot[start:start + model_context_len])
            start += model_context_len

    # Accumulate log-probs for each position across chunks and directions
    # We will collect per-position log-probs for the raw sequence (without BOS/EOS)
    # and average them
    position_log_probs_sum = torch.zeros(L, PROGEN2_NUM_AA, device=device)
    position_counts = torch.zeros(L, device=device)

    for chunk in sequence_chunks:
        for direction_idx, p in enumerate([chunk, chunk[::-1]]):
            ids = torch.tensor(tokenizer.encode(p).ids, device=device)
            log_probs = _score_direction(model, ids, device, fp16)
            # log_probs shape: [num_positions, 25]

            num_pos = log_probs.shape[0]

            # Map chunk positions back to sequence positions
            # The chunk is a substring of "1" + sequence + "2"
            # After removing BOS/EOS from targets, we have pure AA positions
            if direction_idx == 0:
                # Forward direction
                # Find where this chunk starts in the full prot string
                chunk_start_in_prot = prot.index(chunk) if chunk in prot else 0
                # The scored positions correspond to targets (shifted by 1)
                # After removing terminal, scored positions are the AA characters
                # Position in original sequence = position_in_chunk + chunk_start_in_prot - 1
                # (subtract 1 for the "1" BOS character at prot[0])
                for pos_in_chunk in range(num_pos):
                    # The target at pos_in_chunk corresponds to character at
                    # chunk[pos_in_chunk + 1] (because input=ids[:-1], target=ids[1:])
                    char_pos_in_prot = chunk_start_in_prot + pos_in_chunk + 1
                    seq_pos = char_pos_in_prot - 1  # subtract 1 for "1" prefix
                    if 0 <= seq_pos < L:
                        position_log_probs_sum[seq_pos] += log_probs[pos_in_chunk]
                        position_counts[seq_pos] += 1.0
            else:
                # Reverse direction: p = chunk[::-1]
                # The reversed chunk's positions map back in reverse
                # reversed chunk characters: chunk[len-1], chunk[len-2], ..., chunk[0]
                # After tokenization, ids correspond to these characters
                # target positions map to reversed characters starting from index 1
                chunk_start_in_prot = prot.index(chunk) if chunk in prot else 0
                chunk_len = len(chunk)
                for pos_in_chunk in range(num_pos):
                    # In reversed string p, the target at position pos_in_chunk
                    # corresponds to character p[pos_in_chunk + 1]
                    # p[j] = chunk[chunk_len - 1 - j]
                    # So the original chunk position = chunk_len - 1 - (pos_in_chunk + 1)
                    #                                = chunk_len - 2 - pos_in_chunk
                    orig_chunk_pos = chunk_len - 2 - pos_in_chunk
                    char_pos_in_prot = chunk_start_in_prot + orig_chunk_pos
                    seq_pos = char_pos_in_prot - 1  # subtract 1 for "1" prefix
                    if 0 <= seq_pos < L:
                        position_log_probs_sum[seq_pos] += log_probs[pos_in_chunk]
                        position_counts[seq_pos] += 1.0

    # Average across directions and chunks
    position_counts = position_counts.clamp(min=1.0)
    avg_log_probs = position_log_probs_sum / position_counts.unsqueeze(-1)
    # avg_log_probs shape: [L, 25]

    # Project to ESM2 vocab
    progen2_idx_to_esm_id = _build_esm_projection_map(esm_tokenizer)
    projected = torch.full((L, esm_vocab_size), -1e9, device=device)
    for progen2_idx, esm_id in progen2_idx_to_esm_id.items():
        projected[:, esm_id] = avg_log_probs[:, progen2_idx]

    if logger is not None:
        logger.debug(
            f"Progen2 done: output shape {projected.shape}",
            protein=protein_name,
        )

    return projected
