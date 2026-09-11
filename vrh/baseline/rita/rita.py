"""
RITA baseline adapter for VenusREM2.

RITA is an autoregressive protein language model. This adapter scores single
protein sequences using forward + reverse log-probability averaging, then
projects the per-residue log-probs to ESM2 vocabulary for VenusREM2 compatibility.

Reference: Hesslow et al., "RITA: a Study on Scaling Up Generative Protein
Sequence Models", arXiv 2022.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Standard amino acids for projection
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def _get_aa_token_ids(tokenizer):
    """
    Discover which token IDs correspond to each standard amino acid letter
    in the RITA tokenizer.

    RITA uses a custom tokenizer where single amino acid characters are
    individual tokens. We encode each letter individually to find its token ID.

    Returns:
        aa_to_token_id: dict mapping AA letter -> token ID
    """
    aa_to_token_id = {}
    for aa in STANDARD_AA:
        ids = tokenizer.encode(aa, add_special_tokens=False)
        if len(ids) == 1:
            aa_to_token_id[aa] = ids[0]
        else:
            # Fallback: try vocab lookup
            vocab = tokenizer.get_vocab()
            if aa in vocab:
                aa_to_token_id[aa] = vocab[aa]
    return aa_to_token_id


def load_rita_model(
    model_name_or_path: str,
    device: torch.device,
) -> Tuple:
    """
    Load RITA model + tokenizer + ESM2 tokenizer for vocab projection.

    Args:
        model_name_or_path: HF model name like "lightonai/RITA_xl" or local path
        device: torch device

    Returns: (model, tokenizer, esm_tokenizer, model_context_len)
    - model: AutoModelForCausalLM on device
    - tokenizer: AutoTokenizer for RITA
    - esm_tokenizer: AutoTokenizer from "facebook/esm2_t6_8M_UR50D"
    - model_context_len: int (1024 for RITA models)
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
    except AttributeError as e:
        if "all_tied_weights_keys" in str(e):
            # RITA's custom model code is incompatible with newer transformers;
            # patch the missing attribute and retry.
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            model_cls = type(AutoModelForCausalLM.from_config(config, trust_remote_code=True))
            if not hasattr(model_cls, "all_tied_weights_keys"):
                model_cls.all_tied_weights_keys = {}
            model = model_cls.from_pretrained(
                model_name_or_path, config=config, trust_remote_code=True
            )
        else:
            raise
    model = model.float().to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    # RITA context length
    if hasattr(model.config, "n_positions"):
        model_context_len = model.config.n_positions
    elif hasattr(model.config, "max_position_embeddings"):
        model_context_len = model.config.max_position_embeddings
    else:
        model_context_len = 1024

    # Load ESM2 tokenizer for vocab projection
    esm_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t6_8M_UR50D", trust_remote_code=True
    )

    return model, tokenizer, esm_tokenizer, model_context_len


def _build_esm_projection_map(tokenizer, esm_tokenizer):
    """
    Build mapping from RITA AA token IDs to ESM2 token IDs.

    Returns:
        aa_token_id_to_esm_id: dict mapping RITA token ID -> ESM2 token ID
        aa_token_ids_sorted: sorted list of RITA AA token IDs
    """
    aa_to_token_id = _get_aa_token_ids(tokenizer)
    esm_vocab = esm_tokenizer.get_vocab()

    aa_token_id_to_esm_id = {}
    for aa, token_id in aa_to_token_id.items():
        if aa in esm_vocab:
            aa_token_id_to_esm_id[token_id] = esm_vocab[aa]

    aa_token_ids_sorted = sorted(aa_token_id_to_esm_id.keys())
    return aa_token_id_to_esm_id, aa_token_ids_sorted


def _score_direction(model, ids, device):
    """
    Score a single direction (forward or reverse).

    Args:
        model: AutoModelForCausalLM
        ids: token IDs tensor [1, seq_len] (already on device)
        device: torch device

    Returns:
        log_probs: [num_positions, vocab_size] full log-probabilities
    """
    input_ids = ids[:, :-1]
    # targets = ids[:, 1:]

    logits = model(input_ids).logits  # [1, seq_len-1, vocab_size]
    logits = logits.squeeze(0)  # [seq_len-1, vocab_size]

    # Compute log softmax over vocab dimension
    log_probs = F.log_softmax(logits, dim=-1)

    return log_probs


@torch.no_grad()
def forward_rita(
    model,
    tokenizer,
    esm_tokenizer,
    sequence: str,
    device: torch.device,
    model_context_len: int = 1024,
    scoring_mirror: bool = True,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Score a single protein sequence using RITA (autoregressive,
    forward+reverse averaged).

    Returns: torch.Tensor of shape [L, esm_vocab_size] -- log-probability
             logits projected to ESM2 vocab.

    Implementation:
    1. Tokenize the sequence
    2. Split into chunks if longer than model_context_len
    3. For BOTH forward and reverse directions:
       a. input_ids = ids[:-1], targets = ids[1:]
       b. Forward pass -> logits
       c. log_softmax -> per-position log-probs
       d. Extract AA-specific log-probs
    4. Average forward + reverse log-probs
    5. Project AA log-probs to ESM2 vocab indices -> [L, esm_vocab_size]
    """
    L = len(sequence)
    esm_vocab_size = esm_tokenizer.vocab_size

    if logger is not None:
        logger.debug(
            f"RITA forward: seq_len={L}, context_len={model_context_len}",
            protein=protein_name,
        )

    # Build projection map
    aa_token_id_to_esm_id, aa_token_ids_sorted = _build_esm_projection_map(
        tokenizer, esm_tokenizer
    )
    num_aa = len(aa_token_ids_sorted)
    aa_token_ids_tensor = torch.tensor(aa_token_ids_sorted, device=device)

    # Split into chunks if necessary (matching ProteinGym logic)
    sequence_chunks = []
    if len(sequence) < model_context_len:
        sequence_chunks = [sequence]
    else:
        len_target_seq = len(sequence)
        num_windows = 1 + int(len_target_seq / model_context_len)
        start = 0
        for _ in range(1, num_windows + 1):
            chunk = sequence[start:start + model_context_len]
            if chunk:  # avoid empty chunks
                sequence_chunks.append(chunk)
            start += model_context_len

    # Accumulate log-probs for each position
    position_log_probs_sum = torch.zeros(L, num_aa, device=device)
    position_counts = torch.zeros(L, device=device)

    for chunk in sequence_chunks:
        # Find where this chunk starts in the full sequence
        chunk_start_in_seq = sequence.index(chunk)

        directions = [chunk, chunk[::-1]] if scoring_mirror else [chunk]

        for direction_idx, p in enumerate(directions):
            ids = torch.tensor([tokenizer.encode(p)], device=device)
            log_probs = _score_direction(model, ids, device)
            # log_probs shape: [num_tokens-1, vocab_size]

            # Extract only the AA token columns
            aa_log_probs = log_probs[:, aa_token_ids_tensor]  # [num_tokens-1, num_aa]

            num_pos = aa_log_probs.shape[0]
            chunk_len = len(p)

            # Map each scored position back to the original sequence position.
            # The tokenizer encodes each AA character as one token, so
            # token index i in the input predicts character i+1 in the chunk.
            for pos_in_chunk in range(num_pos):
                if direction_idx == 0:
                    # Forward: target at pos_in_chunk predicts character
                    # at position pos_in_chunk + 1 in the chunk string.
                    char_idx_in_chunk = pos_in_chunk + 1
                    if char_idx_in_chunk >= chunk_len:
                        continue
                    seq_pos = chunk_start_in_seq + char_idx_in_chunk
                else:
                    # Reverse: p = chunk[::-1]
                    # target at pos_in_chunk predicts character at
                    # position pos_in_chunk + 1 in the reversed string.
                    rev_char_idx = pos_in_chunk + 1
                    if rev_char_idx >= chunk_len:
                        continue
                    # Map reversed index back to original chunk index
                    orig_chunk_idx = chunk_len - 1 - rev_char_idx
                    seq_pos = chunk_start_in_seq + orig_chunk_idx

                if 0 <= seq_pos < L:
                    position_log_probs_sum[seq_pos] += aa_log_probs[pos_in_chunk]
                    position_counts[seq_pos] += 1.0

    # Average across directions and chunks
    position_counts = position_counts.clamp(min=1.0)
    avg_log_probs = position_log_probs_sum / position_counts.unsqueeze(-1)
    # avg_log_probs shape: [L, num_aa]

    # Project to ESM2 vocab
    projected = torch.full((L, esm_vocab_size), -1e9, device=device)
    for i, token_id in enumerate(aa_token_ids_sorted):
        esm_id = aa_token_id_to_esm_id[token_id]
        projected[:, esm_id] = avg_log_probs[:, i]

    if logger is not None:
        logger.debug(
            f"RITA done: output shape {projected.shape}",
            protein=protein_name,
        )

    return projected
