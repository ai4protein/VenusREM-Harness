"""
Progen3 baseline adapter for VenusREM-Orbit.

Progen3 is an autoregressive protein language model with a Mixture-of-Experts
(MoE) architecture. This adapter scores single protein sequences using
forward + reverse log-probability averaging, then projects the per-residue
log-probs to ESM2 vocabulary for Orbit compatibility.

Progen3 uses a custom model class (ProGen3ForCausalLM) from the progen3
package, which requires megablocks and flash_attn dependencies. The model
is loaded via HuggingFace's from_pretrained mechanism but with a custom
PreTrainedModel subclass.

Reference: Profluent-Bio / ProGen3 (https://huggingface.co/Profluent-Bio)
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoTokenizer

# Progen3 tokenizer vocab layout (from tokenizer.json):
#   0: <pad>, 1: <bos>, 2: <eos>, 3: <bos_glm>, 4: <eos_span>, 5: <mask>
#   6: "1" (N-terminal marker), 7: "2" (C-terminal marker)
#   8-33: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
PROGEN3_BOS_TOKEN = 1
PROGEN3_EOS_TOKEN = 2
PROGEN3_TERM1_TOKEN = 6  # "1" (N-terminal)
PROGEN3_TERM2_TOKEN = 7  # "2" (C-terminal)
PROGEN3_FIRST_AA_TOKEN = 8
PROGEN3_LAST_AA_TOKEN = 33
PROGEN3_NUM_AA = PROGEN3_LAST_AA_TOKEN - PROGEN3_FIRST_AA_TOKEN + 1  # 26

# The amino acid letters in order of Progen3 token indices 8..33
PROGEN3_AA_ORDER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Standard amino acids for projection
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def load_progen3_model(
    model_name_or_path: str,
    device: torch.device,
    fp16: bool = False,
) -> Tuple:
    """
    Load Progen3 model + custom tokenizer + ESM2 tokenizer for vocab projection.

    Progen3 uses a custom ProGen3ForCausalLM model class, which is loaded via
    HuggingFace's from_pretrained with the model class from the progen3 package.
    If the progen3 package is not available, it falls back to HuggingFace's
    AutoModelForCausalLM with trust_remote_code=True.

    Returns: (model, tokenizer, esm_tokenizer, model_context_len)
    - model: ProGen3ForCausalLM on device
    - tokenizer: tokenizers.Tokenizer (ByteLevel BPE from tokenizer.json)
    - esm_tokenizer: AutoTokenizer from "facebook/esm2_t6_8M_UR50D" for vocab projection
    - model_context_len: int from config.max_position_embeddings
    """
    # Try to import the native progen3 model class; fall back to AutoModel
    try:
        from progen3.modeling import ProGen3ForCausalLM

        if fp16:
            model = ProGen3ForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        else:
            model = ProGen3ForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
    except ImportError:
        # Fall back to AutoModel with trust_remote_code
        from transformers import AutoModelForCausalLM

        dtype = torch.float16 if fp16 else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    model = model.to(device).eval()

    # Get context length from model config
    model_context_len = getattr(model.config, "max_position_embeddings", 65536)

    # Load custom ByteLevel BPE tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load ESM2 tokenizer for vocab projection
    esm_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t6_8M_UR50D", trust_remote_code=True
    )

    return model, tokenizer, esm_tokenizer, model_context_len


def _build_esm_projection_map(esm_tokenizer: AutoTokenizer):
    """
    Build mapping from Progen3 AA slice indices to ESM2 token IDs.

    Returns:
        progen3_idx_to_esm_id: dict mapping index within [8:34] slice (0-25)
            to ESM2 token ID, for standard amino acids only.
    """
    esm_vocab = esm_tokenizer.get_vocab()
    progen3_idx_to_esm_id = {}

    for aa in STANDARD_AA:
        # Index within the [8:34] slice
        progen3_idx = PROGEN3_AA_ORDER.index(aa)
        # ESM2 token ID
        if aa in esm_vocab:
            progen3_idx_to_esm_id[progen3_idx] = esm_vocab[aa]

    return progen3_idx_to_esm_id


def _score_direction(model, tokenizer, sequence, device, fp16, reverse=False):
    """
    Score a single direction (forward or reverse) using Progen3.

    Progen3 CLM format: <bos> 1 SEQUENCE 2 <eos>  (forward)
                        <bos> 2 ECNEUQES 1 <eos>  (reverse)

    The model forward pass requires input_ids, position_ids, and sequence_ids.

    Args:
        model: ProGen3ForCausalLM
        tokenizer: tokenizers.Tokenizer
        sequence: raw amino acid sequence (e.g. "MVLSPADK...")
        device: torch device
        fp16: whether to use fp16
        reverse: if True, reverse the sequence

    Returns:
        log_probs: [num_aa_positions, 26] log-probabilities for AA tokens only
        aa_positions: list of int, original sequence positions (0-indexed) for each row
    """
    # Build the CLM string
    if reverse:
        clm_str = "1" + sequence[::-1] + "2"
        clm_str = clm_str[::-1]  # becomes "2" + sequence + "1" reversed
        # Actually the progen3 batch_preparer does:
        #   sequence = "1" + sequence + "2"
        #   if reverse: sequence = sequence[::-1]
        # So: "2" + reversed_seq + "1"
        seq_with_terms = "1" + sequence + "2"
        seq_with_terms = seq_with_terms[::-1]
    else:
        seq_with_terms = "1" + sequence + "2"

    # Tokenize with <bos> and <eos> wrapper
    token_str = "<bos>" + seq_with_terms + "<eos>"
    ids = torch.tensor(tokenizer.encode(token_str).ids, device=device)

    # Build position_ids and sequence_ids
    seq_len = ids.shape[0]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    sequence_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    input_ids = ids.unsqueeze(0)  # [1, seq_len]

    with torch.cuda.amp.autocast(enabled=fp16):
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_ids=sequence_ids,
            return_dict=True,
        )
        logits = output.logits[0]  # [seq_len, vocab_size]

    # For autoregressive scoring: input[:-1] predicts target[1:]
    # The token layout is:
    #   <bos> T1 AA1 AA2 ... AAn T2 <eos>
    #   where T1="1", T2="2" (forward) or T1="2", T2="1" (reverse)
    #
    # Shift for autoregressive: logits[i] predicts token[i+1]
    # We want log-probs for AA positions only (not special tokens)
    #
    # Token indices:
    #   0: <bos>
    #   1: T1 ("1" or "2")
    #   2: first AA
    #   ...
    #   L+1: last AA
    #   L+2: T2 ("2" or "1")
    #   L+3: <eos>
    #
    # For autoregressive, logits[i] predicts token[i+1]:
    #   logits[1] predicts token[2] = first AA
    #   logits[2] predicts token[3] = second AA
    #   ...
    #   logits[L] predicts token[L+1] = last AA
    #
    # So we need logits[1:L+1] for the L amino acid positions.

    L = len(sequence)

    # Slice logits for AA predictions: logits[1:L+1]
    aa_logits = logits[1:L + 1, :]  # [L, vocab_size]

    # Slice to AA-only logits [8:34]
    aa_logits = aa_logits[:, PROGEN3_FIRST_AA_TOKEN:(PROGEN3_LAST_AA_TOKEN + 1)]  # [L, 26]

    # Compute log softmax
    log_probs = F.log_softmax(aa_logits, dim=-1)

    # Map positions back to original sequence positions
    if reverse:
        # In reverse direction, the tokens are in reverse order
        # Position 0 in reversed = position L-1 in original
        aa_positions = list(range(L - 1, -1, -1))
    else:
        aa_positions = list(range(L))

    return log_probs, aa_positions


@torch.no_grad()
def forward_progen3(
    model,
    tokenizer: Tokenizer,
    esm_tokenizer: AutoTokenizer,
    sequence: str,
    device: torch.device,
    model_context_len: int = 65536,
    scoring_mirror: bool = True,
    fp16: bool = False,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Score a single protein sequence using Progen3 (autoregressive, forward+reverse averaged).

    Returns: torch.Tensor of shape [L, esm_vocab_size] -- log-probability logits
             projected to ESM2 vocab.

    Implementation:
    1. Wrap sequence with terminal markers and special tokens:
       Forward: <bos>1{sequence}2<eos>
       Reverse: <bos>2{reversed_sequence}1<eos>
    2. For BOTH forward and reverse directions:
       a. Forward pass with input_ids, position_ids, sequence_ids
       b. Extract logits for AA positions
       c. Slice to [:, 8:34] (AA indices only)
       d. Compute log_softmax
    3. Average forward + reverse log-probs -> [L, 26]
    4. Project 26 AA channels -> ESM2 vocab
    5. Return [L, esm_vocab_size] tensor

    For sequences longer than model_context_len:
    - Split into chunks and average log-probs across overlapping positions
    """
    L = len(sequence)
    esm_vocab_size = esm_tokenizer.vocab_size

    if logger is not None:
        logger.debug(
            f"Progen3 forward: seq_len={L}, context_len={model_context_len}",
            protein=protein_name,
        )

    # Progen3 has a very large context window (65536), so most sequences fit
    # For safety, we still handle chunking
    # The full token sequence is: <bos> T1 AA... T2 <eos> = L + 4 tokens
    full_token_len = L + 4

    if full_token_len <= model_context_len:
        # Single pass: no chunking needed
        position_log_probs_sum = torch.zeros(L, PROGEN3_NUM_AA, device=device)
        position_counts = torch.zeros(L, device=device)

        directions = [False]  # forward
        if scoring_mirror:
            directions.append(True)  # reverse

        for reverse in directions:
            log_probs, aa_positions = _score_direction(
                model, tokenizer, sequence, device, fp16, reverse=reverse
            )
            for i, seq_pos in enumerate(aa_positions):
                if 0 <= seq_pos < L:
                    position_log_probs_sum[seq_pos] += log_probs[i]
                    position_counts[seq_pos] += 1.0
    else:
        # Chunking for very long sequences
        # Split the raw sequence into overlapping chunks
        # Each chunk needs L_chunk + 4 <= model_context_len
        max_chunk_aa = model_context_len - 4
        position_log_probs_sum = torch.zeros(L, PROGEN3_NUM_AA, device=device)
        position_counts = torch.zeros(L, device=device)

        # Create chunks with overlap
        stride = max(1, max_chunk_aa // 2)  # 50% overlap
        chunk_starts = list(range(0, L, stride))
        # Ensure we cover the end
        if chunk_starts[-1] + max_chunk_aa < L:
            chunk_starts.append(max(0, L - max_chunk_aa))

        for start in chunk_starts:
            end = min(start + max_chunk_aa, L)
            chunk_seq = sequence[start:end]
            chunk_len = end - start

            directions = [False]
            if scoring_mirror:
                directions.append(True)

            for reverse in directions:
                log_probs, aa_positions = _score_direction(
                    model, tokenizer, chunk_seq, device, fp16, reverse=reverse
                )
                for i, chunk_pos in enumerate(aa_positions):
                    seq_pos = start + chunk_pos
                    if 0 <= seq_pos < L:
                        position_log_probs_sum[seq_pos] += log_probs[i]
                        position_counts[seq_pos] += 1.0

    # Average across directions and chunks
    position_counts = position_counts.clamp(min=1.0)
    avg_log_probs = position_log_probs_sum / position_counts.unsqueeze(-1)
    # avg_log_probs shape: [L, 26]

    # Project to ESM2 vocab
    progen3_idx_to_esm_id = _build_esm_projection_map(esm_tokenizer)
    projected = torch.full((L, esm_vocab_size), -1e9, device=device)
    for progen3_idx, esm_id in progen3_idx_to_esm_id.items():
        projected[:, esm_id] = avg_log_probs[:, progen3_idx]

    if logger is not None:
        logger.debug(
            f"Progen3 done: output shape {projected.shape}",
            protein=protein_name,
        )

    return projected
