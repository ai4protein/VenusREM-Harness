"""
Tranception baseline adapter for VenusREM-Orbit.

Tranception is an autoregressive protein language model based on GPT-2,
supporting bidirectional scoring (L->R and R->L) and optional MSA retrieval.
This adapter scores single protein sequences using forward + reverse
log-probability averaging, then projects per-residue log-probs to ESM2
vocabulary for Orbit compatibility.

Reference: Notin et al., "Tranception: Protein Fitness Prediction with
Autoregressive Transformers and Inference-time Retrieval", ICML 2022.
"""

import json
import os
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from venus_orbit.baseline.tranception.model import TranceptionConfig, TranceptionLMHeadModel
from venus_orbit.baseline.tranception.model.utils import scoring_utils

# Tranception tokenizer vocab layout (from Basic_tokenizer):
#   0: [UNK], 1: [CLS], 2: [SEP], 3: [PAD], 4: [MASK]
#   5-24: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
TRANCEPTION_FIRST_AA_TOKEN = 5
TRANCEPTION_LAST_AA_TOKEN = 24
TRANCEPTION_NUM_AA = TRANCEPTION_LAST_AA_TOKEN - TRANCEPTION_FIRST_AA_TOKEN + 1  # 20

# The amino acid letters in order of Tranception token indices 5..24
TRANCEPTION_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# Standard amino acids for projection
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def load_tranception_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple:
    """
    Load Tranception model from a local checkpoint directory or HuggingFace Hub.

    Supports both:
      - Local path: ``/path/to/Tranception_Small``
      - HF hub name: ``OATML-Markslab/Tranception_Small``

    Returns: (model, tokenizer, esm_tokenizer, model_context_len)
    - model: TranceptionLMHeadModel on device
    - tokenizer: PreTrainedTokenizerFast (GPT2-style character-level)
    - esm_tokenizer: AutoTokenizer from ESM2 for vocab projection
    - model_context_len: from config.n_positions (typically 1024)
    """
    # Load tokenizer (bundled in this repo, not in HF checkpoint)
    tokenizer_path = os.path.join(
        os.path.dirname(__file__), "model", "utils", "tokenizers", "Basic_tokenizer"
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )

    # Load config — local path or HF hub
    if os.path.isdir(checkpoint_path):
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    else:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(checkpoint_path, "config.json")
        with open(downloaded, "r") as f:
            config_dict = json.load(f)

    config = TranceptionConfig(**config_dict)
    config.attention_mode = "tranception"
    config.position_embedding = "grouped_alibi"
    config.tokenizer = tokenizer
    config.scoring_window = "optimal"
    config.retrieval_aggregation_mode = None

    # Load model weights — local path or HF hub
    model = TranceptionLMHeadModel.from_pretrained(
        pretrained_model_name_or_path=checkpoint_path, config=config
    )
    model = model.to(device).eval()

    model_context_len = config.n_positions if hasattr(config, "n_positions") else 1024

    esm_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t6_8M_UR50D", trust_remote_code=True
    )

    return model, tokenizer, esm_tokenizer, model_context_len


def _build_esm_projection_map(esm_tokenizer: AutoTokenizer):
    """
    Build mapping from Tranception AA token indices to ESM2 token IDs.

    Returns:
        tranception_aa_idx_to_esm_id: dict mapping index within [5:25] slice
            (0-19) to ESM2 token ID.
    """
    esm_vocab = esm_tokenizer.get_vocab()
    tranception_aa_idx_to_esm_id = {}

    for aa in STANDARD_AA:
        # Index within the [5:25] slice (0-based)
        tranception_idx = TRANCEPTION_AA_ORDER.index(aa)
        if aa in esm_vocab:
            tranception_aa_idx_to_esm_id[tranception_idx] = esm_vocab[aa]

    return tranception_aa_idx_to_esm_id


def _score_single_direction(
    model,
    tokenizer,
    sequence: str,
    device: torch.device,
    model_context_len: int,
) -> torch.Tensor:
    """
    Score a sequence in one direction (forward).

    The Tranception tokenizer wraps the sequence with [CLS] and [SEP]:
        [CLS] A1 A2 ... AL [SEP]
    Token indices: 0    1  2  ... L   L+1

    In autoregressive mode:
        input  = tokens[:-1]  -> [CLS] A1 A2 ... AL
        target = tokens[1:]   -> A1 A2 ... AL [SEP]
    So logits[i] predicts tokens[i+1], meaning:
        logits[0] predicts A1 (position 0 in sequence)
        logits[1] predicts A2 (position 1 in sequence)
        ...
        logits[L-1] predicts AL (position L-1 in sequence)
        logits[L] predicts [SEP] (not needed)

    We want per-position log-probs for positions 0..L-1, which are logits[0..L-1].

    For sequences longer than model_context_len, we use a sliding window approach.

    Args:
        model: TranceptionLMHeadModel
        tokenizer: Tranception tokenizer
        sequence: amino acid sequence string
        device: torch device
        model_context_len: max context length

    Returns:
        log_probs: [L, 20] log-probabilities for AA tokens at each position
    """
    L = len(sequence)

    # Tokenize: adds [CLS] prefix and [SEP] suffix
    # The space-separated AA sequence is needed for the tokenizer
    spaced_seq = " ".join(list(sequence))
    encoded = tokenizer(spaced_seq, add_special_tokens=True, return_tensors="pt")
    input_ids = encoded["input_ids"].squeeze(0)  # [seq_len_with_special]
    # input_ids: [CLS] A1 A2 ... AL [SEP]
    # Length should be L + 2

    total_len = input_ids.shape[0]

    # For sequences that fit in context
    # model_context_len is for the model input (without the last token due to shifting)
    # The full tokenized sequence (with special tokens) can be at most model_context_len + 2
    # But let's use the actual context: n_ctx from config
    max_input_len = model_context_len  # this is n_positions from config

    if total_len <= max_input_len:
        # Entire sequence fits
        input_for_model = input_ids[:-1].unsqueeze(0).to(device)  # remove [SEP], add batch
        attention_mask = torch.ones_like(input_for_model)
        outputs = model(input_ids=input_for_model, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits.squeeze(0)  # [seq_len-1, vocab_size]
        # logits[0] predicts A1, logits[1] predicts A2, ..., logits[L-1] predicts AL
        # We want positions 0..L-1
        aa_logits = logits[:L, TRANCEPTION_FIRST_AA_TOKEN:(TRANCEPTION_LAST_AA_TOKEN + 1)]
        log_probs = F.log_softmax(aa_logits, dim=-1)  # [L, 20]
        return log_probs
    else:
        # Sliding window for long sequences
        # We process overlapping windows and average
        position_log_probs_sum = torch.zeros(L, TRANCEPTION_NUM_AA, device=device)
        position_counts = torch.zeros(L, device=device)

        # Window size for the actual input (before shift)
        # input to model is tokens[:-1], so window_size tokens produce window_size logits
        window_size = max_input_len
        stride = window_size // 2  # overlap by half

        # We slide over the full token sequence (with special tokens)
        start = 0
        while start < total_len - 1:
            end = min(start + window_size, total_len)
            # Take window from input_ids, use all but last as input
            window_ids = input_ids[start:end]
            if len(window_ids) < 2:
                break

            input_for_model = window_ids[:-1].unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_for_model)

            outputs = model(input_ids=input_for_model, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits.squeeze(0)  # [window_len-1, vocab_size]

            # Map window positions to sequence positions
            # window_ids[0] is at position start in input_ids
            # logits[i] predicts window_ids[i+1] = input_ids[start + i + 1]
            # input_ids[j] corresponds to sequence position j - 1 (because of [CLS] at index 0)
            for i in range(logits.shape[0]):
                target_token_idx = start + i + 1  # index in input_ids
                seq_pos = target_token_idx - 1  # subtract 1 for [CLS]
                if 0 <= seq_pos < L:
                    aa_logit = logits[i, TRANCEPTION_FIRST_AA_TOKEN:(TRANCEPTION_LAST_AA_TOKEN + 1)]
                    lp = F.log_softmax(aa_logit, dim=-1)
                    position_log_probs_sum[seq_pos] += lp
                    position_counts[seq_pos] += 1.0

            if end >= total_len:
                break
            start += stride

        position_counts = position_counts.clamp(min=1.0)
        return position_log_probs_sum / position_counts.unsqueeze(-1)


@torch.no_grad()
def forward_tranception(
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
    Score a single protein sequence using Tranception (L->R + R->L averaged).
    Projects per-position log-probs to ESM2 vocab.

    Returns: torch.Tensor of shape [L, esm_vocab_size] -- log-probability logits
             projected to ESM2 vocab.

    Args:
        model: TranceptionLMHeadModel
        tokenizer: Tranception's Basic_tokenizer
        esm_tokenizer: ESM2 tokenizer for vocab projection
        sequence: amino acid sequence string
        device: torch device
        model_context_len: max context length for the model
        scoring_mirror: whether to average L->R and R->L scores
        logger: optional logger
        protein_name: optional protein name for logging
    """
    L = len(sequence)
    esm_vocab_size = esm_tokenizer.vocab_size

    if logger is not None:
        logger.debug(
            f"Tranception forward: seq_len={L}, context_len={model_context_len}",
            protein=protein_name,
        )

    # Score forward direction (L -> R)
    log_probs_fwd = _score_single_direction(
        model, tokenizer, sequence, device, model_context_len
    )  # [L, 20]

    if scoring_mirror:
        # Score reverse direction (R -> L)
        reversed_sequence = sequence[::-1]
        log_probs_rev_reversed = _score_single_direction(
            model, tokenizer, reversed_sequence, device, model_context_len
        )  # [L, 20]
        # Flip back to original position order
        log_probs_rev = torch.flip(log_probs_rev_reversed, dims=[0])
        # Average forward and reverse
        avg_log_probs = (log_probs_fwd + log_probs_rev) / 2.0
    else:
        avg_log_probs = log_probs_fwd

    # Project to ESM2 vocab: map 20 AA channels -> ESM2 vocab
    tranception_to_esm = _build_esm_projection_map(esm_tokenizer)
    projected = torch.full((L, esm_vocab_size), -1e9, device=device)
    for tranception_idx, esm_id in tranception_to_esm.items():
        projected[:, esm_id] = avg_log_probs[:, tranception_idx]

    if logger is not None:
        logger.debug(
            f"Tranception done: output shape {projected.shape}",
            protein=protein_name,
        )

    return projected


@torch.no_grad()
def score_tranception_native(
    model,
    tokenizer,
    sequence: str,
    mutant_df: pd.DataFrame,
    device: torch.device,
    model_context_len: int = 1024,
    scoring_mirror: bool = True,
    batch_size_inference: int = 10,
    logger=None,
    protein_name: Optional[str] = None,
) -> List[float]:
    """
    Score mutations using Tranception's native score_mutants() method.

    This computes delta log-likelihood by running each mutated sequence through
    the model and comparing total log-likelihood vs the wild-type sequence.
    Captures downstream context effects that wt-marginals misses.

    Returns: list of float delta scores aligned with mutant_df rows.
    """
    if logger is not None:
        logger.info(
            f"Native Tranception scoring: {len(mutant_df)} mutants, mirror={scoring_mirror}",
            protein=protein_name,
        )

    df = mutant_df[["mutant"]].copy()
    df["mutated_sequence"] = df["mutant"].apply(
        lambda x: scoring_utils.get_mutated_sequence(sequence, x)
    )

    scores_df = model.score_mutants(
        DMS_data=df,
        target_seq=sequence,
        scoring_mirror=scoring_mirror,
        batch_size_inference=batch_size_inference,
        num_workers=0,
    )

    seq_to_score = dict(
        zip(scores_df["mutated_sequence"], scores_df["avg_score"])
    )
    native_scores = [
        seq_to_score.get(seq, 0.0) for seq in df["mutated_sequence"]
    ]

    if logger is not None:
        n_found = sum(1 for s in native_scores if s != 0.0)
        logger.info(
            f"Native scoring complete: {n_found}/{len(native_scores)} scored",
            protein=protein_name,
        )

    return native_scores
