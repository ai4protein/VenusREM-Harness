"""
ESM3 / ESM-C baseline adapter for VenusREM-Orbit.

ESM3 and ESM-C are EvolutionaryScale's protein language models.
ESM-C is a sequence-only masked language model (available in 300M/600M/6B).
ESM3 is a multi-track generative model with sequence, structure, and
functional annotation inputs (available as esm3_sm_open_v1).

Both use the same 33-token sequence vocabulary as ESM2, so no vocab
projection is required -- output logits map directly to ESM2 token IDs.

This adapter uses the ``esm`` Python package from EvolutionaryScale.
Install with:
    pip install esm         # PyPI package from EvolutionaryScale

Reference:
  - ESM-C: https://www.evolutionaryscale.ai/blog/esmc
  - ESM3:  https://www.evolutionaryscale.ai/blog/esm3
  - EvolutionaryScale/esm (GitHub)
"""

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Lazy imports -- ``esm`` is a heavy dependency (flash-attn, etc.).
# We defer the import to load time so that the rest of the codebase can
# reference this module without requiring the package at import time.
# ---------------------------------------------------------------------------
_ESM_AVAILABLE = None  # tri-state: None = not checked, True/False


def _check_esm_available():
    global _ESM_AVAILABLE
    if _ESM_AVAILABLE is not None:
        return _ESM_AVAILABLE
    try:
        import esm  # noqa: F401
        _ESM_AVAILABLE = True
    except ImportError:
        _ESM_AVAILABLE = False
    return _ESM_AVAILABLE


def _require_esm():
    """Raise a clear error if the ``esm`` package is not installed."""
    if not _check_esm_available():
        raise ImportError(
            "The 'esm' package (from EvolutionaryScale) is required for "
            "ESM3/ESM-C scoring but could not be imported.\n"
            "Install it with:\n"
            "    pip install esm\n"
            "For more information see: https://github.com/evolutionaryscale/esm"
        )


# ---- ESM3/ESM-C sequence vocabulary (identical to ESM2) ----
# fmt: off
_ESM_SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]
# fmt: on

_MASK_TOKEN_ID = 32   # <mask>
_CLS_TOKEN_ID = 0     # <cls>  (BOS)
_EOS_TOKEN_ID = 2     # <eos>

# Maximum sequence lengths observed from the models
# ESM-C: tested up to 2048 residues in practice; no hard positional limit
# (uses rotary position embeddings). We default to 2048 as a safe window.
# ESM3: supports sequences up to ~2048 tokens.
_DEFAULT_MAX_RESIDUE_LEN = 2048

# Known model identifiers for convenience
ESMC_MODELS = {
    "esmc_300m": "esmc_300m",
    "esmc_600m": "esmc_600m",
    # "esmc_6b" would go here when available locally
}

ESM3_MODELS = {
    "esm3_sm_open_v1": "esm3_sm_open_v1",
    "esm3-open": "esm3_sm_open_v1",
    "esm3-open-2024-03": "esm3_sm_open_v1",
    "esm3-sm-open-v1": "esm3_sm_open_v1",
}

_ESM3_LOCAL_DIR = os.path.expanduser("~/.cache/huggingface/esm3-sm-open-v1")


def _patch_esm3_data_root():
    if os.path.isdir(_ESM3_LOCAL_DIR):
        import esm.utils.constants.esm3 as _esm3_const
        _esm3_const.data_root = lambda *_args, **_kwargs: Path(_ESM3_LOCAL_DIR)


def _resolve_model_type(model_name_or_path: str) -> str:
    """Classify a model name as 'esmc' or 'esm3'."""
    name_lower = model_name_or_path.lower()
    if "esmc" in name_lower:
        return "esmc"
    if "esm3" in name_lower:
        return "esm3"
    # If it's a path, try to guess from directory name
    if "/" in model_name_or_path or "\\" in model_name_or_path:
        import os
        basename = os.path.basename(model_name_or_path.rstrip("/\\")).lower()
        if "esmc" in basename:
            return "esmc"
        if "esm3" in basename:
            return "esm3"
    # Default to esmc as the simpler model
    return "esmc"


def load_esm3_model(model_name_or_path: str = "esmc_300m", device: str = "cuda",
                    model_type: Optional[str] = None):
    """
    Load an ESM3 or ESM-C model.

    Args:
        model_name_or_path: Model identifier or local path.
            ESM-C: ``"esmc_300m"``, ``"esmc_600m"``
            ESM3:  ``"esm3_sm_open_v1"``, ``"esm3-open"``, etc.
        device: Torch device string.
        model_type: Explicitly set ``"esmc"`` or ``"esm3"``. If ``None``,
            inferred from *model_name_or_path*.

    Returns:
        Tuple of ``(model, tokenizer, esm_tokenizer, max_residue_len)`` where:
        - model: the loaded ESM3 or ESM-C model (``nn.Module``).
        - tokenizer: the model's native ``EsmSequenceTokenizer``.
        - esm_tokenizer: set to ``None`` because ESM3/ESM-C shares the
          same vocabulary as ESM2 -- no projection is needed.  The
          caller can pass this to ``forward_esm3`` unchanged.
        - max_residue_len: maximum sequence length (int).
    """
    _require_esm()

    if model_type is None:
        model_type = _resolve_model_type(model_name_or_path)

    device_obj = torch.device(device) if isinstance(device, str) else device

    if model_type == "esmc":
        from esm.models.esmc import ESMC
        model = ESMC.from_pretrained(model_name_or_path, device=device_obj)
        tokenizer = model.tokenizer
    elif model_type == "esm3":
        from esm.models.esm3 import ESM3
        _patch_esm3_data_root()
        model = ESM3.from_pretrained(model_name_or_path, device=device_obj)
        tokenizer = model.tokenizers.sequence
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Must be 'esmc' or 'esm3'."
        )

    model.eval()

    # ESM3/ESM-C vocabulary is identical to ESM2: same 33 tokens, same
    # indices.  No vocab projection is needed, so esm_tokenizer is None.
    esm_tokenizer = None

    max_residue_len = _DEFAULT_MAX_RESIDUE_LEN

    return model, tokenizer, esm_tokenizer, max_residue_len


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _tokenize_sequence(sequence: str, tokenizer) -> torch.Tensor:
    """
    Tokenize a protein sequence using the ESM sequence tokenizer.

    Returns a 1-D LongTensor: [CLS, tok_1, ..., tok_L, EOS].
    """
    # The EsmSequenceTokenizer is a PreTrainedTokenizerFast.
    # Calling it with add_special_tokens=True adds CLS + EOS.
    encoded = tokenizer(sequence, add_special_tokens=True, return_tensors="pt")
    return encoded["input_ids"].squeeze(0)  # [L+2]


# ---------------------------------------------------------------------------
# Forward functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward_esmc_single_window(
    model, sequence_tokens: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Run a single forward pass through ESM-C.

    Args:
        model: ESMC model instance.
        sequence_tokens: 1-D or 2-D tensor of token IDs (with CLS/EOS).
        device: target device.

    Returns:
        sequence_logits: ``[1, L_tok, vocab_size]`` raw logits.
    """
    if sequence_tokens.dim() == 1:
        sequence_tokens = sequence_tokens.unsqueeze(0)
    sequence_tokens = sequence_tokens.to(device)
    output = model.forward(sequence_tokens=sequence_tokens)
    return output.sequence_logits  # [1, L_tok, 64] (64 = embedding vocab)


@torch.no_grad()
def _forward_esm3_single_window(
    model, sequence_tokens: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Run a single forward pass through ESM3 (sequence-only, all other
    tracks masked).

    Args:
        model: ESM3 model instance.
        sequence_tokens: 1-D or 2-D tensor of token IDs (with CLS/EOS).
        device: target device.

    Returns:
        sequence_logits: ``[1, L_tok, vocab_size]`` raw logits.
    """
    if sequence_tokens.dim() == 1:
        sequence_tokens = sequence_tokens.unsqueeze(0)
    sequence_tokens = sequence_tokens.to(device)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        output = model.forward(sequence_tokens=sequence_tokens)
    return output.sequence_logits  # [1, L_tok, 64]


@torch.no_grad()
def forward_esm3(
    model,
    tokenizer,
    esm_tokenizer,            # unused (kept for API parity)
    sequence: str,
    device,
    max_residue_len: Optional[int] = None,
    long_seq_mode: str = "auto_window",
    long_seq_overlap: int = 256,
    model_type: str = "esmc",
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute wt-marginal log-probabilities for a protein sequence using
    ESM3 or ESM-C (single forward pass, no masking).

    The return tensor has shape ``[L, V]`` where ``L`` is the sequence
    length (residues only, no special tokens) and ``V`` is the model's
    output vocabulary size (64 for ESM3/ESM-C, though only indices 0-32
    are meaningful).  Values are log-softmax probabilities.

    Because ESM3/ESM-C and ESM2 share the same amino acid token mapping,
    the output can be consumed directly by the VenusREM scoring pipeline
    without any vocabulary projection.

    Args:
        model: Loaded ESM3 or ESM-C model.
        tokenizer: The model's native EsmSequenceTokenizer.
        esm_tokenizer: Unused (``None``). Present for API compatibility.
        sequence: Wild-type amino acid sequence (single-letter codes).
        device: Torch device.
        max_residue_len: Maximum residue length for the model window.
            If ``None``, defaults to ``_DEFAULT_MAX_RESIDUE_LEN``.
        long_seq_mode: Strategy for long sequences.
            ``"auto_window"`` -- sliding window with overlap (default).
            ``"error"`` -- raise an error if the sequence is too long.
        long_seq_overlap: Number of overlapping residues between windows.
        model_type: ``"esmc"`` or ``"esm3"`` (determines forward path).
        logger: Optional VenusREM CliLogger for structured logging.
        protein_name: Optional protein name for log messages.

    Returns:
        ``torch.Tensor`` of shape ``[L, V]`` with log-probabilities.
    """
    if max_residue_len is None:
        max_residue_len = _DEFAULT_MAX_RESIDUE_LEN

    device_obj = torch.device(device) if isinstance(device, str) else device

    # Pick the right forward function based on model type
    if model_type == "esm3":
        _forward_fn = _forward_esm3_single_window
    else:
        _forward_fn = _forward_esmc_single_window

    seq_len = len(sequence)

    # ESM3/ESM-C outputs 64-dim vocab but only the first 33 tokens match ESM2.
    # Truncate to 33 for compatibility with the rest of the pipeline.
    esm2_vocab_size = 33

    # ---- Short sequence: single forward pass ----
    if seq_len <= max_residue_len:
        tokens = _tokenize_sequence(sequence, tokenizer).to(device_obj)
        logits = _forward_fn(model, tokens, device_obj)  # [1, L+2, V]
        # Strip CLS (index 0) and EOS (index -1), keep residue positions
        residue_logits = logits[0, 1:-1, :esm2_vocab_size]  # [L, 33]
        return torch.log_softmax(residue_logits, dim=-1)

    # ---- Long sequence: sliding window ----
    if long_seq_mode == "error":
        raise ValueError(
            f"Sequence length {seq_len} exceeds model limit {max_residue_len}. "
            "Use --long_seq_mode auto_window (default) or increase "
            "--max_residue_len."
        )

    overlap = max(0, int(long_seq_overlap))
    if overlap >= max_residue_len:
        overlap = max(0, max_residue_len // 2)
    step = max(1, max_residue_len - overlap)

    msg = (
        f"Long sequence detected: L={seq_len} > {max_residue_len}; "
        f"using sliding-window inference (window={max_residue_len}, "
        f"overlap={overlap})"
    )
    if logger is not None:
        logger.warn(msg, protein=protein_name)
    else:
        print(f">>> {msg}")

    aggregated_logits = None
    counts = None
    start = 0

    while start < seq_len:
        end = min(seq_len, start + max_residue_len)
        window_seq = sequence[start:end]
        tokens = _tokenize_sequence(window_seq, tokenizer).to(device_obj)
        logits = _forward_fn(model, tokens, device_obj)  # [1, W+2, V]
        window_logits = torch.log_softmax(logits[0, 1:-1, :esm2_vocab_size], dim=-1)  # [W, 33]
        window_len = min(window_logits.size(0), end - start)
        window_logits = window_logits[:window_len]

        if aggregated_logits is None:
            aggregated_logits = torch.zeros(
                seq_len,
                window_logits.size(-1),
                device=window_logits.device,
                dtype=window_logits.dtype,
            )
            counts = torch.zeros(
                seq_len, 1,
                device=window_logits.device,
                dtype=window_logits.dtype,
            )

        aggregated_logits[start:start + window_len, :] += window_logits
        counts[start:start + window_len, :] += 1.0

        if end >= seq_len:
            break
        start += step

    counts = counts.clamp_min(1.0)
    return aggregated_logits / counts


@torch.no_grad()
def forward_esm3_masked_marginal(
    model,
    tokenizer,
    esm_tokenizer,            # unused (kept for API parity)
    sequence: str,
    device,
    max_residue_len: Optional[int] = None,
    long_seq_mode: str = "auto_window",
    long_seq_overlap: int = 256,
    model_type: str = "esmc",
    batch_size: int = 1,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute masked-marginal log-probabilities for a protein sequence
    using ESM3 or ESM-C.

    For each position *i*, the token at position *i* is replaced with
    ``<mask>`` and a forward pass is run.  The logits at the masked
    position give the marginal probability conditioned on all other
    (unmasked) positions.

    Args:
        model: Loaded ESM3 or ESM-C model.
        tokenizer: The model's native EsmSequenceTokenizer.
        esm_tokenizer: Unused (``None``). Present for API compatibility.
        sequence: Wild-type amino acid sequence (single-letter codes).
        device: Torch device.
        max_residue_len: Maximum residue length for the model window.
        long_seq_mode: Strategy when sequence exceeds *max_residue_len*.
        long_seq_overlap: Overlap for sliding window (unused here; we
            use per-position windowing for masked marginals).
        model_type: ``"esmc"`` or ``"esm3"``.
        batch_size: Number of masked positions to score simultaneously
            (currently processes one at a time for simplicity).
        logger: Optional VenusREM CliLogger.
        protein_name: Optional protein name for log messages.

    Returns:
        ``torch.Tensor`` of shape ``[L, V]`` with log-probabilities.
    """
    if max_residue_len is None:
        max_residue_len = _DEFAULT_MAX_RESIDUE_LEN

    device_obj = torch.device(device) if isinstance(device, str) else device

    if model_type == "esm3":
        _forward_fn = _forward_esm3_single_window
    else:
        _forward_fn = _forward_esmc_single_window

    seq_len = len(sequence)
    mask_token_id = tokenizer.mask_token_id

    # Truncate to ESM2 vocab (33 tokens) for pipeline compatibility
    esm2_vocab_size = 33
    all_logits = torch.zeros(seq_len, esm2_vocab_size, device=device_obj)

    if seq_len <= max_residue_len:
        # ---- Fits in single window ----
        base_tokens = _tokenize_sequence(sequence, tokenizer).to(device_obj)
        # base_tokens: [L+2] with CLS at 0, EOS at -1

        for pos in range(seq_len):
            token_pos = pos + 1  # +1 for CLS token
            masked_tokens = base_tokens.clone()
            masked_tokens[token_pos] = mask_token_id
            logits = _forward_fn(model, masked_tokens, device_obj)
            all_logits[pos] = logits[0, token_pos, :esm2_vocab_size]
            del logits
    else:
        # ---- Per-position windowed masking for long sequences ----
        if long_seq_mode == "error":
            raise ValueError(
                f"Sequence length {seq_len} exceeds model limit "
                f"{max_residue_len}."
            )
        msg = (
            f"Masked marginal: L={seq_len} > {max_residue_len}; "
            f"using per-position windowed masking"
        )
        if logger is not None:
            logger.warn(msg, protein=protein_name)
        else:
            print(f">>> {msg}")

        half_win = max_residue_len // 2
        for pos in range(seq_len):
            win_start = max(0, pos - half_win)
            win_end = min(seq_len, win_start + max_residue_len)
            win_start = max(0, win_end - max_residue_len)
            win_seq = sequence[win_start:win_end]

            tokens = _tokenize_sequence(win_seq, tokenizer).to(device_obj)
            local_pos = pos - win_start
            token_pos = local_pos + 1  # +1 for CLS
            tokens[token_pos] = mask_token_id

            logits = _forward_fn(model, tokens, device_obj)
            all_logits[pos] = logits[0, token_pos, :esm2_vocab_size]
            del logits

    return torch.log_softmax(all_logits, dim=-1)
