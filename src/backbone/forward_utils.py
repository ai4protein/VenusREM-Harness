import inspect
import os

import torch


def tokenize_structure_sequence(structure_sequence):
    shift_structure_sequence = [i + 3 for i in structure_sequence]
    shift_structure_sequence = [1, *shift_structure_sequence, 2]
    return torch.tensor([shift_structure_sequence], dtype=torch.long)


def backbone_supports_structure_tokens(model):
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return "ss_input_ids" in signature.parameters


def infer_structure_vocab_subdir(model_name):
    name = model_name.split("/")[-1]
    if "ProSST-" in name:
        return name.split("-")[-1]
    return None


def resolve_structure_fasta_path(args, protein_name, model_name):
    if args.struc_seq_dir is None:
        return None
    if args.backbone_mode == "plain_mlm":
        return None

    subdir = args.structure_vocab_subdir
    if subdir is None:
        subdir = infer_structure_vocab_subdir(model_name)

    if subdir is not None:
        candidate = f"{args.struc_seq_dir}/{subdir}/{protein_name}.fasta"
        if os.path.exists(candidate):
            return candidate
    candidate = f"{args.struc_seq_dir}/{protein_name}.fasta"
    if os.path.exists(candidate):
        return candidate
    return None


def infer_model_max_residue_len(model, tokenizer):
    config_candidates = []
    config = getattr(model, "config", None)
    for attr in ["max_position_embeddings", "n_positions", "max_sequence_length"]:
        value = getattr(config, attr, None) if config is not None else None
        if isinstance(value, int) and value > 2:
            config_candidates.append(value - 2)
    if config_candidates:
        return min(config_candidates)

    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 2 < tok_max < 1_000_000:
        return tok_max - 2
    return None


def force_config_max_residue_len(model, residue_len):
    if residue_len is None or residue_len <= 0:
        return
    token_len = int(residue_len) + 2
    config = getattr(model, "config", None)
    if config is None:
        return
    for attr in ["max_position_embeddings", "n_positions", "max_sequence_length"]:
        if hasattr(config, attr):
            setattr(config, attr, token_len)


@torch.no_grad()
def forward_sequence_logits(
    model,
    tokenizer,
    sequence,
    device,
    use_structure=False,
    structure_sequence=None,
    max_residue_len=None,
    long_seq_mode="auto_window",
    long_seq_overlap=256,
    logger=None,
    protein_name=None,
):
    seq_len = len(sequence)
    if max_residue_len is None or seq_len <= max_residue_len:
        tokenized_results = tokenizer([sequence], return_tensors="pt")
        input_ids = tokenized_results["input_ids"].to(device)
        attention_mask = tokenized_results["attention_mask"].to(device)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }
        if use_structure:
            if structure_sequence is None:
                raise ValueError("structure_sequence is required when use_structure=True")
            ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
            model_inputs["ss_input_ids"] = ss_input_ids
        outputs = model(**model_inputs)
        return torch.log_softmax(outputs.logits[0][1:-1, :], dim=-1)

    if long_seq_mode == "error":
        raise ValueError(
            f"Sequence length {seq_len} exceeds model limit {max_residue_len}. "
            "Use --long_seq_mode auto_window (default) or set --max_residue_len."
        )
    if max_residue_len < 1:
        raise ValueError(f"Invalid max_residue_len={max_residue_len}")
    if use_structure and structure_sequence is None:
        raise ValueError("structure_sequence is required when use_structure=True")
    if use_structure and len(structure_sequence) != seq_len:
        raise ValueError(
            f"Structure sequence length mismatch: {len(structure_sequence)} vs residue length {seq_len}"
        )

    overlap = max(0, int(long_seq_overlap))
    if overlap >= max_residue_len:
        overlap = max(0, max_residue_len // 2)
    step = max(1, max_residue_len - overlap)

    msg = (
        f"Long sequence detected: L={seq_len} > {max_residue_len}; "
        f"using sliding-window inference (window={max_residue_len}, overlap={overlap})"
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
        tokenized_results = tokenizer([window_seq], return_tensors="pt")
        input_ids = tokenized_results["input_ids"].to(device)
        attention_mask = tokenized_results["attention_mask"].to(device)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }
        if use_structure:
            window_structure = structure_sequence[start:end]
            ss_input_ids = tokenize_structure_sequence(window_structure).to(device)
            model_inputs["ss_input_ids"] = ss_input_ids

        outputs = model(**model_inputs)
        window_logits = torch.log_softmax(outputs.logits[0][1:-1, :], dim=-1)
        window_len = min(window_logits.size(0), end - start)
        window_logits = window_logits[:window_len]

        if aggregated_logits is None:
            aggregated_logits = torch.zeros(
                seq_len,
                window_logits.size(-1),
                device=window_logits.device,
                dtype=window_logits.dtype,
            )
            counts = torch.zeros(seq_len, 1, device=window_logits.device, dtype=window_logits.dtype)

        aggregated_logits[start : start + window_len, :] += window_logits
        counts[start : start + window_len, :] += 1.0

        if end >= seq_len:
            break
        start += step

    counts = counts.clamp_min(1.0)
    return aggregated_logits / counts
