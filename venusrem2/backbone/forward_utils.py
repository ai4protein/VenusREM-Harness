import torch

from venusrem2.baseline.prosst.structure_tokens import (
    backbone_supports_structure_tokens,
    infer_structure_vocab_subdir,
    resolve_structure_fasta_path,
    tokenize_structure_sequence,
)


def infer_model_max_residue_len(model, tokenizer):
    config_candidates = []
    config = getattr(model, "config", None)
    pos_embed_type = getattr(config, "position_embedding_type", None) if config else None
    for attr in ["max_position_embeddings", "n_positions", "max_sequence_length"]:
        value = getattr(config, attr, None) if config is not None else None
        if isinstance(value, int) and value > 2:
            overhead = 2
            if pos_embed_type == "absolute":
                pos_embed = getattr(getattr(model, "esm", model), "embeddings", None)
                pos_embed = getattr(pos_embed, "position_embeddings", None)
                pad_idx = getattr(pos_embed, "padding_idx", None)
                if isinstance(pad_idx, int) and pad_idx > 0:
                    overhead = pad_idx + 3
            config_candidates.append(value - overhead)
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
def forward_masked_marginal(
    model,
    tokenizer,
    sequence,
    device,
    max_residue_len=None,
    long_seq_mode="auto_window",
    long_seq_overlap=256,
    logger=None,
    protein_name=None,
    batch_size=1,
    use_structure=False,
    structure_sequence=None,
):
    seq_len = len(sequence)
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        name = getattr(tokenizer, "name_or_path", type(tokenizer).__name__)
        raise ValueError(
            f"Tokenizer {name!r} has no mask_token_id; "
            "refusing --scoring_strategy masked-marginals."
        )
    vocab_size = getattr(model.config, "vocab_size", tokenizer.vocab_size)
    all_logits = torch.zeros(seq_len, vocab_size, device=device)

    if use_structure:
        if structure_sequence is None:
            raise ValueError("structure_sequence is required when use_structure=True")
        if len(structure_sequence) != seq_len:
            raise ValueError(
                f"Structure sequence length mismatch: {len(structure_sequence)} vs {seq_len}"
            )

    if max_residue_len is None or seq_len <= max_residue_len:
        tokenized = tokenizer([sequence], return_tensors="pt")
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        ss_input_ids = None
        if use_structure:
            ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)

        for start in range(0, seq_len, batch_size):
            end = min(start + batch_size, seq_len)
            batch_ids = input_ids.expand(end - start, -1).clone()
            batch_mask = attention_mask.expand(end - start, -1)
            for j, pos in enumerate(range(start, end)):
                batch_ids[j, pos + 1] = mask_token_id
            model_inputs = {"input_ids": batch_ids, "attention_mask": batch_mask}
            if use_structure:
                model_inputs["ss_input_ids"] = ss_input_ids.expand(end - start, -1)
            out = model(**model_inputs)
            for j, pos in enumerate(range(start, end)):
                all_logits[pos] = out.logits[j, pos + 1]
            del out
    else:
        if long_seq_mode == "error":
            raise ValueError(
                f"Sequence length {seq_len} exceeds model limit {max_residue_len}."
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
            tokenized = tokenizer([win_seq], return_tensors="pt")
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            local_pos = pos - win_start
            input_ids[0, local_pos + 1] = mask_token_id
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if use_structure:
                win_struc = structure_sequence[win_start:win_end]
                model_inputs["ss_input_ids"] = tokenize_structure_sequence(win_struc).to(device)
            out = model(**model_inputs)
            all_logits[pos] = out.logits[0, local_pos + 1]
            del out

    return torch.log_softmax(all_logits, dim=-1)


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
