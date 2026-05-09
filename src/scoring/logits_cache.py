import os

import torch


def load_cached_logits(
    logits_cache_path,
    sequence,
    reuse_logits_cache,
    cache_miss_policy,
    logits_cache_stage,
    device,
    log_local,
    log_warn_local,
):
    logits = None
    loaded_final_logits = False

    if logits_cache_path and reuse_logits_cache and os.path.exists(logits_cache_path):
        try:
            cached_payload = torch.load(logits_cache_path, map_location="cpu")
            cached_logits = cached_payload.get("logits")
            cached_sequence = cached_payload.get("sequence")
            cached_stage = cached_payload.get("cache_stage", "raw")
            if (
                cached_logits is not None
                and cached_sequence == sequence
                and cached_stage == logits_cache_stage
            ):
                logits = cached_logits.to(device)
                loaded_final_logits = logits_cache_stage == "final"
                log_local(f"Loaded logits cache ({cached_stage}): {logits_cache_path}")
            else:
                log_warn_local(f"Skipped incompatible logits cache: {logits_cache_path}")
        except Exception as cache_error:
            if cache_miss_policy == "error":
                raise RuntimeError(
                    f"Failed to load logits cache and cache_miss_policy=error: {logits_cache_path}"
                ) from cache_error
            log_warn_local(f"Failed to load logits cache ({cache_error}); falling back to forward")

    if logits_cache_path and reuse_logits_cache and not os.path.exists(logits_cache_path):
        if cache_miss_policy == "error":
            raise FileNotFoundError(
                f"Required logits cache missing and cache_miss_policy=error: {logits_cache_path}"
            )
        log_warn_local(f"Logits cache miss; falling back to forward: {logits_cache_path}")

    if cache_miss_policy == "error" and logits is None:
        raise FileNotFoundError(
            f"Required logits cache is missing/invalid: {logits_cache_path}"
        )

    return logits, loaded_final_logits


def save_cached_logits(logits_cache_path, sequence, logits, cache_stage, log_local):
    if not logits_cache_path:
        return
    cache_parent = os.path.dirname(logits_cache_path)
    if cache_parent:
        os.makedirs(cache_parent, exist_ok=True)
    torch.save(
        {"sequence": sequence, "logits": logits.detach().cpu(), "cache_stage": cache_stage},
        logits_cache_path,
    )
    log_local(f"Wrote logits cache ({cache_stage}): {logits_cache_path}")
