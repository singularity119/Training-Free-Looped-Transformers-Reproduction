#!/usr/bin/env python3
"""Engineering-only artificial-token CUDA upper-tail KV capacity evidence."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

from tflt.loopscope.phase11_d8 import MAX_LENGTH, MAX_NEW_TOKENS, REVISION, selected_cells
from tflt.loopscope.phase11_panel import require, write_json_once


def cache_slots(cache):
    """Explicitly support audited HF dynamic-cache layouts; never guess."""
    if hasattr(cache, "layers"):
        return [(layer, "keys", "values") for layer in cache.layers]
    require(hasattr(cache, "key_cache") and hasattr(cache, "value_cache"),
            "unsupported cache layout")
    return [(cache, i, i) for i in range(len(cache.key_cache))]


def tensors(slot):
    owner, key, value = slot
    if isinstance(key, int):
        return owner.key_cache[key], owner.value_cache[value]
    return getattr(owner, key), getattr(owner, value)


def replace(slot, key_tensor, value_tensor):
    owner, key, value = slot
    if isinstance(key, int):
        owner.key_cache[key], owner.value_cache[value] = key_tensor, value_tensor
    else:
        setattr(owner, key, key_tensor)
        setattr(owner, value, value_tensor)


def describe(cache):
    result = []
    for index, slot in enumerate(cache_slots(cache)):
        key, value = tensors(slot)
        require(key.ndim == value.ndim == 4 and key.shape == value.shape,
                "unexpected KV tensor shape")
        result.append({"layer": index, "key_shape": list(key.shape),
                       "value_shape": list(value.shape), "dtype": str(key.dtype),
                       "bytes": (key.numel() + value.numel()) * key.element_size()})
    return result


def run(model_dir, pool_path, output):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tflt.wrapper import looped_model
    from tflt.loopscope.phase11_runtime import make_runtime, validate_calls
    from run_phase11_accuracy import make_loop_config

    require(torch.cuda.is_available(), "capacity probe requires actual CUDA")
    pool = json.loads(pool_path.read_text())
    require(pool.get("target_gold_loaded") is False, "pool must be gold-free")
    prompt_length = max(len(row["input_ids"]) for row in pool["rows"])
    target = prompt_length + MAX_NEW_TOKENS
    require(target <= MAX_LENGTH, "formal prompt plus generation exceeds native context")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    require(model.dtype == torch.bfloat16 and model.config.max_position_embeddings == MAX_LENGTH,
            "capacity model dtype/context differs")
    require(getattr(model.config, "_commit_hash", None) in (None, REVISION), "model revision differs")
    token = tokenizer.encode(" capacity", add_special_tokens=False)[0]
    require(token not in tokenizer.all_special_ids, "artificial token is special")
    device = model.device
    total = torch.cuda.get_device_properties(device).total_memory
    results = []
    for name, cell in selected_cells().items():
        runtime = make_runtime(cell)
        manager = nullcontext() if runtime is None else looped_model(model, make_loop_config(cell, runtime))
        question = nullcontext() if runtime is None else runtime.question(
            "engineering-capacity:" + name, prompt_length - 1, tuple(range(prompt_length)),
            token_ids=(token,) * prompt_length)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), manager, question:
            inputs = torch.full((1, prompt_length), token, dtype=torch.long, device=device)
            ctx = nullcontext() if runtime is None else runtime.forward_context("prefill", prompt_length)
            with ctx:
                prefill = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                use_cache=True, logits_to_keep=1)
            cache = prefill.past_key_values
            require(cache.get_seq_length() == prompt_length, "prefill cache length differs")
            initial = describe(cache)
            require(len(initial) == model.config.num_hidden_layers, "cache missing model layers")
            prefill_peak = torch.cuda.max_memory_allocated(device)
            del prefill, inputs
            # Materialize all layers at the upper tail without computing artificial outputs.
            for slot in cache_slots(cache):
                key, value = tensors(slot)
                shape = list(key.shape)
                shape[-2] = target - 1
                new_key = torch.zeros(shape, dtype=key.dtype, device=key.device)
                new_value = torch.zeros(shape, dtype=value.dtype, device=value.device)
                new_key[..., :prompt_length, :] = key
                new_value[..., :prompt_length, :] = value
                replace(slot, new_key, new_value)
                del key, value, new_key, new_value
            before = describe(cache)
            require(all(row["key_shape"][-2] == target - 1 for row in before),
                    "artificial upper-tail cache incomplete")
            construction_peak = torch.cuda.max_memory_allocated(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = torch.cuda.memory_allocated(device)
            ctx = nullcontext() if runtime is None else runtime.forward_context("decode", 1)
            with ctx:
                tail = model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                    attention_mask=torch.ones((1, target), dtype=torch.long, device=device),
                    cache_position=torch.tensor([target - 1], device=device),
                    position_ids=torch.tensor([[target - 1]], device=device),
                    use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(device)
            require(torch.isfinite(tail.logits).all().item(), "tail logits nonfinite")
            after = describe(tail.past_key_values)
            require(all(row["key_shape"][-2] == target for row in after), "tail layer cache growth differs")
            kv_bytes = sum(row["bytes"] for row in after)
            peak = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            free, _ = torch.cuda.mem_get_info(device)
            # Reserve an extra complete KV copy beyond the measured tail peak, covering
            # retained storage/cat temporaries conservatively across all layers.
            conservative = max(prefill_peak, peak + kv_bytes, construction_peak)
            external = max(0, total - free - torch.cuda.memory_reserved(device))
            passed = conservative + external <= .8 * total and reserved + external <= .8 * total
            results.append({"cell": name, "prefill_cache": initial,
                "artificial_tail_cache_before": before, "real_tail_cache_after": after,
                "prefill_peak_allocated": prefill_peak, "construction_peak_allocated": construction_peak,
                "tail_allocated_before": allocated_before, "tail_peak_allocated": peak,
                "tail_peak_reserved": reserved, "tail_KV_bytes": kv_bytes,
                "conservative_peak_bytes": conservative, "external_device_bytes": external,
                "free_device_bytes": free, "total_device_bytes": total,
                "headroom_fraction": 1 - (conservative + external) / total,
                "pass": passed, "real_loop_forward_count": validate_calls(runtime) if runtime else None})
            del tail, cache
        del runtime
        torch.cuda.empty_cache()
    report = {"status": "CAPACITY_PASS" if all(row["pass"] for row in results) else "CAPACITY_BLOCK",
        "target_gold_loaded": False, "engineering_only": True, "packing": 1,
        "formal_max_prompt_tokens": prompt_length, "max_new_tokens": MAX_NEW_TOKENS,
        "tested_cache_length": target, "artificial_token_id": token,
        "method": "max-prompt artificial prefill; materialized upper-tail KV; real one-token loop/cache forward",
        "inference": "tail_peak plus one extra complete KV copy; prefill/construction peaks also bounded",
        "not_measured": "32768 sequential sampled decoding steps; artificial cache contents are not generated outcomes",
        "cells": results}
    write_json_once(output, report)
    require(report["status"] == "CAPACITY_PASS", "capacity headroom below required 20%; see report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "pool", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    run(args.model_dir, args.pool, args.output)
