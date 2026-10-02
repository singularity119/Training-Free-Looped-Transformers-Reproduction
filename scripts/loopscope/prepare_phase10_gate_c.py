#!/usr/bin/env python3
"""Freeze gold-free MMLU Gate C identities and retainable formal canary shards.

CPU/tokenizers only, using the existing canonical pool and cached tokenizers.
The score manifest keeps all 68 cells and its nine historical reuse references.
Execution indices are scheduling units; they never redefine a complete cell.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from tflt.loopscope.phase10_accuracy import load_pool, require, row_identity, write_json_once
from tflt.loopscope.phase10_panel import load_config, score_manifest, strategy_id


def choose_indices(items):
    """64 length ranks, with canonical tie breaking and canonical output order."""
    require(len(items) >= 64, "formal canary requires at least 64 canonical rows")
    require(all(type(item["context_length"]) is int and item["context_length"] > 0
                for item in items), "positive integer context lengths required")
    ordered = sorted(range(len(items)), key=lambda i: (items[i]["context_length"], i))
    ranks = [round(j * (len(items) - 1) / 63) for j in range(64)]
    indices = sorted(ordered[rank] for rank in ranks)
    require(len(set(indices)) == 64, "canary length ranks must select 64 unique rows")
    return indices, ranks


def measure_lengths(rows, config):
    """Match native letter continuation encoding without model forward or gold."""
    from lm_eval.api.model import TemplateLM
    from transformers import AutoTokenizer

    class Encoder:
        backend = "causal"
        _encode_pair = TemplateLM._encode_pair

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def tok_encode(self, text, **kwargs):
            return self.tokenizer.encode(text, add_special_tokens=False)

    lengths = {}
    for spec in config["models"]:
        encoder = Encoder(AutoTokenizer.from_pretrained(
            spec["model"], revision=spec["revision"], local_files_only=True))
        entries = []
        for row in rows:
            pairs = [encoder._encode_pair(row["prompt"], " " + choice) for choice in "ABCD"]
            require(len({tuple(context) for context, _ in pairs}) == 1,
                    "MMLU candidate context differs before scoring")
            entries.append({"context_length": len(pairs[0][0]),
                            "continuation_lengths": [len(answer) for _, answer in pairs],
                            "prompt_token_length": row["prompt_token_lengths"][spec["model"]]})
        lengths[spec["model"]] = entries
    return lengths


def length_distribution(items):
    counts = Counter(item["context_length"] for item in items)
    ordered = sorted(counts.elements())
    return {"count": len(items), "min_context_tokens": ordered[0],
            "max_context_tokens": ordered[-1],
            "mean_context_tokens": sum(ordered) / len(ordered),
            "context_token_counts": [{"context_length": value, "count": counts[value]}
                                     for value in sorted(counts)],
            "quantiles": [{"fraction": fraction,
                           "context_length": ordered[round(fraction * (len(items) - 1))]}
                          for fraction in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)],
            "max_continuation_tokens": max(max(item["continuation_lengths"]) for item in items),
            "native_context_differs_from_cached_prompt_count": sum(
                item["context_length"] != item["prompt_token_length"] for item in items)}


def build_plan(rows, lengths, pool_path, output_dir, config):
    """Return readable plan and index artifacts; reads no target labels/scores."""
    manifest = score_manifest("mmlu", pool_path, "FORMAL_TEST", config)
    count = len(rows)
    require(count == manifest["sample_count_per_cell"] == 14042,
            "Gate C requires the complete frozen MMLU test pool")
    models = {spec["model"] for spec in config["models"]}
    require(set(lengths) == models and all(len(items) == count for items in lengths.values()),
            "full-population lengths required for both models")
    require((manifest["cell_count"], manifest["new_cell_count"],
             manifest["expected_reuse_cell_count"]) == (68, 59, 9), "Gate C membership differs")
    identities = [row_identity(row, "mmlu") for row in rows]
    require(len(set(identities)) == count, "duplicate canonical identities")
    root = Path(output_dir)
    path = lambda name: str(root / name)
    artifacts = {"mmlu-manifest.json": manifest, "full-identities.json": identities,
                 "full-indices.json": list(range(count))}
    canary_cells = [cell for cell in manifest["score_cells"]
                    if cell["arm"] == "Online" and cell["strength"] == 0.1]
    require(len(canary_cells) == len({strategy_id(cell) for cell in canary_cells}) == 7,
            "seven nonduplicate lambda0.1 strategy canaries required")
    canary_ids = {cell["cell_id"] for cell in canary_cells}
    model_entries = {}
    for spec in config["models"]:
        model = spec["model"]
        tag = "q4" if spec == config["models"][0] else "q17"
        items = lengths[model]
        require(all(len(item["continuation_lengths"]) == 4 and
                    all(type(n) is int and n > 0 for n in item["continuation_lengths"])
                    for item in items), "four full positive continuation lengths required")
        selected, ranks = choose_indices(items)
        selected_set = set(selected)
        remaining = [i for i in range(count) if i not in selected_set]
        canary_name, remainder_name = f"canary-{tag}-indices.json", f"remainder-{tag}-indices.json"
        artifacts[canary_name], artifacts[remainder_name] = selected, remaining
        diagnostics_name = f"lengths-{tag}.json"
        artifacts[diagnostics_name] = [{"canonical_index": i, "identity": identities[i], **item}
                                     for i, item in enumerate(items)]
        model_entries[model] = {"model": model, "tokenizer_revision": spec["revision"],
            "length_source": "native TemplateLM._encode_pair(prompt, full letter continuation); cached tokenizer",
            "lengths_path": path(diagnostics_name), "distribution": length_distribution(items),
            "selection_rule": "sort (context_length, canonical_index), rank round(j*(N-1)/63), j=0..63; canonical sort",
            "selected_ranks": ranks, "canary_indices_path": path(canary_name),
            "remainder_indices_path": path(remainder_name), "selected_count": 64,
            "canary_indices": selected, "canary_identities": [identities[i] for i in selected],
            "remainder_count": len(remaining)}
    cells, groups = [], {}
    for cell in manifest["score_cells"]:
        canary = cell["cell_id"] in canary_ids
        model_entry = model_entries[cell["model"]]
        group_id = strategy_id(cell)
        group = groups.setdefault(group_id, {"group_id": group_id, "model": cell["model"],
            "k": cell["k"], "direction_policy": cell["direction_policy"], "window": cell["window"],
            "new_cell_count": 0, "unstarted_cell_count": 0, "canary_cell_count": 0,
            "canary_sample_records": 0, "remaining_sample_records": 0})
        group["new_cell_count"] += 1
        group["unstarted_cell_count"] += int(not canary)
        group["canary_cell_count"] += int(canary)
        group["canary_sample_records"] += 64 if canary else 0
        group["remaining_sample_records"] += count - 64 if canary else count
        cells.append({"cell_id": cell["cell_id"], "group_id": group_id, "model": cell["model"],
            "full_count": count, "full_identities_path": path("full-identities.json"),
            "full_indices_path": path("full-indices.json"), "canary_count": 64 if canary else 0,
            "canary_indices_path": model_entry["canary_indices_path"] if canary else None,
            "canary_shard_id": "formal-canary64" if canary else None,
            "remaining_count": count - 64 if canary else count,
            "remaining_indices_path": model_entry["remainder_indices_path"] if canary else path("full-indices.json")})
    canaries = [{"cell_id": cell["cell_id"], "group_id": cell["group_id"], "model": cell["model"],
                 "shard_id": cell["canary_shard_id"], "count": cell["canary_count"],
                 "indices_path": cell["canary_indices_path"],
                 "remainder_indices_path": cell["remaining_indices_path"]}
                for cell in cells if cell["canary_count"]]
    plan = {"schema": "loopscope.phase10.gate_c_input_plan.v1", "scope": "FORMAL_TEST",
        "dataset": "mmlu", "pool": str(pool_path), "manifest_path": path("mmlu-manifest.json"),
        "target_gold_loaded": False, "model_forward": False, "canonical_count": count,
        "full_identities_path": path("full-identities.json"), "full_indices_path": path("full-indices.json"),
        "cell_count": 68, "new_cell_count": 59, "expected_reuse_cell_count": 9,
        "display_row_count": 86, "alias_count": 18, "new_sample_records": 59 * count,
        "total_sample_records": 68 * count, "canary_cell_count": 7, "canary_sample_records": 7 * 64,
        "remaining_sample_records": 59 * count - 7 * 64,
        "unstarted_cell_count": 52, "models": model_entries, "groups": list(groups.values()),
        "canaries": canaries, "new_cells": cells,
        "historical_reuse": "use existing Gate B raw-score verification; no historical analysis loaded",
        "closure": "frozen identity membership only; no score shard or complete cell is asserted closed"}
    artifacts["gate_c_input_plan.json"] = plan
    return plan, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Gate C input root already exists")
    config = load_config(args.config)
    rows = load_pool(args.pool, "mmlu", "FORMAL_TEST")
    lengths = measure_lengths(rows, config)
    plan, artifacts = build_plan(rows, lengths, args.pool, args.output_dir, config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in artifacts.items():
        write_json_once(args.output_dir / name, value)
    print(json.dumps({key: plan[key] for key in ("canonical_count", "cell_count", "new_cell_count",
        "expected_reuse_cell_count", "canary_cell_count", "canary_sample_records",
        "remaining_sample_records", "target_gold_loaded")}, sort_keys=True))


if __name__ == "__main__":
    main()
