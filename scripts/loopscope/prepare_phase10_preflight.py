#!/usr/bin/env python3
"""Prepare gold-free validation inputs and canonical bounded Gate B selections.

CPU/tokenizers only. ARC is reused directly from its accepted Gate A pool.
The MMLU safe renderer is reused with only its final row selector replaced;
its source projection, native five demonstrations and prompt rendering stay
unchanged. No protected source file is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from tflt.loopscope.phase10_accuracy import load_pool, require, row_identity, write_json_once
from tflt.loopscope.phase10_panel import load_config, score_manifest
from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION, SEED, _build_pool, identity


def validate_mmlu_pool(pool):
    require(pool.get("schema_version") == "loopscope.phase10.mmlu_preflight_inputs.v1" and
            pool.get("status") == "PREFLIGHT_ONLY_GOLD_FREE", "MMLU preflight schema/scope differs")
    require(pool.get("dataset") == {"repo": DATASET_REPO, "revision": DATASET_REVISION,
            "split": "validation"} and pool.get("seed") == SEED, "MMLU validation recipe differs")
    evidence = pool["source_evidence"]
    counts = evidence["subject_counts"]
    require(len(counts) == 57 and sum(counts.values()) == 1531 and
            all(type(n) is int and n > 0 for n in counts.values()), "MMLU validation population differs")
    expected = [(subject, index) for subject in sorted(counts) for index in range(counts[subject])]
    rows = pool["rows"]
    require([(r["subject"], r["doc_index"]) for r in rows] == expected, "MMLU validation canonical order differs")
    models = {entry["model"] for entry in evidence["tokenizers"]}
    require(len(models) == 2, "two MMLU tokenizer lengths required")
    fields = {"identity", "subject", "doc_index", "split", "question", "choices", "prompt", "prompt_token_lengths"}
    for row in rows:
        require(set(row) == fields and row["split"] == "validation" and
                row["identity"] == identity(row["subject"], row["doc_index"]) and
                isinstance(row["choices"], list) and len(row["choices"]) == 4 and
                all(isinstance(c, str) for c in row["choices"]) and
                isinstance(row["prompt"], str) and row["prompt"] and
                set(row["prompt_token_lengths"]) == models and
                all(type(n) is int and n > 0 for n in row["prompt_token_lengths"].values()),
                "MMLU validation safe fields/lengths differ")
    require(pool["validation_identities"] == [r["identity"] for r in rows], "validation identities differ")
    require(evidence["loaded_splits"] == ["validation", "dev"] and
            evidence["target_gold_loaded"] is False and evidence["lm_eval_version"] == "0.4.11",
            "gold-free native MMLU source required")


def build_mmlu_pool(config, cache_dir):
    model_config = {"models": [dict(spec, observed_revision=spec["revision"]) for spec in config["models"]]}
    with patch("tflt.loopscope.phase8_pool.select_debug", lambda rows, calibration, models: rows):
        original = _build_pool(model_config, cache_dir)
    pool = {key: value for key, value in original.items() if key != "calibration_identities"}
    pool.update(schema_version="loopscope.phase10.mmlu_preflight_inputs.v1",
                status="PREFLIGHT_ONLY_GOLD_FREE",
                validation_identities=[row["identity"] for row in original["rows"]])
    validate_mmlu_pool(pool)
    return pool


def mmlu_lengths(rows, config):
    """Measure retained native request context and full letter continuations."""
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
        encoder = Encoder(AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"], local_files_only=True))
        items = []
        for row in rows:
            pairs = [encoder._encode_pair(row["prompt"], " " + choice) for choice in "ABCD"]
            require(len({tuple(context) for context, _ in pairs}) == 1, "MMLU retained candidate prefix differs")
            items.append({"context_length": len(pairs[0][0]),
                          "continuation_lengths": [len(answer) for _, answer in pairs]})
        lengths[spec["model"]] = items
    return lengths


def arc_lengths(rows):
    return {model: [row["tokenization"][model] for row in rows] for model in rows[0]["tokenization"]}


def choose_indices(rows, lengths, dataset, maximum=16):
    """Canonical first row, each model's longest context/answer, ARC categories."""
    require(1 <= maximum <= 16, "Gate B permits at most sixteen validation questions")
    chosen, reasons = {0}, {"first_canonical": 0}
    for model, items in lengths.items():
        require(len(items) == len(rows), "length metadata count differs")
        for name, key in (("longest_context", lambda item: item["context_length"]),
                          ("longest_continuation", lambda item: max(item["continuation_lengths"]))):
            index = max(range(len(rows)), key=lambda i: (key(items[i]), -i))
            chosen.add(index)
            reasons[model + ":" + name] = index
    if dataset == "arc_challenge":
        for count in (3, 4, 5):
            candidates = [i for i, row in enumerate(rows) if len(row["choices"]) == count]
            if candidates:
                chosen.add(candidates[0])
                reasons["options_" + str(count)] = candidates[0]
        numeric = [i for i, row in enumerate(rows) if any(label.isdigit() for label in row["choice_labels"])]
        if numeric:
            chosen.add(numeric[0])
            reasons["numeric_labels"] = numeric[0]
    require(len(chosen) <= maximum, "coverage constraints exceed bounded selection")
    # A small fixed set covers all requested structure without spending sixteen
    # model evaluations simply to fill the permission ceiling.
    return sorted(chosen), reasons


def input_facts(rows, lengths, indices, reasons, dataset, pool_path):
    return {"dataset": dataset, "pool": str(pool_path), "split": "validation",
            "canonical_count": len(rows), "selected_count": len(indices), "indices": indices,
            "identities": [row_identity(rows[i], dataset) for i in indices], "selection_reasons": reasons,
            "option_counts": sorted({len(row["choices"]) for row in rows}),
            "numeric_label_count": sum(any(label.isdigit() for label in row.get("choice_labels", [])) for row in rows),
            "lengths": {model: {"max_context_tokens": max(item["context_length"] for item in items),
                "max_continuation_tokens": max(max(item["continuation_lengths"]) for item in items),
                "selected": [{"index": i, **items[i]} for i in indices]} for model, items in lengths.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-validation-pool", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "input root already exists")
    config = load_config(args.config)
    arc = load_pool(args.arc_validation_pool, "arc_challenge", "PREFLIGHT_ONLY")
    require(len(arc) == 299 and all(row["split"] == "validation" for row in arc), "accepted ARC validation299 required")
    mmlu = build_mmlu_pool(config, args.cache_dir)
    pools = {"mmlu": args.output_dir / "MMLU-validation-pool.json", "arc_challenge": args.arc_validation_pool}
    rows = {"mmlu": mmlu["rows"], "arc_challenge": arc}
    lengths = {"mmlu": mmlu_lengths(mmlu["rows"], config), "arc_challenge": arc_lengths(arc)}
    selections = {dataset: choose_indices(rows[dataset], lengths[dataset], dataset) for dataset in pools}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_once(pools["mmlu"], mmlu)
    facts = {"schema": "loopscope.phase10.preflight_input_facts.v1", "scope": "PREFLIGHT_ONLY",
             "target_gold_loaded": False, "arc_rebuilt": False, "model_forward": False, "datasets": {}}
    for dataset, pool_path in pools.items():
        indices, reasons = selections[dataset]
        write_json_once(args.output_dir / (dataset + "-indices.json"), indices)
        write_json_once(args.output_dir / (dataset + "-manifest.json"), score_manifest(dataset, pool_path, "PREFLIGHT_ONLY", config))
        facts["datasets"][dataset] = input_facts(rows[dataset], lengths[dataset], indices, reasons, dataset, pool_path)
    write_json_once(args.output_dir / "preflight_input_facts.json", facts)
    print(json.dumps({"output_dir": str(args.output_dir), "target_gold_loaded": False,
        "counts": {dataset: facts["datasets"][dataset]["selected_count"] for dataset in pools}}))


if __name__ == "__main__":
    main()
