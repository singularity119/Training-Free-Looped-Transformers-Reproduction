#!/usr/bin/env python3
"""Check nine historical MMLU raw-score sources without outcome computation.

Reads only frozen mapping/closure, gold-free canonical input, source metadata
and scores.jsonl. Output contains counts and provenance, never raw values or
predictions. No model is loaded and no historical file is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tflt.loopscope.phase10_accuracy import load_pool, require, write_json_once
from tflt.loopscope.phase10_panel import build_panel, load_config
from verify_phase10_scores import read_score_root


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def closure_roots(closure, source_id):
    if "shards" in closure:
        return {item["root"] for item in closure["shards"] if item["cell_id"] == source_id}
    return {item["root"] for cell in closure["cells"] if cell["cell_id"] == source_id for item in cell["roots"]}


def check_reuse(pool_path, config=None):
    config = load_config() if config is None else config
    panel = build_panel("mmlu", config)
    cells = {cell["cell_id"]: cell for cell in panel["cells"]}
    canonical = load_pool(pool_path, "mmlu", "FORMAL_TEST")
    cache, records, counts = {}, [], {}

    def cached(path):
        key = str(path)
        if key not in cache:
            cache[key] = read_json(path)
        return cache[key]

    for entry in panel["reuse_mapping"]:
        source_id = entry["source_cell_id"]
        closure, manifest = cached(entry["source_closure"]), cached(entry["source_manifest"])
        require(closure.get("schema") == entry["source_closure_schema"] and
                closure.get("target_gold_loaded") is False and
                closure.get("status") in ("HISTORICAL_29_CELLS_VERIFIED", "FULL_NEW_SCORE_PANEL_CLOSED", "FULL_GATE_E_SCORE_PANEL_CLOSED"),
                "historical closure schema/scope/gold boundary differs")
        require(manifest.get("schema") == entry["source_manifest_schema"] and
                manifest.get("target_gold_loaded") is False, "historical source manifest schema/gold boundary differs")
        source_cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
        require(source_id in source_cells, "historical source cell absent from source manifest")
        require(set(entry["roots"]) == closure_roots(closure, source_id), "mapped roots differ from previously verified closure")
        source_cell = source_cells[source_id]
        if entry["source_phase"] == 8:
            # Phase 9's history map binds the already frozen Phase 8 cells by
            # source identity/root, rather than embedding the cell recipe.
            require(source_cell["source_cell_ids"] == [source_id] and
                    set(source_cell["roots"]) == set(entry["roots"]), "history source identity/root binding differs")
        covered = set()
        root_facts = []
        target_cell = cells[entry["cell_id"]]
        for root in entry["roots"]:
            metadata, summary, env = (cached(Path(root) / name) for name in ("command_args.json", "summary.json", "env.json"))
            require(metadata.get("pool") == str(pool_path) and closure.get("pool") == str(pool_path),
                    "historical canonical pool differs")
            require(metadata.get("source_commit") and env.get("source_commit") == metadata["source_commit"],
                    "historical runtime source provenance does not close")
            require(env.get("versions", {}).get("torch") == "2.3.1+cu121" and
                    env.get("versions", {}).get("transformers") == "4.51.3",
                    "historical runtime version differs from inherited environment")
            if entry["source_phase"] == 9:
                require(metadata["cell"] == source_cell and metadata.get("manifest") == entry["source_manifest"],
                        "historical runtime cell/source manifest differs")
                require(summary.get("schema") == entry["source_summary_schema"] and
                        metadata.get("target_gold_loaded") is False and env.get("target_gold_loaded") is False,
                        "historical Phase 9 score schema/gold boundary differs")
                require(env.get("tokenizer_revision") == target_cell["tokenizer_revision"],
                        "historical tokenizer revision differs")
            scores, fact = read_score_root(root, target_cell, canonical, "mmlu", "FORMAL_TEST", pool_path,
                                          entry["source_manifest"], source_id=source_id)
            require(not covered.intersection(scores), "historical shards overlap")
            covered.update(scores)
            del scores
            fact.update(source_cell_id=source_id, source_phase=entry["source_phase"],
                        source_summary_schema=summary.get("schema"), runtime_versions=env["versions"],
                        model_revision=env["model_revision"], model_dtype=env["model_dtype"],
                        target_gold_loaded=False)
            root_facts.append(fact)
        require(covered == set(range(14042)), "historical cell misses canonical test identities")
        counts[entry["cell_id"]] = len(covered)
        records.append({"cell_id": entry["cell_id"], "source_cell_id": source_id,
                        "source_closure": entry["source_closure"], "source_manifest": entry["source_manifest"],
                        "count": len(covered), "roots": root_facts})
    require(len(counts) == 9, "nine historical cells required")
    return {"schema": "loopscope.phase10.reuse_verification.v1", "status": "NINE_HISTORICAL_CELLS_RAW_VERIFIED",
            "target_gold_loaded": False, "model_forward": False, "raw_values_emitted": False,
            "pool": str(pool_path), "canonical_count": 14042, "subject_count": 57,
            "cell_count": len(counts), "sample_count": sum(counts.values()), "counts_by_cell": counts, "cells": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, type=Path, help="Existing gold-free canonical MMLU test pool")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "verification output already exists")
    value = check_reuse(args.pool, load_config(args.config))
    write_json_once(args.output, value)
    print(json.dumps({key: value[key] for key in ("status", "cell_count", "sample_count", "target_gold_loaded", "raw_values_emitted")}))


if __name__ == "__main__":
    main()
