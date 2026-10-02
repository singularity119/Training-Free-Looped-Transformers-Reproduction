#!/usr/bin/env python3
"""Close all canonical candidate scores before any Phase 10 gold/outcome analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from tflt.loopscope.phase10_accuracy import (
    choices_for_row, engineering_cell, finite_scores, load_pool, require, row_identity, select_indices, validate_indices, write_json_once,
)
from tflt.loopscope.phase10_panel import load_config, validate_score_manifest
from tflt.loopscope.phase10_runtime import applies_at, direction_schedule


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity_key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def validate_cell(old, target, source_id=None):
    require(old.get("cell_id") == (source_id or target["cell_id"]), "score source cell identity differs")
    for key in ("model", "revision", "dtype", "cache_strategy", "window", "k"):
        require(old.get(key) == target[key], "score source recipe differs: " + key)
    if source_id is None:
        require(old == target, "Phase 10 score source configuration differs from manifest")
    elif target["arm"] == "Online":
        expected_arm = "Online-t0" if target["direction_policy"] == "fixed_t0" else "Lag1-Online"
        require(old.get("arm") == expected_arm and target["strength"] == 0.5,
                "historical Online source differs from frozen lambda0.5 policy")
    else:
        require(old.get("arm") == target["arm"], "historical baseline arm differs")


def read_score_root(root, cell, canonical, dataset, scope, pool_path, manifest_path,
                    expected=None, source_id=None):
    root = Path(root)
    metadata, summary, env = (read_json(root / name) for name in ("command_args.json", "summary.json", "env.json"))
    require(summary.get("status") == "SCORES_COMPLETE" and summary.get("metadata") == metadata,
            "incomplete/inconsistent score attempt: " + str(root))
    require(summary.get("target_gold_loaded") is False and
            (source_id is not None or metadata.get("target_gold_loaded") is False),
            "score attempt crossed target gold boundary")
    validate_cell(metadata.get("cell", {}), cell, source_id)
    require(metadata.get("scope") == scope, "score attempt scope differs from closure")
    require(env.get("model_revision") == cell["revision"] and env.get("model_dtype") == "torch." + cell["dtype"],
            "score environment model revision/dtype differs")
    require(env.get("versions", {}).get("lm_eval") == "0.4.11", "score environment harness version differs")
    if source_id is None:
        acquisition = engineering_cell(cell, metadata.get("engineering_check"), scope)
        require(metadata.get("acquisition_cell", cell) == acquisition, "engineering acquisition configuration differs")
        require(metadata.get("pool") == str(pool_path) and metadata.get("manifest") == str(manifest_path),
                "score attempt source input paths differ")
        require(metadata.get("dataset") == dataset and env.get("dataset") == dataset,
                "score attempt dataset differs")
        require(metadata.get("source_commit") and env.get("source_commit") == metadata["source_commit"],
                "score attempt source revision does not close")
        selection = metadata.get("execution_selection")
        if selection is not None:
            require(selection.get("canonical_pool_count") == len(canonical) and
                    selection.get("indices") == metadata.get("indices"), "execution selection does not close against full pool")
            require(selection.get("kind") in ("canonical_shard", "scope_selection"), "unknown execution selection kind")
            if selection["kind"] == "canonical_shard":
                require(isinstance(selection.get("shard_id"), str) and selection["shard_id"].strip(), "missing readable shard identity")
                indices = validate_indices(metadata.get("indices", []), len(canonical))
                require(set(indices).issubset(expected), "shard selection is outside canonical reference selection")
            else:
                require(selection.get("shard_id") is None, "scope selection cannot carry shard identity")
                require(metadata.get("indices") == expected, "score attempt canonical selection differs")
                indices = expected
        else:
            require(metadata.get("indices") == expected, "score attempt canonical selection differs")
            indices = expected
        runtime = summary.get("positions_and_runtime", {})
        require(runtime.get("unexpected_t0_mutation_count", 0) == runtime.get("nonanswer_mutation_count") == 0,
                "invalid intervention mutation recorded")
        k = int(cell["k"] or 0)
        callbacks, applied = runtime.get("callback_by_t"), runtime.get("applied_by_t")
        require(isinstance(callbacks, dict) and isinstance(applied, dict) and
                set(callbacks) == set(applied) == {str(t) for t in range(k)}, "missing callback schedule evidence")
        if cell["arm"] == "Online":
            contexts = runtime.get("model_context_count")
            require(type(contexts) is int and contexts >= len(indices), "missing Online candidate contexts")
            policy = acquisition["direction_policy"]
            require(all(callbacks[str(t)] == contexts for t in range(k)) and
                    all(applied[str(t)] == (contexts if applies_at(policy,t,acquisition["strength"]) else 0) for t in range(k)),
                    "Online intervention schedule differs")
            if policy != "current_t" or acquisition["strength"] == 0:
                require(runtime.get("t0_mutation_count") == 0, "t0 mutation differs from policy/strength")
            require(runtime.get("direction_schedule") == [list(row) for row in direction_schedule(policy,k)],
                    "direction source/application schedule differs")
            fit, used = runtime.get("direction_fit_by_t"), runtime.get("direction_used_fit_by_t")
            require(isinstance(fit,dict) and isinstance(used,dict) and set(fit)==set(used)==set(callbacks),
                    "missing direction fitting/use timing evidence")
            if policy == "current_t":
                require(all(fit[str(t)] == used[str(t)] == contexts for t in range(k)),
                        "current_t must fit and use every own raw timestep including t0")
        else:
            require(all(value == 0 for value in callbacks.values()) and all(value == 0 for value in applied.values()),
                    "baseline unexpectedly applied a residual transform")
    else:
        if "indices" in metadata:
            indices = metadata["indices"]
        else:
            start, end = metadata.get("start"), metadata.get("end")
            require(type(start) is int and type(end) is int and 0 <= start < end <= len(canonical),
                    "historical canonical shard range is missing")
            indices = list(range(start, end))
        require(indices and indices == sorted(set(indices)) and all(type(i) is int and 0 <= i < len(canonical) for i in indices),
                "historical identity selection differs from canonical pool")
    require(summary.get("count") == len(indices), "score count differs from selected identities")
    records = {}
    allowed = {"identity", "scores", "choice_text_lengths", "continuation_token_lengths", "canonical_index"}
    if dataset == "mmlu":
        allowed |= {"subject", "doc_index"}
    with (root / "scores.jsonl").open(encoding="utf-8") as handle:
        count = 0
        for count, line in enumerate(handle, 1):
            require(count <= len(indices), "extra score records")
            value = json.loads(line)
            require(set(value).issubset(allowed) and {"identity", "scores"}.issubset(value), "score record includes target/outcome fields")
            index, target = indices[count - 1], canonical[indices[count - 1]]
            if source_id is None and metadata.get("execution_selection") is not None:
                require(type(value.get("canonical_index")) is int and value["canonical_index"] == index,
                        "score canonical index differs from shard selection")
            elif "canonical_index" in value:
                require(type(value["canonical_index"]) is int and value["canonical_index"] == index,
                        "score canonical index differs from source selection")
            require(value["identity"] == row_identity(target, dataset), "score canonical identity/order differs")
            if dataset == "mmlu":
                require(value.get("subject") == target["subject"] and value.get("doc_index") == target["doc_index"], "MMLU subject/index differs")
            choices = choices_for_row(target, dataset)
            scores = finite_scores(value["scores"], len(choices))
            require(all(type(score) in (int, float) for score in value["scores"]), "raw scores must be numeric")
            if source_id is None:
                require(value.get("choice_text_lengths") == [len(choice) for choice in choices], "normalization lengths differ from full choice texts")
                lengths = value.get("continuation_token_lengths")
                require(isinstance(lengths, list) and len(lengths) == len(choices) and
                        all(type(length) is int and length > 0 for length in lengths), "candidate token lengths are incomplete")
            records[index] = scores
    require(count == len(indices), "score root is incomplete")
    acquisition_id = acquisition["cell_id"] if source_id is None else cell["cell_id"]
    return records, {"root": str(root), "cell_id": acquisition_id, "canonical_cell_id": cell["cell_id"], "count": len(indices),
        "source_commit": metadata.get("source_commit"), "job_id": env.get("job_id"),
        "execution_selection": metadata.get("execution_selection"), "indices": indices,
        "runtime_max_length": env.get("max_length"), "max_length_requested": metadata.get("max_length_requested"),
        "runtime_versions": env.get("versions"), "engineering_check": metadata.get("engineering_check")}


def verify(cell_roots, manifest_path, pool_path, scope, config_path=None,
           full_panel=False, include_reuse=False, preflight_indices=None, full_cell=False):
    config = load_config(config_path)
    manifest = read_json(manifest_path)
    cells = validate_score_manifest(manifest, config=config, pool_path=pool_path, scope=scope)
    dataset = manifest["dataset"]
    canonical = load_pool(pool_path, dataset, scope)
    explicit = read_json(preflight_indices) if preflight_indices else None
    covered, roots, sources = {}, [], {}
    for root in cell_roots:
        metadata = read_json(Path(root) / "command_args.json")
        cell_id = metadata.get("cell", {}).get("cell_id")
        require(cell_id in cells, "score root cell is outside Phase 10 panel")
        cell = cells[cell_id]
        require(scope != "FORMAL_TEST" or cell["new_in_phase10"], "formal reuse must use frozen historical mapping")
        expected = select_indices(canonical, cell, scope, explicit)
        scores, record = read_score_root(root, cell, canonical, dataset, scope, pool_path, manifest_path, expected)
        existing = covered.setdefault(record["cell_id"], {})
        require(not set(existing).intersection(scores), "overlapping score attempts for one cell")
        source = (record["source_commit"], record["runtime_max_length"], record["max_length_requested"],
                  identity_key(record["runtime_versions"]), record["engineering_check"])
        require(record["cell_id"] not in sources or sources[record["cell_id"]] == source,
                "cell shards contain different source revisions/runtime settings")
        sources[record["cell_id"]] = source
        existing.update(scores)
        roots.append(record)
    if include_reuse:
        require(scope == "FORMAL_TEST" and dataset == "mmlu", "historical reuse is formal MMLU only")
        for entry in manifest["reuse_mapping"]:
            closure = read_json(entry["source_closure"])
            source_manifest = read_json(entry["source_manifest"])
            require(closure.get("schema") == entry["source_closure_schema"] and closure.get("target_gold_loaded") is False,
                    "historical source closure schema/gold boundary differs")
            require(closure.get("status") in ("HISTORICAL_29_CELLS_VERIFIED", "FULL_NEW_SCORE_PANEL_CLOSED", "FULL_GATE_E_SCORE_PANEL_CLOSED"),
                    "historical source panel is not fully closed")
            require(source_manifest.get("schema") == entry["source_manifest_schema"] and source_manifest.get("target_gold_loaded") is False,
                    "historical source manifest schema/gold boundary differs")
            require(entry["source_cell_id"] in {item["cell_id"] for item in source_manifest["cells"]},
                    "historical source cell is absent from source manifest")
            if "shards" in closure:
                verified_roots = {item["root"] for item in closure["shards"] if item["cell_id"] == entry["source_cell_id"]}
            else:
                verified_roots = {item["root"] for old in closure["cells"] if old["cell_id"] == entry["source_cell_id"] for item in old["roots"]}
            require(set(entry["roots"]) == verified_roots, "historical reuse roots differ from previously verified source cell")
            cell = cells[entry["cell_id"]]
            existing = covered.setdefault(cell["cell_id"], {})
            for root in entry["roots"]:
                scores, record = read_score_root(root, cell, canonical, dataset, scope, pool_path, manifest_path,
                    source_id=entry["source_cell_id"])
                require(not set(existing).intersection(scores), "historical score shard overlap")
                existing.update(scores)
                roots.append(record)
    require(covered, "no score roots supplied")
    expected_cells = set(cells) if include_reuse else {cell["cell_id"] for cell in manifest["score_cells"]}
    if full_cell:
        for acquisition_id, scores in covered.items():
            canonical_id = next(record["canonical_cell_id"] for record in roots if record["cell_id"] == acquisition_id)
            expected = select_indices(canonical, cells[canonical_id], scope, explicit)
            require(set(scores) == set(expected), "full cell closure is missing canonical identities")
    if full_panel:
        require(scope == "FORMAL_TEST" and set(covered) == expected_cells, "full closure is missing required configurations")
        require(all(set(scores) == set(range(len(canonical))) for scores in covered.values()), "full closure is missing canonical identities")
        new_ids = {cell["cell_id"] for cell in manifest["score_cells"]}
        require(len({record["source_commit"] for record in roots if record["cell_id"] in new_ids}) == 1,
                "new score panel contains different source revisions")
    all_complete = full_panel and set(covered) == set(cells)
    closure = {"schema": "loopscope.phase10.score_verification.v1",
        "status": "FULL_PHASE10_SCORE_PANEL_CLOSED" if all_complete else ("FULL_PHASE10_NEW_SCORE_PANEL_CLOSED" if full_panel else
            ("FULL_PHASE10_SCORE_CELLS_CLOSED" if full_cell else "SCORE_ROOTS_CLOSED")),
        "target_gold_loaded": False, "dataset": dataset, "dataset_recipe": manifest["dataset_recipe"],
        "scope": scope, "manifest": str(manifest_path), "pool": str(pool_path),
        "cell_count": len(covered), "sample_count": sum(len(values) for values in covered.values()),
        "counts_by_cell": {key: len(value) for key, value in covered.items()}, "roots": roots}
    closure["canonical_pool_count"] = len(canonical)
    closure["indices_by_cell"] = {key: sorted(values) for key, values in covered.items()}
    closure["complete_cells"] = [key for key, values in covered.items() if set(values) == set(range(len(canonical)))]
    aligned = {"schema": "loopscope.phase10.aligned_scores.v1", "dataset": dataset, "target_gold_loaded": False,
        "cells": {cell_id: [scores[index] for index in sorted(scores)] for cell_id, scores in covered.items()},
        "indices_by_cell": closure["indices_by_cell"],
        "identities": [row_identity(row, dataset) for row in canonical],
        "choice_lengths": [[len(choice) for choice in choices_for_row(row, dataset)] for row in canonical]}
    if dataset == "mmlu":
        aligned["subjects"] = [row["subject"] for row in canonical]
    return closure, aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--scope", choices=("PREFLIGHT_ONLY", "FORMAL_TEST"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cell-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--preflight-indices", type=Path)
    parser.add_argument("--full-panel", action="store_true")
    parser.add_argument("--full-cell", action="store_true", help="require exact canonical union for every supplied cell")
    parser.add_argument("--include-reuse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aligned-output", type=Path)
    args = parser.parse_args()
    value, aligned = verify(args.cell_roots, args.manifest, args.pool, args.scope, args.config,
        args.full_panel, args.include_reuse, args.preflight_indices, args.full_cell)
    if args.aligned_output:
        require(value["status"] == "FULL_PHASE10_SCORE_PANEL_CLOSED", "aligned analysis input requires complete frozen-panel closure")
        write_json_once(args.aligned_output, aligned)
    write_json_once(args.output, value)
    print(json.dumps({key: value[key] for key in ("status", "cell_count", "sample_count", "target_gold_loaded")}))


if __name__ == "__main__":
    main()
