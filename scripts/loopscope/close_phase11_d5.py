#!/usr/bin/env python3
"""Close all D5 raw records before saving predictions; never open target gold."""
import argparse
import json
import math
from pathlib import Path

from tflt.loopscope.phase11_panel import require, write_json_once
from tflt.loopscope.phase11_s2 import extract_s2
from tflt.loopscope.phase10_accuracy import load_pool as load_base_pool
import phase11_d5_base as base
import phase11_d5_instruct as instruct

COUNTS = {"p10-base-mmlu": 14042, "p10-base-arc": 1172,
          "p11-instruct-arc": 1172, "p11-instruct-gpqa": 448}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(cell):
    return cell["family"], cell["repeat"], cell["cell_id"]


def validate_recipe(recipe):
    expected = set()
    base_cells = {"p10-base-mmlu": (base.NATIVE, base.LOOP, base.CURRENT),
                  "p10-base-arc": (base.NATIVE, base.LOOP, base.FIXED)}
    for family in COUNTS:
        cells = base_cells[family] if family in base_cells else instruct.selected_cells(instruct.FAMILIES[family])
        expected.update((family, repeat, cell) for repeat in (1, 2) for cell in cells)
    cells = recipe["cells"]
    require(len(cells) == 28 and {key(c) for c in cells} == expected,
            "D5 recipe differs from 28 frozen independent repeat cells")
    require(recipe.get("target_gold_loaded") is False and recipe.get("record_count") == 102796,
            "D5 recipe gold barrier/count differs")
    for cell in cells:
        require(cell["population_count"] == COUNTS[cell["family"]] and
                type(cell["repeat"]) is int and cell["seed"] == {1: 20261014, 2: 20261015}[cell["repeat"]] and
                cell["synthetic"] is False and cell.get("engineering", False) is False,
                "D5 recipe population/seed/formal scope differs")


def membership(recipe, jobs):
    specs = {key(c): c for c in recipe["cells"]}
    require(len(specs) == len(recipe["cells"]), "recipe repeats a cell")
    covered = {k: set() for k in specs}
    roots = set()
    for job in jobs:
        k = key(job)
        require(k in specs and all(job.get(field) == value for field, value in specs[k].items()),
                "plan job differs from frozen recipe")
        indices = job["indices"]
        require(indices and indices == sorted(set(indices)) and
                all(type(i) is int and 0 <= i < specs[k]["population_count"] for i in indices),
                "plan indices differ from canonical population")
        require(not covered[k].intersection(indices), "plan duplicates cell/canonical indices")
        covered[k].update(indices)
        root = str(Path(job["run_root"]).resolve())
        require(root not in roots, "plan repeats an output root")
        roots.add(root)
    require(all(v == set(range(specs[k]["population_count"])) for k, v in covered.items()),
            "plan is missing canonical indices or cells")
    return specs


def load_context(job):
    if job["family"].startswith("p10-"):
        base.validate_job(job)
        dataset = "mmlu" if job["family"] == "p10-base-mmlu" else "arc_challenge"
        # CPU-only safe projection and direct original-manifest binding. The
        # actual immutable producer imports were checked in its process; never
        # mix historical tflt packages into this multi-family closure process.
        manifest = read(job["manifest"])
        matches = [cell for cell in manifest["cells"] if cell["cell_id"] == job["cell_id"]]
        require(len(matches) == 1, "original Base manifest cell absent or repeated")
        cell = matches[0]
        require(cell["model"] == base.MODEL and cell["revision"] == base.REVISION and
                cell["dtype"] == "bfloat16" and cell["cache_strategy"] == "first" and
                (cell["arm"] == "Native" or (cell["window"] == [15, 18] and cell["k"] == 2)),
                "original Base manifest model/window/cache recipe differs")
        rows = load_base_pool(Path(job["pool"]), dataset, "FORMAL_TEST")
        return {"dataset": dataset, "cell": cell, "rows": rows,
                "validator": base.record_validator(job, rows, dataset)}
    dataset, scope, config, cell, pool, indices = instruct.load_job(job)
    meta = instruct.binding(job, dataset, scope, config, cell, indices)
    return {"dataset": dataset, "cell": cell, "rows": pool["rows"], "metadata": meta,
            "validator": lambda i, r, t: instruct.validate_bundle(i, r, t, meta, pool)}


def inspect_attempt(job, context):
    root = Path(job["run_root"])
    command, summary = read(root / "command_args.json"), read(root / "summary.json")
    is_base = job["family"].startswith("p10-")
    require(summary.get("status") == ("SCORES_COMPLETE" if is_base else "RAW_COMPLETE") and
            summary.get("target_gold_loaded") is False and summary.get("count") == len(job["indices"]) and
            summary.get("metadata") == command, "attempt summary is incomplete or bound differently")
    if is_base:
        require(command.get("schema") == "loopscope.phase11.d5.base_attempt.v1" and
                all(command.get("job", {}).get(k) == v for k, v in job.items()) and
                command.get("cell") == context["cell"] and command.get("dataset") == context["dataset"] and
                command.get("indices") == job["indices"] and command.get("target_gold_loaded") is False,
                "Base attempt configuration differs")
    else:
        require(command == context["metadata"], "Instruct attempt configuration differs")
    records, telemetry = read_lines(root / "records.jsonl"), read_lines(root / "telemetry.jsonl")
    indices = job["indices"]
    require([r.get("canonical_index") for r in records] == indices and
            [t.get("canonical_index") for t in telemetry] == indices,
            "attempt records/telemetry are missing, duplicated or out of order")
    complete = root / "complete"
    require({p.name for p in complete.glob("*.json")} == {f"{i:06d}.json" for i in indices},
            "atomic record population differs from declared attempt")
    checked_environments = set()
    for index, record, fact in zip(indices, records, telemetry):
        context["validator"](index, record, fact)
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            require(type(fact.get(name)) is int and fact[name] >= 0, "attempt lacks legal memory telemetry")
        if is_base:
            environment_path = fact["environment_file"]
            if environment_path not in checked_environments:
                environment = read(environment_path)
                require(all(environment.get(k) == v for k, v in base.binding(job).items()) and
                        environment.get("effective_rng_seeds") ==
                        {k: job["seed"] for k in ("python", "numpy", "torch", "cuda_all_devices")},
                        "Base effective process RNG seed/binding differs")
                checked_environments.add(environment_path)
            row = context["rows"][index]
            lengths = [1] * 4 if context["dataset"] == "mmlu" else [len(c) for c in row["choices"]]
            require(record.get("choice_text_lengths") == lengths,
                    "Base normalization differs from original full choices")
            if context["dataset"] == "mmlu":
                require(record.get("subject") == row["subject"] and record.get("doc_index") == row["doc_index"],
                        "Base MMLU subject/doc index differs")
            positions = fact.get("position_metadata", [])
            require(len(positions) == len(record["scores"]) and
                    [p.get("continuation_length") for p in positions] == record["continuation_token_lengths"],
                    "Base scoring position metadata differs")
            frozen = row.get("tokenization", {}).get(context["cell"]["model"])
            if frozen is not None:
                require(record["continuation_token_lengths"] == frozen["continuation_lengths"] and
                        all(p["left_truncated_tokens"] == 0 for p in positions),
                        "Base ARC frozen tokenization/truncation differs")
    if not is_base:
        environments = list((root / "attempts").glob("*/env.json"))
        require(environments, "Instruct attempt lacks process environment")
        for path in environments:
            environment = read(path)
            require(environment.get("metadata") == command and environment.get("effective_rng_seeds") ==
                    {k: job["seed"] for k in ("python", "numpy", "torch", "cuda_all_devices")},
                    "Instruct effective process RNG seed/binding differs")
    return records


def prediction(job, record):
    result = {k: job[k] for k in ("family", "repeat", "cell_id", "seed")}
    result.update(identity=record["identity"], canonical_index=record["canonical_index"])
    if job["family"] == "p11-instruct-gpqa":
        result.update(metric="S2", **extract_s2(record["generated_text"]),
                      generated_token_count=record["generated_token_count"], end_reason=record["end_reason"])
        return result
    scores = record["scores"]
    require(scores and all(type(v) in (float, int) and math.isfinite(v) for v in scores),
            "prediction requires finite full numeric scores")
    raw = max(range(len(scores)), key=scores.__getitem__)
    arc = job["family"].endswith("arc")
    lengths = record["choice_text_lengths"]
    require(len(scores) == len(lengths) and all(type(n) is int and n > 0 for n in lengths),
            "prediction normalization lengths differ")
    predicted = max(range(len(scores)), key=lambda i: scores[i] / lengths[i]) if arc else raw
    result.update(metric="acc_norm" if arc else "acc", prediction_index=predicted,
                  raw_prediction_index=raw)
    if job["family"] == "p10-base-mmlu":
        result.update(subject=record["subject"], doc_index=record["doc_index"])
    return result


def close(plan, recipe):
    validate_recipe(recipe)
    jobs = [job for task in plan["tasks"] for job in task["jobs"]]
    specs = membership(recipe, jobs)
    aligned = {k: {} for k in specs}
    contexts, canonical, roots = {}, {}, {k: [] for k in specs}
    for job in jobs:
        cache_key = tuple(job.get(k) for k in ("family", "cell_id", "pool", "manifest", "producer_source"))
        # Instruct context includes repeat/job metadata, so reuse only Base rows.
        if job["family"].startswith("p10-") and cache_key in contexts:
            cached = contexts[cache_key]
            context = {**cached, "validator": base.record_validator(job, cached["rows"], cached["dataset"])}
        else:
            context = load_context(job)
            if job["family"].startswith("p10-"):
                contexts[cache_key] = context
        require(len(context["rows"]) == specs[key(job)]["population_count"], "original pool count differs")
        identities = [base.identity(row, context["dataset"]) if job["family"].startswith("p10-")
                      else row["identity"] for row in context["rows"]]
        require(len(set(identities)) == len(identities), "original pool duplicates canonical identity")
        if job["family"] in canonical:
            require(identities == canonical[job["family"]], "family cells/repeats use different canonical identities")
        else:
            canonical[job["family"]] = identities
        records = inspect_attempt(job, context)
        for record in records:
            index = record["canonical_index"]
            require(index not in aligned[key(job)], "raw cell duplicates a canonical record")
            aligned[key(job)][index] = record
        roots[key(job)].append({"run_root": job["run_root"], "indices": job["indices"],
                               "records_path": str(Path(job["run_root"]) / "records.jsonl"),
                               "telemetry_path": str(Path(job["run_root"]) / "telemetry.jsonl")})
    require(sum(len(v) for v in aligned.values()) == 102796 and
            all(set(v) == set(range(specs[k]["population_count"])) for k, v in aligned.items()),
            "D5 raw records are missing or duplicated")
    predictions, cells = [], []
    for spec in recipe["cells"]:
        k = key(spec)
        ordered = [aligned[k][i] for i in range(spec["population_count"])]
        predictions.extend(prediction(spec, row) for row in ordered)
        cells.append({**spec, "record_count": len(ordered), "raw_roots": roots[k]})
    return {"schema": "loopscope.phase11.d5.raw_prediction_closure.v1", "status": "GOLD_UNREAD",
            "target_gold_loaded": False, "prediction_closed": True, "record_count": 102796, "cell_count": 28,
            "cells": cells, "prediction_rules": {"p10-base-mmlu": "raw full letter score argmax",
                "arc": "full choice text character-normalized argmax; raw argmax retained",
                "gpqa": "unchanged phase11_s2.extract_s2"}}, predictions


def run(args):
    closure, predictions = close(read(args.plan), read(args.recipe))
    # No output exists until all 102796 raw records have passed closure.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "predictions.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    closure.update(plan=str(args.plan), recipe=str(args.recipe), predictions_path=str(path))
    write_json_once(args.output_dir / "closure.json", closure)
    print(json.dumps({"status": "GOLD_UNREAD", "record_count": closure["record_count"],
                      "predictions_path": str(path), "target_gold_loaded": False}))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(run(p.parse_args()))
