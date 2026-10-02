#!/usr/bin/env python3
"""Validate the Gate C shard path against the same two unsharded MMLU validation rows.

Each combination runs three sequential fresh producer processes through the
ordinary accuracy launcher. No target labels, outcomes, or formal test scoring
are permitted here. The bound executor submits this runner on debug.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tflt.loopscope.phase10_accuracy import finite_scores, load_pool, require, select_indices, write_json_once
from tflt.loopscope.phase10_panel import load_config, validate_score_manifest


CELL_IDS = (
    "q4-w15-18-k2-online-fixed-t0-lambda0.1",
    "q4-w15-18-k3-online-fixed-t0-lambda0.1",
    "q17-w12-15-k2-online-fixed-t0-lambda0.1",
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--pool", type=Path, required=True)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--commit", required=True)
    result.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument("--config", type=Path)
    result.add_argument("--preflight-indices", type=Path, help="Exactly two sorted validation indices; default [0,368]")
    result.add_argument("--cell-id", choices=CELL_IDS, help="One combination per allocation; default runs all three sequentially")
    result.add_argument("--max-length", type=int)
    result.add_argument("--dry-run", action="store_true", help="Validate inputs and print the plan without model loading or writes")
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preflight_plan(args):
    manifest = read_json(args.manifest)
    config = load_config(args.config)
    cells = validate_score_manifest(manifest, config=config, pool_path=args.pool, scope="PREFLIGHT_ONLY")
    require(manifest["dataset"] == "mmlu", "shard preflight requires MMLU")
    rows = load_pool(args.pool, "mmlu", "PREFLIGHT_ONLY")
    require(all(row.get("split") == "validation" for row in rows), "shard preflight permits validation targets only")
    indices = read_json(args.preflight_indices) if args.preflight_indices else [0, 368]
    require(isinstance(indices, list) and len(indices) == 2, "shard preflight requires exactly two canonical indices")
    selected = [args.cell_id] if args.cell_id else list(CELL_IDS)
    for cell_id in selected:
        require(cell_id in cells, "preflight cell is absent from manifest")
        cell = cells[cell_id]
        require(cell["arm"] == "Online" and cell["direction_policy"] == "fixed_t0" and cell["strength"] == 0.1,
                "shard preflight requires the frozen lambda0.1 fixed_t0 configuration")
        require(select_indices(rows, cell, "PREFLIGHT_ONLY", indices) == indices, "preflight selection differs")
    return {"schema": "loopscope.phase10.shard_preflight_plan.v1", "scope": "PREFLIGHT_ONLY",
            "dataset": "mmlu", "target_gold_loaded": False, "indices": indices,
            "identities": [rows[i]["identity"] for i in indices], "cell_ids": selected,
            "source_commit": args.commit, "manifest": str(args.manifest), "pool": str(args.pool),
            "repo": str(args.repo), "python": str(args.python), "packing": 1,
            "producer_process_count": 3 * len(selected)}


def score_records(root):
    """Return complete raw records keyed by identity; verifier closes recipes first."""
    values = {}
    with (Path(root) / "scores.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            identity = json.dumps(value["identity"], sort_keys=True, ensure_ascii=False)
            require(identity not in values, "duplicate preflight score identity")
            require(len(value["scores"]) == 4, "MMLU preflight requires all four candidate scores")
            finite_scores(value["scores"], 4)
            values[identity] = {key: item for key, item in value.items() if key != "canonical_index"}
    require(values, "empty preflight scores")
    return values


def compare_roots(reference, shards):
    expected = score_records(reference)
    merged = {}
    for root in shards:
        records = score_records(root)
        require(not set(merged).intersection(records), "overlapping preflight shards")
        merged.update(records)
    require(set(merged) == set(expected), "preflight shard identity union differs from reference")
    errors = [abs(a - b) for key in expected for a, b in zip(expected[key]["scores"], merged[key]["scores"])]
    require(expected == merged, "sharded candidate scores or record metadata differ from unsharded reference")
    return {"exact": True, "sample_count": len(expected), "candidate_count": len(errors),
            "max_abs_error": max(errors)}


def execute(command, env, log_path):
    # Preserve failures in a fresh attempt; the parent owns any subsequent retry.
    with Path(log_path).open("x", encoding="utf-8") as output:
        subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)


def run(args):
    plan = preflight_plan(args)
    if args.dry_run:
        print(json.dumps({**plan, "status": "PHASE10_SHARD_PREFLIGHT_DRY_RUN"}, sort_keys=True))
        return 0
    require(os.environ.get("SLURM_JOB_ID"), "shard preflight requires an authorized Slurm allocation")
    require(os.environ.get("SLURM_JOB_PARTITION") == "debug", "shard preflight must run on debug")
    args.run_root.mkdir(parents=True, exist_ok=False)
    plan.update(job_id=os.environ["SLURM_JOB_ID"], partition=os.environ["SLURM_JOB_PARTITION"])
    write_json_once(args.run_root / "plan.json", plan)
    indices_path = args.run_root / "indices.json"
    write_json_once(indices_path, plan["indices"])
    env = dict(os.environ)
    # Use the exact production launcher/cache/runtime path without inherited controls.
    for key in ("PHASE10_ENGINEERING_CHECK", "PHASE10_SHARD_INDICES", "PHASE10_SHARD_ID", "DRY_RUN", "PHASE10_CONFIG"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(args.repo / "src"), PYTHONDONTWRITEBYTECODE="1",
               PHASE10_PREFLIGHT_INDICES=str(indices_path))
    if args.config:
        env["PHASE10_CONFIG"] = str(args.config)
    reports = []
    started = time.monotonic()
    for cell_id in plan["cell_ids"]:
        cell_root = args.run_root / cell_id
        cell_root.mkdir()
        reference = cell_root / "reference"
        shards = [cell_root / ("shard-" + str(i)) for i in range(2)]
        for role, root in [("reference", reference)] + [("shard-" + str(i), root) for i, root in enumerate(shards)]:
            child_env = dict(env)
            if role != "reference":
                shard_path = cell_root / (role + "-indices.json")
                write_json_once(shard_path, [plan["indices"][int(role[-1])]])
                child_env.update(PHASE10_SHARD_INDICES=str(shard_path), PHASE10_SHARD_ID=role)
            command = ["bash", str(args.repo / "scripts/loopscope/phase10_accuracy.sbatch"),
                       str(args.repo), str(args.python), "mmlu", str(args.manifest), str(args.pool),
                       cell_id, str(root), args.commit, "PREFLIGHT_ONLY",
                       str(args.max_length) if args.max_length is not None else "-"]
            execute(command, child_env, cell_root / (role + ".log"))
        closures = {}
        for role, roots in (("reference", [reference]), ("merged", shards)):
            closure_path = cell_root / (role + "-closure.json")
            command = [str(args.python), str(args.repo / "scripts/loopscope/verify_phase10_scores.py"),
                       "--manifest", str(args.manifest), "--pool", str(args.pool), "--scope", "PREFLIGHT_ONLY",
                       "--preflight-indices", str(indices_path), "--full-cell", "--cell-roots",
                       *map(str, roots), "--output", str(closure_path)]
            if args.config:
                command += ["--config", str(args.config)]
            execute(command, env, cell_root / (role + "-verification.log"))
            closure = read_json(closure_path)
            require(closure["target_gold_loaded"] is False and closure["counts_by_cell"] == {cell_id: 2},
                    "preflight verifier did not close the two-index cell")
            closures[role] = str(closure_path)
        comparison = compare_roots(reference, shards)
        reports.append({"cell_id": cell_id, "reference_root": str(reference), "shard_roots": list(map(str, shards)),
                        "closures": closures, **comparison})
        print(json.dumps({"status": "SHARD_COMBINATION_VERIFIED", **reports[-1]}), flush=True)
    result = {"schema": "loopscope.phase10.shard_preflight.v1", "status": "PHASE10_SHARD_PREFLIGHT_VERIFIED",
              "scope": "PREFLIGHT_ONLY", "dataset": "mmlu", "target_gold_loaded": False,
              "source_commit": args.commit, "job_id": plan["job_id"], "partition": "debug",
              "indices": plan["indices"], "reports": reports, "elapsed_seconds": time.monotonic() - started}
    write_json_once(args.run_root / "SHARD_PREFLIGHT_VERIFIED.json", result)
    print(json.dumps({key: result[key] for key in ("status", "target_gold_loaded", "elapsed_seconds")}), flush=True)
    return 0


def main(argv=None):
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
