#!/usr/bin/env python3
"""Run the frozen ARC panel in one allocation, without target gold or outcomes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

SAFETY_FACTOR = 1.2
CLEANUP_SECONDS = 120.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def forecast_remaining(cells, measurements, gpu_count):
    """Estimate unmeasured cells from measured matching Online full-cell wall time.

    ``cells`` contains all panel cell dictionaries; ``measurements`` maps measured
    cell IDs to elapsed seconds including producer and per-cell verification.
    """
    require(gpu_count in (1, 2), "GPU count must be one or two")
    by_id = {cell["cell_id"]: cell for cell in cells}
    require(set(measurements).issubset(by_id), "measurement outside panel")
    require(all(math.isfinite(value) and value > 0 for value in measurements.values()),
            "measurements must be positive finite elapsed seconds")
    durations = {}
    for cell in cells:
        if cell["cell_id"] in measurements:
            continue
        def same_recipe(other):
            return all(other.get(key) == cell.get(key) for key in ("model", "window", "k"))
        candidates = [elapsed for cell_id, elapsed in measurements.items()
                      if by_id[cell_id]["arm"] == "Online" and same_recipe(by_id[cell_id])
                      and (cell["arm"] == "Loop" or
                           by_id[cell_id].get("direction_policy") == cell.get("direction_policy"))]
        require(cell["arm"] in ("Online", "Loop") and candidates,
                "missing matching measured Online cell for " + cell["cell_id"])
        durations[cell["cell_id"]] = max(candidates)
    order = sorted(durations, key=lambda cell_id: (-durations[cell_id], cell_id))
    loads = [0.0] * gpu_count
    for cell_id in order:
        slot = min(range(gpu_count), key=lambda index: (loads[index], index))
        loads[slot] += durations[cell_id]
    makespan = max(loads)
    return {"durations_seconds": durations, "order": order,
            "makespan_seconds": makespan, "forecast_gpu_hours": gpu_count * makespan / 3600}


def budget_admission(prior_gpu_hours, allocated_elapsed_seconds, gpu_count,
                     forecast_seconds, walltime_seconds):
    """Charge every allocated GPU, including idle time and cleanup allowance."""
    require(gpu_count in (1, 2), "GPU count must be one or two")
    require(all(math.isfinite(value) and value >= 0 for value in
                (prior_gpu_hours, allocated_elapsed_seconds, forecast_seconds, walltime_seconds)),
            "budget inputs must be nonnegative finite values")
    required = SAFETY_FACTOR * forecast_seconds + CLEANUP_SECONDS
    projected = prior_gpu_hours + gpu_count * (allocated_elapsed_seconds + required) / 3600
    remaining = walltime_seconds - allocated_elapsed_seconds
    reason = "ADMITTED" if projected <= 100 and required <= remaining else (
        "GPU_HOURS_EXCEEDED" if projected > 100 else "WALLTIME_EXCEEDED")
    return {"admitted": reason == "ADMITTED", "projected_gpu_hours": projected,
            "remaining_walltime_seconds": remaining, "required_walltime_seconds": required,
            "reason": reason}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


class Bundle:
    def __init__(self, args, cells, gpu_ids):
        self.args, self.cells, self.gpu_ids = args, cells, gpu_ids
        self.lock = threading.Lock()
        self.processes = set()
        self.stop = threading.Event()
        self.receipts = {}

    def event(self, name, **details):
        value = {"event": name, "epoch": time.time(), "target_gold_loaded": False, **details}
        with self.lock:
            with (self.args.run_root / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def terminate_owned(self):
        self.stop.set()
        with self.lock:
            processes = list(self.processes)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def command(self, command, log, env, deadline):
        if self.stop.is_set():
            return 125
        timeout = deadline - time.time()
        if timeout <= 0:
            return 124
        with Path(log).open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=self.args.repo, env=env,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            with self.lock:
                self.processes.add(process)
            try:
                # Short bounded waits also allow an owned child ignoring TERM to
                # be killed promptly after another worker fails.
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0 or self.stop.is_set():
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        return 124 if remaining <= 0 else 125
                    try:
                        return process.wait(timeout=min(1, remaining))
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                with self.lock:
                    self.processes.discard(process)

    def verify_command(self, roots, output, full_panel=False):
        command = [str(self.args.python), str(self.args.repo / "scripts/loopscope/verify_phase10_scores.py"),
                   "--manifest", str(self.args.manifest), "--pool", str(self.args.pool),
                   "--scope", self.args.scope, "--full-cell", "--cell-roots", *map(str, roots),
                   "--output", str(output)]
        if self.args.preflight_indices:
            command += ["--preflight-indices", str(self.args.preflight_indices)]
        if full_panel:
            command += ["--full-panel", "--aligned-output", str(self.args.run_root / "arc-aligned-scores.json")]
        return command

    def cell(self, cell_id, slot, deadline):
        started = time.time()
        attempt = self.args.run_root / "cells" / cell_id
        root = attempt / "scores"
        attempt.mkdir(exist_ok=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu_ids[slot])
        env.pop("DRY_RUN", None)
        env.pop("PHASE10_CONFIG", None)
        env.pop("PHASE10_SHARD_INDICES", None)
        env.pop("PHASE10_SHARD_ID", None)
        env.pop("PHASE10_ENGINEERING_CHECK", None)
        env.pop("PHASE10_PREFLIGHT_INDICES", None)
        if self.args.preflight_indices:
            env["PHASE10_PREFLIGHT_INDICES"] = str(self.args.preflight_indices)
        producer = ["/bin/bash", str(self.args.repo / "scripts/loopscope/phase10_accuracy.sbatch"),
                    str(self.args.repo), str(self.args.python), "arc_challenge", str(self.args.manifest),
                    str(self.args.pool), cell_id, str(root), self.args.commit, self.args.scope, "32768"]
        verifier = self.verify_command([root], attempt / "closure.json")
        self.event("CELL_START", cell_id=cell_id, gpu_slot=slot, visible_gpu=self.gpu_ids[slot], command=producer)
        producer_exit = verifier_exit = None
        error = None
        try:
            producer_exit = self.command(producer, attempt / "producer.log", env, deadline)
            if producer_exit == 0:
                verifier_exit = self.command(verifier, attempt / "verifier.log", env, deadline)
        except Exception as exc:
            error = repr(exc)
        summary = read_json(root / "summary.json") if (root / "summary.json").exists() else {}
        receipt = {"cell_id": cell_id, "gpu_slot": slot, "visible_gpu": self.gpu_ids[slot],
                   "scope": self.args.scope, "root": str(root), "producer_command": producer,
                   "verifier_command": verifier, "producer_exit": producer_exit, "verifier_exit": verifier_exit,
                   "elapsed_seconds": time.time() - started, "started_epoch": started,
                   "deadline_epoch": deadline,
                   "peak_allocated_bytes": summary.get("peak_allocated_bytes"),
                   "peak_reserved_bytes": summary.get("peak_reserved_bytes"),
                   "model_loading": summary.get("model_loading"), "error": error,
                   "status": "CELL_VERIFIED" if producer_exit == verifier_exit == 0 else "CELL_FAILED",
                   "target_gold_loaded": False}
        write_once(attempt / "receipt.json", receipt)
        self.event(receipt["status"], **receipt)
        if receipt["status"] != "CELL_VERIFIED":
            self.terminate_owned()
        return receipt

    def stage(self, order, deadline):
        pending = iter(order)
        active = {}
        failed = False
        with ThreadPoolExecutor(max_workers=self.args.gpu_count) as executor:
            def dispatch(slot):
                if self.stop.is_set() or time.time() >= deadline:
                    return
                cell_id = next(pending, None)
                if cell_id is not None:
                    active[executor.submit(self.cell, cell_id, slot, deadline)] = slot
            for slot in range(self.args.gpu_count):
                dispatch(slot)
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                finished_slots = []
                for future in done:
                    slot = active.pop(future)
                    finished_slots.append(slot)
                    receipt = future.result()
                    self.receipts[receipt["cell_id"]] = receipt
                    if receipt["status"] != "CELL_VERIFIED":
                        failed = True
                        self.terminate_owned()
                if not failed:
                    for slot in finished_slots:
                        dispatch(slot)
        return not failed and not self.stop.is_set() and all(cell_id in self.receipts for cell_id in order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "python", "manifest", "pool", "plan", "run-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--scope", choices=("PREFLIGHT_ONLY", "FORMAL_TEST"), required=True)
    parser.add_argument("--gpu-count", type=int, choices=(1, 2), required=True)
    parser.add_argument("--prior-gpu-hours", type=float, required=True)
    parser.add_argument("--walltime-hours", type=float, required=True)
    parser.add_argument("--allocation-start-epoch", type=float, required=True)
    parser.add_argument("--preflight-indices", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # These modules use only the standard library; model loading lives in children.
    from tflt.loopscope.phase10_accuracy import load_pool, select_indices
    from tflt.loopscope.phase10_panel import strategy_id, validate_score_manifest
    manifest, plan = read_json(args.manifest), read_json(args.plan)
    cells = validate_score_manifest(manifest, pool_path=args.pool, scope=args.scope)
    require(manifest["dataset"] == "arc_challenge" and len(cells) == 68,
            "bundle requires frozen ARC68 panel")
    require(plan.get("target_gold_loaded") is False, "plan must be gold-free")
    initial, remaining = plan["initial_cells"], plan["remaining_cells"]
    require(len(initial) == 9 and len(remaining) == 59 and
            len(set(initial + remaining)) == 68 and set(initial + remaining) == set(cells),
            "plan must partition ARC68 into first9 and remaining59")
    require(sum(cells[cell_id]["arm"] == "Native" for cell_id in initial) == 2 and
            sum(cells[cell_id]["arm"] == "Online" for cell_id in initial) == 7,
            "first9 must contain two Native and seven transfer Online cells")
    require(len({strategy_id(cells[cell_id]) for cell_id in initial
                 if cells[cell_id]["arm"] == "Online"}) == 7,
            "first9 Online cells must cover all seven strategy groups")
    rows = load_pool(args.pool, "arc_challenge", args.scope)
    explicit = read_json(args.preflight_indices) if args.preflight_indices else None
    if args.scope == "PREFLIGHT_ONLY":
        require(all(row["split"] == "validation" for row in rows), "preflight requires validation pool")
        require(all(len(select_indices(rows, cells[cell_id], args.scope, explicit)) == 6 for cell_id in initial),
                "preflight requires six validation indices per cell")
    else:
        require(explicit is None, "formal cannot select preflight indices")
    require(math.isfinite(args.walltime_hours) and args.walltime_hours > 0 and
            math.isfinite(args.prior_gpu_hours) and 0 <= args.prior_gpu_hours < 40,
            "invalid walltime/prior budget")
    require(args.prior_gpu_hours + args.gpu_count * args.walltime_hours <= 100,
            "allocation reservation exceeds total100GPUh")
    require(math.isfinite(args.allocation_start_epoch) and args.allocation_start_epoch > 0,
            "real allocation start epoch required")
    require(not args.run_root.exists(), "bundle run root must be fresh")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "initial_cells": initial,
                          "remaining_cells": remaining if args.scope == "FORMAL_TEST" else [],
                          "scope": args.scope, "gpu_count": args.gpu_count, "target_gold_loaded": False}))
        return 0
    require(os.environ.get("SLURM_JOB_ID", "").isdigit(), "bundle requires Slurm allocation")
    require(args.allocation_start_epoch <= time.time(), "allocation start is in the future")
    if args.scope == "PREFLIGHT_ONLY":
        require(os.environ.get("SLURM_JOB_PARTITION") == "debug" and args.walltime_hours < 0.5,
                "preflight must use debug under30min")
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    require(len(gpu_ids) == args.gpu_count and all(gpu_ids) and len(set(gpu_ids)) == args.gpu_count,
            "CUDA_VISIBLE_DEVICES must enumerate exactly the allocated GPUs")
    args.run_root.mkdir(parents=True, exist_ok=False)
    (args.run_root / "cells").mkdir()
    bundle = Bundle(args, cells, gpu_ids)
    signal.signal(signal.SIGTERM, lambda *_: bundle.terminate_owned())
    signal.signal(signal.SIGINT, lambda *_: bundle.terminate_owned())
    deadline = args.allocation_start_epoch + args.walltime_hours * 3600 - CLEANUP_SECONDS
    initial_deadline = min(deadline, args.allocation_start_epoch +
                           (40 - args.prior_gpu_hours) / args.gpu_count * 3600 - CLEANUP_SECONDS)
    bundle.event("BUNDLE_START", arguments={key: str(value) if isinstance(value, Path) else value
                                           for key, value in vars(args).items()}, gpu_ids=gpu_ids)
    status, admission = "CELL_FAILURE", None
    if bundle.stage(initial, initial_deadline):
        if args.scope == "PREFLIGHT_ONLY":
            command = bundle.verify_command([Path(bundle.receipts[cell_id]["root"]) for cell_id in initial],
                                            args.run_root / "preflight-closure.json")
            exit_code = bundle.command(command, args.run_root / "preflight-verifier.log", dict(os.environ), deadline)
            status = "PREFLIGHT_COMPLETE" if exit_code == 0 else "CLOSURE_FAILURE"
        else:
            forecast = forecast_remaining(list(cells.values()),
                                          {key: value["elapsed_seconds"] for key, value in bundle.receipts.items()},
                                          args.gpu_count)
            admission = budget_admission(args.prior_gpu_hours, time.time() - args.allocation_start_epoch,
                                         args.gpu_count, forecast["makespan_seconds"], args.walltime_hours * 3600)
            bundle.event("BUDGET_ADMISSION", forecast=forecast, admission=admission)
            if not admission["admitted"]:
                status = "BUDGET_STOP"
            elif bundle.stage(forecast["order"], deadline):
                command = bundle.verify_command([Path(bundle.receipts[cell_id]["root"]) for cell_id in initial + remaining],
                                                args.run_root / "arc-full-panel-closure.json", full_panel=True)
                exit_code = bundle.command(command, args.run_root / "panel-verifier.log", dict(os.environ),
                                           args.allocation_start_epoch + args.walltime_hours * 3600 - 10)
                status = "FULL_ARC_PANEL_COMPLETE" if exit_code == 0 else "CLOSURE_FAILURE"
    if status == "CELL_FAILURE":
        failures = [value for value in bundle.receipts.values() if value["status"] == "CELL_FAILED"]
        genuine_failure = any(value["error"] or
                              value["producer_exit"] not in (0, 124, 125) or
                              value["verifier_exit"] not in (None, 0, 124, 125)
                              for value in failures)
        if not genuine_failure and any(124 in (value["producer_exit"], value["verifier_exit"])
                                       for value in failures):
            status = "BUDGET_STOP"
        elif not failures and not bundle.stop.is_set():
            status = "BUDGET_STOP"
        elif not genuine_failure and bundle.stop.is_set():
            status = "INTERRUPTED"
    elapsed = time.time() - args.allocation_start_epoch
    result = {"status": status, "scope": args.scope, "target_gold_loaded": False,
              "verified_cells": [key for key, value in bundle.receipts.items() if value["status"] == "CELL_VERIFIED"],
              "unstarted_cells": [cell_id for cell_id in initial + remaining if cell_id not in bundle.receipts],
              "receipts": bundle.receipts, "admission": admission, "allocated_elapsed_seconds": elapsed,
              "allocated_gpu_hours": args.gpu_count * elapsed / 3600,
              "cumulative_gpu_hours": args.prior_gpu_hours + args.gpu_count * elapsed / 3600}
    write_once(args.run_root / "bundle-summary.json", result)
    bundle.event("BUNDLE_END", status=status, cumulative_gpu_hours=result["cumulative_gpu_hours"])
    return 0 if status in ("PREFLIGHT_COMPLETE", "FULL_ARC_PANEL_COMPLETE", "BUDGET_STOP") else 1


if __name__ == "__main__":
    raise SystemExit(main())
