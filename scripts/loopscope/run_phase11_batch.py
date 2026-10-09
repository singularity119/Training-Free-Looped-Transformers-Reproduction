#!/usr/bin/env python3
"""Run bounded independent Phase 11 multi-question producers on one GPU."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from tflt.loopscope.phase11_panel import require, write_json_once

PACKINGS = (1, 2, 4, 6, 8)


def thread_environment(packing, environment=None):
    require(packing in PACKINGS, "unsupported packing")
    env = dict(os.environ if environment is None else environment)
    threads = str(max(1, 8 // packing))
    env.update({key: threads for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def command(job, scope, commit):
    result = [sys.executable, "scripts/loopscope/run_phase11_accuracy.py",
              "--dataset", job["dataset"], "--manifest", job["manifest"],
              "--pool", job["pool"], "--cell-id", job["cell_id"],
              "--run-root", job["run_root"], "--commit", commit, "--scope", scope]
    if job.get("indices_file") is not None:
        result += ["--indices-file", str(job["indices_file"])]
    elif scope == "FORMAL_TEST":
        result += ["--shard", f'{job["canonical_index"]}/{job["population_count"]}']
    if scope == "PREFLIGHT_ONLY" and job["dataset"] != "arc_challenge":
        result += ["--engineering-generate-steps", "3"]
    return result


def prepare_jobs(jobs, batch_root):
    """Write inline index batches once; keep canonical singleton compatibility."""
    require(jobs and len({j["run_root"] for j in jobs}) == len(jobs),
            "worklist roots must be unique")
    require(all(not Path(j["run_root"]).exists() for j in jobs),
            "existing attempts cannot be replaced")
    prepared = []
    membership = {}
    for number, original in enumerate(jobs):
        job = dict(original)
        require(job["dataset"] in ("arc_challenge", "gpqa_main"), "Gate D supports ARC and GPQA only")
        indices = job.get("indices")
        if indices is not None:
            require(job.get("indices_file") is None, "declare inline indices or an indices file")
            declaration = {"schema": "loopscope.phase11.index_batch.v1",
                           "dataset": job["dataset"], "cell_id": job["cell_id"],
                           "population_count": job["population_count"], "indices": indices}
            path = batch_root / f"attempt-{number:05d}-indices.json"
            job["indices_file"] = str(path)
        elif job.get("indices_file") is not None:
            declaration = json.loads(Path(job["indices_file"]).read_text(encoding="utf-8"))
            require(declaration.get("schema") == "loopscope.phase11.index_batch.v1" and
                    declaration.get("dataset") == job["dataset"] and
                    declaration.get("cell_id") == job["cell_id"] and
                    declaration.get("population_count") == job["population_count"],
                    "indices file differs from worklist identity")
            indices = declaration["indices"]
        elif "canonical_index" in job:
            indices = [job["canonical_index"]]
        if indices is not None:
            population = job["population_count"]
            require(type(population) is int and population > 0 and indices and
                    all(type(i) is int and 0 <= i < population for i in indices) and
                    indices == sorted(set(indices)), "invalid canonical index batch")
            key = (job["dataset"], job["cell_id"])
            seen = membership.setdefault(key, set())
            require(not seen.intersection(indices), "worklist repeats canonical indices within a cell")
            seen.update(indices)
            job["indices"] = indices
        if original.get("indices") is not None:
            write_json_once(path, declaration)
        job["attempt_number"] = number
        prepared.append(job)
    return prepared


def device_query(environment):
    # No CUDA import/context in the parent. Resolve this command before workers
    # start so telemetry never inspects worker CUDA initialization state.
    device = environment.get("CUDA_VISIBLE_DEVICES", "")
    require(device and "," not in device, "exactly one allocated visible GPU is required")
    return ["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total",
            "--format=csv,noheader,nounits", "--id=" + device]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--scope", choices=("PREFLIGHT_ONLY", "FORMAL_TEST"), required=True)
    parser.add_argument("--packing", type=int, choices=PACKINGS, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "allocation required")
    require(args.scope != "PREFLIGHT_ONLY" or os.environ.get("SLURM_JOB_PARTITION") == "debug",
            "synthetic work must use debug")
    require(args.packing != 1 or args.scope == "PREFLIGHT_ONLY", "packing 1 is debug only")
    jobs = json.loads(args.worklist.read_text(encoding="utf-8"))["jobs"]
    args.batch_root.mkdir(parents=True, exist_ok=False)
    jobs = prepare_jobs(jobs, args.batch_root)
    env = thread_environment(args.packing)
    captured_env = {key: value for key, value in env.items() if key.startswith("SLURM_") or key in (
        "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
        "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")}
    query = device_query(env)
    write_json_once(args.batch_root / "command_args.json", {
        "worklist": str(args.worklist), "scope": args.scope, "packing": args.packing,
        "commit": args.commit, "batch_root": str(args.batch_root),
        "job_id": env["SLURM_JOB_ID"], "jobs": jobs, "target_gold_loaded": False})
    write_json_once(args.batch_root / "env.json", captured_env)
    stop = threading.Event()
    telemetry_failed = threading.Event()
    started = time.monotonic()

    def sample(handle):
        try:
            result = subprocess.run(query, capture_output=True, text=True, timeout=10, env=env)
            fact = {"elapsed_seconds": time.monotonic() - started,
                    "exit_code": result.returncode, "devices_csv": result.stdout.strip(),
                    "stderr": result.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            fact = {"elapsed_seconds": time.monotonic() - started, "exit_code": -1,
                    "error": str(error)}
        handle.write(json.dumps(fact) + "\n")
        handle.flush()
        if fact["exit_code"] != 0:
            telemetry_failed.set()

    def sample_device(handle):
        next_sample = time.monotonic() + 1
        while not stop.wait(max(0, next_sample - time.monotonic())):
            sample(handle)
            next_sample = time.monotonic() + 1 if time.monotonic() >= next_sample + 1 else next_sample + 1

    def run_job(job):
        cmd = command(job, args.scope, args.commit)
        name = f'attempt-{job["attempt_number"]:05d}'
        child_started = time.monotonic()
        fact = {"dataset": job["dataset"], "cell_id": job["cell_id"],
                "indices": job.get("indices"), "canonical_index": job.get("canonical_index"),
                "run_root": job["run_root"], "command": cmd,
                "started_elapsed_seconds": child_started - started, "environment": captured_env}
        write_json_once(args.batch_root / (name + "-command.json"), fact)
        with (args.batch_root / (name + ".out")).open("x") as out, \
                (args.batch_root / (name + ".err")).open("x") as err:
            try:
                result = subprocess.run(cmd, stdout=out, stderr=err, env=env)
                fact["producer_exit_code"] = result.returncode
                fact["exit_code"] = result.returncode
                if result.returncode == 0 and args.scope == "PREFLIGHT_ONLY":
                    verify = [sys.executable, "scripts/loopscope/verify_phase11_scores.py",
                              "--manifest", job["manifest"], "--pool", job["pool"],
                              "--scope", args.scope, "--attempt", job["run_root"],
                              "--output-dir", job["run_root"] + "-verification"]
                    fact["verification_command"] = verify
                    fact["exit_code"] = subprocess.run(verify, stdout=out, stderr=err, env=env).returncode
                    fact["verification_exit_code"] = fact["exit_code"]
            except OSError as error:
                fact.update(exit_code=-1, error=str(error))
        fact["elapsed_seconds"] = time.monotonic() - child_started
        return fact

    completed = []
    launched = 0
    failed = False
    with (args.batch_root / "device_memory.jsonl").open("x") as device_handle:
        sample(device_handle)  # Synchronous device check before any CUDA worker starts.
        sampler = threading.Thread(target=sample_device, args=(device_handle,), daemon=True)
        sampler.start()
        try:
            with (args.batch_root / "attempt_facts.jsonl").open("x") as facts_handle, \
                    concurrent.futures.ThreadPoolExecutor(max_workers=args.packing) as executor:
                active = {}
                while active or (launched < len(jobs) and not failed):
                    failed |= telemetry_failed.is_set()
                    while not failed and len(active) < args.packing and launched < len(jobs):
                        job = jobs[launched]
                        active[executor.submit(run_job, job)] = job
                        launched += 1
                    if not active:
                        break
                    done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        del active[future]
                        fact = future.result()
                        completed.append(fact)
                        facts_handle.write(json.dumps(fact) + "\n")
                        facts_handle.flush()
                        failed |= fact["exit_code"] != 0
        except Exception:
            failed = True
            raise
        finally:
            stop.set()
            sampler.join(timeout=12)
            failed |= telemetry_failed.is_set()
            write_json_once(args.batch_root / "batch_summary.json", {
                "status": "FAILED" if failed else "BATCH_COMPLETE", "scope": args.scope,
                "packing": args.packing, "threads_per_worker": max(1, 8 // args.packing),
                "expected_attempts": len(jobs), "launched_attempts": launched,
                "completed_attempts": len(completed), "not_started_attempts": len(jobs) - launched,
                "elapsed_seconds": time.monotonic() - started, "attempts": completed,
                "telemetry_failed": telemetry_failed.is_set(), "target_gold_loaded": False})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
