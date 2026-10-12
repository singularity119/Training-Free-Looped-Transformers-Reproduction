#!/usr/bin/env python3
"""Bounded independent D-3 processes; no shared model, batch or cache state."""
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
from run_phase11_batch import device_query


def command(job, commit):
    cmd = [sys.executable, "scripts/loopscope/phase11_length_pilot.py", "acquire",
        "--membership", job["membership"], "--pool", job["pool"], "--cell-id", job["cell_id"],
        "--run-root", job["run_root"], "--commit", commit]
    if job.get("synthetic"):
        cmd += ["--synthetic"]
    else:
        cmd += ["--indices", ",".join(map(str, job["indices"]))]
    return cmd


def run(args):
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == "debug",
            "D-3 batch requires debug allocation")
    jobs = json.loads(args.worklist.read_text())["jobs"]
    require(jobs and len({j["run_root"] for j in jobs}) == len(jobs) and
            all(not Path(j["run_root"]).exists() for j in jobs), "attempt roots must be fresh and unique")
    seen = set()
    for job in jobs:
        require(job.get("synthetic", False) == args.synthetic, "worklist synthetic scope differs")
        if not args.synthetic:
            require(job["indices"] and len(job["indices"]) == len(set(job["indices"])), "indices repeat")
            pairs = {(job["cell_id"], i) for i in job["indices"]}
            require(not seen.intersection(pairs), "worklist repeats completed cell/sample pairs")
            seen.update(pairs)
    args.batch_root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    threads = str(max(1, 8 // args.packing))
    env.update({k: threads for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    query = device_query(env)
    write_json_once(args.batch_root / "command_args.json", {"worklist": str(args.worklist),
        "jobs": jobs, "packing": args.packing, "synthetic": args.synthetic, "commit": args.commit,
        "job_id": env["SLURM_JOB_ID"], "target_gold_loaded": False})
    start = time.monotonic()
    stop = threading.Event()
    telemetry_failed = threading.Event()

    def sample(handle):
        result = subprocess.run(query, capture_output=True, text=True, timeout=10, env=env)
        handle.write(json.dumps({"elapsed_seconds": time.monotonic() - start,
            "exit_code": result.returncode, "devices_csv": result.stdout.strip()}) + "\n")
        handle.flush()
        if result.returncode:
            telemetry_failed.set()

    def sampler(handle):
        while not stop.wait(1):
            try:
                sample(handle)
            except Exception as exc:
                telemetry_failed.set()
                handle.write(json.dumps({"exit_code": -1, "exception_type": type(exc).__name__}) + "\n")
                handle.flush()
                return

    def child(number, job):
        cmd = command(job, args.commit)
        began = time.monotonic()
        with (args.batch_root / f"child-{number:03d}.out").open("x") as out, \
             (args.batch_root / f"child-{number:03d}.err").open("x") as err:
            result = subprocess.run(cmd, stdout=out, stderr=err, env=env)
        fact = {"cell_id": job["cell_id"], "indices": job.get("indices"), "run_root": job["run_root"],
            "command": cmd, "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - began}
        write_json_once(args.batch_root / f"child-{number:03d}-fact.json", fact)
        return fact

    completed, launched, failed = [], 0, False
    with (args.batch_root / "device_memory.jsonl").open("x") as device:
        sample(device)
        thread = threading.Thread(target=sampler, args=(device,), daemon=True)
        thread.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.packing) as executor:
                active = set()
                while active or (launched < len(jobs) and not failed):
                    failed |= telemetry_failed.is_set()
                    while not failed and len(active) < args.packing and launched < len(jobs):
                        active.add(executor.submit(child, launched, jobs[launched]))
                        launched += 1
                    if not active:
                        break
                    done, active = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        fact = future.result()
                        completed.append(fact)
                        failed |= fact["exit_code"] != 0
        except Exception:
            failed = True
            raise
        finally:
            stop.set()
            thread.join(timeout=12)
            failed |= telemetry_failed.is_set()
            write_json_once(args.batch_root / "batch_summary.json", {"status": "FAILED" if failed else "BATCH_COMPLETE",
                "packing": args.packing, "synthetic": args.synthetic, "expected_attempts": len(jobs),
                "launched_attempts": launched, "completed_attempts": len(completed),
                "attempts": completed, "elapsed_seconds": time.monotonic() - start,
                "telemetry_failed": telemetry_failed.is_set(), "target_gold_loaded": False})
    return 1 if failed else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worklist", type=Path, required=True)
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--packing", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--synthetic", action="store_true")
    raise SystemExit(run(p.parse_args()))
