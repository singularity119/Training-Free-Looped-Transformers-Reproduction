#!/usr/bin/env python3
"""Run bounded independent singleton producer processes and device telemetry."""
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


def command(job, scope, commit):
    result = [sys.executable, "scripts/loopscope/run_phase11_accuracy.py",
              "--dataset", job["dataset"], "--manifest", job["manifest"],
              "--pool", job["pool"], "--cell-id", job["cell_id"],
              "--run-root", job["run_root"], "--commit", commit, "--scope", scope]
    if scope == "FORMAL_TEST":
        result += ["--shard", f'{job["canonical_index"]}/{job["population_count"]}']
    elif job["dataset"] != "arc_challenge":
        result += ["--engineering-generate-steps", "3"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--scope", choices=("PREFLIGHT_ONLY", "FORMAL_TEST"), required=True)
    parser.add_argument("--packing", type=int, choices=(1, 2), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("SLURM_JOB_ID"), "allocation required")
    require(args.scope != "PREFLIGHT_ONLY" or os.environ.get("SLURM_JOB_PARTITION") == "debug",
            "synthetic work must use debug")
    jobs = json.loads(args.worklist.read_text())["jobs"]
    require(jobs and len({j["run_root"] for j in jobs}) == len(jobs), "worklist roots must be unique")
    require(all(not Path(j["run_root"]).exists() for j in jobs), "existing attempts cannot be replaced")
    args.batch_root.mkdir(parents=True, exist_ok=False)
    write_json_once(args.batch_root / "command_args.json", {**vars(args),
        "worklist": str(args.worklist), "batch_root": str(args.batch_root),
        "job_id": os.environ["SLURM_JOB_ID"], "target_gold_loaded": False})
    stop = threading.Event()
    started = time.monotonic()

    def sample_device():
        with (args.batch_root / "device_memory.jsonl").open("x") as handle:
            while not stop.is_set():
                # Slurm exposes its allocated physical GPU ordinal/UUID here.
                device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                query = ["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total", "--format=csv,noheader,nounits"]
                if device:
                    query += ["--id=" + device]
                result = subprocess.run(query, capture_output=True, text=True, timeout=10)
                handle.write(json.dumps({"elapsed_seconds": time.monotonic() - started,
                    "exit_code": result.returncode, "devices_csv": result.stdout.strip()}) + "\n")
                handle.flush()
                stop.wait(1)

    def run_job(job):
        cmd = command(job, args.scope, args.commit)
        name = f'{job["dataset"]}-{job["cell_id"]}-{job.get("canonical_index", "synthetic")}'
        child_started = time.monotonic()
        with (args.batch_root / (name + ".out")).open("x") as out, (args.batch_root / (name + ".err")).open("x") as err:
            result = subprocess.run(cmd, stdout=out, stderr=err)
            if result.returncode == 0 and args.scope == "PREFLIGHT_ONLY":
                verify = [sys.executable, "scripts/loopscope/verify_phase11_scores.py",
                    "--manifest", job["manifest"], "--pool", job["pool"], "--scope", args.scope,
                    "--attempt", job["run_root"], "--output-dir", job["run_root"] + "-verification"]
                result = subprocess.run(verify, stdout=out, stderr=err)
        return {"dataset": job["dataset"], "cell_id": job["cell_id"],
            "canonical_index": job.get("canonical_index"), "run_root": job["run_root"],
            "command": cmd, "exit_code": result.returncode,
            "elapsed_seconds": time.monotonic() - child_started}

    sampler = threading.Thread(target=sample_device, daemon=True)
    sampler.start()
    completed = []
    iterator = iter(jobs)
    failed = False
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.packing) as executor:
            active = {executor.submit(run_job, j): j for j in [next(iterator, None) for _ in range(args.packing)] if j is not None}
            while active:
                done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    del active[future]
                    fact = future.result()
                    completed.append(fact)
                    failed |= fact["exit_code"] != 0
                if not failed:
                    for _ in range(args.packing - len(active)):
                        job = next(iterator, None)
                        if job is not None:
                            active[executor.submit(run_job, job)] = job
    except Exception:
        failed = True
        raise
    finally:
        stop.set()
        sampler.join(timeout=12)
        write_json_once(args.batch_root / "batch_summary.json", {
            "status": "FAILED" if failed else "BATCH_COMPLETE", "scope": args.scope,
            "packing": args.packing, "expected_attempts": len(jobs), "completed_attempts": len(completed),
            "elapsed_seconds": time.monotonic() - started, "attempts": completed,
            "target_gold_loaded": False})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
