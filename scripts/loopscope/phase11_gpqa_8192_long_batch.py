#!/usr/bin/env python3
"""D-4 resumable independent worker pool; parent reads checkpoint counts only."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from tflt.loopscope.phase11_panel import require, write_json_once
from run_phase11_batch import device_query


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update_json(path, value):
    temp = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def select_task(plan, task_index=None):
    if task_index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        require(value is not None, "task index or SLURM_ARRAY_TASK_ID required")
        task_index = int(value)
    tasks = plan["tasks"]
    require(tasks and len({task["number"] for task in tasks}) == len(tasks), "plan task numbers repeat")
    matches = [task for task in tasks if task["number"] == task_index]
    require(len(matches) == 1, "task index outside plan")
    return matches[0]


def command(job, commit, partition):
    cmd = [sys.executable, "scripts/loopscope/phase11_gpqa_8192_resume.py", "acquire",
           "--membership", job["membership"], "--pool", job["pool"], "--cell-id", job["cell_id"],
           "--run-root", job["run_root"], "--commit", commit, "--partition", partition]
    if job.get("synthetic", False):
        cmd.append("--synthetic")
    else:
        cmd += ["--indices", ",".join(map(str, job["indices"]))]
    return cmd


def aggregate(jobs, synthetic):
    """Progress is descriptive; the producer revalidates bundles before skipping."""
    done, target = 0, 0
    for job in jobs:
        indices = [0] if synthetic else job["indices"]
        target += len(indices)
        path = Path(job["run_root"]) / "checkpoint.json"
        if path.exists():
            completed = read(path)["done"]
            require(isinstance(completed, list) and len(set(completed)) == len(completed)
                    and all(type(i) is int and i in indices for i in completed), "child checkpoint membership differs")
            done += len(completed)
    return {"done_record_count": done, "target_count": target}


def run(args):
    task = select_task(read(args.plan), args.task_index)
    synthetic = task.get("synthetic", False)
    require(type(synthetic) is bool, "task synthetic flag must be boolean")
    partition = task.get("partition", "debug" if synthetic else None)
    require(isinstance(partition, str) and partition and (not synthetic or partition == "debug"),
            "formal task requires explicit partition; synthetic task requires debug")
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_PARTITION") == partition,
            "allocation partition differs from exact plan task")
    packing = task["packing"]
    require(type(packing) is int and 1 <= packing <= 5 and (synthetic or packing == 5),
            "formal packing must remain measured 5; debug packing may be lower")
    jobs = read(task["worklist"])["jobs"]
    require(jobs and len({job["run_root"] for job in jobs}) == len(jobs), "worker roots must be unique")
    seen = set()
    for job in jobs:
        require(job.get("synthetic", False) == synthetic, "worklist synthetic scope differs")
        if not synthetic:
            indices = job["indices"]
            require(indices and len(set(indices)) == len(indices)
                    and all(type(i) is int and 0 <= i < 448 for i in indices), "worker indices invalid or duplicated")
            pairs = {(job["cell_id"], i) for i in indices}
            require(not seen.intersection(pairs), "worklist repeats cell/sample pairs")
            seen.update(pairs)
    root = Path(task["batch_root"])
    root.mkdir(parents=True, exist_ok=True)
    attempt = root / "attempts" / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
              + "-" + os.environ["SLURM_JOB_ID"] + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    threads = str(8 // packing)
    env.update({key: threads for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    write_json_once(attempt / "command_args.json", {"plan": str(args.plan), "task": task, "jobs": jobs,
        "commit": args.commit, "job_id": env["SLURM_JOB_ID"], "threads_per_child": int(threads), "target_gold_loaded": False})
    start = time.monotonic()
    stop, telemetry_failed = threading.Event(), threading.Event()
    completed, launched, failed, thread, exception_type = [], 0, False, None, None
    aggregate_lock = threading.Lock()

    def progress():
        with aggregate_lock:
            value = aggregate(jobs, synthetic)
            update_json(root / "checkpoint.json", value)
            return value

    def sample(handle, query):
        result = subprocess.run(query, capture_output=True, text=True, timeout=10, env=env)
        handle.write(json.dumps({"elapsed_seconds": time.monotonic() - start,
            "exit_code": result.returncode, "devices_csv": result.stdout.strip()}) + "\n")
        handle.flush()
        progress()
        if result.returncode:
            telemetry_failed.set()

    def sampler(handle, query):
        while not stop.wait(1):
            try:
                sample(handle, query)
            except Exception as exc:
                telemetry_failed.set()
                handle.write(json.dumps({"exit_code": -1, "exception_type": type(exc).__name__, "reason": str(exc)}) + "\n")
                handle.flush()
                return

    def child(number, job):
        cmd = command(job, args.commit, partition)
        began, exit_code, error = time.monotonic(), None, None
        try:
            with (attempt / f"child-{number:03d}.out").open("x") as out, \
                 (attempt / f"child-{number:03d}.err").open("x") as err:
                exit_code = subprocess.run(cmd, stdout=out, stderr=err, env=env).returncode
        except Exception as exc:
            error = {"exception_type": type(exc).__name__, "reason": str(exc)}
            raise
        finally:
            fact = {"cell_id": job["cell_id"], "indices": job.get("indices"), "run_root": job["run_root"],
                    "command": cmd, "exit_code": exit_code, "elapsed_seconds": time.monotonic() - began, "error": error}
            write_json_once(attempt / f"child-{number:03d}-fact.json", fact)
        return fact

    try:
        counts = progress()
        update_json(root / "batch_summary.json", {"status": "RUNNING", "task_number": task["number"],
            "job_id": env["SLURM_JOB_ID"], "attempt_root": str(attempt), "packing": packing,
            "synthetic": synthetic, "partition": partition, **counts, "target_gold_loaded": False})
        query = device_query(env)
        with (attempt / "device_memory.jsonl").open("x") as device:
            try:
                sample(device, query)
                thread = threading.Thread(target=sampler, args=(device, query), daemon=True)
                thread.start()
                with concurrent.futures.ThreadPoolExecutor(max_workers=packing) as executor:
                    active = set()
                    while active or (launched < len(jobs) and not failed):
                        failed |= telemetry_failed.is_set()
                        while not failed and len(active) < packing and launched < len(jobs):
                            active.add(executor.submit(child, launched, jobs[launched]))
                            launched += 1
                        if not active:
                            break
                        done, active = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            fact = future.result()
                            completed.append(fact)
                            failed |= fact["exit_code"] != 0
            finally:
                stop.set()
                if thread is not None:
                    thread.join(timeout=12)
    except Exception as exc:
        failed, exception_type = True, type(exc).__name__
        raise
    finally:
        failed |= telemetry_failed.is_set()
        try:
            counts = progress()
        except Exception as exc:
            failed, exception_type = True, type(exc).__name__
            # Retain a failed-attempt receipt before propagating a corrupt checkpoint.
            counts = {"done_record_count": None, "target_count": sum(1 if synthetic else len(j["indices"]) for j in jobs)}
            error = exc
        else:
            error = None
        failed |= len(completed) != len(jobs) or counts["done_record_count"] != counts["target_count"]
        summary = {"status": "FAILED" if failed else "BATCH_COMPLETE", "packing": packing, "synthetic": synthetic,
            "partition": partition, "task_number": task["number"], "job_id": env["SLURM_JOB_ID"],
            "attempt_root": str(attempt), "expected_attempts": len(jobs), "launched_attempts": launched,
            "completed_attempts": len(completed), "attempts": completed, "elapsed_seconds": time.monotonic() - start,
            "telemetry_failed": telemetry_failed.is_set(), "exception_type": exception_type,
            **counts, "target_gold_loaded": False}
        write_json_once(attempt / "batch_summary.json", summary)
        update_json(root / "batch_summary.json", summary)
        if error is not None:
            raise error
    return 1 if failed else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--task-index", type=int)
    p.add_argument("--commit", required=True)
    return p


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
