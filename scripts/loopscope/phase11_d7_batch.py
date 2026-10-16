#!/usr/bin/env python3
"""D7 independent-process scheduling; acquisition stays in source-specific drivers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value, once=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if once and path.exists():
        raise FileExistsError(path)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8") as h:
        json.dump(value, h, ensure_ascii=False, allow_nan=False)
        h.write("\n")
        h.flush()
        os.fsync(h.fileno())
    os.replace(temp, path)


def validate_task(task):
    packing = task["packing"]
    if type(packing) is not int or not 1 <= packing <= 5:
        raise ValueError("packing must be admitted independently, within 1..5")
    jobs = task["jobs"]
    if not jobs or len({j["run_root"] for j in jobs}) != len(jobs):
        raise ValueError("job roots must be unique")
    seen = set()
    for j in jobs:
        from phase11_d7 import validate_job
        validate_job(j)
        ids = j["indices"]
        if not ids or ids != sorted(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("indices must preserve canonical order")
        pairs = {(j["cell_key"], i) for i in ids}
        if seen & pairs:
            raise ValueError("duplicate cell/question within task")
        seen |= pairs
    return len(seen)


def child_command(job, job_file, controller):
    driver = "phase11_d7.py"
    env = dict(os.environ)
    source = job.get("producer_source", str(controller))
    env.update(PYTHONPATH=str(Path(source) / "src"), PYTHONDONTWRITEBYTECODE="1")
    return [sys.executable, str(controller / "scripts/loopscope" / driver), "--job-file", str(job_file)], env


FIRST_KEYS = ("native", "current_t_k2_l05", "fixed_t0_k3_l09")
LOOP_KEYS = ("loop_k2", "loop_k3")


def validate_first_closure(receipt, first_cells):
    """Bind the outcome-blind first-batch receipt before any formal Loop work."""
    if (receipt.get("schema") != "loopscope.phase11.d7.first_batch_raw_closure.v1"
        or receipt.get("status") != "FIRST_BATCH_RAW_CLOSED_GOLD_UNREAD"
        or receipt.get("target_gold_loaded") is not False
        or receipt.get("record_count") != 36096 or receipt.get("cell_count") != 3
        or set(receipt.get("cell_keys", [])) != set(FIRST_KEYS)):
        raise ValueError("D7 first-batch raw closure is absent or incomplete")
    expected = {c["cell_key"]: c for c in first_cells}
    observed = {c["cell_key"]: c for c in receipt.get("cells", [])}
    if set(expected) != set(FIRST_KEYS) or set(observed) != set(FIRST_KEYS):
        raise ValueError("D7 first-batch receipt cell membership differs")
    for key, cell in expected.items():
        for field in ("cell_id", "pool", "source_commit", "seed", "prompt_mode",
                      "damping_scope", "population_count", "raw_roots"):
            if observed[key].get(field) != cell.get(field):
                raise ValueError("D7 first-batch receipt binding differs: " + key + "/" + field)
    return receipt


def validate_plan_barrier(plan):
    if not 1 <= len(plan["tasks"]) <= 8:
        raise ValueError("D7 permits at most eight GPU tasks")
    formal_loops = [j for t in plan["tasks"] for j in t["jobs"]
                    if j["cell_key"] in LOOP_KEYS and not j["synthetic"]]
    if formal_loops:
        barrier = plan.get("first_batch_barrier")
        if plan.get("stage") != "last" or not barrier:
            raise ValueError("D7 formal Loop requires the closed first batch")
        validate_first_closure(read(barrier["receipt_path"]), barrier["first_cells"])


def run(args):
    plan = read(args.plan)
    validate_plan_barrier(plan)
    number = args.task_index if args.task_index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    tasks = [t for t in plan["tasks"] if t["number"] == number]
    if len(tasks) != 1:
        raise ValueError("task identity absent or repeated")
    task = tasks[0]
    target = validate_task(task)
    if os.environ.get("SLURM_JOB_PARTITION") != task["partition"]:
        raise ValueError("allocation differs from plan")
    if task["partition"] == "debug" and not all(j.get("engineering", False) for j in task["jobs"]):
        raise ValueError("debug requires engineering scope")
    controller = Path(__file__).resolve().parents[2]
    root = Path(task["batch_root"])
    attempt = root / "attempts" / (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    save(attempt / "task.json", {"task": task, "target": target, "job_id": os.environ["SLURM_JOB_ID"], "target_gold_loaded": False}, once=True)
    stop = threading.Event()
    telemetry_failed = threading.Event()
    began = time.monotonic()
    facts = []

    def telemetry():
        with (attempt / "device_memory.jsonl").open("x") as h:
            while not stop.is_set():
                try:
                    device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                    if not device or "," in device:
                        raise ValueError("one allocated GPU must be visible")
                    p = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total", "--format=csv,noheader,nounits", "--id=" + device], capture_output=True, text=True, timeout=10)
                    fact = {"elapsed_seconds": time.monotonic()-began, "exit_code": p.returncode, "devices_csv": p.stdout.strip()}
                    if p.returncode:
                        telemetry_failed.set()
                except Exception as exc:
                    fact = {"exit_code": -1, "exception_type": type(exc).__name__, "reason": str(exc)}
                    telemetry_failed.set()
                h.write(json.dumps(fact) + "\n")
                h.flush()
                if telemetry_failed.is_set():
                    return
                if stop.wait(5):
                    return

    def child(n, job, packing):
        file = attempt / f"job-{n:03d}.json"
        save(file, {**job, "controller_source": str(controller)}, once=True)
        cmd, env = child_command(job, file, controller)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(max(1, 8 // packing))
        start = time.monotonic()
        with (attempt / f"child-{n:03d}.out").open("x") as out, (attempt / f"child-{n:03d}.err").open("x") as err:
            code = subprocess.run(cmd, env=env, stdout=out, stderr=err, cwd=controller).returncode
        fact = {"job": job, "command": cmd, "exit_code": code, "elapsed_seconds": time.monotonic()-start}
        save(attempt / f"child-{n:03d}-fact.json", fact, once=True)
        return fact

    thread = threading.Thread(target=telemetry, daemon=True)
    thread.start()
    failed = False
    try:
        groups = task.get("groups", [{"packing": task["packing"], "jobs": task["jobs"]}])
        if [j for g in groups for j in g["jobs"]] != task["jobs"]:
            raise ValueError("workload groups differ from task membership")
        offset = 0
        for group in groups:
            packing = group["packing"]
            if type(packing) is not int or not 1 <= packing <= task["packing"]:
                raise ValueError("workload group packing exceeds admission")
            jobs = group["jobs"]
            with ThreadPoolExecutor(max_workers=packing) as pool:
                active, launched = set(), 0
                while active or (launched < len(jobs) and not failed):
                    while not failed and len(active) < packing and launched < len(jobs):
                        active.add(pool.submit(child, offset + launched, jobs[launched], packing))
                        launched += 1
                    done, active = wait(active, return_when=FIRST_COMPLETED)
                    for f in done:
                        fact = f.result()
                        facts.append(fact)
                        failed |= fact["exit_code"] != 0 or telemetry_failed.is_set()
                    save(root / "batch_progress.json", {"finished_processes": len(facts), "target_processes": len(task["jobs"]), "failed": failed, "target_gold_loaded": False})
            offset += len(jobs)
            if failed:
                break
    finally:
        stop.set()
        thread.join(timeout=12)
        summary = {"status": "FAILED" if failed or telemetry_failed.is_set() or len(facts) != len(task["jobs"]) else "BATCH_COMPLETE", "target_count": target, "attempt_root": str(attempt), "job_id": os.environ["SLURM_JOB_ID"], "elapsed_seconds": time.monotonic()-began, "packing": task["packing"], "facts": facts, "target_gold_loaded": False}
        save(attempt / "batch_summary.json", summary, once=True)
        save(root / "batch_summary.json", summary)
    return 1 if summary["status"] == "FAILED" else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--task-index", type=int)
    raise SystemExit(run(p.parse_args()))
