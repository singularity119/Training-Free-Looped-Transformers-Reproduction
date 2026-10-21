#!/usr/bin/env python3
"""Plan or launch independent D9 workers from an OSS-mounted source bundle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tflt.loopscope.phase11_thinking2507 import CELLS, COUNT, SEED, selected_cells, validate_job
from tflt.loopscope.phase11_panel import require, write_json_once


def make_plan(output_root, pool, source_commit, total_gpus=8, synthetic=False):
    require(total_gpus in (1, 8, 16), "D9 uses one preflight GPU or eight/sixteen formal GPUs")
    require(synthetic or total_gpus != 1, "one GPU is reserved for synthetic preflight")
    root, cells = Path(output_root), selected_cells()
    tasks, jobs = [], []
    for rank in range(total_gpus):
        worker_jobs = []
        for key in CELLS:
            c = cells[key]
            job = {"cell_key": key, "cell_id": c["cell_id"], "prompt_mode": c["prompt_mode"],
                   "damping_scope": c["damping_scope"], "seed": SEED, "source_commit": source_commit,
                   "pool": str(pool), "run_root": str(root / f"worker-{rank:02d}" / key),
                   "indices": [0] if synthetic else list(range(rank, COUNT, total_gpus)),
                   "synthetic": synthetic, "engineering": synthetic}
            validate_job(job)
            worker_jobs.append(job)
            jobs.append(job)
        tasks.append({"rank": rank, "jobs": worker_jobs})
    return {"schema": "loopscope.phase11.thinking2507.lingjing_plan.v1", "total_gpus": total_gpus,
            "gpus_per_node": 1 if total_gpus == 1 else 8, "packing": 1, "source_commit": source_commit,
            "tasks": tasks, "registry": {"schema": "loopscope.phase11.thinking2507.full_registry.v1", "jobs": jobs}}


def topology(env, plan):
    size, per_node = plan["total_gpus"], plan["gpus_per_node"]
    require(int(env["WORLD_SIZE"]) == size, "platform world size differs from total requested cards")
    rank, local = int(env["RANK"]), int(env["LOCAL_RANK"])
    require(0 <= rank < size and local == rank % per_node, "unexpected outer rank topology")
    if "LOCAL_WORLD_SIZE" in env:
        require(int(env["LOCAL_WORLD_SIZE"]) == per_node, "unexpected cards per node")
    return rank, local, rank // per_node


def child_environment(env, visible, outer_rank, worker_rank):
    result = dict(env)
    for key in list(result):
        if key.startswith("TORCHELASTIC_") or key in {
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
            "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}:
            result.pop(key)
    result.update(CUDA_VISIBLE_DEVICES=visible, D9_OUTER_RANK=str(outer_rank),
                  D9_WORKER_RANK=str(worker_rank), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    return result


def node_devices(env, expected):
    """Discover the container's allocation; an absent CUDA mask means all exposed GPUs."""
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name",
        "--format=csv,noheader"], text=True)
    devices = [tuple(part.strip() for part in line.split(",", 1))
               for line in inventory.splitlines() if line.strip()]
    require(all(len(row) == 2 and row[0].startswith("GPU-") and "H20" in row[1]
                for row in devices), "unexpected node GPU inventory: " + repr(devices))
    raw_mask = env.get("CUDA_VISIBLE_DEVICES")
    visible = [value.strip() for value in raw_mask.split(",")] if raw_mask is not None else [row[0] for row in devices]
    require(len(visible) == expected and len(set(visible)) == expected and all(visible),
            "leader must see every distinct card on its node")
    # A short child verifies CUDA visibility without retaining CUDA contexts in the launcher.
    cuda_count = int(subprocess.check_output([sys.executable, "-c",
        "import torch; print(torch.cuda.device_count())"], env=dict(env), text=True).strip())
    require(cuda_count == expected, "CUDA device count differs from requested node allocation")
    if raw_mask is None:
        require(len(devices) == expected, "unmasked container exposes a different node allocation")
    return visible, {"cuda_visible_devices_environment": raw_mask,
        "selection_source": "explicit_environment" if raw_mask is not None else "container_gpu_inventory",
        "inventory": [{"uuid": uuid, "name": name} for uuid, name in devices],
        "cuda_device_count": cuda_count, "worker_devices": visible}


def launch(plan_file, model_dir, admission, output, platform_job_id, timeout_seconds):
    plan = json.loads(Path(plan_file).read_text())
    rank, local, node = topology(os.environ, plan)
    if local != 0:
        return
    root = Path(output) / f"node-{node}"
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("phase11_thinking2507_worker.py")
    children, logs = [], []
    sampler = None
    began = time.monotonic()
    try:
        visible, discovery = node_devices(os.environ, plan["gpus_per_node"])
        write_json_once(root / "topology.json", {"outer_rank": rank, "local_rank": local,
            "node": node, "world_size": plan["total_gpus"], **discovery})
        print(json.dumps({"event": "D9_NODE_GPU_ROUTING", "node": node, **discovery}), flush=True)
        memory_log = (root / "device-memory.csv").open("x")
        memory_err = (root / "device-memory.err").open("x")
        logs.extend((memory_log, memory_err))
        sampler = subprocess.Popen(["nvidia-smi", "--id=" + ",".join(visible),
            "--query-gpu=timestamp,uuid,name,memory.used,memory.total",
            "--format=csv,noheader,nounits", "--loop-ms=1000"], stdout=memory_log, stderr=memory_err)
        for gpu, device in enumerate(visible):
            worker = node * plan["gpus_per_node"] + gpu
            task = plan["tasks"][worker]
            env = child_environment(os.environ, device, rank, worker)
            env["D9_PLATFORM_JOB_ID"] = platform_job_id
            # Each GPU processes its three cells sequentially, with independent models.
            task_file = root / f"task-{worker:02d}.json"
            write_json_once(task_file, task)
            log = (root / f"worker-{worker:02d}.log").open("x")
            logs.append(log)
            cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--task", str(task_file),
                   "--model-dir", str(model_dir), "--admission", str(admission)]
            children.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            require(sampler.poll() is None, "device-memory sampler stopped; inspect its error log")
            require(not any(p.poll() not in (None, 0) for p in children), "worker failed; inspect its log")
            if time.monotonic()-began >= timeout_seconds:
                raise TimeoutError("D9 node exceeded its explicit wall-time bound")
            time.sleep(1)
        require(all(p.returncode == 0 for p in children), "worker failed at completion")
        require(sampler.poll() is None, "device-memory sampler failed")
        write_json_once(root / "complete.json", {"status": "NODE_COMPLETE", "node": node,
            "platform_job_id": platform_job_id, "elapsed_seconds": time.monotonic()-began})
    except BaseException as exc:
        write_json_once(root / "failure.json", {"exception_type": type(exc).__name__, "reason": str(exc)})
        raise
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            p.wait()
        if sampler is not None:
            if sampler.poll() is None:
                sampler.terminate()
            sampler.wait()
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("plan")
    for key in ("output-root", "pool", "source-commit", "plan-file"):
        a.add_argument("--" + key, required=True)
    a.add_argument("--preflight", action="store_true")
    a.add_argument("--total-gpus", type=int, choices=(8, 16), default=8)
    a = sub.add_parser("launch")
    for key in ("plan", "model-dir", "admission", "launcher-output", "platform-job-id"):
        a.add_argument("--" + key, required=True)
    a.add_argument("--timeout-seconds", type=int, required=True)
    a = sub.add_parser("worker")
    for key in ("task", "model-dir", "admission"):
        a.add_argument("--" + key, required=True)
    args = p.parse_args()
    if args.command == "plan":
        plan = make_plan(args.output_root, args.pool, args.source_commit, 1 if args.preflight else args.total_gpus, args.preflight)
        write_json_once(Path(args.plan_file), plan)
    elif args.command == "launch":
        launch(args.plan, args.model_dir, args.admission, args.launcher_output,
               args.platform_job_id, args.timeout_seconds)
    else:
        task = json.loads(Path(args.task).read_text())
        from phase11_thinking2507_worker import run
        for n, job in enumerate(task["jobs"]):
            path = Path(args.task).with_name(f"job-{task['rank']:02d}-{n}.json")
            write_json_once(path, job)
            run(path, args.model_dir, args.admission)


if __name__ == "__main__":
    main()
