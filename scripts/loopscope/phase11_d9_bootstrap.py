#!/usr/bin/env python3
"""Short OSS-mounted entry; unpack code and launch the explicit Lingjing plan."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("D9 bootstrap total deadline exhausted")
    return seconds


def wait_file(path, deadline, description):
    wait_deadline = min(deadline, time.monotonic() + 120)
    while not path.exists():
        try:
            seconds = remaining(wait_deadline)
        except TimeoutError as exc:
            raise TimeoutError("bootstrap deadline waiting for " + description) from exc
        time.sleep(min(1, seconds))
    remaining(deadline)


def run_step(command, deadline, **kwargs):
    """One shared budget; a timed-out step loses its whole owned process group."""
    remaining(deadline)
    child = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        code = child.wait(timeout=remaining(deadline))
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        # Pip, input preparation and launcher descendants inherit this group.
        # SIGKILL also bounds cleanup if a child ignores normal termination.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        raise
    remaining(deadline)


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "plan", "model-dir", "admission", "launcher-output", "receipt-file"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-seconds", type=int, required=True)
    p.add_argument("--verify-model-source", action="store_true")
    a = p.parse_args()
    if a.timeout_seconds <= 0:
        raise ValueError("bootstrap total time limit must be positive")
    deadline = started + a.timeout_seconds
    receipt_path = Path(a.receipt_file)
    wait_file(receipt_path, deadline, "real accepted training receipt")
    receipt = json.loads(receipt_path.read_text())
    job_id = receipt["modelTrainingJobId"]
    if not isinstance(job_id, int) or job_id <= 0:
        raise ValueError("receipt lacks a real Lingjing modelTrainingJobId")
    root = Path(tempfile.mkdtemp(prefix="loopscope-d9-"))
    with tarfile.open(a.bundle) as archive:
        archive.extractall(root, filter="data")
    remaining(deadline)
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    plan = json.loads(Path(a.plan).read_text())
    synthetic = all(job["synthetic"] for task in plan["tasks"] for job in task["jobs"])
    log_root = Path(a.launcher_output)
    log_root.mkdir(parents=True, exist_ok=True)
    if a.verify_model_source:
        if not synthetic:
            raise ValueError("model source preparation is a preflight-only action")
        admitted_file = log_root / "admission-source-verified.json"
        if int(os.environ["RANK"]) == 0:
            evidence = log_root / "model-source.json"
            run_step([sys.executable, str(root / "scripts/loopscope/phase11_d9_model_source.py"),
                "--model-dir", a.model_dir, "--output", str(evidence),
                "--timeout-seconds", str(min(900, remaining(deadline)))], deadline, env=env)
            source = json.loads(evidence.read_text())
            if source["status"] != "PASS":
                raise ValueError("mounted model differs or source comparison failed")
            admitted = json.loads(Path(a.admission).read_text())
            admitted["model_source_evidence"] = str(evidence)
            with admitted_file.open("x") as handle:
                json.dump(admitted, handle, indent=2)
        # Source comparison may take more than the usual short receipt wait.
        while not admitted_file.exists():
            time.sleep(min(1, remaining(deadline)))
        a.admission = str(admitted_file)
    admission = json.loads(Path(a.admission).read_text())
    bundle_info = json.loads((root / "bundle-info.json").read_text())
    if (not admission.get("model_source_evidence") or
            admission.get("model_revision") != "1cfa9a7208912126459214e8b04321603b3df60c" or
            admission.get("source_commit") != bundle_info["source_commit"]):
        raise ValueError("resolve exact model provenance and code admission before starting the bundle")
    # Only each node's leader installs dependencies; accelerator packages stay image-owned.
    if int(os.environ["LOCAL_RANK"]) == 0:
        protected = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                     if d.metadata["Name"].lower().replace("_", "-").startswith(
                         ("torch", "triton", "nvidia", "cuda", "flash"))}
        constraints = root / "image-constraints.txt"
        constraints.write_text("".join(f"{k}=={v}\n" for k, v in sorted(protected.items())))
        with (log_root / f"rank-{os.environ['RANK']}-install.log").open("x") as log:
            run_step([sys.executable, "-m", "pip", "install", "-r", str(root / "requirements.txt"),
                "--constraint", str(constraints)], deadline, stdout=log, stderr=subprocess.STDOUT)
        for name, version in protected.items():
            if importlib.metadata.version(name) != version:
                raise RuntimeError("image accelerator package changed: " + name)
        pools = {job["pool"] for task in plan["tasks"] for job in task["jobs"]}
        if len(pools) != 1:
            raise ValueError("all D9 tasks must use the same frozen input pool")
        pool_file = Path(next(iter(pools)))
        if pool_file.name != "thinking-pool.json":
            raise ValueError("use the input builder's canonical output filename")
        ready = pool_file.parent / "alignment.json"
        if int(os.environ["RANK"]) == 0 and not ready.exists():
            command = [sys.executable, str(root / "scripts/loopscope/prepare_phase11_d9_inputs.py"),
                       "--model-dir", a.model_dir, "--output-dir", str(pool_file.parent)]
            if synthetic:
                command.append("--synthetic")
            else:
                config = json.loads((root / "configs/loopscope/phase11_d9_lingjing.json").read_text())
                command += ["--source-csv", config["data"]["source_csv_mount"]]
            run_step(command, deadline, env=env)
        wait_file(ready, deadline, "shared input preparation")
    cmd = [sys.executable, str(root / "scripts/loopscope/phase11_d9_lingjing.py"), "launch"]
    for key in ("plan", "model_dir", "admission", "launcher_output"):
        cmd += ["--" + key.replace("_", "-"), str(getattr(a, key))]
    cmd += ["--timeout-seconds", str(max(1, int(remaining(deadline))))]
    cmd += ["--platform-job-id", str(job_id)]
    run_step(cmd, deadline, env=env)
    if synthetic and int(os.environ["RANK"]) == 0:
        run_step([sys.executable, str(root / "scripts/loopscope/check_phase11_d9_gpu.py"),
            "--plan", a.plan, "--model-dir", a.model_dir,
            "--output", str(log_root / "preflight-verified.json")], deadline, env=env)
        config = json.loads((root / "configs/loopscope/phase11_d9_lingjing.json").read_text())
        formal_pool = Path(config["cloud_root"]) / "inputs/formal/thinking-pool.json"
        if not (formal_pool.parent / "alignment.json").exists():
            run_step([sys.executable, str(root / "scripts/loopscope/prepare_phase11_d9_inputs.py"),
                "--model-dir", a.model_dir, "--output-dir", str(formal_pool.parent),
                "--source-csv", config["data"]["source_csv_mount"]], deadline, env=env)
        capacity_env = dict(env, CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
        run_step([sys.executable, str(root / "scripts/loopscope/phase11_d9_capacity.py"),
            "--model-dir", a.model_dir, "--pool", str(formal_pool),
            "--output", str(log_root / "capacity.json")], deadline, env=capacity_env)
        with (log_root / "preflight-complete.json").open("x") as handle:
            json.dump({"status": "PREFLIGHT_COMPLETE", "source_commit": bundle_info["source_commit"],
                "platform_job_id": job_id, "elapsed_seconds": time.monotonic() - started,
                "total_gpus": plan["total_gpus"], "formal_gold_scored": False,
                "model_source_evidence": admission["model_source_evidence"],
                "capacity_evidence": str(log_root / "capacity.json")}, handle, indent=2)


if __name__ == "__main__":
    main()
