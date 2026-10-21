#!/usr/bin/env python3
"""Short OSS-mounted entry; unpack code and launch the explicit Lingjing plan."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "plan", "model-dir", "admission", "launcher-output", "receipt-file"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-seconds", type=int, required=True)
    a = p.parse_args()
    receipt_path = Path(a.receipt_file)
    deadline = time.monotonic() + 120
    while not receipt_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("publish the real accepted training receipt to the declared shared path")
        time.sleep(1)
    receipt = json.loads(receipt_path.read_text())
    job_id = receipt["modelTrainingJobId"]
    if not isinstance(job_id, int) or job_id <= 0:
        raise ValueError("receipt lacks a real Lingjing modelTrainingJobId")
    root = Path(tempfile.mkdtemp(prefix="loopscope-d9-"))
    with tarfile.open(a.bundle) as archive:
        archive.extractall(root, filter="data")
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
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
        log_root = Path(a.launcher_output)
        log_root.mkdir(parents=True, exist_ok=True)
        with (log_root / f"rank-{os.environ['RANK']}-install.log").open("x") as log:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(root / "requirements.txt"),
                "--constraint", str(constraints)], stdout=log, stderr=subprocess.STDOUT, check=True)
        for name, version in protected.items():
            if importlib.metadata.version(name) != version:
                raise RuntimeError("image accelerator package changed: " + name)
        plan = json.loads(Path(a.plan).read_text())
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
            if plan["total_gpus"] == 1:
                command.append("--synthetic")
            subprocess.run(command, env=env, check=True)
        deadline = time.monotonic() + 120
        while not ready.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("node zero did not finish shared input preparation")
            time.sleep(1)
    cmd = [sys.executable, str(root / "scripts/loopscope/phase11_d9_lingjing.py"), "launch"]
    for key in ("plan", "model_dir", "admission", "launcher_output", "timeout_seconds"):
        cmd += ["--" + key.replace("_", "-"), str(getattr(a, key))]
    cmd += ["--platform-job-id", str(job_id)]
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
