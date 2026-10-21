#!/usr/bin/env python3
"""Cloud entry: bounded synthetic path check, then frozen greedy32768 whole-S2 panel."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import signal
import subprocess
import tarfile
import tempfile
import time

FAILURE_ROOT = None
NODE_COUNT = 2

def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("D12 bootstrap total deadline exhausted")
    return seconds


def wait_file(path, deadline, description):
    wait_deadline = deadline
    while not path.exists():
        failures = [] if FAILURE_ROOT is None else [p for p in (
            *[FAILURE_ROOT / f"launch/bootstrap-node-{n}-failure.json" for n in range(NODE_COUNT)],
            *[FAILURE_ROOT / f"launch/{stage}/node-{n}/failure.json"
              for stage in ("synthetic", "formal-attempt1") for n in range(NODE_COUNT)]) if p.exists()]
        if failures:
            raise RuntimeError("peer bootstrap failed: " + str(failures[0]))
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
    global FAILURE_ROOT, NODE_COUNT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--timeout-seconds", type=int, default=86400)
    a = p.parse_args()
    started = time.monotonic()
    deadline = started + a.timeout_seconds
    # The platform creates one outer process per card; only the node leader owns workers.
    if int(os.environ["LOCAL_RANK"]) != 0:
        return
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    if size not in (8, 16) or rank not in range(0, size, 8):
        raise ValueError("Qwen3NonThinking requires eight-card nodes with valid leaders")
    NODE_COUNT = size // 8
    node = rank // 8
    cloud = a.root
    FAILURE_ROOT = cloud
    wait_file(cloud / "receipt.json", deadline, "accepted real Lingjing receipt")
    receipt = json.loads((cloud / "receipt.json").read_text())
    job_id = receipt["modelTrainingJobId"]
    if type(job_id) is not int or job_id <= 0:
        raise ValueError("real modelTrainingJobId required")
    work = Path(tempfile.mkdtemp(prefix="qwen3_nothink-"))
    with tarfile.open(cloud / "runtime.tar.gz") as archive:
        archive.extractall(work, filter="data")
    env = dict(os.environ, PYTHONPATH=str(work / "src"), PYTHONDONTWRITEBYTECODE="1")
    log_root = cloud / "launch/formal-attempt1"
    log_root.mkdir(parents=True, exist_ok=True)
    smoke_deadline = min(deadline, started + 1740)
    protected = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                 if d.metadata["Name"].lower().replace("_", "-").startswith(
                     ("torch", "triton", "nvidia", "cuda", "flash"))}
    constraints = work / "image-constraints.txt"
    constraints.write_text("".join(f"{k}=={v}\n" for k, v in sorted(protected.items())))
    with (log_root / f"install-node-{node}.log").open("x") as log:
        run_step([sys.executable, "-m", "pip", "install", "-r", str(work / "requirements.txt"),
                  "--constraint", str(constraints)], smoke_deadline, stdout=log,
                 stderr=subprocess.STDOUT)
    for name, version in protected.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("image accelerator package changed: " + name)
    config = json.loads((cloud / "experiment.json").read_text())
    if config["total_cards"] != size:
        raise ValueError("platform card count differs from frozen submission")
    info = json.loads((work / "bundle-info.json").read_text())
    if config["source_commit"] != info["source_commit"]:
        raise ValueError("runtime/config source differs")
    scripts = work / "scripts/loopscope"
    prepare = scripts / "prepare_phase11_qwen3_nothink_inputs.py"
    launcher = scripts / "phase11_qwen3_nothink_launch.py"
    model = config["model_mount"]
    admission = cloud / "admission.json"
    def build(stage, synthetic, until):
        cmd = [sys.executable, str(prepare), "--model-dir", model,
               "--output-dir", str(cloud / "inputs" / stage),
               "--model-provenance-file", str(admission)]
        cmd += ["--synthetic"] if synthetic else ["--source-csv", config["source_csv_mount"]]
        run_step(cmd, until, env=env)
    def launch(stage, output, until):
        run_step([sys.executable, str(launcher), "launch", "--plan", str(cloud / f"plan-{stage}.json"),
            "--model-dir", model, "--admission", str(admission), "--launcher-output", str(output),
            "--platform-job-id", str(job_id), "--timeout-seconds", str(max(1, int(remaining(until))))],
            until, env=env)
    if node == 0:
        build("synthetic", True, smoke_deadline)
    else:
        wait_file(cloud / "inputs/synthetic/alignment.json", smoke_deadline, "synthetic input preparation")
    launch("synthetic", cloud / "launch/synthetic", smoke_deadline)
    if node != 0:
        wait_file(log_root / "runtime-smoke.json", smoke_deadline, "both-node synthetic validation")
        wait_file(cloud / "inputs/formal/alignment.json", deadline, "formal input preparation")
        launch("formal", log_root, deadline)
        wait_file(log_root / "formal-complete.json", deadline, "rank-zero closure and scoring")
        return
    for peer in range(NODE_COUNT):
        wait_file(cloud / f"launch/synthetic/node-{peer}/complete.json", smoke_deadline,
                  "synthetic node completion")
    smoke = json.loads((cloud / "plan-synthetic.json").read_text())
    for task in smoke["tasks"]:
        for job in task["jobs"]:
            summary = json.loads((Path(job["run_root"]) / "summary.json").read_text())
            if summary["status"] != "RAW_COMPLETE" or summary["count"] != 1:
                raise ValueError("synthetic three-arm worker path incomplete")
    with (log_root / "runtime-smoke.writing").open("x") as handle:
        json.dump({"status": "RUNTIME_SMOKE_COMPLETE", "source_commit": info["source_commit"],
                   "elapsed_seconds": time.monotonic()-started, "workers": size, "arms_per_worker": 3,
                   "long_sequence_capacity_status": "NOT_MEASURED", "target_gold_loaded": False}, handle, indent=2)
    (log_root / "runtime-smoke.writing").rename(log_root / "runtime-smoke.json")
    build("formal", False, deadline)
    launch("formal", log_root, deadline)
    for peer in range(NODE_COUNT):
        wait_file(log_root / f"node-{peer}/complete.json", deadline, "formal node completion")
    analyzer = scripts / "analyze_phase11_qwen3_nothink.py"
    closure, analysis = log_root / "closure", log_root / "analysis"
    run_step([sys.executable, str(analyzer), "--registry", str(cloud / "registry-formal.json"),
        "--model-dir", model, "--output-dir", str(closure)], deadline, env=env)
    run_step([sys.executable, str(analyzer), "--closure", str(closure / "closure.json"),
        "--model-dir", model, "--gold", str(cloud / "inputs/formal/sealed/gpqa_main-gold.jsonl"),
        "--output-dir", str(analysis)], deadline, env=env)
    with (log_root / "formal-complete.writing").open("x") as handle:
        json.dump({"status": "FORMAL_AND_ANALYSIS_COMPLETE", "record_count": 1344,
                   "platform_job_id": job_id, "source_commit": info["source_commit"],
                   "model_provenance": config["model_provenance"], "max_new_tokens": 32768,
                   "do_sample": False, "scoring_region": "whole_generated_output_as_final",
                   "closure": str(closure / "closure.json"), "analysis": str(analysis / "analysis.json"),
                   "elapsed_seconds": time.monotonic()-started}, handle, indent=2)
    (log_root / "formal-complete.writing").rename(log_root / "formal-complete.json")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if "--root" in sys.argv:
            node = int(os.environ.get("RANK", "0")) // 8
            failure = Path(sys.argv[sys.argv.index("--root") + 1]) / f"launch/bootstrap-node-{node}-failure.json"
            failure.parent.mkdir(parents=True, exist_ok=True)
            with failure.open("x") as handle:
                json.dump({"status": "FAILED", "exception_type": type(exc).__name__, "reason": str(exc)}, handle, indent=2)
        raise
