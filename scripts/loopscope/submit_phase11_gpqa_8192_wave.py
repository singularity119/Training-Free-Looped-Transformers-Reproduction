#!/usr/bin/env python3
"""Executor-invoked bounded D-4 submissions; never used by the read-only observer."""
import argparse
import json
from pathlib import Path
import subprocess


def run(args):
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    packing = plan["packing"]
    assert packing in (4, 5) and plan["target_gold_loaded"] is False
    assert 0 <= args.first_batch < len(plan["batches"]) and 1 <= args.batch_count <= 48
    source = Path(args.source)
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=source, text=True).strip() == "loopscope"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() == args.commit
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    queue = subprocess.check_output(["/opt/slurm/bin/squeue", "-h", "-u", "xhuang225", "-o", "%i"], text=True)
    existing = len(queue.splitlines())
    batches = plan["batches"][args.first_batch:args.first_batch + args.batch_count]
    assert existing + len(batches) <= 50, "normal-user MaxSubmitJobs=50 admission exceeded"
    receipt_dir = plan_path.parent / "submission-receipts"
    receipt_dir.mkdir(exist_ok=True)
    for batch in batches:
        receipt = receipt_dir / f'batch-{batch["number"]:03d}.json'
        assert not receipt.exists(), "batch already has a submission attempt receipt"
        assert not Path(batch["batch_root"]).exists(), "batch root exists; preserve prior attempt"
        work = json.loads(Path(batch["worklist"]).read_text())
        assert all(not Path(j["run_root"]).exists() for j in work["jobs"]), "child root exists"
    submitted = []
    for batch in batches:
        number = batch["number"]
        command = ["/opt/slurm/bin/sbatch", "--parsable", "--partition=emergency_gpu",
            "--qos=emergency_gpu", "--gres=gpu:a800:1", "--cpus-per-task=8", "--mem=96G",
            "--time=02:00:00", f"--job-name=ls-p11-d4-full-p{packing}-b{number:03d}",
            f"--output={args.logs}/full-p{packing}-b{number:03d}-%j.out",
            f"--error={args.logs}/full-p{packing}-b{number:03d}-%j.err",
            "scripts/loopscope/phase11_gpqa_8192_batch.sbatch", str(source), args.python,
            "--worklist", batch["worklist"], "--batch-root", batch["batch_root"],
            "--packing", str(packing), "--commit", args.commit]
        result = subprocess.run(command, cwd=source, capture_output=True, text=True)
        fact = {**batch, "packing": packing, "command": command, "exit_code": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "target_gold_loaded": False}
        if result.returncode == 0:
            job_id = result.stdout.strip().split(";")[0]
            assert job_id.isdecimal(), "unrecognized sbatch result; inspect before any retry"
            fact["job_id"] = job_id
        with (receipt_dir / f"batch-{number:03d}.json").open("x") as handle:
            json.dump(fact, handle, indent=2)
        print(json.dumps({k: fact[k] for k in ("number", "exit_code", "stdout")}), flush=True)
        if result.returncode:
            raise RuntimeError("submission rejected; receipt preserved, stop this wave")
        submitted.append(fact)
    with (plan_path.parent / f"wave-{args.first_batch:03d}-submission.json").open("x") as handle:
        json.dump({"packing": packing, "first_batch": args.first_batch, "batch_count": len(submitted),
                   "job_ids": [j["job_id"] for j in submitted], "jobs": submitted,
                   "reserved_gpuh": len(submitted) * 2, "target_gold_loaded": False}, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "source", "python", "commit", "logs"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--first-batch", type=int, required=True)
    parser.add_argument("--batch-count", type=int, required=True)
    run(parser.parse_args())
