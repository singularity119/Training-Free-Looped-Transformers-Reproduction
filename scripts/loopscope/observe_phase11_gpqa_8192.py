#!/usr/bin/env python3
"""One bounded read-only D-4 scheduler/aggregate check of an exact jobset."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import subprocess


def run(jobset_path):
    jobset = json.loads(Path(jobset_path).read_text())
    jobs = jobset["jobs"]
    ids = [j["job_id"] for j in jobs]
    assert jobs and len(ids) == len(set(ids)) and ids == jobset["active_job_ids"]
    queue_result = subprocess.run(["/opt/slurm/bin/squeue", "-h", "-j", ",".join(ids),
        "-o", "%i|%T|%R"], capture_output=True, text=True, timeout=15)
    account_result = subprocess.run(["/opt/slurm/bin/sacct", "-X", "-n", "-P", "-j", ",".join(ids),
        "-o", "JobID,State,ExitCode,ElapsedRaw"], capture_output=True, text=True, timeout=15)
    assert queue_result.returncode == account_result.returncode == 0, (queue_result.stderr, account_result.stderr)
    queue = [line.split("|") for line in queue_result.stdout.splitlines() if line.strip()]
    facts = {}
    for line in account_result.stdout.splitlines():
        row = line.split("|")
        if row[0] in ids:
            facts[row[0]] = {"state": row[1], "exit_code": row[2], "allocation_seconds": int(row[3])}
    bad_states = {"FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
    failed = [{"job_id": i, **v} for i, v in facts.items() if v["state"].split()[0] in bad_states
              or (v["state"] == "COMPLETED" and v["exit_code"] != "0:0")]
    complete_records = 0
    failed_parents, complete_parents = [], []
    for job in jobs:
        root = Path(job["batch_root"])
        summary = root / "batch_summary.json"
        if summary.exists():
            value = json.loads(summary.read_text())
            (complete_parents if value["status"] == "BATCH_COMPLETE" else failed_parents).append(job["job_id"])
        for child in json.loads(Path(job["worklist"]).read_text())["jobs"]:
            path = Path(child["run_root"]) / "summary.json"
            if path.exists():
                value = json.loads(path.read_text())
                if value["status"] == "GPQA_8192_ATTEMPT_COMPLETE":
                    complete_records += value["count"]
    missing = sorted(set(ids) - set(facts))
    all_terminal = not queue and not missing and all(v["state"].split()[0] in bad_states | {"COMPLETED"} for v in facts.values())
    needs_refill = len(queue) <= 8 and jobset["unsubmitted_records"] > 0 and not missing
    return {"checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "active_job_ids": ids, "jobset_file": str(jobset_path), "scheduler_exit_codes": [0, 0],
        "queue_state_counts": dict(Counter(row[1] for row in queue)),
        "accounting_state_counts": dict(Counter(v["state"] for v in facts.values())),
        "missing_accounting_ids": missing, "failed_jobs": failed, "failed_parent_jobs": failed_parents,
        "complete_parent_jobs": complete_parents, "current_inflight_count": len(queue),
        "current_wave_complete_child_records": complete_records,
        "verified_complete_formal_records": jobset["prior_verified_complete_records"] + complete_records,
        "all_current_terminal": all_terminal, "needs_refill": needs_refill,
        "trigger_resume": bool(failed or failed_parents or all_terminal or needs_refill),
        "completed_allocation_gpuh": sum(v["allocation_seconds"] for v in facts.values() if v["state"] == "COMPLETED") / 3600,
        "target_gold_loaded": False, "mmlu_accessed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobset", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.jobset)))
