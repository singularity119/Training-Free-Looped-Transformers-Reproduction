#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 RUN_ROOT" >&2; exit 2; fi

repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_looppilot
python_bin=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction/.venv/bin/python
workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot
gate_c="$workspace/gates/gate_c/20260711T125126Z_98ed1f06"
gate_d="$workspace/gates/gate_d/20260711T135708Z_dfcab688"
run_root=$1

cd "$repo"
head_commit=$(git rev-parse HEAD)
test "$(git rev-parse --abbrev-ref HEAD)" = looppilot
test "$(git rev-parse origin/looppilot)" = "$head_commit"
test -z "$(git status --short)"
case "$run_root" in "$workspace"/gates/gate_e/*_"${head_commit:0:8}") ;; *) exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo/src"
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

if [[ ! -e "$run_root" ]]; then
  mkdir -p "$workspace/gates/gate_e"
  mkdir "$run_root" "$run_root/control" "$run_root/preflight" "$run_root/shards" "$run_root/receipts"
  cp scripts/looppilot/gate_e_gpu_array.sbatch "$run_root/control/gate_e_gpu_array.sbatch"
  cp scripts/looppilot/gate_e_analysis.sbatch "$run_root/control/gate_e_analysis.sbatch"
  cp configs/looppilot/gate_e_full.json "$run_root/control/gate_e_full.json"
  cp configs/looppilot/gate_e_analysis_contract.json "$run_root/control/gate_e_analysis_contract.json"
  cp configs/looppilot/gate_e_artifact_schema.json "$run_root/control/gate_e_artifact_schema.json"
  cp configs/looppilot/gate_c_question_mask_schema.json "$run_root/control/question_mask_schema.json"
  "$python_bin" scripts/looppilot/prepare_gate_e_full.py --gate-c-run "$gate_c" --gate-d-run "$gate_d" --output-dir "$run_root/preflight/build"
  mv "$run_root/preflight/build"/* "$run_root/preflight"/
  rmdir "$run_root/preflight/build"
  (cd "$run_root/preflight" && find . -type f -print | LC_ALL=C sort | while IFS= read -r path; do sha256sum "$path"; done > "$run_root/control/preflight_sha256.txt")
else
  test -d "$run_root/control" -a -d "$run_root/preflight" -a -d "$run_root/shards" -a -d "$run_root/receipts"
  test -f "$run_root/control/preflight_sha256.txt"
  (cd "$run_root/preflight" && sha256sum -c "$run_root/control/preflight_sha256.txt")
fi

# Once this marker exists, the normal array may never be invoked again, regardless
# of whether the sbatch client returned a JobID.
test ! -e "$run_root/receipts/gpu-array-submit-started.txt"
test ! -e "$run_root/receipts/gpu-array-job_id.txt"

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
sinfo_raw="$run_root/receipts/gpu-capacity-sinfo-$timestamp.txt"
nodes_raw="$run_root/receipts/gpu-capacity-nodes-$timestamp.txt"
squeue_raw="$run_root/receipts/gpu-capacity-squeue-$timestamp.txt"
snapshot="$run_root/receipts/gpu-capacity-snapshot-$timestamp.json"
for path in "$sinfo_raw" "$nodes_raw" "$squeue_raw" "$snapshot" "$snapshot.sha256"; do test ! -e "$path"; done

set -C
/opt/slurm/bin/sinfo -p i64m1tga40u -N -o '%N|%P|%t|%G' > "$sinfo_raw" 2>&1
/opt/slurm/bin/scontrol show node -o > "$nodes_raw" 2>&1
/opt/slurm/bin/squeue -u xhuang225 -o '%.18i %.9P %.30j %.2t %.12M %.30R %.30b' > "$squeue_raw" 2>&1
set +C

concurrency=$("$python_bin" - "$sinfo_raw" "$nodes_raw" "$squeue_raw" "$snapshot" "$timestamp" <<'PY_CAPACITY'
import hashlib
import json
import re
import sys
from pathlib import Path

sinfo_path, nodes_path, squeue_path, snapshot_path = map(Path, sys.argv[1:5])
timestamp = sys.argv[5]

def read(path):
    return path.read_text(encoding="utf-8")

def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def fields(line):
    result = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            if key in result:
                raise RuntimeError("duplicate scontrol field %s" % key)
            result[key] = value
    return result

def a40_count(tres, required):
    matches = re.findall(r"(?:^|,)gres/gpu:a40=(\d+)(?:,|$)", tres)
    if len(matches) > 1 or (required and len(matches) != 1):
        raise RuntimeError("ambiguous gres/gpu:a40 TRES: %s" % tres)
    return int(matches[0]) if matches else 0

sinfo_text = read(sinfo_path)
nodes_text = read(nodes_path)
squeue_text = read(squeue_path)
eligible = []
for line in nodes_text.splitlines():
    if not line.strip():
        continue
    row = fields(line)
    partitions = row.get("Partitions", "").split(",")
    if "i64m1tga40u" not in partitions:
        continue
    state = row.get("State", "").upper()
    if any(flag in state for flag in ("DOWN", "DRAIN", "FAIL", "MAINT", "NOT_RESPOND")):
        continue
    cfg_text = row.get("CfgTRES", "")
    cfg_matches = re.findall(r"(?:^|,)gres/gpu:a40=(\d+)(?:,|$)", cfg_text)
    if not cfg_matches:
        continue
    cfg = a40_count(cfg_text, True)
    alloc = a40_count(row.get("AllocTRES", ""), False)
    free = cfg - alloc
    if free < 0:
        raise RuntimeError("negative scheduler-visible free A40 on %s" % row.get("NodeName"))
    eligible.append({"node": row.get("NodeName"), "state": state, "cfg_a40": cfg, "alloc_a40": alloc, "free_a40": free})

aggregate = sum(row["free_a40"] for row in eligible)
chosen_concurrency = 8
payload = {
    "schema_version": 1,
    "timestamp_utc": timestamp,
    "partition": "i64m1tga40u",
    "excluded_states": ["DOWN", "DRAIN", "FAIL", "MAINT", "NOT_RESPOND"],
    "commands": {
        "sinfo": "/opt/slurm/bin/sinfo -p i64m1tga40u -N -o %N|%P|%t|%G",
        "scontrol": "/opt/slurm/bin/scontrol show node -o",
        "squeue": "/opt/slurm/bin/squeue -u xhuang225",
    },
    "raw_outputs": {
        "sinfo": {"stdout": sinfo_text, "sha256": sha(sinfo_text)},
        "scontrol_nodes": {"stdout": nodes_text, "sha256": sha(nodes_text)},
        "squeue": {"stdout": squeue_text, "sha256": sha(squeue_text)},
    },
    "eligible_nodes": eligible,
    "aggregate_free_a40": aggregate,
    "chosen_concurrency": chosen_concurrency,
    "submission_mode": "pending_allowed",
}
with snapshot_path.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
print(chosen_concurrency)
PY_CAPACITY
)
(set -C; sha256sum "$snapshot" > "$snapshot.sha256")
test "$concurrency" = 8

array="0-7%8"
submit_command="$run_root/receipts/gpu-array-submit-command.txt"
submit_stdout="$run_root/receipts/gpu-array-submit.stdout"
submit_stderr="$run_root/receipts/gpu-array-submit.stderr"
submit_exit="$run_root/receipts/gpu-array-submit.exitcode"
for path in "$submit_command" "$submit_stdout" "$submit_stderr" "$submit_exit"; do test ! -e "$path"; done
submit_argv=(/opt/slurm/bin/sbatch --parsable --array="$array" --output="$run_root/receipts/slurm-%A_%a.out" --error="$run_root/receipts/slurm-%A_%a.err" --export=ALL,RUN_ROOT="$run_root",EXPECTED_COMMIT="$head_commit" "$run_root/control/gate_e_gpu_array.sbatch")
(set -C; printf '%q ' "${submit_argv[@]}" > "$submit_command"; printf '\n' >> "$submit_command")
(set -C; printf '%s\n' "$timestamp" > "$run_root/receipts/gpu-array-submit-started.txt")
set +e
set -C
"${submit_argv[@]}" > "$submit_stdout" 2> "$submit_stderr"
submit_code=$?
set +C
set -e
(set -C; printf '%s\n' "$submit_code" > "$submit_exit")
if [[ "$submit_code" -ne 0 ]]; then exit "$submit_code"; fi
job_line=$(tr -d '\r\n' < "$submit_stdout")
job_id=${job_line%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]]
(set -C; printf '%s\n' "$job_id" > "$run_root/receipts/gpu-array-job_id.txt")
set -C
/opt/slurm/bin/scontrol show job "$job_id" -o > "$run_root/receipts/gpu-array-scontrol.txt"
sed -n 's/.*SubmitLine=\([^ ]*.*\)/SubmitLine=\1/p' "$run_root/receipts/gpu-array-scontrol.txt" > "$run_root/receipts/gpu-array-SubmitLine.txt"
set +C
printf '%s\n' "$job_id"
