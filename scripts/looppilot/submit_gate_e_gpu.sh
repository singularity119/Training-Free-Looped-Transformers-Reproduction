#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then echo "usage: $0 RUN_ROOT A|B" >&2; exit 2; fi
repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_looppilot
python_bin=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction/.venv/bin/python
workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot
gate_c="$workspace/gates/gate_c/20260711T125126Z_98ed1f06"
gate_d="$workspace/gates/gate_d/20260711T135708Z_dfcab688"
run_root=$1
batch=$2
cd "$repo"
head_commit=$(git rev-parse HEAD)
test "$(git rev-parse --abbrev-ref HEAD)" = looppilot
test "$(git rev-parse origin/looppilot)" = "$head_commit"
test -z "$(git status --short)"
case "$run_root" in "$workspace"/gates/gate_e/*_"${head_commit:0:8}") ;; *) exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo/src"
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [[ "$batch" = A ]]; then
  test ! -e "$run_root"
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
  find "$run_root/preflight" -type f -print | LC_ALL=C sort | while IFS= read -r path; do sha256sum "$path"; done > "$run_root/control/preflight_sha256.txt"
  array=0-3%2
  receipt=batch-a
elif [[ "$batch" = B ]]; then
  test -d "$run_root"
  test -f "$run_root/receipts/batch-a-terminal-verified.txt"
  for i in 0 1 2 3; do test -f "$run_root/shards/shard-$i/shard_complete.txt"; done
  array=4-7%2
  receipt=batch-b
else exit 2; fi
test ! -e "$run_root/receipts/$receipt-job_id.txt"
printf 'sbatch --array=%s --export=ALL,RUN_ROOT=%q,EXPECTED_COMMIT=%q %q\n' "$array" "$run_root" "$head_commit" "$run_root/control/gate_e_gpu_array.sbatch" > "$run_root/receipts/$receipt-submit-command.txt"
job_id=$(sbatch --parsable --array="$array" --output="$run_root/receipts/slurm-%A_%a.out" --error="$run_root/receipts/slurm-%A_%a.err" --export=ALL,RUN_ROOT="$run_root",EXPECTED_COMMIT="$head_commit" "$run_root/control/gate_e_gpu_array.sbatch")
(set -C; printf '%s\n' "$job_id" > "$run_root/receipts/$receipt-job_id.txt")
scontrol show job "$job_id" -o | sed -n 's/.*SubmitLine=\([^ ]*.*\)/SubmitLine=\1/p' > "$run_root/receipts/$receipt-SubmitLine.txt" || true
printf '%s\n' "$job_id"
