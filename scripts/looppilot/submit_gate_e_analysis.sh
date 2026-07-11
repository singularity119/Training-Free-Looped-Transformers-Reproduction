#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "usage: $0 RUN_ROOT" >&2; exit 2; fi
repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_looppilot
run_root=$1
cd "$repo"
head_commit=$(git rev-parse HEAD)
test "$(git rev-parse origin/looppilot)" = "$head_commit"
test -z "$(git status --short)"
for i in 0 1 2 3 4 5 6 7; do test -f "$run_root/shards/shard-$i/shard_complete.txt"; done
test -f "$run_root/receipts/gpu-array-terminal-verified.txt"
test ! -e "$run_root/receipts/analysis-job_id.txt"
job_id=$(/opt/slurm/bin/sbatch --parsable --output="$run_root/receipts/slurm-analysis-%j.out" --error="$run_root/receipts/slurm-analysis-%j.err" --export=ALL,RUN_ROOT="$run_root",EXPECTED_COMMIT="$head_commit" "$run_root/control/gate_e_analysis.sbatch")
(set -C; printf '%s\n' "$job_id" > "$run_root/receipts/analysis-job_id.txt")
/opt/slurm/bin/scontrol show job "$job_id" -o | sed -n 's/.*SubmitLine=\([^ ]*.*\)/SubmitLine=\1/p' > "$run_root/receipts/analysis-SubmitLine.txt" || true
printf '%s\n' "$job_id"
