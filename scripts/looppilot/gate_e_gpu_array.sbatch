#!/usr/bin/env bash
#SBATCH --job-name=looppilot-gate-e
#SBATCH --partition=i64m1tga40u
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_looppilot
python_bin=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction/.venv/bin/python
source_venv=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction/.venv
: "${RUN_ROOT:?}"
: "${EXPECTED_COMMIT:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
case "$RUN_ROOT" in
  /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot/gates/gate_e/*_*) ;;
  *) exit 2 ;;
esac
cd "$repo"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/looppilot)" = "$EXPECTED_COMMIT"
test "$(git rev-parse --abbrev-ref HEAD)" = looppilot
test -z "$(git status --short)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo/src"
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
shard="$RUN_ROOT/shards/shard-$SLURM_ARRAY_TASK_ID"
test ! -e "$shard"
mkdir "$shard"
find "$source_venv" -type f -printf '%P %s %T@\n' | LC_ALL=C sort | sha256sum > "$RUN_ROOT/receipts/venv-before-shard-$SLURM_ARRAY_TASK_ID.sha256"
nvidia-smi > "$RUN_ROOT/receipts/nvidia-smi-before-shard-$SLURM_ARRAY_TASK_ID.txt"
"$python_bin" scripts/looppilot/run_gate_e_shard.py --config configs/looppilot/gate_e_full.json --preflight-dir "$RUN_ROOT/preflight" --shard-id "$SLURM_ARRAY_TASK_ID" --output-dir "$shard/results"
find "$source_venv" -type f -printf '%P %s %T@\n' | LC_ALL=C sort | sha256sum > "$RUN_ROOT/receipts/venv-after-shard-$SLURM_ARRAY_TASK_ID.sha256"
cmp "$RUN_ROOT/receipts/venv-before-shard-$SLURM_ARRAY_TASK_ID.sha256" "$RUN_ROOT/receipts/venv-after-shard-$SLURM_ARRAY_TASK_ID.sha256"
nvidia-smi > "$RUN_ROOT/receipts/nvidia-smi-after-shard-$SLURM_ARRAY_TASK_ID.txt"
mv "$shard/results"/* "$shard"/
rmdir "$shard/results"
(cd "$shard" && find . -type f ! -name results_manifest.txt -print | LC_ALL=C sort | while IFS= read -r path; do sha256sum "$path"; done > results_manifest.txt)
printf 'complete\n' > "$shard/shard_complete.txt"
