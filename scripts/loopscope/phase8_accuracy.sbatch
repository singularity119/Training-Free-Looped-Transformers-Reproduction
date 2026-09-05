#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
set -euo pipefail
repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
python_runtime="$repo/.venv-loopscope-cu121-20260711/bin/python"
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
manifest=$1; pool=$2; group=$3; commit=$4; scope=$5; packing=$6; start=$7; end=$8
shift 8
[[ "$packing" =~ ^[123]$ ]]
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / packing ))
mkdir "$group"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu --format=csv -l 1 > "$group/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
pids=(); roots=(); failed=0
wait_wave() {
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  pids=()
  [[ "$failed" == 0 ]]
}
for cell in "$@"; do
  root="$group/$cell"
  roots+=("$root")
  "$python_runtime" scripts/loopscope/run_phase8_accuracy.py --manifest "$manifest" --pool "$pool" --cell-id "$cell" --run-root "$root" --commit "$commit" --scope "$scope" --start "$start" --end "$end" > "$group/$cell.log" 2>&1 &
  pids+=("$!")
  if [[ ${#pids[@]} == "$packing" ]]; then wait_wave; fi
done
wait_wave
"$python_runtime" scripts/loopscope/verify_phase8_accuracy.py --manifest "$manifest" --pool "$pool" --cell-roots "${roots[@]}" --output "$group/verification.json"
