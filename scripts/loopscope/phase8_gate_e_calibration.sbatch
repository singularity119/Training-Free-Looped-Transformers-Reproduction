#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
set -euo pipefail
repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
python_runtime="$repo/.venv-loopscope-cu121-20260711/bin/python"
cd "$repo"
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
# pool, fresh group root, producer commit, scope, packing, both cell indices
pool=$1; group=$2; commit=$3; scope=$4; packing=$5
shift 5
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / packing ))
[[ "$packing" =~ ^[123]$ ]]
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
  root="$group/cell-$cell"
  roots+=("$root")
  "$python_runtime" scripts/loopscope/run_phase8_gate_e_calibration.py --cell-index "$cell" --pool "$pool" --run-root "$root" --commit "$commit" --scope "$scope" > "$group/cell-$cell.log" 2>&1 &
  pids+=("$!")
  if [[ ${#pids[@]} == "$packing" ]]; then wait_wave; fi
done
wait_wave
"$python_runtime" scripts/loopscope/verify_phase8_gate_e_calibration.py --cell-roots "${roots[@]}" --pool "$pool" --scope "$scope" --require-all --output "$group/verification.json"
