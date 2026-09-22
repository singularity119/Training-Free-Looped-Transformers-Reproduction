#!/usr/bin/env bash
# Phase 9 new-arm score acquisition.  Each child is an independent batch-one
# model process; packing never changes the frozen scientific batch shape.
# Usage: phase9_accuracy.sbatch REPO MANIFEST POOL GROUP COMMIT SCOPE PACKING SELECTION CANARY_INDEX_FILE CELL_ID...
set -euo pipefail

if [[ $# -lt 10 ]]; then
  echo "usage: $0 REPO MANIFEST POOL GROUP COMMIT SCOPE PACKING SELECTION CANARY_INDEX_FILE CELL_ID..." >&2
  exit 2
fi

repo=$1
manifest=$2
pool=$3
group=$4
commit=$5
scope=$6
packing=$7
selection=$8
canary_indices=$9
shift 9

python_runtime="$repo/.venv-loopscope-cu121-20260711/bin/python"
[[ "$packing" =~ ^[1-8]$ ]]
[[ "$selection" == preflight || "$selection" == canary || "$selection" == remaining ]]
cd "$repo"
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / packing ))
(( OMP_NUM_THREADS > 0 )) || export OMP_NUM_THREADS=1

mkdir "$group"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu --format=csv -l 5 > "$group/gpu.csv" &
gpu_monitor=$!
trap 'kill "$gpu_monitor" 2>/dev/null || true' EXIT

pids=()
roots=()
failed=0
wait_wave() {
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  pids=()
  [[ "$failed" == 0 ]]
}

for cell in "$@"; do
  root="$group/$cell"
  roots+=("$root")
  extra=()
  if [[ "$canary_indices" != "-" ]]; then
    extra+=(--canary-indices "$canary_indices")
  fi
  "$python_runtime" scripts/loopscope/run_phase9_accuracy.py \
    --manifest "$manifest" --pool "$pool" --cell-id "$cell" \
    --run-root "$root" --commit "$commit" --scope "$scope" \
    --selection "$selection" "${extra[@]}" > "$group/$cell.log" 2>&1 &
  pids+=("$!")
  if [[ ${#pids[@]} -eq "$packing" ]]; then wait_wave; fi
done
wait_wave

verify_args=(
  --manifest "$manifest" --pool "$pool" --scope "$scope"
  --selection "$selection" --output "$group/verification.json"
  --cell-roots "${roots[@]}"
)
if [[ "$canary_indices" != "-" ]]; then
  verify_args+=(--canary-indices "$canary_indices")
fi
"$python_runtime" scripts/loopscope/verify_phase9_scores.py "${verify_args[@]}"
