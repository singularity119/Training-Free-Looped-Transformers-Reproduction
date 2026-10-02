#!/usr/bin/env bash
# Resource placeholders: Gate B uses debug; formal Gates supply highest legal
# A800 partition/QoS. Each Gate supplies GRES/memory/time at
# submission after current scheduler and measured resource evidence. No submit.
# Usage: REPO PYTHON DATASET MANIFEST POOL CELL_ID RUN_ROOT COMMIT SCOPE MAX_LENGTH
# MAX_LENGTH '-' keeps native HFLM model max length. DRY_RUN=1 is CPU-only.
set -euo pipefail
if [[ $# -ne 10 ]]; then
  echo "usage: $0 REPO PYTHON DATASET MANIFEST POOL CELL_ID RUN_ROOT COMMIT SCOPE MAX_LENGTH" >&2
  exit 2
fi
repo=$1
python_runtime=$2
dataset=$3
manifest=$4
pool=$5
cell_id=$6
run_root=$7
commit=$8
scope=$9
max_length=${10}
[[ -x "$python_runtime" ]]
[[ "$dataset" == mmlu || "$dataset" == arc_challenge ]]
[[ "$scope" == PREFLIGHT_ONLY || "$scope" == FORMAL_TEST ]]
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
fi
cd "$repo"
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
extra=()
if [[ "$max_length" != - ]]; then
  extra+=(--max-length "$max_length")
fi
if [[ "${PHASE10_CONFIG:-}" != "" ]]; then
  extra+=(--config "$PHASE10_CONFIG")
fi
if [[ "${PHASE10_PREFLIGHT_INDICES:-}" != "" ]]; then
  extra+=(--preflight-indices "$PHASE10_PREFLIGHT_INDICES")
fi
if [[ "${PHASE10_SHARD_INDICES:-}" != "" ]]; then
  extra+=(--shard-indices "$PHASE10_SHARD_INDICES")
fi
if [[ "${PHASE10_SHARD_ID:-}" != "" ]]; then
  extra+=(--shard-id "$PHASE10_SHARD_ID")
fi
if [[ "${PHASE10_ENGINEERING_CHECK:-}" != "" ]]; then
  extra+=(--engineering-check "$PHASE10_ENGINEERING_CHECK")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  extra+=(--dry-run)
fi
exec "$python_runtime" scripts/loopscope/run_phase10_accuracy.py \
  --dataset "$dataset" --manifest "$manifest" --pool "$pool" --cell-id "$cell_id" \
  --run-root "$run_root" --commit "$commit" --scope "$scope" "${extra[@]}"
