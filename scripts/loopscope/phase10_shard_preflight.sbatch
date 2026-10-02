#!/usr/bin/env bash
# Gate C supplies debug partition/GRES/memory. Exactly one GPU, packing1.
# Usage: REPO PYTHON MANIFEST POOL RUN_ROOT COMMIT [CELL_ID]
# Omitting CELL_ID runs all three model/K combinations sequentially.
#SBATCH --time=00:29:00
set -euo pipefail
[[ $# -eq 6 || $# -eq 7 ]]
repo=$1
python_runtime=$2
[[ -x "$python_runtime" ]]
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
  [[ "${SLURM_JOB_PARTITION:-}" == debug ]]
fi
cd "$repo"
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
extra=()
if [[ $# -eq 7 ]]; then
  extra+=(--cell-id "$7")
fi
if [[ "${PHASE10_CONFIG:-}" != "" ]]; then
  extra+=(--config "$PHASE10_CONFIG")
fi
if [[ "${PHASE10_PREFLIGHT_INDICES:-}" != "" ]]; then
  extra+=(--preflight-indices "$PHASE10_PREFLIGHT_INDICES")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  extra+=(--dry-run)
fi
exec "$python_runtime" scripts/loopscope/check_phase10_shard_preflight.py \
  --repo "$repo" --python "$python_runtime" --manifest "$3" --pool "$4" \
  --run-root "$5" --commit "$6" "${extra[@]}"
