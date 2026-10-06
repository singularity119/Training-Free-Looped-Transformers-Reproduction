#!/usr/bin/env bash
# Submission supplies partition/QoS/GRES/time; one sequential worker per GPU.
# Usage: REPO PYTHON MANIFEST POOL PLAN RUN_ROOT COMMIT SCOPE GPU_COUNT PRIOR_GPUH WALLTIME_HOURS PREFLIGHT_INDICES_OR_DASH
set -euo pipefail
if [[ "${1:-}" == --help || $# -ne 12 ]]; then
  echo "usage: $0 REPO PYTHON MANIFEST POOL PLAN RUN_ROOT COMMIT SCOPE GPU_COUNT PRIOR_GPUH WALLTIME_HOURS PREFLIGHT_INDICES_OR_DASH"
  [[ "${1:-}" == --help ]] && exit 0
  exit 2
fi
repo=$1
python_runtime=$2
[[ -x "$python_runtime" ]]
cd "$repo"
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
extra=()
if [[ "${12}" != - ]]; then
  extra+=(--preflight-indices "${12}")
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  extra+=(--dry-run)
  allocation_start=1
else
  [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
  # Slurm supplies the real allocation epoch without a compute-node CLI.
  allocation_start=${SLURM_JOB_START_TIME:?Slurm allocation start epoch missing}
  [[ "$allocation_start" =~ ^[0-9]+$ && "$allocation_start" -gt 0 ]]
  echo "allocation_start_epoch=$allocation_start source=SLURM_JOB_START_TIME" >&2
fi
exec "$python_runtime" scripts/loopscope/run_phase10_arc_bundle.py \
  --repo "$repo" --python "$python_runtime" --manifest "$3" --pool "$4" --plan "$5" \
  --run-root "$6" --commit "$7" --scope "$8" --gpu-count "$9" \
  --prior-gpu-hours "${10}" --walltime-hours "${11}" \
  --allocation-start-epoch "$allocation_start" "${extra[@]}"
