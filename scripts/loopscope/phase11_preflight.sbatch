#!/usr/bin/env bash
# Gate B submission must be separately authorized. Supply GPU/memory/QoS
# after current scheduler inspection; no submit is performed by this file.
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
[[ $# -ge 3 ]]
source_dir=$1
python_runtime=$2
entry=$3
shift 3
[[ "$entry" == check_phase11_gpu.py || "$entry" == run_phase11_accuracy.py ]]
[[ -x "$python_runtime" ]]
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
  [[ "${SLURM_JOB_PARTITION:-}" == debug ]]
fi
cd "$source_dir"
export PYTHONPATH="$source_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
extra=()
if [[ "$entry" == run_phase11_accuracy.py ]]; then
  extra+=(--scope PREFLIGHT_ONLY)
  arguments=("$@")
  for ((i=0; i<${#arguments[@]}; i++)); do
    if [[ "${arguments[i]}" == --dataset && "${arguments[i+1]:-}" != arc_challenge ]]; then
      extra+=(--engineering-generate-steps 3)
      break
    fi
  done
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  extra+=(--dry-run)
fi
exec "$python_runtime" "scripts/loopscope/$entry" "$@" "${extra[@]}"
