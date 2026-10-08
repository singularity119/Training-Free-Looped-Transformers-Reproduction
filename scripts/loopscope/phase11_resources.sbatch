#!/usr/bin/env bash
# One independent batch-one case. Submitter supplies GPU type, memory and QoS
# only after scheduler admission; this launcher never submits another job.
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
[[ $# -ge 2 ]]
source_dir=$1
python_runtime=$2
shift 2
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
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  extra+=(--dry-run)
fi
exec "$python_runtime" scripts/loopscope/measure_phase11_resources.py "$@" "${extra[@]}"
