#!/usr/bin/env bash
# Nonformal GPU checks only. Resources are supplied at submission on debug.
# Usage: SOURCE PYTHON CHECK_SCRIPT [script arguments...]
#SBATCH --time=00:29:00
set -euo pipefail
[[ $# -ge 3 ]]
source_dir=$1
python_runtime=$2
check_script=$3
shift 3
[[ "$check_script" == check_phase10_gpu.py || "$check_script" == measure_phase10_resources.py ]]
[[ "${SLURM_JOB_PARTITION:-}" == debug ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
[[ -x "$python_runtime" ]]
cd "$source_dir"
export PYTHONPATH="$source_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
exec "$python_runtime" "scripts/loopscope/$check_script" "$@"
