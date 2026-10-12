#!/bin/bash
# D-3 debug-only launcher. Original model/runtime/cache path remains unchanged.
set -euo pipefail
source_dir=${1:?source}
python_runtime=${2:?pinned Python}
shift 2
[[ -x "$python_runtime" && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_JOB_PARTITION:-}" == debug ]]
cd "$source_dir"
export PYTHONPATH="$source_dir/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$python_runtime" scripts/loopscope/phase11_length_pilot.py "$@"
