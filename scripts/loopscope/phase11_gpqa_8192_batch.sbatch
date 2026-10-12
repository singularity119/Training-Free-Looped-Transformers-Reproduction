#!/bin/bash
set -euo pipefail
source_dir=${1:?source}
python_runtime=${2:?pinned Python}
shift 2
[[ -x "$python_runtime" && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && ("${SLURM_JOB_PARTITION:-}" == debug || "${SLURM_JOB_PARTITION:-}" == emergency_gpu) ]]
cd "$source_dir"
export PYTHONPATH="$source_dir/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec "$python_runtime" scripts/loopscope/phase11_gpqa_8192_batch.py "$@"
