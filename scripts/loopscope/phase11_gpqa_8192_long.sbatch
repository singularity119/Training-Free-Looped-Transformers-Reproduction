#!/bin/bash
set -euo pipefail
source_dir=${1:?source}
python_runtime=${2:?pinned Python}
plan_file=${3:?plan}
source_commit=${4:?commit}
[[ -x "$python_runtime" && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ && -n "${SLURM_JOB_PARTITION:-}" ]]
cd "$source_dir"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/local/cuda/bin
export PYTHONPATH="$source_dir/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec "$python_runtime" scripts/loopscope/phase11_gpqa_8192_long_batch.py --plan "$plan_file" --commit "$source_commit"
