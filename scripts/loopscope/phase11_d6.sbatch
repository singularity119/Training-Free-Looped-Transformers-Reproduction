#!/bin/bash
set -euo pipefail
source_dir=${1:?source}
python_runtime=${2:?pinned Python}
plan_file=${3:?plan}
[[ -x "$python_runtime" && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
cd "$source_dir"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/usr/local/cuda/bin
export PYTHONPATH="$source_dir/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
"$python_runtime" scripts/loopscope/phase11_d6_batch.py --plan "$plan_file"
if [[ "${SLURM_JOB_PARTITION:-}" == debug ]]; then
    exec "$python_runtime" scripts/loopscope/check_phase11_d6_gpu.py --plan "$plan_file"
fi
