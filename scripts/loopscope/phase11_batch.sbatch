#!/bin/bash
# Gate D same-path debug/formal launcher. Submitter owns scheduler admission.
# Request 1 GPU / 8 CPUs; packing and worklist are explicit arguments.
set -euo pipefail
source_dir=${1:?source}
python_runtime=${2:?pinned Python}
shift 2
[[ -x "$python_runtime" && "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
cd "$source_dir"
export PYTHONPATH="$source_dir/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# run_phase11_batch sets and captures thread counts separately for each packing.
exec "$python_runtime" scripts/loopscope/run_phase11_batch.py "$@"
