#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
set -euo pipefail
repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
python_runtime="$repo/.venv-loopscope-cu121-20260711/bin/python"
cd "$repo"
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# Arguments: model index, pool path, fresh run root, checked commit, optional cell index.
args=(--model-index "$1" --pool "$2" --run-root "$3" --commit "$4")
if [[ $# == 5 ]]; then args+=(--cell-index "$5"); fi
"$python_runtime" scripts/loopscope/run_phase8_debug.py "${args[@]}"
"$python_runtime" scripts/loopscope/verify_phase8_debug.py --run-root "$3"
