#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
set -euo pipefail
repo=$1; pool=$2; group=$3; commit=$4; scope=$5
python_runtime=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
"$python_runtime" scripts/loopscope/run_phase8_gate_g_calibration.py --pool "$pool" --run-root "$group" --commit "$commit" --scope "$scope"
"$python_runtime" scripts/loopscope/verify_phase8_gate_g_calibration.py --root "$group" --pool "$pool" --scope "$scope" --output "$group/verification.json"
