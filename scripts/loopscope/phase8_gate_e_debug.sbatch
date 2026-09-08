#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
set -euo pipefail
repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
python_runtime="$repo/.venv-loopscope-cu121-20260711/bin/python"
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
calibration_pool=$1; test_pool=$2; group=$3; commit=$4
[[ "$SLURM_JOB_PARTITION" == debug ]]
mkdir "$group"
bash scripts/loopscope/phase8_gate_e_calibration.sbatch "$calibration_pool" "$group/calibration" "$commit" PREFLIGHT_ONLY 1 0 1
"$python_runtime" scripts/loopscope/build_phase8_gate_e_accuracy_manifest.py --pool "$test_pool" --basis-root "$group/calibration" --scope PREFLIGHT_ONLY --output "$group/panel.json"
bash scripts/loopscope/phase8_gate_e_accuracy.sbatch "$group/panel.json" "$test_pool" "$group/scores" "$commit" PREFLIGHT_ONLY 1 0 14042 q4-w12-15-k3-loop q4-w12-15-k3-spectral q4-w13-16-k3-loop q4-w13-16-k3-spectral
