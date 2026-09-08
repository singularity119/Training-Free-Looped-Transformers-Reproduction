#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
set -euo pipefail
repo=$1; pool=$2; group=$3; commit=$4
python_runtime=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir "$group"
for cell in 0 1 2 3; do
  prefix=q4; if [[ "$cell" -ge 2 ]]; then prefix=q17; fi
  saved=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase8-gate-c-20260905T100000Z-formal-$prefix-a1/cell-$(( (cell % 2) * 2 ))/residuals.pt
  "$python_runtime" scripts/loopscope/run_phase8_gate_f_calibration.py --mode t0-from-saved --cell-index "$cell" --pool "$pool" --run-root "$group/t0/cell-$cell" --saved-residuals "$saved" --commit "$commit" --scope FORMAL_CALIBRATION
done
for wi in 0 1; do
  "$python_runtime" scripts/loopscope/run_phase8_gate_f_calibration.py --mode t1 --cell-index "$((wi+2))" --pool "$pool" --run-root "$group/t1/cell-$wi" --commit "$commit" --scope FORMAL_CALIBRATION
done
"$python_runtime" scripts/loopscope/verify_phase8_gate_f_calibration.py --root "$group" --pool "$pool" --output "$group/verification.json"
