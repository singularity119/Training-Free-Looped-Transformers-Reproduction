#!/usr/bin/env bash
# Gate E real-model engineering preflight; each job is strictly <30 minutes.
# Usage: phase9_gate_e_debug.sbatch POOL RUN_ROOT SOURCE_COMMIT MODEL_INDEX
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:29:00
set -euo pipefail

repo=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/phase9-gate-e-source
python_runtime=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
if [[ $# -ne 4 ]]; then
  echo "usage: $0 POOL RUN_ROOT SOURCE_COMMIT MODEL_INDEX" >&2
  exit 2
fi
pool=$1
run_root=$2
source_commit=$3
model_index=$4
[[ -x "$python_runtime" ]]
[[ "$SLURM_JOB_PARTITION" == debug ]]
cd "$repo"
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
exec "$python_runtime" scripts/loopscope/run_phase9_gate_e_debug.py \
  --pool "$pool" --run-root "$run_root" --commit "$source_commit" \
  --model-index "$model_index" \
  --direction-policies fixed_t0 lag1 \
  --intervention-modes spectral matched_norm
