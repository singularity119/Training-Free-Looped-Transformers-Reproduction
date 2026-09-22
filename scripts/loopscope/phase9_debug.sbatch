#!/usr/bin/env bash
# Gate B: one real model/window/K debug preflight on the debug partition.
# Usage: phase9_debug.sbatch MODEL_INDEX POOL RUN_ROOT SOURCE_COMMIT CELL_INDEX
# Use submit_phase9_debug_jobs.sh for multi-cell submission so indexes come
# from the frozen model/window/K manifest rather than a hand-written range.
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:29:00
set -euo pipefail

repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
python_runtime="$repo/.venv-loopscope-cu121-20260711/bin/python"
if [[ $# -ne 5 ]]; then
  echo "usage: $0 MODEL_INDEX POOL RUN_ROOT SOURCE_COMMIT CELL_INDEX" >&2
  exit 2
fi
model_index=$1
pool=$2
run_root=$3
source_commit=$4
cell_index=$5
cd "$repo"
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
exec "$python_runtime" scripts/loopscope/run_phase9_debug.py \
  --mode debug \
  --scope DEBUG_PREFLIGHT \
  --model-index "$model_index" \
  --pool "$pool" \
  --run-root "$run_root" \
  --commit "$source_commit" \
  --cell-index "$cell_index"
