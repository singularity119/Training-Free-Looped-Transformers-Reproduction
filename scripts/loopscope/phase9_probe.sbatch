#!/usr/bin/env bash
# Gate B: all frozen diagnostic cells for one model on a formal user partition.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00

# The caller selects the ordinary-user legal partition/QoS at submission time.
# Usage: phase9_probe.sbatch MODEL_INDEX POOL RUN_ROOT SOURCE_COMMIT
set -euo pipefail

repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
python_runtime="$repo/.venv-loopscope-cu121-20260711/bin/python"
if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODEL_INDEX POOL RUN_ROOT SOURCE_COMMIT" >&2
  exit 2
fi
model_index=$1
pool=$2
run_root=$3
source_commit=$4
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
  --mode diagnostic \
  --scope FORMAL_DIAGNOSTIC \
  --model-index "$model_index" \
  --pool "$pool" \
  --run-root "$run_root" \
  --commit "$source_commit"
