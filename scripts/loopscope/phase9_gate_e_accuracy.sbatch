#!/usr/bin/env bash
# Gate E formal score worker. Each process keeps scientific batch_size=1.
# Usage: phase9_gate_e_accuracy.sbatch MANIFEST POOL CELL_ID RUN_ROOT COMMIT SCOPE SELECTION CANARY_FILE DIRECTION_POLICY INTERVENTION_MODE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 MANIFEST POOL CELL_ID RUN_ROOT COMMIT SCOPE SELECTION CANARY_FILE DIRECTION_POLICY INTERVENTION_MODE" >&2
  exit 2
fi
repo=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/phase9-gate-e-source
python_runtime=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
manifest=$1
pool=$2
cell_id=$3
run_root=$4
commit=$5
scope=$6
selection=$7
canary_file=$8
direction_policy=$9
intervention_mode=${10}
[[ -x "$python_runtime" ]]
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
[[ "$selection" == preflight || "$selection" == canary || "$selection" == remaining ]]
[[ "$direction_policy" == fixed_t0 || "$direction_policy" == lag1 ]]
[[ "$intervention_mode" == spectral || "$intervention_mode" == matched_norm ]]
cd "$repo"
export PYTHONPATH="$repo/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
extra=()
if [[ "$canary_file" != - ]]; then
  extra+=(--canary-indices "$canary_file")
fi
exec "$python_runtime" scripts/loopscope/run_phase9_gate_e_accuracy.py \
  --manifest "$manifest" --pool "$pool" --cell-id "$cell_id" \
  --run-root "$run_root" --commit "$commit" --scope "$scope" \
  --selection "$selection" --direction-policy "$direction_policy" \
  --intervention-mode "$intervention_mode" "${extra[@]}"
