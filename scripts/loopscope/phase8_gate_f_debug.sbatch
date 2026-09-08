#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
set -euo pipefail
repo=$1; calibration_pool=$2; test_pool=$3; group=$4; commit=$5; model_index=$6
python_runtime=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export TRANSFORMERS_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
[[ "$SLURM_JOB_PARTITION" == debug ]]
mkdir "$group"
for wi in 0 1; do
  cell=$(( model_index * 2 + wi ))
  extra=()
  if [[ "$model_index" == 1 ]]; then extra=(--t1-root "$group/t1/cell-$wi"); fi
  "$python_runtime" scripts/loopscope/run_phase8_gate_f_calibration.py --mode preflight --cell-index "$cell" --pool "$calibration_pool" --run-root "$group/t0/cell-$cell" --commit "$commit" --scope PREFLIGHT_ONLY "${extra[@]}"
  prefix=q4; if [[ "$model_index" == 1 ]]; then prefix=q17; fi
  saved=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase8-gate-c-20260905T100000Z-formal-$prefix-a1/cell-$(( wi * 2 ))/residuals.pt
  "$python_runtime" scripts/loopscope/run_phase8_gate_f_calibration.py --mode t0-from-saved --cell-index "$cell" --pool "$calibration_pool" --run-root "$group/saved-fit/cell-$cell" --saved-residuals "$saved" --commit "$commit" --scope PREFLIGHT_ONLY
done
"$python_runtime" scripts/loopscope/build_phase8_gate_f_accuracy_manifest.py --pool "$test_pool" --t0-basis-root "$group/t0" --t1-basis-root "$group/t1" --scope PREFLIGHT_ONLY --output "$group/panel.json"
mapfile -t cells < <("$python_runtime" -c 'import json,sys; print("\n".join(c["cell_id"] for c in json.load(open(sys.argv[1]))["cells"] if c["model_index"]==int(sys.argv[2])))' "$group/panel.json" "$model_index")
bash scripts/loopscope/phase8_gate_f_accuracy.sbatch "$repo" "$group/panel.json" "$test_pool" "$group/scores" "$commit" PREFLIGHT_ONLY 1 0 14042 "${cells[@]}"
