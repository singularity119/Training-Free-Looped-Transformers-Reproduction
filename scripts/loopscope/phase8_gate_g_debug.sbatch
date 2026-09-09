#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
set -euo pipefail
repo=$1; calibration_pool=$2; test_pool=$3; group=$4; commit=$5
python_runtime=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
cd "$repo"
export PYTHONPATH="$repo/src" PYTHONDONTWRITEBYTECODE=1
[[ "$SLURM_JOB_PARTITION" == debug ]]
mkdir "$group"
bash scripts/loopscope/phase8_gate_g_calibration.sbatch "$repo" "$calibration_pool" "$group/calibration" "$commit" PREFLIGHT_ONLY
"$python_runtime" scripts/loopscope/build_phase8_gate_g_accuracy_manifest.py --pool "$test_pool" --basis-root "$group/calibration" --scope PREFLIGHT_ONLY --output "$group/panel.json"
mapfile -t cells < <("$python_runtime" -c 'import json,sys; print("\n".join(c["cell_id"] for c in json.load(open(sys.argv[1]))["cells"]))' "$group/panel.json")
bash scripts/loopscope/phase8_gate_g_accuracy.sbatch "$repo" "$group/panel.json" "$test_pool" "$group/scores" "$commit" PREFLIGHT_ONLY 1 0 14042 "${cells[@]}"
