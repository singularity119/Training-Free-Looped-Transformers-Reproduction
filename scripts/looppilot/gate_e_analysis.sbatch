#!/usr/bin/env bash
#SBATCH --job-name=looppilot-gate-e-analysis
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_looppilot
python_bin=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction/.venv/bin/python
: "${RUN_ROOT:?}"
: "${EXPECTED_COMMIT:?}"
cd "$repo"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/looppilot)" = "$EXPECTED_COMMIT"
test -z "$(git status --short)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo/src"
test ! -e "$RUN_ROOT/analysis"
"$python_bin" scripts/looppilot/verify_gate_e_shards.py --run-root "$RUN_ROOT" --output "$RUN_ROOT/receipts/shards_verified.json"
"$python_bin" scripts/looppilot/analyze_gate_e.py --config configs/looppilot/gate_e_full.json --run-root "$RUN_ROOT" --output-dir "$RUN_ROOT/analysis"
"$python_bin" scripts/looppilot/verify_gate_e_final.py --analysis-dir "$RUN_ROOT/analysis"
