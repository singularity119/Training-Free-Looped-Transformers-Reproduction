#!/usr/bin/env bash
# Submit only the Phase 9 debug cells that exist in the frozen model config.
# Usage: submit_phase9_debug_jobs.sh MODEL_INDEX POOL RUN_ROOT_PREFIX SOURCE_COMMIT [CELL_INDEX ...]
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_runtime="${repo}/.venv-loopscope-cu121-20260711/bin/python"
config="${repo}/configs/loopscope/phase9_model_windows.json"

if [[ $# -lt 4 ]]; then
  echo "usage: $0 MODEL_INDEX POOL RUN_ROOT_PREFIX SOURCE_COMMIT [CELL_INDEX ...]" >&2
  exit 2
fi

model_index=$1
pool=$2
run_root_prefix=$3
source_commit=$4
shift 4

cell_indexes_text=$("$python_runtime" - "$config" "$model_index" "$@" <<'PY'
import json
import sys

config_path, model_index, *requested = sys.argv[1:]
with open(config_path, encoding="utf-8") as stream:
    config = json.load(stream)
models = config.get("models", ())
index = int(model_index)
if not 0 <= index < len(models):
    raise SystemExit("model index is outside the frozen Phase 9 model manifest")
cell_count = len(models[index].get("windows", ())) * len(config.get("K", ()))
selected = list(range(cell_count)) if not requested else [int(value) for value in requested]
if any(value < 0 or value >= cell_count for value in selected):
    raise SystemExit("requested cell index is outside the frozen model cell manifest")
if len(set(selected)) != len(selected):
    raise SystemExit("duplicate cell index")
for value in selected:
    print(value)
PY
)
mapfile -t cell_indexes <<<"$cell_indexes_text"

log_dir="${run_root_prefix}-logs"
mkdir -p "$log_dir"
for cell_index in "${cell_indexes[@]}"; do
  run_root="${run_root_prefix}-m${model_index}-c${cell_index}"
  sbatch \
    --job-name="p9b-m${model_index}-c${cell_index}" \
    --output="${log_dir}/m${model_index}-c${cell_index}-%j.out" \
    --error="${log_dir}/m${model_index}-c${cell_index}-%j.err" \
    "${repo}/scripts/loopscope/phase9_debug.sbatch" \
    "$model_index" "$pool" "$run_root" "$source_commit" "$cell_index"
done
