#!/usr/bin/env bash
#SBATCH --job-name=looppilot-gate-c
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_looppilot
python_bin=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction/.venv/bin/python
workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot

: "${RUN_ROOT:?RUN_ROOT must be exported by submit_gate_c_probe.sh}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must be exported by submit_gate_c_probe.sh}"
case "$RUN_ROOT" in
  "$workspace"/gates/gate_c/*_*) ;;
  *) echo "invalid Gate C run root: $RUN_ROOT" >&2; exit 2 ;;
esac

cd "$repo"
test "$(git rev-parse --show-toplevel)" = "$repo"
test "$(git rev-parse --abbrev-ref HEAD)" = looppilot
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --short)"
git merge-base --is-ancestor 4f59bd93eca4da3cbf458a93508f91c5b23912bc HEAD
test -x "$python_bin"
test -f "$RUN_ROOT/control/preflight/preflight_manifest.json"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src"
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export UV_CACHE_DIR=/hpc2hdd/home/xhuang225/shared/uv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

hostname > "$RUN_ROOT/control/compute_hostname.txt"
"$python_bin" -V > "$RUN_ROOT/control/job_python_version.txt" 2>&1
nvidia-smi > "$RUN_ROOT/control/nvidia_smi_before.txt"
(cd "$RUN_ROOT/control" && sha256sum -c preflight_sha256.txt)

"$python_bin" scripts/looppilot/run_gate_c_probe.py \
  --config configs/looppilot/gate_c_probe.json \
  --preflight-manifest "$RUN_ROOT/control/preflight/preflight_manifest.json" \
  --output-dir "$RUN_ROOT/results"

nvidia-smi > "$RUN_ROOT/control/nvidia_smi_after.txt"
find "$RUN_ROOT/results" -type f -print | LC_ALL=C sort | while IFS= read -r path; do
  sha256sum "$path"
done > "$RUN_ROOT/control/job_results_sha256.txt"
(set -C; printf '%s\n' "complete" > "$RUN_ROOT/control/job_complete.txt")
