#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot/gates/gate_c/<UTC>_<commit8>" >&2
  exit 2
fi

repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_looppilot
python_bin=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction/.venv/bin/python
workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot
run_root=$1

cd "$repo"
test "$(git rev-parse --show-toplevel)" = "$repo"
test "$(git rev-parse --abbrev-ref HEAD)" = looppilot
test -z "$(git status --short)"
head_commit=$(git rev-parse HEAD)
test "$(git rev-parse origin/looppilot)" = "$head_commit"
git merge-base --is-ancestor 4f59bd93eca4da3cbf458a93508f91c5b23912bc HEAD
case "$run_root" in
  "$workspace"/gates/gate_c/*_"${head_commit:0:8}") ;;
  *) echo "run root must be a fresh Gate C UTC timestamp suffixed by ${head_commit:0:8}" >&2; exit 2 ;;
esac
test ! -e "$run_root"

mkdir -p "$workspace/gates/gate_c"
mkdir "$run_root"
mkdir "$run_root/control"
cp scripts/looppilot/gate_c_probe.sbatch "$run_root/control/job.sbatch"
cp configs/looppilot/gate_c_probe.json "$run_root/control/probe_config.json"
cp configs/looppilot/gate_c_action_schema.json "$run_root/control/action_schema.json"
cp configs/looppilot/gate_c_question_mask_schema.json "$run_root/control/question_mask_schema.json"

{
  git status --short --branch
  git rev-parse --show-toplevel
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  git rev-parse origin/looppilot
  git merge-base --is-ancestor 4f59bd93eca4da3cbf458a93508f91c5b23912bc HEAD
} > "$run_root/control/git_provenance.txt"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo/src"
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export UV_CACHE_DIR=/hpc2hdd/home/xhuang225/shared/uv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

env | LC_ALL=C sort | grep -E '^(HF_|TRANSFORMERS_|UV_CACHE_DIR|PYTHONPATH|PYTHONDONTWRITEBYTECODE)=' \
  > "$run_root/control/environment_whitelist.txt"
"$python_bin" scripts/looppilot/prepare_gate_c_probe.py \
  --config configs/looppilot/gate_c_probe.json \
  --output-dir "$run_root/control/preflight"

cd "$run_root/control"
sha256sum preflight/preflight_manifest.json probe_config.json action_schema.json question_mask_schema.json \
  > preflight_sha256.txt
bash -n job.sbatch
bash -n "$repo/scripts/looppilot/submit_gate_c_probe.sh"
printf '%q ' "$python_bin" "$repo/scripts/looppilot/run_gate_c_probe.py" \
  --config "$repo/configs/looppilot/gate_c_probe.json" \
  --preflight-manifest "$run_root/control/preflight/preflight_manifest.json" \
  --output-dir "$run_root/results" > command_argv_shell.txt
printf '\n' >> command_argv_shell.txt

job_id=$(sbatch --parsable \
  --output="$run_root/control/slurm-%j.out" \
  --error="$run_root/control/slurm-%j.err" \
  --export=ALL,RUN_ROOT="$run_root",EXPECTED_COMMIT="$head_commit" \
  "$run_root/control/job.sbatch")
(set -C; printf '%s\n' "$job_id" > "$run_root/control/job_id.txt")
printf '%s\n' "$job_id"
