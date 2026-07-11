#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot/gates/gate_d/<UTC>_<commit8>" >&2
  exit 2
fi

repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_looppilot
python_bin=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction/.venv/bin/python
source_venv=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction/.venv
workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot
gate_c_run="$workspace/gates/gate_c/20260711T125126Z_98ed1f06"
run_root=$1

cd "$repo"
test "$(git rev-parse --show-toplevel)" = "$repo"
test "$(git rev-parse --abbrev-ref HEAD)" = looppilot
test -z "$(git status --short)"
head_commit=$(git rev-parse HEAD)
test "$(git rev-parse origin/looppilot)" = "$head_commit"
git merge-base --is-ancestor 4f59bd93eca4da3cbf458a93508f91c5b23912bc HEAD
git merge-base --is-ancestor 98ed1f06545c8b2d04145bd232cec12a991aac7e HEAD
case "$run_root" in
  "$workspace"/gates/gate_d/*_"${head_commit:0:8}") ;;
  *) echo "run root must be a fresh Gate D UTC timestamp suffixed by ${head_commit:0:8}" >&2; exit 2 ;;
esac
test ! -e "$run_root"
test -x "$python_bin"
test -f "$gate_c_run/control/job_results_sha256.txt"
test "$(sha256sum "$gate_c_run/control/job_results_sha256.txt" | cut -d' ' -f1)" = \
  deb16bc4f0f8968a735485e3db33eb96dc798f136f39c3722f57dec6e86edede
(cd "$gate_c_run/control" && sha256sum -c preflight_sha256.txt)
(cd "$gate_c_run/control" && sha256sum -c job_results_sha256.txt)

mkdir -p "$workspace/gates/gate_d"
mkdir "$run_root"
mkdir "$run_root/control"
cp scripts/looppilot/gate_d_limit.sbatch "$run_root/control/job.sbatch"
cp configs/looppilot/gate_d_limit.json "$run_root/control/limit_config.json"
cp configs/looppilot/gate_d_join_schema.json "$run_root/control/join_schema.json"
cp configs/looppilot/gate_c_question_mask_schema.json "$run_root/control/question_mask_schema.json"
cp configs/looppilot/criterion_v0.json "$run_root/control/criterion_v0.json"

{
  git status --short --branch
  git rev-parse --show-toplevel
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  git rev-parse origin/looppilot
  git merge-base --is-ancestor 4f59bd93eca4da3cbf458a93508f91c5b23912bc HEAD
  git merge-base --is-ancestor 98ed1f06545c8b2d04145bd232cec12a991aac7e HEAD
  git log --format='%H %s' 844c148bda521d29e48a3c6734a918494d02ef0f..HEAD
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
find "$source_venv" -type f -printf '%P %s %T@\n' | LC_ALL=C sort | sha256sum \
  > "$run_root/control/source_venv_before.sha256"
"$python_bin" scripts/looppilot/prepare_gate_d_limit.py \
  --config configs/looppilot/gate_d_limit.json \
  --join-schema configs/looppilot/gate_d_join_schema.json \
  --mask-schema configs/looppilot/gate_c_question_mask_schema.json \
  --criterion configs/looppilot/criterion_v0.json \
  --gate-c-run "$gate_c_run" \
  --output-dir "$run_root/control/preflight"

cd "$run_root/control"
sha256sum \
  preflight/preflight_manifest.json \
  limit_config.json \
  join_schema.json \
  question_mask_schema.json \
  criterion_v0.json \
  job.sbatch \
  > preflight_sha256.txt
bash -n job.sbatch
bash -n "$repo/scripts/looppilot/submit_gate_d_limit.sh"
printf '%q ' "$python_bin" "$repo/scripts/looppilot/run_gate_d_limit.py" \
  --config "$repo/configs/looppilot/gate_d_limit.json" \
  --preflight-manifest "$run_root/control/preflight/preflight_manifest.json" \
  --output-dir "$run_root/results" > run_argv_shell.txt
printf '\n' >> run_argv_shell.txt
printf '%q ' "$python_bin" "$repo/scripts/looppilot/verify_gate_d_limit.py" \
  --config "$repo/configs/looppilot/gate_d_limit.json" \
  --preflight-manifest "$run_root/control/preflight/preflight_manifest.json" \
  --results-dir "$run_root/results" > verify_argv_shell.txt
printf '\n' >> verify_argv_shell.txt
cp "$gate_c_run/control/job_results_sha256.txt" gate_c_accepted_results_sha256.txt
sha256sum gate_c_accepted_results_sha256.txt > gate_c_linkage_sha256.txt

job_id=$(sbatch --parsable \
  --output="$run_root/control/slurm-%j.out" \
  --error="$run_root/control/slurm-%j.err" \
  --export=ALL,RUN_ROOT="$run_root",EXPECTED_COMMIT="$head_commit" \
  "$run_root/control/job.sbatch")
(set -C; printf '%s\n' "$job_id" > "$run_root/control/job_id.txt")
printf '%s\n' "$job_id"
