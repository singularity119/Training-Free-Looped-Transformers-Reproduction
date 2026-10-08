#!/usr/bin/env bash
# Two independent batch-one processes on the same allocated A40, then ARC pair.
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == debug ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
source_dir=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-b-20261008T101100Z-ffb1e38/cuda-source
python_runtime=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
run_prefix=${1:?fresh Gate B run prefix required}
[[ "$run_prefix" == /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase11-gate-b-* ]]
log_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/logs/phase11-gate-b-20261008T100500Z
export PYTHONPATH="$source_dir/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PHASE11_PACKING=2
cd "$source_dir"
for dataset in mmlu_pro arc_challenge; do
  started=$(/usr/bin/date +%s)
  "$python_runtime" scripts/loopscope/measure_phase11_resources.py \
    --dataset "$dataset" --family OnlineK3 --run-root "$run_prefix-$dataset-process0" \
    --commit ffb1e3824e85484e346b1090b7b5f6f572f6b4f3 \
    >"$log_root/pack2-$SLURM_JOB_ID-$dataset-process0.out" \
    2>"$log_root/pack2-$SLURM_JOB_ID-$dataset-process0.err" &
  first_pid=$!
  "$python_runtime" scripts/loopscope/measure_phase11_resources.py \
    --dataset "$dataset" --family OnlineK3 --run-root "$run_prefix-$dataset-process1" \
    --commit ffb1e3824e85484e346b1090b7b5f6f572f6b4f3 \
    >"$log_root/pack2-$SLURM_JOB_ID-$dataset-process1.out" \
    2>"$log_root/pack2-$SLURM_JOB_ID-$dataset-process1.err" &
  second_pid=$!
  if wait "$first_pid"; then first_exit=0; else first_exit=$?; fi
  if wait "$second_pid"; then second_exit=0; else second_exit=$?; fi
  finished=$(/usr/bin/date +%s)
  "$python_runtime" - "$run_prefix" "$dataset" "$started" "$finished" "$first_exit" "$second_exit" "$first_pid" "$second_pid" <<'PY'
import json, os, sys
from pathlib import Path
prefix, dataset, started, finished, first_exit, second_exit, first_pid, second_pid = sys.argv[1:]
children = []
for index in range(2):
    root = Path(prefix + "-" + dataset + "-process" + str(index))
    if (root / "summary.json").exists():
        s = json.loads((root / "summary.json").read_text())
        children.append({"run_root": str(root), "status": s["status"], "stages": s["stages"],
                         "evidence": s["evidence"], "elapsed_seconds": s["elapsed_seconds"], "oom": s["oom"]})
    else:
        failure = json.loads((root / "failure.json").read_text()) if (root / "failure.json").exists() else None
        children.append({"run_root": str(root), "status": "INCOMPLETE", "failure": failure})
complete = first_exit == second_exit == "0" and all(c["status"] == "RESOURCE_COMPLETE" for c in children)
wall = int(finished) - int(started)
result = {"status": "PACKING2_COMPLETE" if complete else "PACKING2_FAILED", "scope": "PREFLIGHT_ONLY",
          "dataset": dataset, "packing": 2, "batch_size_per_process": 1, "model_tensor_sharing": False,
          "job_id": os.environ["SLURM_JOB_ID"], "allocated_gpu_count": 1,
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "threads_per_process": 4,
          "pids": [first_pid, second_pid], "child_exit_codes": [int(first_exit), int(second_exit)],
          "pair_wall_seconds_including_startup": wall, "children": children, "target_gold_loaded": False}
if complete:
    result["aggregate_question_tokens_per_second"] = sum(c["evidence"].get("generated_token_count", c["evidence"].get("scored_continuation_tokens", 0)) for c in children) / max(c["evidence"]["question_seconds"] for c in children)
    result["pair_tokens_per_second_including_startup"] = sum(c["evidence"].get("generated_token_count", c["evidence"].get("scored_continuation_tokens", 0)) for c in children) / wall
with Path(prefix + "-" + dataset + "-pair.json").open("x") as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write("\n")
print(json.dumps({k: v for k, v in result.items() if k != "children"}), flush=True)
if not complete:
    raise SystemExit(1)
PY
done
