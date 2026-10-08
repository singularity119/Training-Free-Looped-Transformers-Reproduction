#!/usr/bin/env bash
# Gate B synthetic closure followed by representative packing-one resources.
# Submit only after checker completion. All cases remain PREFLIGHT_ONLY.
#SBATCH --partition=debug
#SBATCH --time=00:29:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == debug ]]
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
source_dir=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase11-gate-b-20261008T101100Z-ffb1e38/cuda-source
python_runtime=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
input_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-a-20261008T094044Z
run_prefix=${1:?fresh Gate B run prefix required}
[[ "$run_prefix" == /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase11-gate-b-* ]]
export PYTHONPATH="$source_dir/src" PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
cd "$source_dir"
for dataset in arc_challenge mmlu_pro gpqa_main; do
  manifest="$input_root/manifests/$dataset-synthetic.json"
  pool="$input_root/synthetic/$dataset-pool.json"
  /bin/bash scripts/loopscope/phase11_preflight.sbatch "$source_dir" "$python_runtime" \
    run_phase11_accuracy.py --dataset "$dataset" --manifest "$manifest" --pool "$pool" \
    --cell-id "$dataset-q4-w15-18-k3-online-current-t-lambda0.9" \
    --run-root "$run_prefix-$dataset-attempt" --commit ffb1e3824e85484e346b1090b7b5f6f572f6b4f3
  "$python_runtime" scripts/loopscope/verify_phase11_scores.py \
    --manifest "$manifest" --pool "$pool" --scope PREFLIGHT_ONLY \
    --attempt "$run_prefix-$dataset-attempt" --output-dir "$run_prefix-$dataset-verification"
done
for family in Native LoopK3 OnlineK3; do
  /bin/bash scripts/loopscope/phase11_resources.sbatch "$source_dir" "$python_runtime" \
    --dataset mmlu_pro --family "$family" \
    --run-root "$run_prefix-resource-mmlu_pro-$family" \
    --commit ffb1e3824e85484e346b1090b7b5f6f572f6b4f3
done
/bin/bash scripts/loopscope/phase11_resources.sbatch "$source_dir" "$python_runtime" \
  --dataset arc_challenge --family OnlineK3 \
  --run-root "$run_prefix-resource-arc_challenge-OnlineK3" \
  --commit ffb1e3824e85484e346b1090b7b5f6f572f6b4f3
