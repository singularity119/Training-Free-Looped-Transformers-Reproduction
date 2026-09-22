#!/usr/bin/env bash
# Phase 9 full-score path preflight on the debug partition.
# Usage: phase9_accuracy_debug.sbatch REPO MANIFEST POOL GROUP COMMIT SCOPE PACKING SELECTION CANARY_INDEX_FILE CELL_ID...
# The preflight must use SCOPE=PREFLIGHT_ONLY, SELECTION=preflight, and '-'.
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --account=root
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:29:00
set -euo pipefail

if [[ $# -lt 10 ]]; then
  echo "usage: $0 REPO MANIFEST POOL GROUP COMMIT SCOPE PACKING SELECTION CANARY_INDEX_FILE CELL_ID..." >&2
  exit 2
fi
repo=$1
exec bash "$repo/scripts/loopscope/phase9_accuracy.sbatch" "$@"
