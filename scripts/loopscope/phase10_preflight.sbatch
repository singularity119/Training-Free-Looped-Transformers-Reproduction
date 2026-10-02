#!/usr/bin/env bash
# Gate B supplies the authorized debug partition/QoS and
# measured memory/GRES. The preflight is always less than thirty minutes.
# Usage: same ten arguments as phase10_accuracy.sbatch; scope PREFLIGHT_ONLY.
#SBATCH --time=00:29:00
set -euo pipefail
[[ $# -eq 10 ]]
[[ "$9" == PREFLIGHT_ONLY ]]
repo=$1
exec bash "$repo/scripts/loopscope/phase10_accuracy.sbatch" "$@"
