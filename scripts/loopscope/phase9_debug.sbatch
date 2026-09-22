#!/usr/bin/env bash
set -euo pipefail

# Gate B owns actual model/debug execution.  Gate A only checks that this
# self-contained entry validates a write-once Phase 9 logical manifest.
if [[ $# -ne 1 ]]; then
  echo "usage: $0 MANIFEST" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 "${SCRIPT_DIR}/run_phase9_debug.py" \
  --manifest "$1" \
  --mode manifest
