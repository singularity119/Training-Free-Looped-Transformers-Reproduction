#!/usr/bin/env bash
set -euo pipefail

echo "== host =="
hostname
date

echo "== python =="
python -V
python - <<'PY'
import importlib
for name in ["torch", "transformers", "lm_eval"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "unknown"))
PY

echo "== gpu =="
nvidia-smi

echo "== paths =="
echo "HF_HOME=${HF_HOME:-unset}"
echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-unset}"
echo "RESULT_ROOT=${RESULT_ROOT:-unset}"
df -h .
