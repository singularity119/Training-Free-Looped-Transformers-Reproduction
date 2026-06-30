#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: hpc2_freeze_env.sh [options]

Freeze an existing TFLT hpc2 virtualenv into a timestamped report directory.

Options:
  --venv PATH       Virtualenv path relative to repo root or absolute. Default: .venv
  --out-dir PATH    Output directory. Default: envs/hpc2/freeze/<timestamp>
  -h, --help        Show this help.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
venv_path="$repo_root/.venv"
timestamp="$(date -u '+%Y%m%dT%H%M%SZ')"
out_dir="$repo_root/envs/hpc2/freeze/$timestamp"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      venv_path="$2"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$venv_path" in
  /*) ;;
  *) venv_path="$repo_root/$venv_path" ;;
esac
case "$out_dir" in
  /*) ;;
  *) out_dir="$repo_root/$out_dir" ;;
esac

venv_python="$venv_path/bin/python"
if [[ ! -x "$venv_python" ]]; then
  echo "Python not found in venv: $venv_python" >&2
  exit 2
fi
export VIRTUAL_ENV="$venv_path"
export PATH="$VIRTUAL_ENV/bin:$PATH"

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +e
  source /etc/profile.d/modules.sh >/dev/null 2>&1
  set -e
fi
if [[ "${TFLT_SKIP_MODULE_LOAD:-0}" != "1" ]] && command -v module >/dev/null 2>&1; then
  module load anaconda3 cuda/12.4 uv
fi

mkdir -p "$out_dir"

if command -v uv >/dev/null 2>&1; then
  uv pip freeze --python "$venv_python" | sed '/^-e file:/d' > "$out_dir/requirements-eval.freeze.txt"
else
  "$venv_python" -m pip freeze --all | sed '/^-e file:/d' > "$out_dir/requirements-eval.freeze.txt"
fi

"$venv_python" - <<'PY' > "$out_dir/core_versions.json"
import importlib, json, os, platform, sys

mods = {}
for name in ["torch", "transformers", "lm_eval", "accelerate", "tokenizers", "datasets", "evaluate", "huggingface_hub", "safetensors", "numpy"]:
    try:
        mod = importlib.import_module(name)
        mods[name] = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        mods[name] = "%s: %s" % (type(exc).__name__, exc)

try:
    import torch
    cuda = {
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
except Exception as exc:
    cuda = {"error": "%s: %s" % (type(exc).__name__, exc)}

print(json.dumps({
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "mods": mods,
    "cuda": cuda,
    "env": {k: os.environ.get(k) for k in [
        "HF_ENDPOINT",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HUB_DISABLE_XET",
        "UV_CACHE_DIR",
        "VIRTUAL_ENV",
    ]},
}, indent=2, sort_keys=True))
PY

{
  echo "# TFLT hpc2 freeze report"
  printf 'timestamp_utc='
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf 'host='
  hostname
  echo "repo=$repo_root"
  echo "venv=$venv_path"
  echo "freeze=$out_dir/requirements-eval.freeze.txt"
  echo "core_versions=$out_dir/core_versions.json"
  echo
  "$venv_python" -V
  if command -v uv >/dev/null 2>&1; then
    uv --version
  fi
} > "$out_dir/env_report.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$out_dir/nvidia_smi.txt" 2>&1 || true
fi

echo "Wrote $out_dir"
