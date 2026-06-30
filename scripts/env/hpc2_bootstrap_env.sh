#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: hpc2_bootstrap_env.sh [options]

Create or rebuild a project-local TFLT hpc2 virtualenv from envs/hpc2 lock.

Options:
  --venv PATH              Virtualenv path relative to repo root or absolute.
                           Default: .venv
  --python PATH            Python interpreter from the hpc2 base module.
                           Default: python
  --lock PATH              Requirements lock file.
                           Default: envs/hpc2/requirements-eval.lock.txt
  --recreate               Remove and recreate the target venv. Refuses the
                           main .venv unless ALLOW_RECREATE_MAIN_VENV=1.
  --reuse                  Reuse an existing target venv and sync locked deps.
  --offline-wheelhouse     Install only from TFLT_WHEELHOUSE/--wheelhouse.
  --wheelhouse PATH        Wheelhouse path for offline install.
  --skip-project-install   Do not install this checkout editable with --no-deps.
  -h, --help               Show this help.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

venv_path="$repo_root/.venv"
python_bin="${PYTHON_BIN:-python}"
lock_file="$repo_root/envs/hpc2/requirements-eval.lock.txt"
cache_root="${TFLT_HPC2_CACHE_ROOT:-/hpc2hdd/home/xhuang225/shared}"
wheelhouse="${TFLT_WHEELHOUSE:-$cache_root/wheelhouse/tflt-cu121}"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
uv_index_strategy="${UV_INDEX_STRATEGY:-unsafe-best-match}"
report_file="${TFLT_ENV_REPORT:-$repo_root/envs/hpc2/env_report.txt}"
recreate=0
reuse=0
offline=0
install_project=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      venv_path="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
      shift 2
      ;;
    --lock)
      lock_file="$2"
      shift 2
      ;;
    --recreate)
      recreate=1
      shift
      ;;
    --reuse)
      reuse=1
      shift
      ;;
    --offline-wheelhouse)
      offline=1
      shift
      ;;
    --wheelhouse)
      wheelhouse="$2"
      shift 2
      ;;
    --skip-project-install)
      install_project=0
      shift
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
case "$lock_file" in
  /*) ;;
  *) lock_file="$repo_root/$lock_file" ;;
esac
case "$wheelhouse" in
  /*) ;;
  *) wheelhouse="$repo_root/$wheelhouse" ;;
esac

if [[ ! -f "$lock_file" ]]; then
  echo "Lock file not found: $lock_file" >&2
  exit 2
fi
if [[ "$recreate" -eq 1 && "$reuse" -eq 1 ]]; then
  echo "--recreate and --reuse are mutually exclusive" >&2
  exit 2
fi

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +e
  source /etc/profile.d/modules.sh >/dev/null 2>&1
  set -e
fi
if [[ "${TFLT_SKIP_MODULE_LOAD:-0}" != "1" ]] && command -v module >/dev/null 2>&1; then
  module load anaconda3 cuda/12.4 uv
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. On hpc2 run: module load anaconda3 cuda/12.4 uv" >&2
  exit 2
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$cache_root/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$cache_root/datasets}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache_root/uv}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR"

if [[ -e "$venv_path" ]]; then
  if [[ "$recreate" -eq 1 ]]; then
    if [[ "$venv_path" == "$repo_root/.venv" && "${ALLOW_RECREATE_MAIN_VENV:-0}" != "1" ]]; then
      echo "Refusing to recreate the active .venv without ALLOW_RECREATE_MAIN_VENV=1." >&2
      echo "Use --venv .venv-test or a versioned path for rebuild validation." >&2
      exit 2
    fi
    rm -rf "$venv_path"
  elif [[ "$reuse" -ne 1 ]]; then
    echo "Target venv already exists: $venv_path" >&2
    echo "Use --reuse, --recreate, or choose a fresh --venv path." >&2
    exit 2
  fi
fi

if [[ ! -x "$venv_path/bin/python" ]]; then
  uv venv --python "$python_bin" "$venv_path"
fi

venv_python="$venv_path/bin/python"
export VIRTUAL_ENV="$venv_path"
export PATH="$VIRTUAL_ENV/bin:$PATH"
install_args=(--python "$venv_python")
if [[ "$offline" -eq 1 ]]; then
  if [[ ! -d "$wheelhouse" ]]; then
    echo "Wheelhouse not found: $wheelhouse" >&2
    exit 2
  fi
  install_args+=(--no-index --find-links "$wheelhouse")
else
  install_args+=(--extra-index-url "$torch_index_url" --index-strategy "$uv_index_strategy")
fi

uv pip install "${install_args[@]}" -r "$lock_file"
if [[ "$install_project" -eq 1 ]]; then
  uv pip install --python "$venv_python" --no-deps -e "$repo_root"
fi

mkdir -p "$(dirname "$report_file")"
{
  echo "# TFLT hpc2 env report"
  printf 'timestamp_utc='
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf 'host='
  hostname
  echo "repo=$repo_root"
  echo "venv=$venv_path"
  echo "lock=$lock_file"
  if command -v sha256sum >/dev/null 2>&1; then
    echo "lock_sha256=$(sha256sum "$lock_file" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    echo "lock_sha256=$(shasum -a 256 "$lock_file" | awk '{print $1}')"
  fi
  echo
  echo "== tools =="
  uv --version
  "$venv_python" -V
  echo
  echo "== core imports =="
  "$venv_python" - <<'PY'
import importlib, json, os, platform, sys

mods = {}
for name in ["torch", "transformers", "lm_eval", "accelerate", "datasets", "numpy"]:
    mod = importlib.import_module(name)
    mods[name] = getattr(mod, "__version__", "unknown")

import torch

payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "executable": sys.executable,
    "mods": mods,
    "torch_cuda": getattr(torch.version, "cuda", None),
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_device_count": torch.cuda.device_count(),
    "env": {k: os.environ.get(k) for k in [
        "HF_ENDPOINT",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HUB_DISABLE_XET",
        "UV_CACHE_DIR",
        "VIRTUAL_ENV",
    ]},
}
print(json.dumps(payload, indent=2, sort_keys=True))

expected = {"torch": "2.3.1", "transformers": "4.51.3", "lm_eval": "0.4.11"}
for name, expected_prefix in expected.items():
    actual = mods[name]
    if not actual.startswith(expected_prefix):
        raise SystemExit("%s version mismatch: expected %s, got %s" % (name, expected_prefix, actual))
PY
  echo
  echo "== nvidia-smi =="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found"
  fi
} > "$report_file"

echo "Wrote $report_file"
