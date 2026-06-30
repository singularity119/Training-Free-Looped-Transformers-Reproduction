#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: hpc2_download_wheelhouse.sh [options]

Download wheels for the hpc2 eval lock into the shared TFLT wheelhouse.

Options:
  --lock PATH          Requirements lock file.
                      Default: envs/hpc2/requirements-eval.lock.txt
  --wheelhouse PATH    Destination wheelhouse.
                      Default: /hpc2hdd/home/xhuang225/shared/wheelhouse/tflt-cu121
  -h, --help          Show this help.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cache_root="${TFLT_HPC2_CACHE_ROOT:-/hpc2hdd/home/xhuang225/shared}"
lock_file="$repo_root/envs/hpc2/requirements-eval.lock.txt"
wheelhouse="${TFLT_WHEELHOUSE:-$cache_root/wheelhouse/tflt-cu121}"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
uv_index_strategy="${UV_INDEX_STRATEGY:-unsafe-best-match}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lock)
      lock_file="$2"
      shift 2
      ;;
    --wheelhouse)
      wheelhouse="$2"
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

case "$lock_file" in
  /*) ;;
  *) lock_file="$repo_root/$lock_file" ;;
esac
case "$wheelhouse" in
  /*) ;;
  *) wheelhouse="$repo_root/$wheelhouse" ;;
esac

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

export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache_root/uv}"
mkdir -p "$UV_CACHE_DIR" "$wheelhouse"
uv pip download \
  -r "$lock_file" \
  --dest "$wheelhouse" \
  --extra-index-url "$torch_index_url" \
  --index-strategy "$uv_index_strategy"
echo "Wrote wheelhouse: $wheelhouse"
