# hpc2-hkustgz Remote Workflow

Local repository policy:

- Keep source code, configs, scripts, and docs locally.
- Do not keep model weights, datasets, full eval logs, or result directories on
  this Mac.
- Use `tflt collect` only for small summaries or selected artifacts.

## Remote Layout

Recommended source checkout:

```bash
$SCRATCH/training_free_looped_transformer
```

Recommended hpc2 layout:

```bash
export TFLT_HPC2_CACHE_ROOT=/hpc2hdd/home/xhuang225/shared
export HF_HOME="$TFLT_HPC2_CACHE_ROOT/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$TFLT_HPC2_CACHE_ROOT/datasets"
export UV_CACHE_DIR="$TFLT_HPC2_CACHE_ROOT/uv"
export RESULT_ROOT=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs
```

## Environment

On the remote side:

```bash
module load anaconda3 cuda/12.4 uv
bash scripts/env/hpc2_bootstrap_env.sh --venv .venv-test --recreate
```

The frozen hpc2 spec is in `envs/hpc2/`: Python 3.11.9, torch 2.3.1+cu121,
transformers 4.51.3, and lm_eval 0.4.11. Build a fresh versioned venv and
validate it before switching the active `.venv` symlink. Do not upgrade an
active environment in place while jobs may be using it.

## Preflight

Run before submitting real evals:

```bash
bash scripts/tflt-hpc2-preflight.sh
```

Check:

- GPU is visible through `nvidia-smi`
- Python imports `torch`, `transformers`, and `lm_eval`
- `HF_HOME`, `TRANSFORMERS_CACHE`, and `RESULT_ROOT` point to remote storage
- Enough disk space is available

## Slurm Dry Run

```bash
tflt launch --backend slurm \
  --profile configs/profiles/hpc2-hkustgz.toml \
  --model qwen3-1.7b-base \
  --tasks sciq,mmlu \
  --limit 10 \
  --loop --window 12:15 --k 2 --strategy damped_euler \
  --dry-run
```

Inspect `<RESULT_ROOT>/<timestamp>/control/job.sbatch` before submitting.

## Plain SSH/Tmux Dry Run

```bash
tflt launch --backend ssh \
  --model qwen3-1.7b-base \
  --tasks sciq \
  --limit 10 \
  --loop \
  --dry-run
```

This writes a `run_ssh.sh` control script that starts a tmux session when run on
the remote host.
