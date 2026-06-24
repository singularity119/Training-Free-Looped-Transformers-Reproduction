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

Recommended cache roots:

```bash
export HF_HOME="$SCRATCH/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export RESULT_ROOT="$SCRATCH/training_free_looped_transformer/runs"
```

## Environment

On the remote side:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[eval]"
```

The reproduction plan pins `lm-eval==0.4.11` and uses
`transformers>=4.51,<4.52` for Qwen3 compatibility. This is an engineering
reproduction setting and must be reported separately from any bit-exact paper
claim.

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

Inspect `runs/<timestamp>/control/job.sbatch` on the remote checkout before
submitting.

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
