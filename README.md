# Training-Free Looped Transformers

Local codebase for reproducing the paper "Training-Free Looped Transformers".

This repository is intentionally code-only. Model weights, datasets, dependency
environments, eval logs, and result tables should live on the lab server or
`hpc2-hkustgz`, not on this Mac.

## Scope

- Package: `tflt`
- CLI: `tflt inspect-model`, `tflt smoke`, `tflt eval`, `tflt launch`,
  `tflt collect`, `tflt report`
- Default first-stage models: Qwen3 dense checkpoints
- Remote backends: Slurm and plain SSH
- Local tests: pure Python, no torch or transformers required

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Remote eval environments should install the eval extras:

```bash
python -m pip install -e ".[eval]"
```

## Dry Runs

Generate the exact command for a smoke eval without downloading anything:

```bash
tflt eval --model qwen3-1.7b-base --tasks sciq,mmlu --limit 10 --dry-run
```

Generate a Slurm control script:

```bash
tflt launch --backend slurm --profile configs/profiles/hpc2-hkustgz.toml \
  --model qwen3-1.7b-base --tasks mmlu --limit 10 --dry-run
```

See [docs/remote_hpc2.md](docs/remote_hpc2.md) for the remote workflow.
