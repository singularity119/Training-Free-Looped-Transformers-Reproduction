# Training-Free Looped Transformers

LoopPilot 第一阶段的冻结配置、信号 schema 与本地 Gate A 契约见
[`docs/looppilot_phase1.md`](docs/looppilot_phase1.md)。该增量默认关闭；未提供 controller 时
原有 wrapper/strategy 热路径保持不变。

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

Remote eval environments should be rebuilt from the hpc2 lock:

```bash
module load anaconda3 cuda/12.4 uv
bash scripts/env/hpc2_bootstrap_env.sh --venv .venv-test --recreate
```

See [envs/hpc2](envs/hpc2) for the locked eval environment, shared cache
layout, freeze script, and optional wheelhouse workflow.

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
