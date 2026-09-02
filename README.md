# Training-Free Looped Transformers

Local codebase for reproducing the paper "Training-Free Looped Transformers".

This repository is intentionally code-only. Model weights, datasets, dependency
environments, eval logs, and result tables should live on the lab server or
`hpc2-hkustgz`, not on this Mac.

## LoopScope 项目接续入口

在 `loopscope` 分支接续研究时，请先读 [AGENTS.md](AGENTS.md) 和
[PROJECT_MEMORY.md](PROJECT_MEMORY.md)，再查 [.planning 索引](.planning/README.md)
及当前 control。历史 Gate 记录不授予新的执行权限。

换设备请按 [新设备接续说明](docs/new_device_handoff.md) 获取既有分支并建立 planning
兼容链接。原 `AGENTS.md` 按用户要求保持原样；当前阶段与迁移后的路径解释由上述入口提供。
本分支除代码、配置和测试外也保存项目文档与 planning 历史；模型、数据和完整运行工件仍留在 HPC。

## Scope

- Package: `tflt`
- CLI: `tflt inspect-model`, `tflt smoke`, `tflt eval`, `tflt launch`,
  `tflt collect`, `tflt report`, and the gated `tflt probe-layers`,
  `tflt probe-window`, `tflt make-window-grid`, `tflt score-windows`,
  `tflt analyze-window-grid` LoopScope commands
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

LoopScope Qwen3-1.7B 第一阶段的门控、无标签校准池与不可覆盖运行流程见
[docs/loopscope_phase1.md](docs/loopscope_phase1.md)。
