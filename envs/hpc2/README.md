# hpc2 Environment Spec

This directory freezes the first known-good TFLT eval environment on
`hpc2-hkustgz`.

## Policy

- Use one project-local virtualenv per project. For TFLT the active env is
  `<project>/.venv`.
- Do not share one global conda env across TFLT, LoopPilot, LoopScope, or later
  projects.
- Do not upgrade an active `.venv` in place while jobs may be using it. Build a
  new versioned env such as `.venv-cu121-20260701`, validate it, then switch the
  `.venv` symlink only after jobs are idle.
- Keep `.venv`, model caches, eval caches, and `runs/` out of git. Commit only
  specs, lock files, bootstrap scripts, and report examples.

## hpc2 Baseline

- Modules: `anaconda3 cuda/12.4 uv`
- Python: `3.11.9`
- uv: `0.9.5`
- torch import version: `2.3.1+cu121`
- transformers: `4.51.3`
- lm_eval: `0.4.11`

The lock file was derived from the existing successful remote `.venv` with
`uv pip freeze --python .venv/bin/python`. The editable project install line is
not included in the lock; the bootstrap script installs this checkout with
`--no-deps` after locked dependencies are installed.

The bootstrap uses PyPI plus the PyTorch CUDA 12.1 wheel index with
`UV_INDEX_STRATEGY=unsafe-best-match` by default. This is intentional for this
frozen environment because both indexes are required to reconstruct the current
CUDA-enabled package set.

## Shared Cache Layout

```bash
export TFLT_HPC2_CACHE_ROOT=/hpc2hdd/home/xhuang225/shared
export HF_HOME=$TFLT_HPC2_CACHE_ROOT/hf_home
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$TFLT_HPC2_CACHE_ROOT/datasets
export UV_CACHE_DIR=$TFLT_HPC2_CACHE_ROOT/uv
export TFLT_WHEELHOUSE=$TFLT_HPC2_CACHE_ROOT/wheelhouse/tflt-cu121
```

Legacy project-local cache paths under
`/hpc2hdd/home/xhuang225/projects/.cache` are compatibility symlinks to this
shared layout. New profiles and scripts should use `/hpc2hdd/home/xhuang225/shared`
directly.

## Bootstrap

From the repository root on hpc2:

```bash
module load anaconda3 cuda/12.4 uv
bash scripts/env/hpc2_bootstrap_env.sh --venv .venv-test --recreate
```

For the production environment, build a versioned venv first:

```bash
bash scripts/env/hpc2_bootstrap_env.sh --venv .venv-cu121-20260701 --recreate
```

After validation and only when no jobs are using the old env, point `.venv` to
the validated versioned env.

## Freeze

```bash
bash scripts/env/hpc2_freeze_env.sh --venv .venv
```

The freeze script writes a timestamped directory under `envs/hpc2/freeze/`.
Those snapshots are ignored by git unless intentionally promoted into
`requirements-eval.lock.txt`.

## Optional Wheelhouse

When hpc2 networking is unreliable:

```bash
bash scripts/env/hpc2_download_wheelhouse.sh
bash scripts/env/hpc2_bootstrap_env.sh --venv .venv-test --recreate --offline-wheelhouse
```

This downloads wheels into
`/hpc2hdd/home/xhuang225/shared/wheelhouse/tflt-cu121` and rebuilds
from that wheelhouse without touching model caches or existing run outputs.

## Validation

Use import and help checks, not model evals:

```bash
. .venv-test/bin/activate
python -c "import torch, transformers, lm_eval; print(torch.__version__, transformers.__version__, lm_eval.__version__)"
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m tflt.cli audit-loop-effect --help
```

For GPU visibility, submit a short Slurm import/CUDA check only. Do not run
MMLU, SciQ, audit prompts, or any model generation as part of environment
rebuild validation.
