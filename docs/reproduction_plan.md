# Reproduction Plan

## Phase 0: Local Engineering

- Keep the Mac checkout code-only.
- Run pure-Python tests for config parsing, loop strategy formulas, wrapper
  patch/restore behavior, and command generation.
- Do not download checkpoints locally.

## Phase 1: Remote Smoke

- Models: `qwen3-0.6b-base`, `qwen3-1.7b-base`
- Tasks: small `--limit` runs for SciQ, ARC, and MMLU slices
- Loop: window `[12, 15]`, `k=2`, damped Euler, block-mode for dense
- Acceptance: baseline and loop both finish with matching result schema

## Phase 2: Main Small-Model Reproduction

- Model: `qwen3-1.7b-base`
- Tasks: 16-task macro or MMLU 5-shot
- Report checkpoint revision, transformers version, lm-eval version, task alias,
  prompt settings, loop config, hardware, and cache/decode settings.

## Phase 3: Dense Headline

- Models: `qwen3-4b-base`, `qwen3-4b-instruct-2507`
- Tasks: MMLU-Pro, GPQA-Main, CommonsenseQA, MMLU, SciQ
- Mark `-2507` experiments as current-checkpoint reproductions.

## Phase 4: Ablation

- Window width and position
- K value
- Naive loop versus damped Euler and RK variants
- Cache strategy: first, last, none
- Block-mode versus layer-mode

## Phase 5: Extensions

- Qwen1.5-MoE and Qwen3-30B-A3B use layer-mode by default.
- DeepSeek/Moonlight support should be added only after a dedicated DynamicCache
  compatibility pass.
