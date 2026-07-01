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

### Next target: Table 6 blue-box MMLU-Pro cell

- Reproduction type: current-checkpoint reproduction, because the local model
  alias maps to `Qwen/Qwen3-4B-Instruct-2507` rather than a bit-exact paper
  checkpoint.
- Model alias: `qwen3-4b-instruct-2507`
- Benchmark: MMLU-Pro 5-shot CoT
- lm-eval task alias to start from: `mmlu_pro`
- Paper target: baseline `57.14`, loop `59.79`, delta `+2.64 pp`
- Loop config:
  - window: `[15, 18]`
  - `k=3`
  - `iteration_mode=block`
  - `strategy=euler`
  - `cache_strategy=first`
  - `decode_mode=full`
- Required reporting: model repo and commit revision, transformers version,
  lm-eval version, exact task alias/prompt setting, few-shot count, loop config,
  dtype, hardware, and cache/decode settings.
- Execution order:
  1. Run a small `--limit` smoke pair first: baseline then loop.
  2. Inspect logs/results to confirm the task prompt is the intended 5-shot CoT
     MMLU-Pro setting.
  3. Only after the smoke pair passes, run the full baseline and full loop in a
     fresh run root such as
     `runs/phase3-qwen4b-instruct2507-mmlupro-<timestamp>`.
- Command shape:

```bash
python -m tflt.eval_runner \
  --model qwen3-4b-instruct-2507 \
  --tasks mmlu_pro \
  --batch-size auto \
  --dtype bfloat16 \
  --output-dir <run_root>/baseline \
  --num-fewshot 5

python -m tflt.eval_runner \
  --model qwen3-4b-instruct-2507 \
  --tasks mmlu_pro \
  --batch-size auto \
  --dtype bfloat16 \
  --output-dir <run_root>/loop \
  --num-fewshot 5 \
  --loop \
  --window 15:18 \
  --k 3 \
  --iteration-mode block \
  --strategy euler \
  --cache-strategy first \
  --decode-mode full
```

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
