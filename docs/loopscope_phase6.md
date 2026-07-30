# LoopScope Phase 6: Pre-answer trajectory window selection

Phase 6 is activated under a standing phase-continuation mandate. Gate A alone is currently
authorized to its exact executor; Gates B--G remain locked. The authoritative mutable state is
`../../.planning/loopscope_phase6_control.md`; this file is a runbook-level summary and never grants
authority by itself.

## Scientific question

Phase 6 keeps the Phase 4 model, task, renderer, generation, and TFLT configuration. It moves the
trajectory probe from the last rendered prompt token to the final generated token immediately
before the first token of the final answer content.

The experiment is a two-pass, dataset-level offline selector:

1. Run deterministic native no-loop CoT generation.
2. Resolve the unique final-answer content span and freeze the preceding token as the probe anchor.
3. Replay the prompt plus generated tokens strictly before the answer token with
   `use_cache=false`, `output_hidden_states=true`, and zero loop insertions.
4. Aggregate the 12,032 MMLU-Pro test identities and freeze one global V2 selector decision.
5. Run independent loop full decoding only after the selector and panel are byte-frozen.

It is not a same-pass online or per-sample dynamic selector.

The anchor extractor is span-preserving. The pinned lm-eval 0.4.11 source closure establishes that
`utils.py@74ab409c...` defines the renderer, while
`_default_template_yaml@356e937a...` defines `custom-extract` as the case-sensitive regex
`answer is \(?([ABCDEFGHIJ])\)?` followed by the outcome filter `take_first`. Phase 6 uses the
literal regex and capture group 1, but requires exactly one match span; it must not silently apply
`take_first` to multiple possible anchors. A separately versioned aligner maps that span start to
the generated token carrying its first content character. A start exactly on a half-open token
boundary belongs to the token on the right. Regex and alignment rules are frozen on synthetic
fixtures before any real model completion is observed. Missing, ambiguous, or unalignable spans
are engineering BLOCKs; they are never dropped or repaired post hoc.

## Frozen cell

```text
model=Qwen/Qwen3-4B-Instruct-2507
revision=cdbee75f17c01a7cc42f958dc650907174af0554
task=mmlu_pro
num_fewshot=5
population=test-12032
dtype=bfloat16
k=3
iteration_mode=block
strategy=euler
step_size=1/3
total_horizon=1
cache_strategy=first
decode_mode=full
do_sample=false
temperature=0.0
max_gen_toks=2048
```

All provenance fields must be reclosed from live immutable artifacts before activation. Historical
values in the control file are not a substitute for Gate B evidence.

## Probe outputs

At the pre-answer anchor, extract raw residual boundaries `B_0...B_36`.

The sanitized trajectory record keeps only span offsets, token indices, and extractor/aligner
hashes. It must not retain the answer content, an answer-span hash, a parsed prediction, gold,
correctness, or any outcome field.

Selector inputs:

- full-vocabulary next-token entropy;
- `KL(p_l || p_36)`.

Diagnostic-only outputs:

- final-normalized RMS-L2 to `B_36`;
- final-normalized cosine to `B_36`;
- cosine distance derived as `1-cosine`;
- raw adjacent angular distance `acos(clamp(cos(B_l,B_l+1)))/pi`.

Hidden metrics must not enter eligibility, ranking, selection frequency, or panel membership.

## V2 selector

Use the complete central-40% width `3/4/5/6` domain for 36 layers:

```text
central blocks=11...24
w3 starts=11...22
w4 starts=11...21
w5 starts=11...20
w6 starts=11...19
candidate count=42
```

The selector preserves V1 `NewEligible` and uses V2 `RateStable` plus
`S_RATE=sqrt(G_H*G_K)`. The only selector terminal states are one stable selected window or one of
the three frozen scientific ABSTAIN states. Non-finite values, anchor mismatch, population drift,
or verifier disagreement are engineering BLOCKs.

External Phase 6 window names are half-open block intervals: width 3 at start 15 is `15:18`.
The row's `end` field remains the inclusive final block (17), while `boundary_exit` is `B_18`.
This convention is required to compare candidates against the frozen same-cell registry without
representation drift.

## Outcome panel

After selector freeze, run a fresh baseline plus the unique union of:

- fixed comparator `15:18`;
- registry-excluded diagnostic `S_RATE` High3;
- registry-excluded diagnostic `S_RATE` Low3;
- selected window when distinct.

The historical same-cell Phase 4 cells remain in a known-outcome registry. A selected historical
window is a known-window hit, not new prospective evidence. The final analysis reports selector
decision, absolute selected gain, ranking enrichment, and known-outcome status separately.

## Gate sequence

| Gate | Purpose |
|---|---|
| A | Local card, schema, anchor, producer, V2 analyzer, and focused tests |
| B | Remote CPU/import and provenance closure |
| C | Four-identity deterministic two-pass GPU smoke |
| D | Full test-12032 no-loop generation and trajectory acquisition |
| E | CPU-only V2 selector/panel freeze and diagnostic reporting |
| F | Frozen loop-panel full-decode acquisition with outcomes sealed |
| G | One-time unseal, paired analysis, final report, and phase audit |

Each Gate requires a new independent user-visible executor. No executor may cross a Gate boundary.
The planning/audit task alone issues `PASS`, `PASS_WITH_FIXES`, or `BLOCK`.

## Gate A local artifacts

Gate A provides one machine-readable card, closed trajectory and selector-freeze schemas,
span/ID anchor utilities, pure NumPy producer reductions, a V1-compatible V2 selector adapter,
panel freezing, and an independent verifier. The prepare command is intentionally a dry-run
skeleton: it validates and prints the frozen two-pass plan but cannot load a model or dataset.

```text
configs/loopscope/phase6_pre_answer_v2_card.json
configs/loopscope/phase6_trajectory_schema.json
configs/loopscope/phase6_selector_freeze_schema.json
src/tflt/loopscope/phase6_anchor.py
src/tflt/loopscope/phase6_schema.py
src/tflt/loopscope/phase6_acquisition.py
src/tflt/loopscope/phase6_selector.py
src/tflt/loopscope/phase6_verifier.py
scripts/loopscope/run_qwen4_phase6_prepare.py
scripts/loopscope/verify_phase6_gate_a.py
tests/test_loopscope_phase6_*.py
```

## Current authorization

Only Gate A is authorized to executor
`019fb3fd-8cb5-7980-a0fb-6baaa48bd0d0` under
`../../.planning/loopscope_phase6_gate_a_handoff.md` revision 3. Gate A may perform its local
allowlisted implementation, targeted CPU tests, explicit-path commit, and push. It may additionally
use bounded read-only SSH to the pinned HPC2 lm-eval MMLU-Pro source directory and package METADATA
solely for exact source/hash closure. It may not write remotely, sync remote Git, load/download a
model or dataset, read caches/outcomes, use CUDA/GPU/Slurm, run real generation/replay, or enter
Gate B. Exact current authority and any later transition are owned only by the control file plus
the matching Gate handoff.
