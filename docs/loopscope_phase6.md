# LoopScope Phase 6: Pre-answer trajectory window selection

Phase 6 is activated under a standing phase-continuation mandate. The authoritative mutable state
is `../../.planning/loopscope_phase6_control.md`; this runbook never records current Gate/executor
state and never grants execution authority.

## Current MMLU 0-shot prefix branch

The current Phase 6 continuation is the MMLU 0-shot prefix-trajectory contract in
`configs/loopscope/phase6_mmlu0_prefix_card.json`. It is a separate local contract from the
historical MMLU-Pro pre-answer/D-2 material below.

```text
model=Qwen/Qwen3-4B-Instruct-2507
revision=cdbee75f17c01a7cc42f958dc650907174af0554
task=cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe
lm_eval=0.4.11
num_fewshot=0
prompt=plain standard task renderer terminal
apply_chat_template=false
fewshot_as_multiturn=false
generation=false
population=validation-1531 / 57 subjects
choice_continuations=" A", " B", " C", " D"; each exactly one unique tokenizer ID
probe=last non-padding rendered-prefix token; one native no-loop forward; use_cache=false
boundaries=B_0...B_36; final normalization is applied once to B_0...B_35
selector=RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE; central blocks 11...24; widths 3/4/5/6; 42 candidates
bootstrap=2000 subject-stratified joint draws; seed=20260801; frequency threshold=0.80
```

Gate H only closes the local card, schema, pure metrics, selector projection, verifier, and
no-execution CLI. A trajectory record contains identity/subject, four-choice probabilities,
choice entropy, KL to the final choice distribution, hidden-to-final scalar diagnostics, and 36
adjacent angular diagnostics; raw logits, full-vocabulary probabilities, and hidden tensors are
not persisted. The selector projection is exactly `identity, category, H, D`, so hidden
diagnostics cannot enter ranking. Audited Phase 5 validation identity provenance may be reused,
but zero-shot prompt hashes require a fresh closure.

Gate H does not load a model or data, run a forward/generation call, read gold/target/label/
correctness/outcome fields, execute the formal selector, run a loop, or create a panel. Gate I is
limited to CPU/import and four-identity A40 debug admission; Gate J performs the validation-1531
V2 selector and ends this branch with a selected window or legal `ABSTAIN`. No full-loop result,
panel, or full-loop result is produced by this branch. The mutable authority
and live handoff remain the planning control file and matching handoff named above.

## MMLU 5-shot follow-up: Gate K contract

The authorized follow-up is a separate five-shot contract in
`configs/loopscope/phase6_mmlu5_prefix_card.json`. It keeps the accepted Qwen3-4B model,
MMLU validation-1531 identity order, exact ` A/B/C/D` continuation surfaces, 37 no-loop
boundaries, diagnostic metrics, V2 candidate domain, bootstrap, and TFLT identity. Its only
scientific change is standard `num_fewshot=5` with `fewshot_split=dev` and the resulting fresh
rendered-prefix projection.

Gate K uses the real lm-eval standard MMLU renderer. Validation rows are selected only with
`question`, `choices`, and `subject`; dev rows may provide five example demonstrations and their
answers to the renderer. Prompt text and token IDs are hashed in memory and are not persisted.
The closed projection records exact demonstration order/count, prompt/tokenization hashes, the
last non-padding probe position, renderer/source hashes, and no target/gold/test/outcome fields.
The trajectory shape is separately versioned as `phase6_mmlu5_trajectory_schema.json` so 0-shot
records cannot enter the five-shot population.

Gate K has no GPU, CUDA, Slurm, model-forward, selector, loop, test-split, validation-target, or
outcome authority. Its CPU projection and verifier only unlock Gate L after planning acceptance.
Gate L remains a new executor's debug-partition four-identity native smoke; Gate M alone may
acquire validation-1531 trajectories and run the frozen V2 selector, ending with selected window
or legal `ABSTAIN`. No test-14042 or full-loop outcome belongs to this follow-up.

## Scientific question

Phase 6 keeps the Phase 4 model, task, renderer, generation, and TFLT configuration. It moves the
trajectory probe from the last rendered prompt token to the final generated token immediately
before the first token of evaluator-selected answer content.

The experiment is an eligibility-aware two-pass, dataset-level offline selector:

1. Run deterministic native no-loop CoT generation.
2. Enumerate final-answer matches, select ordinal zero, map its first content character to the
   generated token that carries it, and classify canonical anchor eligibility.
3. Freeze one canonical-order eligibility mask and its category coverage receipt across all
   12,032 identities before selector execution.
4. For eligible identities only, run one cache-aligned incremental replay: establish the prompt
   KV cache with `use_cache=true`, then feed generated-prefix tokens strictly in original order
   with exact cache-position and attention-mask closure. Capture hidden states only on the final
   generated token immediately before the answer token, with zero loop insertions. Full-prefix
   `use_cache=false` replay is forbidden.
5. Record `replay_argmax_matches_generated` for every eligible trajectory. A mismatch is retained
   in the trajectory aggregation and rate denominator. The eligible overall match rate must be
   `>=0.995`, and every frozen-category match rate must be `>=0.98`; structural replay/cache
   failures remain fail-closed engineering defects.
6. Aggregate the near-complete eligible trajectory subset and freeze one global V2 selector
   decision together with the frozen coverage receipt.
7. Run independent full-population loop decoding only after the selector and panel are
   byte-frozen. Every canonical identity remains in sealed baseline and outcome membership.

It is not a same-pass online or per-sample dynamic selector.

The anchor extractor is span-preserving. The pinned lm-eval 0.4.11 source closure establishes that
`utils.py@74ab409c...` defines the renderer, while
`_default_template_yaml@356e937a...` defines `custom-extract` as the case-sensitive regex
`answer is \(?([ABCDEFGHIJ])\)?` followed by the outcome filter `take_first`. Phase 6 uses that
literal regex and capture group 1, enumerates every match in generated-text order, and selects
ordinal zero. Multiple matches are legal. Zero exact matches, or a match whose answer character is
carried by the first generated token so that no preceding generated token exists, is
`ANCHOR_NOT_EXPRESSED`: the identity remains in the canonical and sealed population but has no
replay and no trajectory. No alternate parser, case folding, cue broadening, EOS/prompt/final-token
fallback, identity replacement, or fabricated anchor is allowed.

A separately versioned aligner proves that the frozen generated IDs decode exactly to the frozen
generated text and uniquely maps the selected span start to its generated token. It does not assume
that every intermediate prefix decode is a string prefix: an unrelated unstable prefix is legal
when the adjacent decoded boundaries at the selected character prove a unique token carrier. A
start exactly on a half-open token boundary belongs to the token on the right. A special token may
occur before or after the selected answer when it follows the actual frozen decode path and full
closure still holds; the token carrying answer content itself cannot be special. In particular, a
later special token does not invalidate an already resolved answer unless it breaks full
generated-ID/text closure. If an exact match exists, any alignment, replay, or runtime failure is an
engineering BLOCK and cannot be converted into an eligibility exclusion.

The frozen mask must satisfy `eligible_count / 12032 >= 0.995` and coverage in every one of the 14
frozen categories `>= 0.98`; otherwise the terminal state is
`BLOCK_ANCHOR_COVERAGE_BELOW_FLOOR`.

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

Each canonical identity has one minimal eligibility row containing only canonical identity/category,
eligibility state, generation count, sealed-payload membership/hash reference, and frozen
provenance. In particular, an `ANCHOR_NOT_EXPRESSED` row contains no completion text, answer
letter/content, token IDs, span values, target/gold/label, correctness/outcome, logits,
probabilities, or hidden tensors.

Eligible identities additionally have one closed sanitized trajectory record. It keeps only match
count, selected-match ordinal zero, selected span offsets, token indices, extractor/aligner hashes,
and the frozen trajectory scalars. It must not retain answer content, an answer-span hash, a parsed
prediction, gold, correctness, or any outcome field. Human-facing claims must describe a
near-complete eligible trajectory subset, never exact-12,032 trajectory coverage.

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
| D | Historical exact-population trajectory attempt; invalid outputs are diagnostic only |
| D-2 | Eligibility-aware full-population generation, mask freeze, and eligible trajectory recovery |
| E | CPU-only V2 selector/panel freeze and diagnostic reporting |
| F | Frozen loop-panel full-decode acquisition with outcomes sealed |
| G | One-time unseal, paired analysis, final report, and phase audit |

Each Gate requires a new independent user-visible executor. No executor may cross a Gate boundary.
The planning/audit task alone issues `PASS`, `PASS_WITH_FIXES`, or `BLOCK`.

The Gate D-2 real-CUDA debug cohort validates the eligibility partition it actually observes and
may legally contain zero or more `ANCHOR_NOT_EXPRESSED` rows. Focused synthetic verification must
separately close both the all-eligible case and a mixed eligible/not-expressed case, including zero
replay and zero trajectory for every not-expressed row. Formal test-12032 acquisition is the
decisive coverage measurement; debug does not search for or require a permanently classified
identity.

## Gate A local artifacts

Gate A provides one machine-readable card, closed trajectory and selector-freeze schemas,
span/ID anchor utilities, pure NumPy producer reductions, a V1-compatible V2 selector adapter,
panel freezing, and an independent verifier. The prepare command is intentionally a dry-run
skeleton: it validates and prints the frozen two-pass plan but cannot load a model or dataset.

```text
configs/loopscope/phase6_pre_answer_v2_card.json
configs/loopscope/phase6_eligibility_schema.json
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

## Authorization

This runbook never records live Gate or executor state and does not grant execution authority.
Consult `../../.planning/loopscope_phase6_control.md` and the matching current-Gate handoff. If
they disagree, stop.
