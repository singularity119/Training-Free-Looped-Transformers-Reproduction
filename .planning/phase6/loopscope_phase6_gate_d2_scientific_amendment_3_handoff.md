# GATE_D2_SCIENTIFIC_AMENDMENT_3_HANDOFF

Project/phase: LoopScope Phase 6  
Gate: D-2 — cache-aligned eligibility-aware full trajectory recovery  
Planning thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor: `019fb8f3-8cb7-7c93-a8a0-bae811735601`  
Executor title: `execute-LoopScope-PreAnswer-RBRV2-第6阶段-Gate D-2`  
Handoff state: `AUTHORIZED`  

## 1. Superseding authority

This handoff supersedes the Gate D-2 revocation associated with
`BLOCK_GENERATION_REPLAY_MISMATCH`. The user approved Scientific Amendment 3 on 2026-08-01:
cache-aligned incremental replay is required, while per-identity argmax equality is replaced by a
frozen population/category rate closure. Gate E...G remain locked.

Admission must match:

- live control SHA-256:
  `4add8017e2be44bb2e67a4b95e7e7a186ad28ca463c89469c0323da671a0d909`;
- protected `AGENTS.md` SHA-256:
  `c82dae577b166d5481928b1d22af2b8a7cbdf2c7ae11f06532c3939e62a8ba08`;
- repo/branch: `loopscope-tflt` / `loopscope`;
- exact starting commit and `origin/loopscope`:
  `1a365a5c1fd90ca2e2af4e6ac54a803487a3b4ec`;
- expected local dirty state: only protected unstaged `M AGENTS.md`;
- HPC2 dedicated clone must be clean and fast-forwarded to the exact implementation commit before
  any GPU run.

Stop only on a real admission mismatch. Do not modify, stage, or commit `AGENTS.md` or planning-owned
control/handoff files.

## 2. Frozen scientific amendment

Keep model, tokenizer, dataset, renderer, generation parameters, TFLT configuration, first-match
anchor, eligibility mask, metrics, candidate windows, selector rule, and information barrier exactly
as already frozen, except for the replay/argmax closure below.

Pass 1 remains native greedy `model.generate` with its incremental KV cache. Pass 2 must be one
logical cache-aligned incremental replay:

1. run the exact rendered prompt with `use_cache=true` to establish `past_key_values`;
2. feed generated-prefix tokens strictly in original order, one incremental step at a time, using
   the same model, dtype, attention implementation, cache position/mask, and generation-critical
   kwargs;
3. capture B0...B36 and the frozen metrics only at the final generated token immediately before the
   selected answer-content token;
4. record sanitized incremental step count/hash closure and `replay_count=1`;
5. forbid full-prefix `use_cache=false` replay and forbid changing generation to `use_cache=false`.

For every `ANCHOR_ELIGIBLE` identity, record Boolean
`replay_argmax_matches_generated`. A `false` value is not an identity failure: do not delete, mask,
replace, or downgrade it, and retain its trajectory in the global aggregation and its identity in the
rate denominator. Formal acceptance requires:

- eligible overall argmax match rate `>=0.995`;
- every frozen category argmax match rate `>=0.98`.

Below either floor is scientific
`BLOCK_REPLAY_ARGMAX_MATCH_RATE_BELOW_FLOOR`. Token/order/cache-state/position/mask/step-count/hash,
anchor, hidden-capture, schema, finite, or replay-structure failures remain fail-closed engineering
errors and are not covered by this tolerance.

Scientific Amendment 2 remains unchanged: exact case-sensitive regex, first match ordinal 0,
`ANCHOR_NOT_EXPRESSED` only for zero match or no preceding generated token, overall eligibility
coverage `>=0.995`, each category `>=0.98`, and no fallback or identity substitution.

## 3. Implementation and verification scope

Modify only the smallest materially required Phase 6 card/schema/runtime/runner/verifier/tests/runbook
paths. Add focused regression coverage for:

- native cache-aligned incremental replay and exact cache/prefix/step closure;
- mixed match/mismatch rows and rate recomputation;
- mismatch trajectory retention and denominator membership;
- overall and per-category rate-floor rejection;
- structural cache failure remaining fail-closed;
- all existing eligibility, information-barrier, and trajectory-shape invariants.

Run focused local tests, compile/help/dry-run, diff checks, explicit-path commit/push, clean remote
fast-forward sync, and the same focused remote tests. Low-risk engineering repair and
evidence-preserving fresh retries remain unlimited while the frozen science and information barrier
are unchanged.

## 4. GPU execution

First run one fresh `<30min` A40 `debug` preflight using a representative cohort that includes the
outcome-safe identities/positions corresponding to the prior formal mismatch path. Real debug may
produce either match or mismatch; acceptance requires valid Boolean recording, structural closure,
trajectory retention, and a fresh-process verifier PASS. Synthetic focused tests must independently
exercise an actual mismatch-retention case.

Only after debug PASS, create a completely fresh formal root and submit the full 12,032 canonical
population to the highest legal normal-user A800 partition/QoS. Do not reuse or mix any old Gate D or
D-2 shard. After the formal job ID is verified, activate exactly one 60-minute executor-owned
heartbeat immediately, including while PENDING. Debug heartbeat is created only after stable RUNNING
for about 60 seconds and only if continued waiting remains useful.

Any job failure is an attempt-level event, not a Gate terminal. Pause the heartbeat, resume this exact
executor, diagnose safely, repair within contract, use a fresh attempt/root when required, and
continue. Emit `BLOCK` only at a genuine science, information, safety, destructive/external-authority,
or demonstrated no-progress boundary.

## 5. Gate D-2 acceptance

The final producer and independent fresh-process verifier must close:

- canonical identity/category/order and sealed baseline membership exactly 12,032;
- `eligible + not-expressed = 12,032`, no missing/extra/duplicate;
- eligibility overall/category floors;
- exact regex ordinal 0 for all eligible rows and no fallback for not-expressed rows;
- one generation and one logical replay per eligible identity, zero loop insertion;
- cache-aligned prompt/prefix/order/state/step hash closure;
- Boolean argmax field on every eligible row, mismatch trajectories retained, exact overall/category
  rate recomputation, and both rate floors;
- B0...B36, full-vocabulary entropy/KL, Hidden RMS-L2/cosine/cosine-distance, adjacent angular
  distance, finite/endpoints/schema/hash closure;
- immutable receipts/manifests and preserved information barrier.

Do not run the selector, read sealed completion/gold/label/correctness/outcome, or enter Gate E...G.
Old failed/cancelled roots remain immutable historical evidence only.

Send exactly one terminal `GATE_D2_FINAL_AUDIT` or protocol-valid `BLOCK` to the planning thread.
