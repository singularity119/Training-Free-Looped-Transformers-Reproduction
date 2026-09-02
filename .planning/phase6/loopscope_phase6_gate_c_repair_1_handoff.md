# GATE_C_REPAIR_1_HANDOFF

Project/phase: LoopScope Phase 6  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor: `019fb452-e2af-7bf2-a411-612c81971245`  
Gate: C repair 1 — replace the false-positive import-state guard and rerun the exact smoke  
Decision: `PASS_WITH_FIXES`; Gate C remains open and Gate D...G remain locked.

This revision applies the user's latest parallel-race rule: keep the already queued full smoke and
also submit an exact same-commit 4×2 smoke on `debug`, with separate write-once outputs.

The later mandatory debug-before-long rule is prospective. Job `10105004` was submitted before it
and remains grandfathered; do not retroactively cancel it. The parallel debug leg supplies the
current commit/runtime preflight without changing either job.

## Audited blocker

- Exact base/implementation commit: `fd6c311ea11b37fbb68b9c6cc20ac343409ab8f7`
- Job `10101747`: pre-science `/bin/sh` `pipefail` failure.
- Job `10104916`: pre-model/data failure at
  `assert_native_no_loop_runtime()` with
  `loop implementation modules are loaded in Gate C runtime`.
- Normal `import tflt.loopscope.phase6_runtime` executes `tflt/__init__.py`, whose public API
  imports `tflt.wrapper` and therefore `tflt.cache`/`tflt.strategies`. Module presence is not
  evidence that a loaded model has been wrapped; the current predicate is unreachable on the
  normal producer path.

## Authorized repair

Modify only:

- `src/tflt/loopscope/phase6_runtime.py`
- `tests/test_loopscope_phase6_gate_c.py`
- `scripts/loopscope/run_qwen4_phase6_gate_c.py` only if strictly required to expose the same
  application-state evidence in the existing sanitized receipt.

Replace only the `sys.modules` presence predicate with a model-object application-state check.
The repaired normal path must allow harmless package imports but fail if any registered decoder
module is actually a `tflt.wrapper` implementation or equivalent loop wrapper has been applied.
Keep explicit evidence that generation/replay use the native model and
`loop_insertions=0`. Do not change `tflt/__init__.py`, `tflt.wrapper`, model/data, identities,
generation/replay, metrics, tolerances, schemas, sanitizer, or information barrier.

Add one focused regression proving:

1. ordinary `tflt` package imports no longer fail the native runtime admission;
2. an actually wrapped/synthetic wrapper-bearing model fails closed;
3. all prior focused Gate C tests still pass.

Run only focused tests, compile/help/dry-run, and `git diff --check`; no full suite. Commit and push
the explicit repair paths, then ff-only sync the dedicated HPC2 clone.

## User-directed debug-first authority

Job `10105004` was submitted before the debug-first hold and was reported
`PENDING(Resources)`, elapsed `00:00:00`, with no node or artifacts when the hold arrived.
Preserve this already queued job and its queue position. Do not cancel, hold, requeue, replace, or
modify its fresh root, exact 4×2 membership, resource envelope, or repair commit.

In parallel, submit exactly one additional smoke on partition `debug`:

- exact commit `5e4a5eb9e094fee2c7a500c98d6fa7a0f1d6e92c`;
- the unchanged existing `smoke` command, all four frozen identities, two replicates each;
- one A40-compatible GPU, 8 CPU, at most 64 GiB, walltime at most `00:30:00`;
- a distinct root `phase6-gate-c-smoke-debug-race-<UTC>`;
- no code change, new commit, alternate membership, shortened generation, or shared output.

This is a resource race, not a scientific retry. If `debug` rejects the unchanged request,
times out, or fails, preserve its evidence and let `10105004` continue. Do not resubmit the debug
leg. Use the single existing 10-minute heartbeat to monitor both job IDs as one Gate C job set;
update its prompt in place and do not create a second automation. The first valid exact 4×2
receipt is decision-eligible, but no job is automatically cancelled under this supplement.

## Fresh full smoke authority

Job `10105004` is the authorized fresh full smoke: original four identities × two replicates,
unchanged A40/8 CPU/64 GiB/02:00:00 envelope, and existing write-once root. No additional full
submission is authorized beyond the single debug race leg above. The previously granted one
additional retry remains limited to a preserved pre-model/data scheduler or launch-infrastructure
failure of the normal leg; it does not authorize cancellation merely to switch partitions or a
debug resubmission. Any guard, anchor, replay, numeric, determinism, sanitizer, verifier,
OOM-after-load, or other science-path failure is `BLOCK`.

When waiting on a real job, use exactly one executor-owned smoke heartbeat at 10-minute cadence;
update the existing paused heartbeat or create one for the new job, never both. Terminal handling
remains pause/delete first, then `AUTOMATION_TERMINAL_RESUME` to this exact executor.

## Terminal

Return one visible `GATE_C_FINAL_AUDIT` or `BLOCK` to planning. Include the repair commit/diff,
focused tests, new job/root and terminal evidence, all original 4×2 must-pass closures, retry
accounting, information-barrier confirmation, and requested planning decision. Do not enter Gate D.
