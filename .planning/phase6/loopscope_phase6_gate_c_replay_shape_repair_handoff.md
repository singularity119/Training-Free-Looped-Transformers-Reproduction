# GATE_C_REPLAY_SHAPE_REPAIR_HANDOFF

Project/phase: LoopScope Phase 6  
Gate: C — low-risk replay tensor-shape repair continuation  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Executor: `019fb452-e2af-7bf2-a411-612c81971245`  
Decision requested: `PASS / PASS_WITH_FIXES / BLOCK`

## Admission and materiality

Scientific amendment 1 remains frozen at commit
`62cea9bbf606c919bd3e312c0d79c32a11218ded`. Debug Job `10105218` reached the real
replay path and failed in `_replay_once` because a 2D raw boundary tensor was indexed as 3D.
No non-log scientific artifact was produced and the information barrier remained intact.

This is an authorized low-risk engineering repair in the same Gate, not a new scientific
amendment. Re-read the live control, this handoff, and protected `AGENTS.md`; verify exact
thread/Git/dirty-state admission before mutation.

## Repair policy

Executor-owned low-risk engineering repair loops are unlimited. Each loop must:

- preserve the evaluator-aligned ordinal-0 anchor, exact regex, model/data/generation/TFLT,
  four identities × two replicates, trajectory metrics, schemas, information barrier, and Gate
  boundaries;
- address one reproducible engineering cause with targeted regression coverage;
- preserve every failed job/root and use a fresh write-once path for any real rerun;
- make material progress and stop before any scientific, destructive, protected-path, resource,
  outcome-unsealing, or next-Gate choice.

Unlimited engineering loops do not permit unchanged resubmission or formal-partition execution.
After each locally/remote verified material repair, exactly one fresh `debug` 4×2 smoke may be
submitted to exercise the repaired path. Further low-risk failures may be repaired and validated
the same way without a numeric loop cap.

## Immediate repair objective

Normalize the replay raw-boundary extraction so every B0...B36 vector is selected correctly
whether the frozen FinalNorm pre-hook capture is stored as `[1,H]` while model hidden states are
`[1,S,H]`. Keep the existing raw-B36/FinalNorm closure and fail closed on any unexpected rank or
shape. Add the smallest focused regression test that reproduces the 2D/3D mix.

Run targeted Phase 6 Gate C/anchor/acquisition tests, compile/help/dry-run, and
`git diff --check`; do not run an unrelated full suite. Commit/push explicit paths, ff-only sync
the clean HPC2 clone, and run focused remote checks.

Then submit one fresh `<30min` `debug` smoke with the exact existing 4×2 command and a new
write-once root. Do not enable a heartbeat on submission or while `PENDING`. Enable exactly one
10-minute smoke heartbeat only after bounded executor checks confirm the job is genuinely
`RUNNING` for about 60 seconds with no launcher/startup error.

On terminal success, pause/delete the heartbeat, run the fresh-process verifier, and deliver one
`GATE_C_FINAL_AUDIT`. On a material non-low-risk/scientific blocker, deliver `BLOCK`. Gate D...G
remain locked.
