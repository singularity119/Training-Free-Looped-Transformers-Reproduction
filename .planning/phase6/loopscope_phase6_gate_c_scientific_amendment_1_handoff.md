# GATE_C_SCIENTIFIC_AMENDMENT_1_HANDOFF

Project/phase: LoopScope Phase 6  
Gate: C — evaluator-aligned first-match anchor repair and 4×2 debug smoke  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Executor: `019fb452-e2af-7bf2-a411-612c81971245`  
Decision requested: `PASS / PASS_WITH_FIXES / BLOCK`

## Admission and objective

The user approved scientific amendment 1 after debug Job `10105143` proved that a normal
completion can contain two exact answer spans. Replace the obsolete unique-span anchor with the
evaluator-aligned first valid match, cancel obsolete queued Job `10105004`, and rerun only the
frozen four-identity × two-replicate smoke on `debug`.

Re-read the live control, this handoff, and `loopscope-tflt/AGENTS.md`; verify exact hashes,
thread binding, branch/HEAD/origin, and the protected dirty state before mutation.

## Frozen revised anchor contract

- Regex remains exactly case-sensitive `answer is \(?([ABCDEFGHIJ])\)?`, capture group 1.
- Enumerate all matches in generated-text order.
- Zero matches: fail closed as `ANCHOR_UNRESOLVED`.
- One or more matches: select ordinal `0`, exactly matching lm-eval
  `custom-extract -> take_first`.
- Map the selected capture start to the unique generated token carrying that byte; the probe is
  the immediately preceding generated token.
- Missing preceding generated token, token/byte ambiguity, decode round-trip failure, replay-ID
  mismatch, or next-token closure failure remains BLOCK.
- Multiple matches are legal and must not drop or replace the identity.
- Persist only sanitized `answer_match_count`, `selected_match_ordinal=0`, selected byte offsets,
  token indices, hashes, lengths, metrics, and provenance. Do not persist generated text, answer
  letters, parsed predictions, gold, labels, correctness, accuracy, gain, logits, probabilities,
  or hidden tensors.

Everything else remains frozen: model/data/revisions, four identities, two replicates,
renderer/generation kwargs, two-pass replay, B0...B36 capture, H/KL, hidden diagnostics, TFLT,
selector, panel, and information barrier.

## Ordered execution

1. Cancel queued obsolete-contract Job `10105004` with `scancel`; verify `CANCELLED` via bounded
   scheduler evidence. Do not delete or reuse its root.
2. Pause or delete the existing heartbeat `loopscope-p6-gate-c-job-10101747`; do not create a
   monitor until a new real debug job exists.
3. Make the smallest contract implementation change in Phase 6 card/schema/anchor/runtime/
   verifier, focused tests, and the timeless Phase 6 runbook. Preserve unrelated paths and the
   protected unstaged `AGENTS.md`; do not stage or commit `AGENTS.md`.
4. Add focused fixtures for zero, one, repeated-identical, and corrected multiple matches;
   verify ordinal 0, count, byte/token alignment, replay closure, and closed sanitized schema.
5. Run only targeted Phase 6 anchor/Gate C/schema tests, compile/help/dry-run, and
   `git diff --check`. Stop hardening when these pass.
6. Explicit-path commit and push; fast-forward the clean dedicated HPC2 clone to that exact
   commit and rerun the focused remote checks.
7. Submit exactly one fresh `<30min` `debug` job with the existing four frozen identities × two
   replicates, same environment/resources/runtime/producer/verifier, and a fresh write-once root.
8. Once the job exists, create or update exactly one 10-minute smoke heartbeat. On terminal,
   pause/delete it, resume this exact executor, and either run the fresh-process verifier or emit
   BLOCK.

## Acceptance

- Old Job `10105004` is terminal `CANCELLED` and its root was not overwritten or deleted.
- Contract diff implements ordinal-0 selection without regex/model/data/metric drift.
- Targeted local and remote checks pass.
- Debug 4×2 smoke completes for all eight records, including the previously multi-match normal
  path, with deterministic duplicate hashes.
- Replay next-token closure, raw B0...B36/FinalNorm hook closure, all trajectory shapes/ranges,
  count=1 generation/replay, loop_insertions=0, sanitized namespace, and fresh-process verifier
  all PASS.
- Information barrier remains intact.

Gate D...G remain locked. Do not submit a formal-partition Gate C job, test-12032 acquisition,
selector, panel, or outcome work.

## Repair and terminal route

Agility budget: one minimal contract patch, targeted checks, then the earliest real debug smoke.
Executor-owned low-risk engineering repair loops are unlimited while they remain inside this
Gate, frozen science, permission/information-barrier envelope, protected-state rules, and the
authorized fresh-debug retry semantics. Unlimited loops do not authorize scientific choices,
regex/identity changes, formal-partition submission, outcome access, destructive action, or
repeated work that no longer makes material progress.

Send exactly one `GATE_C_FINAL_AUDIT` or `BLOCK` to planning thread
`019fb3de-2298-75f2-a083-0dca453ea79c` with visible delivery confirmation.
