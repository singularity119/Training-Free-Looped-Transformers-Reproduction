# GATE_D2_HANDOFF

Project/phase: LoopScope Phase 6  
Gate: D-2 — eligibility-aware pre-answer trajectory recovery  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor: `019fb8f3-8cb7-7c93-a8a0-bae811735601`  
Executor title: `execute-LoopScope-PreAnswer-RBRV2-第6阶段-Gate D-2`  
Decision requested: `PASS / PASS_WITH_FIXES / BLOCK`

## Authority and admission

The user approved Phase 6 Scientific Amendment 2 on 2026-08-01. Gate D was planning-accepted
`BLOCK` after Job `10105466` produced zero valid shards: one exact-regex zero-match and seven
generated-ID/text alignment failures. The former Gate D executor is revoked. Gate D-2 is a new
independent recovery Gate; Gate E...G remain locked.

Before any mutation, re-read:

- policy: `loopscope-tflt/AGENTS.md`, expected SHA-256
  `e4373fd0513a312026004d807869a59b8857b032e06abed5b81a93ad96a5c628`;
- control: `.planning/loopscope_phase6_control.md`, expected SHA-256
  `f7a68e2b264a18ba4298217de00f94645dd0a4828933ddfe85c1d7b1ce98f4b7`;
- this handoff and the timeless `loopscope-tflt/docs/loopscope_phase6.md` runbook.

Admission requires exact executor/title/Gate agreement, local repo `loopscope-tflt`, branch
`loopscope`, local HEAD and `origin/loopscope`
`3da3c075a034660bf479f16c05aefa0bbcb1ddf2`, required Phase 6 ancestry, and starting dirty state
exactly the protected unstaged `M AGENTS.md`. The HPC2 dedicated clone must be clean before
fast-forward synchronization. Do not stage, commit, overwrite, restore, stash, or otherwise alter
the protected `AGENTS.md` planning change.

## Frozen Scientific Amendment 2

All model, tokenizer, dataset, renderer, generation, TFLT, boundary, metric, V2 selector,
candidate, seed, panel, and information-barrier contracts remain unchanged except this exact
anchor-coverage amendment:

1. Primary anchor remains literal case-sensitive
   `answer is \(?([ABCDEFGHIJ])\)?`, capture group 1, all matches in generated-text order,
   selected ordinal `0`, aligned with lm-eval `custom-extract -> take_first`.
2. One or more exact matches means `ANCHOR_ELIGIBLE`. The selected answer token must be mapped
   from actual generated IDs and the selected span; replay is prompt plus generated IDs strictly
   before that token.
3. Zero exact matches, or an exact match at the first generated token such that no preceding
   generated token exists, means `ANCHOR_NOT_EXPRESSED`. Keep the identity in the exact canonical
   test-12032 manifest and sealed baseline membership, but perform no replay and produce no
   trajectory for it.
4. No regex broadening, case folding, alternate cue parser, prompt/EOS/final-token fallback,
   identity removal/replacement, or fabricated anchor is allowed.
5. Freeze exactly one canonical-order eligibility mask before selector execution. Require
   `eligible_count / 12032 >= 0.995` and each of the 14 frozen category coverages `>= 0.98`.
   Falling below either floor is `BLOCK_ANCHOR_COVERAGE_BELOW_FLOOR`.
6. If an exact match exists, alignment/replay/runtime failure is an engineering defect, not an
   eligibility exclusion. It must fail closed and be root-cause repaired; the mask must not absorb
   it.
7. Every identity remains in sealed baseline and later full-population outcome. Gate E will read
   only eligible sanitized trajectories plus the frozen coverage receipt. Human claims must say
   near-complete eligible trajectory subset, never exact-12032 trajectory coverage.

`ANCHOR_NOT_EXPRESSED` records may contain only canonical identity/category, eligibility state,
generation count, sealed-payload membership/hash reference, and frozen provenance. They must not
contain completion text, answer letter/content, token IDs, span values, gold, label, correctness,
outcome, logits, probabilities, or hidden tensors. Eligible records retain the existing closed
sanitized trajectory schema.

## Smallest authorized implementation

Modify only the Phase 6 paths materially affected by the amendment and alignment root cause:

- `configs/loopscope/phase6_*` card/schema;
- `src/tflt/loopscope/phase6_anchor.py`, `phase6_runtime.py`, `phase6_schema.py`,
  `phase6_verifier.py`, and the smallest directly required Phase 6 helper;
- `scripts/loopscope/run_qwen4_phase6_gate_d.py` and Phase 6 verification/preparation CLI only as
  required;
- focused `tests/test_loopscope_phase6_*`;
- `docs/loopscope_phase6.md` timeless contract/runbook wording.

Do not build a general tokenizer-alignment framework or refactor Phase 4/5. Repair the normal
Qwen3 tokenizer path so it does not assume every intermediate `decode(ids[:k])` must be an exact
string prefix when that property is not guaranteed. The replacement must still prove an exact
full generated-ID/text closure and a unique mapping of the selected answer span to a generated
token. Special-token behavior must match the actual frozen generation decode path; later special
tokens must not invalidate an already resolved answer unless they break full closure.

The preserved failed Gate D root may be processed only by bounded outcome-blind diagnostic code
that emits sanitized exception class/count/identity-hash/coverage facts. Neither executor nor logs
may display or persist completion text, answer content, token IDs, prompt content, logits, hidden
tensors, gold, correctness, or outcome. Old partial shards are diagnostic evidence only and may
never be reused as valid Gate D-2 science.

## Execution sequence

1. Run admission and add focused red/characterization tests for zero-match masking, coverage
   floors, expressed-vs-not-expressed schema closure, and the seven alignment normal-path classes.
2. Implement the smallest card/schema/anchor/runtime/verifier/runbook change. Run only targeted
   Phase 6 tests, compile/help/dry-run, and diff checks.
3. Commit/push explicit authorized paths; fast-forward the clean HPC2 clone to the exact commit;
   rerun focused remote CPU/import tests.
4. Run one fresh `<30min` `debug` A40 end-to-end preflight at the same commit/launcher/env/runtime.
   It must cover real model/data/CUDA, at least one deterministic previously failing
   `ANCHOR_NOT_EXPRESSED` identity, representative eligible identities, sealed/sanitized
   separation, eligibility receipt, trajectory writer, and fresh verifier. A completed job before
   about 60 seconds needs no heartbeat; otherwise create the debug heartbeat only after stable
   RUNNING for about 60 seconds.
5. Only after debug PASS, freeze a new full write-once root, exact canonical population, shard
   manifest, and coverage contract. Submit one formal array using the highest-priority legal
   normal-user partition/QoS within the prior envelope: at most 8 concurrent GPUs, 8 CPU, 64 GiB,
   and 6 hours per task. The old Gate D root remains immutable and is not reused.
6. Create exactly one executor-owned `full` heartbeat at 60-minute cadence immediately after a
   verified formal job ID exists, including while PENDING. On terminal evidence, pause/delete it
   and trigger this exact executor with `AUTOMATION_TERMINAL_RESUME`; relay planning only if exact
   continuation delivery is unavailable.
7. Preserve valid fresh shards. Low-risk scheduler/engineering recovery may rerun only invalid or
   missing shards on fresh attempt paths when producer semantics and commit remain compatible.
   If decision-critical producer semantics change after any valid shard, use a new full root; do
   not mix commits. Merge once in Gate B canonical order and run one fresh-process verifier.

## Must-pass evidence

- exact 12,032 canonical identities/categories/order in source and sealed membership;
- `eligible + not_expressed = 12032`, missing/extra/duplicate zero;
- overall eligible coverage `>=0.995`; every category coverage `>=0.98`;
- no exact-match identity excluded because of alignment/replay/runtime failure;
- eligible: selected ordinal zero, one generation and one replay, zero loop insertions, replay
  next-token closure, 37 raw boundaries, one FinalNorm pre-hook, full-vocabulary entropy/KL,
  hidden diagnostics, 36 adjacent-angular transitions, finite/range/endpoint checks;
- not-expressed: one generation, zero replay, zero trajectory, no fallback, sealed membership
  retained;
- canonical frozen eligibility receipt, shard receipts, merged sanitized manifest, opaque sealed
  membership, scheduler receipt, and fresh verifier receipt are immutable and hash-closed;
- no generated payload, target/gold/label, correctness, accuracy, gain, Phase 4/6 outcome,
  selector, or Gate E action is exposed or executed.

## Repair, agility, and stop rules

Executor-owned low-risk engineering repair loops are unlimited inside this Gate, frozen science,
paths, information barrier, scheduler envelope, protected state, and fresh-retry semantics. The
executor may use up to three concurrent bounded subagents for isolated implementation/tests or
read-only evidence extraction; only the bound executor may integrate, commit/push, write remotely,
submit/control Slurm, own automation, or send the terminal packet.

Do not change regex/ordinal, eligibility definitions or floors, model/data/generation/TFLT,
metrics, candidates, selector/panel, or information access. These require new planning/user
authority. Stop hardening after focused tests, debug preflight, complete full acquisition, merge,
and fresh verifier pass.

Send exactly one `GATE_D2_FINAL_AUDIT` or `BLOCK` to planning thread
`019fb3de-2298-75f2-a083-0dca453ea79c` with visible delivery confirmation. Gate E...G remain
locked.
