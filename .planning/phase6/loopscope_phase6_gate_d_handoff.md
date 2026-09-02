# GATE_D_HANDOFF

Project/phase: LoopScope Phase 6  
Gate: D — test-12032 outcome-blind pre-answer trajectory acquisition  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor: `019fb6d3-9485-7f33-90c1-ece73c3863bc`  
Executor title: `execute-LoopScope-PreAnswer-RBRV2-第6阶段-Gate D`  
Decision requested: `PASS / PASS_WITH_FIXES / BLOCK`

## Objective and admission

Gate C is planning-accepted `PASS` at commit
`09fb90762df1369d4768426e1cd1d6e65563b218`. Implement the smallest sharded Gate D
producer/launcher/merge/verifier layer, then acquire exactly the Gate B frozen MMLU-Pro test
population of 12,032 identities using one native no-loop generation and one pre-answer replay per
identity. Persist selector-safe scalar trajectories and a separate sealed baseline completion
namespace without reading its content.

Before mutation, re-read the live control, this handoff, `loopscope-tflt/AGENTS.md`, and the
timeless runbook. Verify executor/thread binding, repository/branch, local HEAD and
`origin/loopscope`, required base ancestry, remote clean checkout, and the sole protected local
dirty path `M AGENTS.md`.

## Frozen provenance

- Local repo: `loopscope-tflt`, branch `loopscope`, base
  `09fb90762df1369d4768426e1cd1d6e65563b218`.
- HPC2 clone:
  `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`.
- Venv:
  `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711`.
- Gate B root:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-b-provenance-20260730T183107Z`.
- Gate B manifest SHA-256:
  `ccb148bd61971d24ad81b240cbe1b5f0810e06e620fb568473a61b2147b44a1d`.
- Exact population: 12,032 unique identities in Gate B canonical order, ordered identity SHA-256
  `ac52d6e43c693bd0b47b567c2955dae0c6be895ce3230e854d4e1538f05503a5`.
- Gate C passing preflight evidence remains immutable under root
  `phase6-gate-c-smoke-lmhead-dtype-repair-20260731T061226Z`; it proves the scientific runtime,
  but it does not waive the new Gate D serialization/shard/write/merge preflight.

Model, tokenizer, dataset, renderer, generation, evaluator-aligned ordinal-0 anchor, replay,
raw-B0...B36, entropy/KL, hidden diagnostics, TFLT, candidate/selector, and information-barrier
contracts remain exactly frozen by the card/control. No science change is authorized.

## Smallest implementation

Add only the Gate D orchestration needed for:

1. deterministic population and shard-manifest freeze from the exact Gate B canonical order;
2. a representative debug acquisition using the same Gate D record writer and verifier path;
3. per-shard native generation/replay with separate write-once sanitized and sealed-baseline
   outputs;
4. canonical-order merge, completeness/identity/hash closure, and a fresh-process independent
   verifier.

Prefer a new Gate D runner/module and focused tests. Reuse Gate C runtime, Gate B identity
projection, Phase 6 closed schemas, and existing write-once helpers. Do not refactor a general
framework, re-audit Gate A/B/C, or alter Phase 4/5 code. `AGENTS.md` is protected and must never
be staged or committed.

## Persistence and information barrier

Sanitized records may contain only the closed Phase 6 trajectory schema, including match count,
selected ordinal zero, span/token offsets and hashes, H/D arrays, diagnostic arrays, counts, and
provenance. They must not contain prompt/generated text, token IDs, answer content/prediction,
gold/label/correctness/accuracy/gain/flip/outcome, logits/probabilities, or hidden tensors.

The sealed baseline namespace may store identity-bound raw generated completion payloads needed
for Gate G, but no gold/label/correctness/outcome. The executor, tests, merge, verifier, logs, and
planning packet must treat payloads as opaque bytes: only membership, byte count, and hashes may
be inspected. Gate E will receive only sanitized trajectories and their manifest.

Any zero-match, alignment/replay mismatch, non-finite/endpoint failure, forbidden field,
population drift, duplicate/missing identity, or cross-namespace leakage is material BLOCK. Never
drop, replace, or complete-case filter an identity.

## Execution sequence and resources

1. Implement focused Gate D paths and targeted tests; run compile/help/dry-plan and
   `git diff --check`.
2. Explicit-path commit/push; fast-forward the clean HPC2 clone to the exact commit and rerun
   focused remote checks.
3. Use a fresh write-once debug root and one `<30min` `debug` A40 job covering real model/data
   load, CUDA, the new Gate D writer, sealed/sanitized namespace separation, and fresh-process
   verification on the four Gate B smoke identities. No formal job may be submitted before PASS.
4. Freeze one formal run root and deterministic shard manifest before submission. Partition the
   canonical order by frozen ordinal modulo shard count and merge back to exact Gate B order.
   Choose and record 8–32 shards and at most 8 concurrent GPUs from measured debug throughput;
   use A40/A800 and the highest-priority partition/QoS legally available to this normal user to
   minimize time-to-result, with at most 8 CPU, 64 GiB, and 6 hours per task. This does not
   authorize administrator-only priority, invented QoS, or bypass of cluster policy.
5. Submit one formal shard array at the exact preflight-passing commit/launcher/env/runtime.
   Preserve valid shards. Scheduler/transient or low-risk engineering failures may be repaired
   without a numeric loop cap; rerun only missing/invalid shards on fresh attempt paths. If
   scientific producer semantics or commit changes after any valid formal shard, do not mix
   commits—start a new full write-once root.
6. After all shards are terminal, stop monitoring, merge once in canonical order, then run one
   fresh-process verifier over the merged sanitized records, opaque sealed membership, receipts,
   source closure, scheduler evidence, and exact Git commit.

For a `debug`-partition task, enable a heartbeat only after it is genuinely `RUNNING` for about
60 seconds with no launcher/startup error; a debug job that terminates sooner needs none. For the
formal non-debug array, enable exactly one executor-owned `full` heartbeat at 60-minute cadence
immediately after a verified real job ID exists, including while `PENDING`. The same heartbeat
observes queue-to-running transition and subsequent execution; never create a second monitor.
Each wake is one bounded read-only scheduler/root-membership check. On terminal or material
blocker, pause/delete it and trigger this exact executor through `AUTOMATION_TERMINAL_RESUME`;
if delivery fails, relay planning.

Operational supplement, 2026-07-31: formal Job `10105406` currently has 8/8 tasks
`PENDING (Priority)` under run root
`phase6-gate-d-preanswer-trajectory-20260731T064903Z`. Create its unique 60-minute full heartbeat
now. The executor may inspect the legal partition/QoS options and move this exact frozen formal
array to the highest-priority option available to the normal user when that materially improves
admission. Before changing queues, confirm the old job has not started; preserve its job/root
evidence, avoid duplicate live compute, and use a fresh write-once root for any resubmission.
No scientific, manifest, commit, shard-membership, resource-envelope, or information-barrier
change is authorized.

## Acceptance

- Exact 12,032 canonical identities/categories/order; missing/extra/duplicate zero.
- Exactly one generation and replay per identity; zero loop insertions.
- Every identity has at least one exact-regex match and selected ordinal zero; zero unresolved.
- Replay next-token closure, 37 raw boundaries, 36 angular transitions, one FinalNorm pre-hook,
  finite/range/endpoint and schema checks all pass.
- Sanitized trajectory, opaque sealed baseline membership, shard manifests/receipts, merged
  manifest, scheduler receipt, and fresh verifier receipt are immutable and hash-closed.
- No protected target/outcome access, selector execution, Gate E work, artifact overwrite, or
  Phase 4/5 modification occurred.

## Agility, repair, and terminal

Executor-owned low-risk engineering repair loops are unlimited within this Gate, frozen science,
permissions, protected state, information barrier, and fresh-retry rules. Stop once the focused
implementation, debug preflight, exact full acquisition, merge, and verifier pass; do not add
optional hardening or run an unrelated full suite.

The executor may use up to three concurrent bounded subagents for isolated implementation/tests
or read-only evidence extraction. Only the bound executor may perform Git integration, remote
writes, GPU/Slurm actions, monitor lifecycle, and terminal delivery.

Send exactly one `GATE_D_FINAL_AUDIT` or `BLOCK` to planning thread
`019fb3de-2298-75f2-a083-0dca453ea79c` with visible delivery confirmation. Gate E...G remain
locked.
