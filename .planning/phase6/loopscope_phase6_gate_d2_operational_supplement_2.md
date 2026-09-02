# GATE_D2_OPERATIONAL_SUPPLEMENT_2

Project/phase: LoopScope Phase 6  
Gate: D-2  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor: `019fb8f3-8cb7-7c93-a8a0-bae811735601`  
Purpose: persistent engineering recovery after failed job attempts  

Before the next mutation, submission, or terminal packet, re-read:

- control SHA-256:
  `73e44355ad161cadd64006ac5d7213d825fefb25f6bc7088ebd86ccdb844ca83`;
- protected `AGENTS.md` SHA-256:
  `e4373fd0513a312026004d807869a59b8857b032e06abed5b81a93ad96a5c628`;
- original D-2 handoff and operational supplement 1 SHA-256
  `761ca7a113daf82b46124fb5c83d539d303bbc1319d090093b653d76098f2055`.

This supplement supersedes any interpretation that a failed Slurm/debug/smoke/probe/formal/shard
job automatically terminates Gate D-2. A terminal job is an attempt-level event. The monitor must
pause/delete itself and resume this exact executor; the executor must then diagnose, repair, test,
and retry on a fresh write-once path while a materially progressive low-risk path remains.

## Required persistence

- Continue autonomously through launcher, shell, import, environment, dtype/device, CUDA,
  tokenizer/alignment, runtime, serialization, schema, verifier, scheduler/resource, and other
  engineering failures inside the frozen Gate D-2 contract.
- Executor-owned low-risk repair loops and evidence-preserving retries have no numeric cap.
  Preserve every failed root/job/attempt; never overwrite or disguise it.
- A first failed diagnostic hypothesis, a failed debug/formal job, absent receipts from that failed
  attempt, or an as-yet-unlocalized engineering error is not sufficient for Gate-level `BLOCK`.
  Continue bounded characterization and the next safest material diagnostic.
- After a replacement job is submitted, create or retarget exactly one heartbeat under the frozen
  debug/formal activation and cadence rules. A terminal heartbeat resumes the executor; it does
  not decide the Gate.
- If an engineering fix changes a decision-critical producer/launcher/runtime path, rerun focused
  tests and the required debug preflight before formal. Use a new full root rather than mixing
  incompatible valid shards.

Gate-level `BLOCK` is permitted only when continuation requires changing frozen science,
eligibility/coverage semantics, model/data/generation/TFLT/selector/panel, opening sealed
information, a destructive/unsafe/external authority, an operational permission planning cannot
grant, a preregistered scientific stop branch, or when safe bounded characterization has
established that no low-risk path can make material progress. State which exact boundary is met;
do not report only the latest failed job.

All information barriers, protected-state rules, exact scientific contract, fresh-path semantics,
and Gate E-G locks remain unchanged. Continue from the current legitimate executor-owned Gate D-2
commit/state; do not discard or revert already completed valid work merely because this supplement
arrived mid-turn. Emit exactly one final `GATE_D2_FINAL_AUDIT` after the Gate objectives pass, or a
true Gate-level `BLOCK` under the threshold above.
