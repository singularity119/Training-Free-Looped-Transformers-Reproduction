# LoopScope Phase 7 Gate C Executor Recovery Decision

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=C
RECOVERY_CLASS=EXECUTOR_TERMINAL_DELIVERY_AND_CONTINUATION_ROUTING_FAILURE
SUPERSEDED_EXECUTOR=01a00215-a3a1-7841-9079-e87cf78571e0
SUPERSEDED_TITLE=execute-LoopScope-Full-Acquisition-第7阶段-Gate C
REPLACEMENT_EXECUTOR=01a0036f-fc1d-7412-b043-b93b5f088398
REPLACEMENT_TITLE=execute-LoopScope-Full-Acquisition-第7阶段-Gate C
MODEL_CONFIGURATION=INHERIT_CODEX_CONFIG_DEFAULT_NO_OVERRIDE
GATE_C_SCIENCE_STATE=UNCHANGED
GATE_D_AUTHORITY=NONE
```

## Incident

The superseded Gate C executor completed the authorized Phase 7 full-acquisition glue commit and
three-model debug preflight, then encountered a formal-submission approval rejection that compared
the accepted Gate B base commit with the newer Gate C commit. It ended by asking the user to approve
continuation directly. It did not send `GATE_C_FINAL_AUDIT`, a genuine `BLOCK`, or
`TERMINAL_DELIVERY_UNCONFIRMED` to the planning thread and did not retain visible terminal-delivery
confirmation.

That is a material protocol failure under `research-gate-orchestrator`: ordinary operational
permission and continuation routing belong to the planning thread, and an executor must not become
idle without the required terminal route. The old executor is therefore revoked and retained as
visible read-only provenance under project policy. It has no current Gate authority.

## Preserved valid work

The protocol failure does not invalidate the already completed normal-path engineering evidence:

- local, `origin/loopscope`, and the clean HPC2 checkout all resolve to
  `feb8a9f67546db22658d2d62053d8b69b1c0b286`;
- that commit contains only the five Phase 7 manifest/shard/merge/verifier paths reported by the old
  executor;
- common validation manifest and two-shard-per-model layouts exist under the preserved Gate C root
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase7-gate-c-20260814T212327Z`;
- debug jobs `10217835`, `10217836`, and `10217837` are `COMPLETED 0:0` on the accepted Gate C commit,
  with elapsed times 29, 30, and 20 seconds;
- no formal shard job was submitted and no formal record was produced.

The replacement executor must cheaply re-read these facts before use but must not repeat the passing
preflight unless a decision-critical commit, launcher, environment, renderer, producer, serialization,
runtime, or verifier path has changed.

## Recovery decision

Gate C remains open and scientifically unchanged. The current authority is rebound to independent
executor `01a0036f-fc1d-7412-b043-b93b5f088398` under
`.planning/phase7/loopscope_phase7_gate_c_recovery_handoff_2.md`. Its canonical title remains
`execute-LoopScope-Full-Acquisition-第7阶段-Gate C`; replacement iteration is represented only by the
exact thread ID and recovery artifact filename, never by changing the Gate field. Its model/reasoning
settings inherit the user's Codex configuration without override.

The replacement continues from `feb8a9f...`, submits the still-unsubmitted formal six-shard acquisition,
uses the single required 60-minute full heartbeat after verified job IDs exist, merges and verifies the
three 1,531/57 seals, and sends exactly one visibly confirmed terminal event to planning. It must route
ordinary operational blockers to planning rather than ending with a direct user approval request.
