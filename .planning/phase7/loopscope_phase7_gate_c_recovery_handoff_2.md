# GATE_C_RECOVERY_HANDOFF_2

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=C
RECOVERY_ITERATION=2
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a0036f-fc1d-7412-b043-b93b5f088398
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-Full-Acquisition-第7阶段-Gate C
SUPERSEDED_EXECUTOR_THREAD=01a00215-a3a1-7841-9079-e87cf78571e0
AUTHORIZATION_STATE=AUTHORIZED
TERMINAL_EVENT=GATE_C_FINAL_AUDIT_OR_BLOCK
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
MODEL_CONFIGURATION=INHERIT_CODEX_CONFIG_DEFAULT_NO_OVERRIDE
GATE_D_AUTHORITY=NONE
GATE_E_AUTHORITY=NONE
```

## 1. Mandatory collaboration protocol

This handoff and every action under it must follow the complete collaboration and lifecycle protocol in
`/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md` and its
`references/protocol.md`. This requirement is not merely a source citation:

- one independent executor owns only this Gate;
- the executor may use at most three concurrently active, narrowly bounded subagents, but remains the
  sole owner of integration, remote writes, Git, GPU/Slurm, heartbeat lifecycle, and terminal delivery;
- attempt-level scheduler or engineering failure is diagnosed and repaired inside this frozen Gate when
  a low-risk path remains; it is not automatically a Gate-level `BLOCK`;
- the formal job set uses exactly one executor-owned 60-minute full heartbeat, including while PENDING;
- the executor must send exactly one `GATE_C_FINAL_AUDIT` or genuine `BLOCK` to the planning thread and
  retain visible delivery confirmation before it stops;
- if delivery cannot be confirmed, it must emit the complete paste-ready packet as
  `TERMINAL_DELIVERY_UNCONFIRMED`; it must not silently end, ask the user directly for an ordinary
  operational permission, or leave planning uninformed.

The old Gate C executor violated the final delivery rule and is revoked but retained as visible read-only
evidence only. Do not message, resume, or delegate to it.

## 2. Objective and non-objectives

Continue the already-started Gate C normal path from the accepted Gate C commit and passing preflights.
Submit, acquire, merge, independently verify, and seal the complete common MMLU validation population
for all three frozen Base models.

Gate C ends only when all three models close at 1,531 unique identities / 57 subjects with exact common
membership and fresh verifier PASS, or when a genuine Gate-level blocker is visibly delivered to
planning. Do not run V3, score or rank windows, draw figures, generate, execute loops, access the test
split, read validation gold/correctness, evaluate outcomes, or enter Gate D.

## 3. Authority and source order

Before mutation, completely re-read:

- the `research-gate-orchestrator` skill and protocol named above;
- `/Users/huangxutao/.codex/skills/hpc2-hkustgz-ssh/SKILL.md`;
- `loopscope-tflt/AGENTS.md`, including Phase 7 policy;
- `.planning/phase7/loopscope_phase7_control.md`;
- `.planning/phase7/loopscope_phase7_scientific_contract.md`;
- `.planning/phase7/loopscope_phase7_plan.md`;
- `.planning/phase7/loopscope_phase7_gate_c_executor_recovery_decision_20260815.md`;
- `loopscope-tflt/docs/loopscope_phase7.md`;
- this recovery handoff.

Authority order is the user's latest instruction, live Git/HPC facts, `AGENTS.md`, current control,
this handoff/contract/plan/runbook, then history. Confirm the exact bound executor ID before any action.
If control and handoff disagree, stop before mutation and deliver the conflict to planning.

## 4. Accepted recovery starting state

```text
local repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
required branch=loopscope
current authorized Gate C commit=feb8a9f67546db22658d2d62053d8b69b1c0b286
accepted Gate B ancestor=a95772a6bded50ae62e4ffcbb8b0b24c374b7b20
required ancestry base=4f59bd93eca4da3cbf458a93508f91c5b23912bc
origin/loopscope=feb8a9f67546db22658d2d62053d8b69b1c0b286
remote host=hpc2-hkustgz
remote checkout=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
remote HEAD=feb8a9f67546db22658d2d62053d8b69b1c0b286
remote worktree=clean
remote python=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
HF cache=/hpc2hdd/home/xhuang225/shared/hf_home/hub
dataset cache=/hpc2hdd/home/xhuang225/shared/datasets
preserved Gate C root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase7-gate-c-20260814T212327Z
```

Commit `feb8a9f...` is the authorized current Gate C commit, not a provenance mismatch. It contains
only the Gate C manifest, merge, membership-verifier, and targeted-test paths:

```text
scripts/loopscope/run_phase7_build_validation_manifest.py
scripts/loopscope/run_phase7_merge_trajectory.py
scripts/loopscope/verify_phase7_base_trajectory.py
src/tflt/loopscope/phase7_manifest.py
tests/test_phase7_manifest.py
```

Local staged membership is empty. Preserve these pre-existing tracked dirty paths exactly:

```text
AGENTS.md
docs/loopscope_phase2.md
docs/loopscope_phase4.md
docs/loopscope_phase5.md
docs/loopscope_phase6.md
scripts/loopscope/run_qwen4_phase4_p4b.py
src/tflt/loopscope/phase4_outcome.py
```

Never reset, checkout, stash, clean, overwrite, stage, or commit them. Do not use broad `git add`.

## 5. Preserved manifest and preflight evidence

The existing Gate C root already contains:

- `manifests/common_validation_manifest.json` with the common safe 1,531/57 membership;
- two contiguous manifest shards for each of `qwen25_3b`, `llama32_3b`, and `gemma2_2b`;
- fresh one-identity preflight trajectories and PASS receipts for all three models.

Accepted debug jobs on commit `feb8a9f...` are:

```text
10217835 COMPLETED 0:0 elapsed=00:00:29
10217836 COMPLETED 0:0 elapsed=00:00:30
10217837 COMPLETED 0:0 elapsed=00:00:20
partition=debug
node=gpu3-9
```

No formal shard job was submitted and no formal result exists. Re-read the manifest/receipt summaries,
Git state, and scheduler records cheaply before reuse. Do not repeat the preflight unless the exact
commit, launcher, environment, renderer, model/runtime path, producer, hook, serialization, or verifier
has changed. Do not overwrite or expand the preflight directories.

## 6. Frozen science and information barrier

```text
models=Qwen/Qwen2.5-3B;meta-llama/Llama-3.2-3B;google/gemma-2-2b
revisions=3aab1f1954e9cc14eb9509a215f9e5ca08227a9b;13afe5124825b4f3751f836b40dafda64c1ed062;c5ebcd40d208330abc697524c919956e692655cf
layer_counts=36;28;26
dataset=cais/mmlu
split=validation
population=1531 identities / 57 subjects
num_fewshot=5
fewshot_split=dev
renderer=lm-eval 0.4.11 compatible standard plain non-chat MMLU
probe=rendered current-query Answer: prefix final non-padding token before continuation
runtime=bfloat16,batch_size=1,no quantization,no CPU offload
forward=one native zero-loop use_cache=false forward per identity
choice_surfaces=" A";" B";" C";" D"
```

Use exactly the preserved common manifest and shard membership for all models. Persist only safe
identity/subject/task/index and the sanitized trajectory schema. Never persist or expose validation
gold/target/label/correctness, demonstration content, prompt text, input/token IDs, hidden tensors, full
logits/probabilities, test data, or outcome. Demonstration answers are renderer-only in-memory inputs.

## 7. Authorized actions and operational authority

The bound replacement executor is authorized to:

- perform bounded read-only local/HPC drift checks;
- use the current clean `feb8a9f...` checkout and existing caches/manifests/preflights;
- write formal Gate C launch scripts/logs and fresh write-once model/shard/merge/receipt artifacts under
  the preserved Gate C root or a clearly named fresh sibling recovery root;
- inspect current legal partitions/QoS/GPU queues and submit, monitor, cancel, or evidence-preservingly
  retry only Gate C jobs within the frozen science;
- perform minimum low-risk launcher, environment, resource, serialization, merge, or verifier repair in
  authorized Phase 7 paths, with targeted tests and a fresh purpose-specific commit when code changes;
- fast-forward `origin/loopscope` and the clean dedicated remote checkout when needed.

Formal jobs use the highest-priority partition/QoS legally available to the normal user and a technically
adequate GPU chosen from memory fit, runtime, availability, queue estimate, and time-to-result. No
privileged QoS or science change is authorized. All engineering/debug work remains on `debug`.

The user has delegated ordinary Gate-scoped operational permission decisions to planning. If a tool or
approval layer rejects an authorized normal-path action, preserve the rejection and route a concise
operational blocker to planning task `01a0013e-71c3-7c90-a547-4059b462dc7e`; do not ask the user directly
and become idle. Continue all other safe diagnostics. Planning will issue a supplement if necessary.

## 8. Ordered execution path

1. Confirm exact executor binding, `feb8a9f...` local/origin/remote closure, protected dirty state, safe
   manifest counts, three preflight receipts, and scheduler completion.
2. Refresh legal formal resource/queue information and freeze the six formal shard commands and fresh
   output/log paths without changing membership or science.
3. Submit the two shards for each of three models. Verify real job IDs, then immediately create exactly
   one executor-owned 60-minute full heartbeat covering all six jobs, including PENDING.
4. The heartbeat performs one bounded read-only status check per wakeup. On any terminal attempt it
   pauses/deletes itself and visibly sends `AUTOMATION_TERMINAL_RESUME` to this exact executor; if that
   delivery cannot be confirmed it sends `AUTOMATION_RELAY_REQUIRED` to planning and emits the packet.
5. Treat failed jobs as attempt events: characterize and use fresh write-once retry paths for only
   invalid/missing work when the frozen decision-critical path is unchanged. Re-preflight affected
   models after any decision-critical code/runtime change.
6. After all shards close, merge each model once into a fresh sealed record, run fresh-process membership/
   schema/finite-value verification, close scheduler accounting, and delete the heartbeat.
7. Deliver exactly one terminal packet to planning and stop. Do not begin V3 or Gate D.

## 9. Must-pass per model

```text
record_count=1531
unique_identity_count=1531
subject_count=57
forward_count=1531
loop_insertions=0
membership=exact equality to common_validation_manifest.json
missing=0
duplicate=0
extra=0
partial=0
boundary_count=L+1
transition_count=L
final_norm_prehook_calls_per_forward=1
native FinalNorm/logit closure=true
four choice surfaces single-token and distinct
all six scalar arrays finite with exact lengths
forbidden persisted fields absent
fresh_process_data_verifier=PASS
```

All three seals must pass. Gate C artifacts remain trajectory-only and outcome-blind.

## 10. Repair, subagents, and agility budget

Low-risk Gate-scoped engineering repair and fresh retry may continue while science, permissions,
protected state, and information barriers remain unchanged. A model/data/split/membership/prompt/probe/
dtype/metric change, information unsealing, destructive action, or cross-Gate action requires planning.

At most three active subagents may be used for concrete bounded tasks that materially shorten Gate C.
They may inspect code, run focused local tests, or extract authorized read-only evidence. They may not
perform Git integration, remote writes, GPU/Slurm/job control, scientific changes, terminal delivery,
or spawn further subagents. The bound executor reviews and owns every result.

```text
smallest remaining increment=submit six already-preflighted formal shards, merge, and verify
minimum decisive checks=1531/57 exact membership,L+1/L shape,finite sanitized scalar schema,fresh verifier
deferred=V3/bootstrap,ranking,plots,test/outcome,loop/generation,extra models,general refactor
stop rule=three complete sealed PASS receipts -> terminal delivery; add no optional hardening
```

## 11. Terminal delivery requirement

Send exactly one structured `GATE_C_FINAL_AUDIT` or genuine `BLOCK` to planning task
`01a0013e-71c3-7c90-a547-4059b462dc7e` and retain the successful delivery response. Include executor
identity, final local/origin/remote state, protected state, commits/files/tests, manifest and shard roots,
Slurm jobs/resources/exits/runtimes, heartbeat lifecycle, per-model seals and fresh receipts, deviations/
repairs, actions deliberately not taken, and the requested planning decision.

Do not include credentials, prompts, token IDs, gold/correctness, full tensors/logits, test/outcome, or
digest evidence. If direct delivery is unavailable or unconfirmed, output the same complete packet as
`TERMINAL_DELIVERY_UNCONFIRMED` with the intended planning thread ID and failed mechanism. Ending with
only an approval question, local final answer, or “ready to resume” is forbidden.
