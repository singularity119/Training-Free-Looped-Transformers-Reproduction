# GATE_C_HANDOFF

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=C
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a00215-a3a1-7841-9079-e87cf78571e0
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-Full-Acquisition-第7阶段-Gate C
AUTHORIZATION_STATE=SUPERSEDED_REVOKED_AFTER_TERMINAL_DELIVERY_FAILURE
TERMINAL_EVENT=GATE_C_FINAL_AUDIT_OR_BLOCK
GATE_D_AUTHORITY=NONE
GATE_E_AUTHORITY=NONE
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
```

本 handoff 的协作与生命周期必须完整遵守 `research-gate-orchestrator` 的 `SKILL.md` 和
`references/protocol.md`：一 Gate 一独立 executor、最多三个有界 subagents、executor 独占 Git/
remote-write/GPU/Slurm/terminal 权限、正式任务只使用一个 60-minute full heartbeat，并且退出前必须
向 planning 线程投递且确认唯一 `GATE_C_FINAL_AUDIT` 或真正的 `BLOCK`；无法确认时必须使用
`TERMINAL_DELIVERY_UNCONFIRMED` fallback，不得静默结束、直接向用户请求 ordinary operational
permission，或跨 Gate。

## 1. Objective and stop condition

Gate C only acquires, merges, verifies, and seals the complete MMLU validation trajectory population
for the three frozen Base models. It does not run V3, rank windows, plot, generate, loop, access the
test split, or evaluate outcomes.

Send one `GATE_C_FINAL_AUDIT` after all three models independently close at 1,531 identities and 57
subjects with an exact common membership and fresh verifier PASS. If a material blocker cannot be
removed by normal low-risk Gate-scoped repair, send one structured `BLOCK`. Stop after delivery and do
not enter or create Gate D.

## 2. Mandatory protocol and authority check

Before mutation, completely read and follow:

- `research-gate-orchestrator/SKILL.md` and `references/protocol.md`;
- `hpc2-hkustgz-ssh/SKILL.md`;
- `loopscope-tflt/AGENTS.md`, including Phase 7 policy;
- `.planning/phase7/loopscope_phase7_control.md`;
- `.planning/phase7/loopscope_phase7_scientific_contract.md`;
- `.planning/phase7/loopscope_phase7_plan.md`;
- `loopscope-tflt/docs/loopscope_phase7.md`;
- this handoff.

Confirm the exact executor ID, Gate, branch, base, dirty/staged state, remote checkout, environment,
and Gate B accepted commit before any write or submission. A mismatch is a pre-mutation `BLOCK`; do
not reconcile authority or science silently.

## 3. Starting provenance and protected state

```text
local repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
required branch=loopscope
accepted Gate B HEAD=a95772a6bded50ae62e4ffcbb8b0b24c374b7b20
required ancestry base=4f59bd93eca4da3cbf458a93508f91c5b23912bc
remote host=hpc2-hkustgz
remote checkout=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
remote python=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
HF cache=/hpc2hdd/home/xhuang225/shared/hf_home/hub
dataset cache=/hpc2hdd/home/xhuang225/shared/datasets
```

The remote checkout was clean at the accepted commit. Local staged membership was empty. These local
tracked dirty paths are protected pre-existing state:

```text
AGENTS.md
docs/loopscope_phase2.md
docs/loopscope_phase4.md
docs/loopscope_phase5.md
docs/loopscope_phase6.md
scripts/loopscope/run_qwen4_phase4_p4b.py
src/tflt/loopscope/phase4_outcome.py
```

Do not reset, checkout, stash, clean, overwrite, stage, or commit protected paths. Do not use broad
`git add`. Planning files are planning-owned and must not enter executor commits.

## 4. Frozen science and exact membership

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
probe=full rendered current-query Answer: prefix final non-padding token before continuation
runtime=bfloat16,batch_size=1,no quantization,no CPU offload
forward=one native zero-loop use_cache=false forward per identity
choice_surfaces=" A";" B";" C";" D"
```

Build one canonical safe validation manifest in natural task/subject/index order and reuse it exactly
for every model and shard. Persist only the safe identity, subject, task name, and non-sensitive
validation index needed by the renderer. The manifest and trajectory records must not persist or expose
validation gold/target/label/correctness, dev demonstration content, prompt text, input/token IDs,
hidden tensors, full logits/probabilities, test data, or outcome fields. Demonstration answers may be
used only in memory by the standard five-shot renderer.

Raw boundaries, temporary FinalNorm capture, native-logit closure, choice distribution, and the six
scalar metric arrays remain exactly as accepted at Gate B. No candidate domain or V3 computation is a
Gate C action.

## 5. Authorized implementation and Git integration

If the full-population path needs minimal manifest, sharding, merge, launcher, or verifier glue, edits
are authorized only in these Phase 7 paths:

```text
configs/loopscope/phase7_*
docs/loopscope_phase7.md
scripts/loopscope/run_phase7_*.py
scripts/loopscope/verify_phase7_*.py
src/tflt/loopscope/phase7_*.py
tests/test_phase7_*.py
```

Use the smallest focused tests, CLI checks, and `git diff --check` that could change the acquisition
decision. Explicitly stage only authorized files, inspect staged membership, make purpose-specific
commits, and push only an ordinary fast-forward to `origin/loopscope`. The dedicated remote checkout
may only `git pull --ff-only` while clean and on the correct branch/ancestry. Never merge, rebase,
force-push, modify main, or commit planning/protected state.

If no decision-critical path changes, reuse the accepted Gate B three-model preflight. Any change to
producer, renderer, model inputs, probe, hook, serialization, runtime/device binding, verifier, or the
actual shard launch command invalidates the affected preflight and requires a fresh `<30min` debug
preflight for every affected model before formal submission. Pure orchestration that does not alter a
model job's exact command/data path need not trigger redundant smoke.

## 6. HPC2, resources, formal scheduling, and heartbeat

All SSH commands use `BatchMode=yes`, bounded `ConnectTimeout`, and `ClearAllForwardings=yes`. Existing
model snapshots and dataset cache are ready; use cache-only normal paths. Do not inspect, print, copy,
change, or delete credentials, and do not initiate new authentication, repository substitution,
environment installation, or science-changing download.

Before submission, read the current legal Slurm partitions/QoS, GPU inventory, queue, and estimated
time-to-result. Formal jobs must use the highest-priority partition/QoS legally available to the normal
user, then choose a technically adequate GPU using Gate B memory/runtime evidence, current availability,
queue estimate, and total time-to-result. Do not request privileged QoS or lock the plan to A40 when a
different legal resource is better. Shard count and concurrency are operational choices: prefer the
smallest layout that closes promptly, keeps roots/model/shards unambiguous, and avoids avoidable queue
or wall-time risk.

After exact formal job IDs exist, create exactly one executor-owned 60-minute full heartbeat covering
the entire Gate C job set, including PENDING jobs. Do not create one monitor per model or overlapping
monitors. The heartbeat only observes and wakes this executor; it has no Gate continuation authority.
Stop/delete it immediately after every Gate C job is terminal and before the final packet.

Model jobs, logs, manifests, shards, retries, merged records, and receipts must use fresh, write-once,
model-scoped Gate C roots. Preserve failed attempts. Low-risk repair uses a fresh sibling root and
retries only invalid/missing shards when the accepted decision-critical path is unchanged; if that path
changes, rerun the affected preflight and all affected data. Never append to, expand, or overwrite Gate B
or failed Gate C roots.

## 7. Shortest credible execution path

1. Recheck authority, local/remote provenance, environment, snapshots, dataset cache, and whether Gate B
   preflight remains valid.
2. Build and independently validate the safe canonical 1,531/57 manifest before any formal forward.
3. Add only necessary Phase 7 full-population/shard/merge glue; run focused checks and integrate once.
4. If a decision-critical path changed, run fresh per-model one-identity debug preflights and verify
   each in a new root.
5. Freeze per-model shard manifests from the common membership, submit the three formal acquisitions,
   and start one 60-minute heartbeat for their exact job set.
6. As shards finish, verify safe membership/schema/finite values only. Do not inspect partial V3 scores
   or outcomes.
7. Merge each model into a fresh sealed root, run a fresh-process independent verifier, close scheduler
   accounting, stop the heartbeat, and send the terminal packet.

## 8. Must-pass for each model

```text
record_count=1531
unique_identity_count=1531
subject_count=57
forward_count=1531
loop_insertions=0
membership=exact equality to the one canonical validation manifest
missing=0
duplicate=0
extra=0
partial=0
probe=last non-padding token of rendered current-query Answer: prefix
boundary_count=L+1
transition_count=L
final_norm_prehook_calls_per_forward=1
native FinalNorm/logit closure=true
four choice surfaces single-token and distinct
all six scalar arrays finite with exact lengths
forbidden persisted fields absent
fresh_process_data_verifier=PASS
```

All three independent seals must pass. A valid seal contains trajectory data and safe provenance only;
it does not contain selector, ranking, plot, test, gold, correctness, generation, loop, or outcome data.

## 9. Repair boundary, subagents, and agility budget

Job failure is attempt-level, not automatically Gate-level. Diagnose ordinary launcher, environment,
OOM, scheduling, serialization, shard, merge, or verifier errors and use bounded fresh retries. A change
to model membership/revision, dataset/split/population, renderer/few-shot content, prompt, probe, dtype,
metric definition, information barrier, or any outcome access is a science/authority change and must
`BLOCK` for planning; do not improvise a fallback.

The executor may use at most three simultaneously active subagents only when they materially shorten
this Gate. Give each a narrow Gate C task. Subagents may inspect code, run focused local tests, or extract
read-only authorized evidence; they may not independently perform Git integration, remote writes,
GPU/Slurm/job control, cross-Gate work, or terminal delivery. The bound executor owns integration,
external actions, and the single terminal event.

```text
earliest real experiment=formal acquisition after a still-valid or refreshed per-model preflight
minimum decisive checks=1531/57 exact membership,L+1/L shapes,finite sanitized scalar schema,fresh verifier
deferred=V3/bootstrap,ranking,plots,test/outcome,loop/generation,extra models,prompt variants,general refactor
deletion test=remove any check that cannot change the three data seals or Gate D admission
```

## 10. Terminal delivery

Send exactly one structured `GATE_C_FINAL_AUDIT` or `BLOCK` to planning task
`01a0013e-71c3-7c90-a547-4059b462dc7e`. Include:

- final local/remote branch, commit, ancestry, dirty/staged and protected-state confirmation;
- any Phase 7 commits, focused checks, push, and remote fast-forward;
- exact common manifest path and safe membership summary;
- model/revision/snapshot, shard manifests, run roots, merged records, and verifier receipts;
- Slurm job IDs, legal partition/QoS, GPUs, requested resources, states, exits, runtimes, and the resource
  choice rationale;
- exact command classes and exit codes, repair/retry history, final per-model 1,531/57/forward/shape/
  finite/barrier results, and heartbeat removal;
- an explicit request for `PASS` or a narrow blocker.

Do not include credentials, prompts, token IDs, gold/labels/correctness, full tensors/logits, test/outcome,
or digest evidence. After terminal delivery stop; Gate D authority is `NONE`.
