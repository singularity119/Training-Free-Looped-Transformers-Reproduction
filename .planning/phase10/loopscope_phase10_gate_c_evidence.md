# Phase10 Gate C execution evidence

Executor `01a0fba5-4ee0-7262-83d4-5bde515f3a94`; planning `01a0fae2-12df-74e0-a874-190518ef0501`.
Current state: batch1/2 full-cell closures complete; budget-admitted batch3 submitted and monitored.
No target gold/outcome access.

## Source and implementation

Starting branch `loopscope`, commit `cc623e5e2cec22a56c6fa211574c9cd8de9e3b94`.
Initial dirty four planning documents were preserved and committed unchanged as `c02e1d0`.
Gate C source `2644034a9d9c806099693e810c1fada9d73cd750`, pushed to origin/loopscope.
Protected ancestor check passed. Phase8/9 and wrapper/strategy/cache/config/eval_runner unchanged.

Three bounded workers owned shard implementation, gold-free input planning, and shard preflight;
parent reviewed/integrated, committed/deployed/submitted. No worker remote/GPU/Git mutation.

Producer supports explicit scheduling shards from the full frozen pool/manifest, with exact canonical
indices in metadata and records. Verifier checks shard identities, configuration, source/runtime,
overlap and exact full-cell/full-panel union; merged scores use canonical ordering. Separate
label-free telemetry records per-row CUDA-synchronized wall time and context length.
CPU preparation freezes all59 new cells and full14042 identities; seven lambda0.1 canaries each64
at fixed length ranks,448 total, remaining828030. Historical9 references are unchanged.
Cost estimator uses larger adjacent measured row costs by context length, model-loading cost per
planned shard and the handoff's1.2 reserve. This is an operational estimate, not a result claim.

## Local validation

All commands below exit0 with `PYTHONPATH=src` and bundled Python
`/Users/huangxutao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`:

- `-m unittest discover -s tests -p test_phase10_shard*.py`:10 tests.
- `-m unittest discover -s tests -p test_phase10_gate_c_plan.py`:6 tests.
- `-m unittest discover -s tests -p test_phase10_accuracy.py`:7 tests.
- `-m unittest discover -s tests -p test_phase10_analysis.py`:5 tests.
- Producer/verifier/preparation/preflight/cost CLI help; focused cost envelope checks.
- `git diff --check`; `bash -n` accuracy and shard-preflight launchers.

## Remote roots and preflight

Workspace `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope`.
Tag `phase10-gate-c-20261002T081604Z`.
Complete git archive deployed to `staging/<tag>/source-2644034a9d9c806099693e810c1fada9d73cd750`;
immutable source not edited. Inputs under `inputs/<tag>/prepared-attempt1`; runs under `runs/<tag>`.
Old source remains clean `phase9-gate-c-20260922` / `a2c9e58a32163e1ed1527d3ab57798bc7378f486`.
Original audited Python/cache/model dependencies used; no installation/download/upgrade.
SSH strict host checking, BatchMode, ConnectTimeout10, ClearAllForwardings and UpdateHostKeys=no.

Debug job `12902763`: `--partition=debug --qos=debug --gres=gpu:a40:1 --cpus-per-task=8
--mem=64G --time=00:29:00`, launcher `phase10_shard_preflight.sbatch` with source, original Python,
B MMLU validation manifest/pool, `debug-shard-preflight-attempt1`, source commit. Expected3–10min,
three model/K combinations, each unsharded2 rows vs two1-row shards, exact candidate score equality
and real verifier closure. Source/launcher/runtime identical to subsequent formal producer path.

Before submission, scheduler partition/association/QoS snapshots saved under the new run root.
Highest legal A800 partition `emergency_gpu` has PriorityTier/JobFactor300, RootOnly=NO,
AllowGroups/Accounts/QoS=ALL; user association includes emergency_gpu; available QoS priorities0.
First snapshot command used unsupported multi-partition `scontrol show partition emergency_gpu debug`,
exit1 before sbatch; corrected to separate single-partition reads. No job created by failed command.

One bounded check confirmed job RUNNING89s, 4B K2/K3 shard/unsharded exact equality (max error0);
1.7B pending completion. This is partial engineering evidence, not full preflight admission.
After stable RUNNING>60s, created the sole10-minute heartbeat
`phase10-gate-c-debug-continuation`, confirmed ACTIVE by automation tool. It checks job/short logs
and input-plan existence only; terminal first disables itself then visibly sends
AUTOMATION_TERMINAL_RESUME to exact executor, or AUTOMATION_RELAY_REQUIRED to planning.

No Gate terminal event emitted. C remains active; D and ARC formal/outcome remain locked.

## Preflight closure and formal canary dispatch

Debug12902763 COMPLETED/0:0, elapsed117s, allocated0.0325GPUh. Independently read the complete
SHARD_PREFLIGHT_VERIFIED.json: all3 model/K combinations exact, max error0; source2644034 unchanged.
Prepared input plan exists and pre-submit CPU reread closed full14042 identities, full manifest68,
59new/9reuse, seven unique lambda0.1 groups, and exact64 length-ranked indices/model.

The debug monitor paused and sent its self-resume packet, but the executor stayed idle until
planning's continuation-relay recovery. Executor then resumed, verified no existing C jobs or
formal submissions, and dispatched the following seven jobs exactly once:
`12903430,12903431,12903432,12903433,12903434,12903435,12903436`.
All `emergency_gpu` partition/QoS, A8001GPU,8CPU,64G,1h, packing1/batch1; two afterok serial
chains enforce max2 simultaneous GPUs. Initial upper bound0.0325+7=7.0325GPUh, below16GPUh.
Actual source remains2644034; evidence-only later commits do not change deployed producer/runtime.
Fresh submission plan/commands/stdout/stderr/job IDs are in remote `formal-canary-submission-plan.json`,
`job_submissions.jsonl`, `formal-canary-jobs.json`. All7 confirmed PENDING, first2 Priority and
remaining5 Dependency; highest lawful partition/QoS credentials reread before submission.

Updated the same automation ID (no duplicate) to `Phase10 Gate C formal continuation`, ACTIVE,
60-minute cadence. Frozen all7jobs and terminal-failure/all-complete criteria; strengthened actual
executor continuation or planning relay to avoid the observed self-delivery/idle defect.
The next section supersedes this historical canary-wait state.

## A800 admission and remaining batch1

On the next terminal heartbeat, executor received its self-resume in the active continuation,
switched roles, reread policy/control/handoff and independently continued without planning relay.
All7 canary jobs COMPLETED/0:0, elapsed88/103/35/52/69/38/42s,427A800 GPU-seconds.
Same-source normal verifier closed7×64=448 candidate-complete records; target_gold_loaded=false.
Total C allocation544GPU-seconds=0.151111GPUh including debug. No OOM or GPU failure.
Observed GPU NVIDIA A800-SXM4-80GB; max CUDA reserved10.546875GiB(4B)/6.1796875GiB(1.7B),
including model loading. Runtime torch2.3.1+cu121/transformers4.51.3/lm_eval0.4.11 verified.
See compact resource evidence `loopscope_phase10_gate_c_resources.json`.

Original1024-row cost estimate134.672831GPUh, reserve-inclusive161.758508GPUh passed240GPUh.
Full remaining cell-shard sizing reduces repeated loading:130.064149GPUh remaining,
consumed+1.2×remaining=156.228090GPUh. Both estimates use the same measured seven strategies and
population native context lengths, larger adjacent row-time envelope; later strengths/system
contention remain operational uncertainty, never a reason to alter scientific membership.

Frozen `formal-remaining-work-plan-attempt1.json` has59 canonical remaining/full-cell execution
units,828030 records, each complementing retained canary where present. One unit per cell avoids
826 small jobs and repeated model loading; time=ceil(1.5×estimated cell cost)hours, observed2–6h,
within24h handoff limit. Source/launcher/env/runtime/scoring path unchanged, so prior debug applies.

First bounded batch14 units covers lambda0.1 remaining13978×7 and lambda0.2 full14042×7=196140.
Jobs `12903718` through `12903731`, emergency_gpu/A8001GPU/sameQoS/8CPU/64G/packing1; two afterok
serial chains cap2GPUs. Times4/2/4/3/6/3/4/4/2/4/3/6/3/4h, queued/run upper bound52GPUh;
consumed+all batch time upper bound52.151111GPUh<240. Highest legal credentials reread.
Exact commands/response/errors in remote job_submissions.jsonl; plan/IDs in
formal-remaining-batch1-plan.json and formal-remaining-batch1-jobs.json. Confirmed PENDING
(first2 Priority, remaining Dependency). Same sole full60-minute heartbeat retargeted ACTIVE.

Next executor step: terminal/GPUh and gold-free full-cell closure for completed batch, then next
bounded workplan batch with updated budget. No full panel closure, gold/outcome or ARC/D yet.

## Batch1 closure and remaining batch2

2026-10-03 terminal heartbeat confirmed all14 batch1 jobs COMPLETED/0:0. Automation tool confirmed
the sole monitor PAUSED; send_message_to_thread confirmed delivery to the exact executor. The
self-resume event actually arrived in the active turn, and executor continued independently.
Policy/control/handoff and Git provenance reread: loopscope clean, HEAD1ba3134, protected ancestor
present. Source2644034 and preflight unchanged. A local read helper initially assumed job_id on
old canary journal entries; it exited before verification or mutation, then correctly used their
retained sbatch stdout IDs. No failed GPU attempt or repeated submission resulted.

Independent sacct closed all22 C jobs (debug1, canary7, batch1 fourteen). Batch1 allocation99450s
=27.625GPUh; total99994s=27.776111GPUh. Same immutable verifier --full-cell closed14 cells,
196588 records, exact14042 identities/cell, source/runtime agreement and target_gold_loaded=false.
Seven canaries448 records are included exactly once. Remote decisive closure:
formal-remaining-batch1-closure-attempt1.json, status FULL_PHASE10_SCORE_CELLS_CLOSED.

Label-free allocation costs by model/K/policy were compared with the frozen canary-length envelope;
the larger estimate gives remaining45 units99.665563GPUh, consumed+1.2remaining147.374786GPUh.
Later strengths/contention remain estimate uncertainty. Allocation/next-admission evidence is
batch1-allocation-and-next-admission-attempt1.json. No score outcomes or target labels accessed.

Batch2 is the next14 unsubmitted frozen units: lambda0.3/0.4, seven groups each,196588 records.
Jobs12907693–12907706, emergency_gpu/QoS, A8001GPU/8CPU/64G, packing1, same immutable launcher
and runtime. Current partition/association/QoS snapshots confirm emergency_gpu is the highest
eligible normal-user A800 tier300. Two afterok chains enforce max2 GPUs. Time limits remain
4/2/4/3/6/3/4/4/2/4/3/6/3/4h, total52GPUh; consumed+queued upper bound79.776111GPUh<240.
Fresh roots, canonical full indices and no existing submissions checked before sbatch. Plan/IDs:
formal-remaining-batch2-plan.json, formal-remaining-batch2-jobs.json; exact commands/responses
appended to job_submissions.jsonl and batch2-submission-00 through13.json. Confirmed first2
PENDING/Priority and remaining12 Dependency. No cancellation, retry or scientific change.

Same unique60-minute monitor retargeted to batch2; automation_update confirmed ACTIVE. Its prompt
requires actual executor receipt/continuation or planning relay, and retains gold/full-panel
barriers. Next step is batch2 terminal and full-cell closure, then the next bounded workplan batch.
C remains active. No Gate terminal event, full-panel seal, MMLU analysis, transfer selection or ARC/D.

## Batch2 closure and remaining batch3

2026-10-04 batch2 terminal heartbeat found all14 jobs COMPLETED/0:0 and all summaries present.
Monitor PAUSED and exact-executor delivery were confirmed by tools; the self-resume event arrived
and executor actually continued. Live C authority, clean loopscope HEADba5321a and protected
ancestor independently confirmed. Source/launcher/runtime unchanged, original preflight applies.

Default SSH source selection then failed with Can't assign requested address before remote work.
Read-only local route/ifconfig showed current VPN utun7/source10.21.0.3. A strict, forwarding-disabled
probe explicitly bound this address and succeeded; executor continued using that command-only
BindAddress. No SSH configuration, VPN/Clash service or network settings changed. Failed transport
attempts neither ran verification nor created GPU jobs, and do not indicate a Slurm failure.

Independent sacct closed all36 C jobs: debug1/canary7/batch1 fourteen/batch2 fourteen. Batch2
allocation101736s=28.26GPUh; total201730s=56.036111GPUh. Same-source --full-cell verifier closed
batch2 fourteen×14042=196588 records, exact identities and recipe/runtime, target_gold_loaded=false.
Closure formal-remaining-batch2-closure-attempt1.json is FULL_PHASE10_SCORE_CELLS_CLOSED. Together
batch1/2 cover28 new cells/393176 records, with canary448 included exactly once.

Using the larger frozen canary envelope or batch1/2 group maximum allocation seconds/record gives
31 unsubmitted units69.200229GPUh; consumed+1.2remaining139.076386GPUh<240. New immutable account:
batch2-allocation-and-next-admission-attempt1.json. No gold, accuracy or old analysis read.

Batch3 next14 frozen units: lambda0.5 three current_t cells (historical fixed_t0/lag1 reused),
lambda0.6 seven groups, lambda0.7 firstfour groups. Each14042,196588 records. Jobs12912814–12912827,
highest legal emergency_gpu tier300/QoS, A8001GPU/8CPU/64G, same source2644034, packing1/max2GPU.
Time limits4/4/6/4/2/4/3/6/3/4/4/2/4/3h total53GPUh; consumed+queued upper bound109.036111GPUh.
Current legal partition/association/QoS snapshots preserved; exact canonical full indices, fresh
roots and absence of active earlier jobs checked before submission. Exact commands and replies in
job_submissions.jsonl and batch3-submission-00 through13.json, plan/IDs in
formal-remaining-batch3-plan.json/formal-remaining-batch3-jobs.json. All confirmed PENDING:
first2 Priority, remaining12 Dependency, with two afterok chains.

Same unique full60-minute monitor confirmed ACTIVE for batch3, with current VPN source discovery
and single-SSH boundary documented. Next action: batch3 terminal/full-cell closure, then next
bounded frozen units within budget. C remains active; full-panel seal, gold/analysis, transfer
selection and ARC/D remain unavailable.
