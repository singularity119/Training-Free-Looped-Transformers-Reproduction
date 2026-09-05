# Phase 8 Gate D execution evidence

Executor `01a071e9-d133-7742-8826-e0cf5993a1f1`; planning
`01a06fe9-c7bb-7d72-a906-234f301de317`. Gate execution in progress, not a decision.

READY binding and clean starting `loopscope` HEAD
`563a443ec2685f46eb970dc51d6a494ed641b34b` checked. Applicable AGENTS,
PROJECT_MEMORY, planning index, current control/handoff, contract v1,
Gate C acceptance, research-gate-orchestrator skill/protocol and HPC2 SSH skill read.
Protected ancestor check exit0. No protected runtime source changed.

Three bounded implementation agents own gold-free test pool, pre-outcome verifier,
and statistics respectively; exact executor owns integration, commits/pushes,
remote operations, scheduler and terminal delivery. No nested delegation.

D producer preserves original HFLM scorer through existing position adapter.
Native has no loop wrapper; original Loop uses residual_transform=None;
Spectral loads matching FORMAL_CALIBRATION basis without changing provenance.
Only four finite raw scores plus canonical identities are persisted; no target
labels or correctness are acquired during scoring. Runtime callback call records
are counted and cleared per identity rather than retaining full-run metadata.

Remote first contact: strict hpc2-hkustgz alias, forwards disabled, mgmt-4;
remote clean C producer `6d6214f6ad912b1984b1e0994015270acf627f59`, no user jobs.
Local origin connectivity initially timed out, second bounded query confirmed
planning commit563a443 on origin/loopscope. No transport/config change.
Account root is ordinary user association; debug QOS caps total16CPU/2GPU,
so each dual-model debug requests CPU8/GPU1/time00:29:00. A40 emergency partition
priority tier300 is above regular/extended, and emergency_gpu A800 also tier300;
both are normal-user authorized QOS. Packing remains evidence-dependent.

Required local checks: `PYTHONPATH=src python3 -m unittest discover -s tests -p
 'test_phase8_accuracy*.py'`, adapter targeted tests, launcher bash syntax,
CLI help/import checks and diff check. Actual counts and runtime outcomes follow.
No test gold/outcome unsealed; no formal jobs submitted yet.

## Producer checks

D targeted suite:17 tests exit0, including synthetic full18×14042 verifier,
paired arithmetic/CI/McNemar/Holm and gold barrier. Existing adapter tests:5,
exit0, covering untouched scorer delegation, truncation and multi-token position.
Runner CLI help, launcher `bash -n`, Python compilation and `git diff --check`
exit0. These are focused checks, no broad historical audit or model rerun.

## Gold-free pool and same-producer preflight complete

Producer `2dd3dfd59caf547a3abbc0a1516bc4a3207d9af0` committed/pushed and remote
clean fast-forward confirmed. Remote SciPy1.17.1 import, runner CLI/help,
launcher syntax and7 statistics tests exit0. No dependency replacement.
Workspace prefix W remains the handoff's dedicated workspace.
`inputs/phase8-gate-d-20260905T142900Z-a1/test_pool.json` and `panel.json`:
14042 ordered test identities,57 subjects,18 cells,8 matching admitted C bases;
source gold_loaded=false. Both model token lengths median524,p99 2670,max3096.
Builder and manifest CLI exit0; output/log path is
`staging/phase8-gate-d-20260905T142900Z-a1/pool.log`.

Debug jobs12650702(q4)/12650703(q17) COMPLETED0:0,61s/54s, A40 gpu3-9,
CPU8/GPU1/mem128G, partition/QOS debug,29min,packing1. Roots
`runs/phase8-gate-d-20260905T142900Z-debug-{q4,q17}-a1`.
Each covers native,window12:15 K4 Loop/Spectral, first2+longest4 test identities;
all six shards closed with6 finite-score rows each, no gold. Maximum actual
input3096, no naturally truncated/multi-token requests; adapter's targeted
synthetic truncation/multi-token test protects that unchanged scoring path.
Spectral records24 callbacks and18 applied steps for6 identities, as expected.
GPU.csv peak q4=10928MiB/q17=6116MiB. K4 spectral throughput2.23/3.03samples/s
on upper-tail-heavy debug rows. These short runs do not establish sustained packing.
Debug finished before stable-running monitor handoff; no debug heartbeat created.
Consumed debug GPU time115seconds=0.031944GPUh.

## Formal canary queued with single monitor

First formal submission rejected before job creation: scheduler enforces at most
8CPU/GPU (requested16). No allocation/output/attempt data existed; reduced CPU
request to8 under handoff without scientific change. Both verified real jobs:
-12650740 q4,packing2, nine cells, canonical interval[0,512).
-12650741 q17,packing3, nine cells, canonical interval[0,512).
Partition/QOS emergency_gpua40,account root,GPU:a40:1,CPU8,mem128G,time01:00:00
per job, maximum concurrent2GPU. Latest state both PENDING(Priority).
Run roots `runs/phase8-gate-d-20260905T142900Z-canary-{q4,q17}-a1`.
Scheduler logs `staging/phase8-gate-d-20260905T142900Z-a1/canary-{q4,q17}-%j.log`.
All invoke `phase8_accuracy.sbatch panel.json test_pool.json root producer
FORMAL_TEST packing 0 512` followed by each model's nine frozen cell IDs.

Unique automation `loopscope-phase8-gated-monitor` ACTIVE,full60min,
including queue phase. Tool creation visibly confirmed. It is read-only, quiet
when healthy; terminal/failure pauses itself and actively delivers
AUTOMATION_TERMINAL_RESUME to exact executor01a071e9-d133-7742-8826-e0cf5993a1f1;
fallback AUTOMATION_RELAY_REQUIRED goes to planning01a06fe9-c7bb-7d72-a906-234f301de317.
It cannot retry/submit/cancel/read gold or advance a Gate.

Next authorized executor step: inspect existing canary jobs and closures, sustained
GPU.csv peak/15percent headroom and runtime. Preserve valid first512 per cell.
Only after valid canary, expand remaining[512,14042), respecting two GPUs,
6h/job,64GPUh cumulative and fresh failed-shard paths. Retarget this same monitor.
No formal remainder, target gold unseal, analysis or Gate terminal occurred yet.

## Canary complete; remaining formal panel submitted

User completion notification triggered exact-executor continuation. Fresh sacct:
12650740 COMPLETED0:0 elapsed545s,12650741 COMPLETED0:0 elapsed241s,
batch/extern also0:0. Paused the unique monitor with visible confirmation before
continuing. Fresh `phase8_accuracy_verify.verify` read saved raw scores and
closed9cells×512 per model,9216 total scores, without outcomes.
GPU.csv sustained q4 packing2 peak22460/49140MiB (54.29percent headroom);
q17 packing3 peak20301/49140MiB (58.69percent). All18 summaries OOM0;
concurrent worst per-cell throughput about5.25/7.01samples/s respectively.
Preserve all first512 per cell; no resubmission or outcome read.
Debug+canary elapsed901GPU-seconds=0.250278GPUh. Remainder estimate from canary
throughput and waves is roughly3.6h q4/1.6h q17 plus length/load variation;
6h/job bounds and at most12 additionalGPUh stay below64GPUh total.

Remote remains clean producer2dd3dfd; later local evidence-only commit need not
replace this unchanged preflight/runtime checkout. Fresh user queue empty.
Submitted same producer/launcher on highest eligible A40 tier300 emergency
partition/QOS,account root,GPU1/CPU8/mem128G/time06:00:00 each:
-12651219: q4 nine cells,packing2,canonical[512,14042).
-12651220: q17 nine cells,packing3,canonical[512,14042).
Roots `runs/phase8-gate-d-20260905T173200Z-remaining-{q4,q17}-a1`;
logs `staging/phase8-gate-d-20260905T173200Z-remaining-a1/{q4,q17}-%j.log`.
Both real job IDs verified PENDING(Resources/Priority). Pool/panel paths unchanged.
Exactly the same monitor retargeted and ACTIVE full60min, including PENDING;
visible update confirmation retained. Terminal continuation must combine these
remainders with the preserved canaries and verify18×14042 before gold access.
No Gate-level final event yet; current state is waiting for formal remainder.
