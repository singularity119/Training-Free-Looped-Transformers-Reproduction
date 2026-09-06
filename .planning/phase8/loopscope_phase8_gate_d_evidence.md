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

## User-requested migration to A800

Read planning operational resume, live control and A800 supplement at1c8554b;
ordinary push of that planning commit succeeded. Paused unique monitor before
resource migration. Fresh scheduler showed both12651219/12651220 PENDING and
both planned remainder roots absent. `scancel --state=PENDING` cancelled only
these jobs; subsequent sacct confirmed both CANCELLED by204599,elapsed0.
Classification USER_REQUESTED_RESOURCE_MIGRATION, not engineering failure.
All prior valid A40 first512/cell remain preserved. No duplicate scores acquired.

Fresh association permits emergency_gpu,account root(ordinary user); partition
UP,RootOnlyNO,PriorityTier/JobFactor300. No admin priority changes. Remote clean
producer2dd3dfd/launcher/runtime unchanged; same-producer functional debug reused.
Submitted A800 resource canaries with original launcher,FORMAL_TEST [512,1024),
nine cells/model,1GPU:a800,CPU8,mem128G,time1h,emergency_gpu partition/QoS:
-12652573 q4,packing2,latest RUNNING gpu1-11 after36s.
-12652574 q17,packing3,latest PENDING(Resources).
Roots `runs/phase8-gate-d-20260906T031200Z-a800-canary-{q4,q17}-a1`;
logs `staging/phase8-gate-d-20260906T031200Z-a800-a1/{q4,q17}-%j.log`.
Actual A800 memory/runtime will be measured before expanding[1024,14042).
Pair arms share the same per-model hardware/interval schedule; mixed A40/A800
provenance will be reported without interpreting hardware as treatment effect.
Unique monitor retargeted ACTIVE full60min,with terminal self-wake unchanged.
Planning received switch packet via send_message_to_thread with visible success.
Budget before A8000.250278GPUh; no gold or partial outcome access.

## A800 canary closed; final remainder submitted

User/planning completion notice resumed exact executor. Current control/supplement
and clean local cb7181e checked. Both A800 canaries12652573/12652574 and batch/extern
COMPLETED0:0,369s/183s. Paused same monitor before continuation. Fresh verifier
combined prior[0,512) and new[512,1024) for each model:9cells×1024 unique finite
scores,18432 total, no outcome access. All summaries actual NVIDIA A800-SXM4-80GB,
OOM0. GPU.csv q4 packing2 peak29427/81920MiB,64.08percent headroom;
q17 packing3 peak30722/81920MiB,62.50percent. Concurrent K4 throughput about
8.33/9.44samples/s respectively. Retain same measured packing for final remainder.
Cumulative debug/canaries1453GPU-seconds=0.403611GPUh; user-cancelled queued jobs0.

Remote unchanged clean producer2dd3dfd,queue empty before submit. Submitted only
canonical[1024,14042),13018/model-cell,all9cells/model,original launcher:
-12652622 q4,packing2; latest RUNNING gpu1-11 at28s.
-12652623 q17,packing3; latest PENDING(Resources).
Both emergency_gpu partition/QoS Tier300,ordinary account root,GPU:a800:1,CPU8,
mem128G,time06:00:00. Based on measured rates,roughly2.2h/1.2h active execution
plus loading/length variation;6h bound permits at most12 additionalGPUh.
Roots `runs/phase8-gate-d-20260906T032700Z-a800-remaining-{q4,q17}-a1`;
logs `staging/phase8-gate-d-20260906T032700Z-a800-remaining-a1/{q4,q17}-%j.log`.
Unique monitor updated ACTIVE full60min with these jobs and exact self-wake;
visible confirmation. Planning received canary/remaining-submission packet.

Final closure must join all three disjoint intervals[0,512),[512,1024),[1024,14042)
for all18cells before gold access. No rerun of valid outputs, no extra arms or
Gate transition; current task waits for final A800 remainder.

## Full panel closure and one-shot analysis complete

Monitor bounded read confirmed final12652622/12652623 COMPLETED0:0,9103s/5061s,
batch/extern0:0. It PAUSED itself via tool success and actively delivered
AUTOMATION_TERMINAL_RESUME to exact executor with visible confirmation. Executor
fresh-read current control, clean local269bc8e and remote clean producer2dd3dfd,
all scheduler elapsed/exit states. All8 allocated debug/formal jobs COMPLETED0:0;
2 queued migration cancellations consumed0. Total15617GPU-seconds=4.33805556GPUh.

Fresh CPU `phase8_accuracy_verify.verify(...,full_panel=True)` combined all54
shards across three disjoint canonical intervals. FULL_PANEL_CLOSED,18cells,
252756samples,57subjects,target_gold_loaded=false; no overlap/missing identities,
configuration/basis/raw finite-score/runtime revision checks passed. Preserved at
`artifacts/phase8-gate-d-20260906T060000Z-closure-a1/verification.json` with
resources.json. Directory timestamp is a path label; actual chronology is saved_at.
Final A800 peak q4 packing2=42948/81920MiB (47.57percent headroom),q17 packing3=
50794/81920MiB (38.00percent),all54shards OOM0. No extra GPU work required.

Executed once,exit0,under unchanged remote producer:
`PYTHONPATH=src .venv-loopscope-cu121-20260711/bin/python scripts/loopscope/analyze_phase8_accuracy.py --closure W/artifacts/phase8-gate-d-20260906T060000Z-closure-a1/verification.json --output-dir W/artifacts/phase8-gate-d-20260906T131600Z-analysis-a1`
with frozen existing HF_HOME/HF_DATASETS_CACHE and offline env. Log:
`staging/phase8-gate-d-20260906T032700Z-a800-remaining-a1/analysis.log`.
Analyzer fresh pre-outcome closure saved2026-09-06T13:16:36.606559Z;
gold_load_started13:16:38.081938Z,completed13:16:56.522255Z. No partial outcome,
no outcome-driven repair/tuning/resubmission. One analysis18cells/24contrasts,
fresh verification VERIFIED: scalar argmax/correctN,discordance/pp,subjectmacro,
independent scipy exact McNemar,sorted Holm,and bootstrap from fresh flags using
the same tested algorithm. The latter is not a second independent implementation.

K2 Spectral−Loop ordered q4 12:15,q4 13:16,q17 12:15,q17 6:9:
-+0.021364pp,114gains/111losses,CI[-0.185159,+0.235009],Holm1.
-+0.170916pp,154/130,CI[-0.064093,+0.405925],Holm0.688832.
--0.064093pp,156/165,CI[-0.306224,+0.192280],Holm1.
--0.049850pp,61/68,CI[-0.206523,+0.113944],Holm1.
K4 same order:+0.014243,-0.064093,+0.128187,+0.028486pp;allCIcross0,Holm1.
All8nominal CIs cross0,neither primary nor secondary family significant. No
reliable spectral improvement claim. q4 native10262/14042; q17 native8813/14042.
Largest K2positive point remains below native and is uncertain.

Human report and compact18cell/24contrast CSVs copied as aggregate-only artifacts
to outer `资产/报告/phase8/`; no raw per-sample scores/gold/basis copied into Git
or local report. Report names source/version,offline512fit,Kfixedhorizon,mixed
A40/A800 intervals,negative/uncertain outcomes,nominalCI/multiplicity and absence
ofmatchednorm control. docs/loopscope_phase8.md updated with result/navigation.
An initial SCP brace path did not expand and copied nothing; explicit four paths
succeeded. No scientific or acquisition retry. Original scores remain HPC.

Completion: full frozen panel,statistics,fresh verification,resource accounting,
Chinese report and versioned evidence. Unique monitor PAUSED. No self-PASS,
no Phase9/newarm/science changes,protected runtime unchanged. Terminal packet
will request planning acceptance and phase-end audit; task-local final is fallback.
