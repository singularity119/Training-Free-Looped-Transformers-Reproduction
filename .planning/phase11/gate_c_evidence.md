# Phase11 Gate C execution evidence (final audit requested)

Executor `01a11b40-07d3-7342-9d01-f809b480f49e/local`; planning
`01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local`. Current authority is the
216-record Gate C handoff, with budget12 allocated GPUh, at most2 GPUs,
1 GPU/job and formal jobs at most2h. No full-panel or outcome authority.

## Minimal implementation and source

Implementation/source commit `447f64be3172dae403f7cd0c07ae17399a6197b4`
was normally pushed on loopscope. Core Phase11 runtime and protected wrapper,
strategies/cache/config have no changes relative to Gate B. The producer gains
scalar CUDA-event forward timings without per-decode synchronization; frozen
model, input, sampling, SVD, residual transform and cache semantics are unchanged.
The same producer must pass the C debug path before formal use.

Worker canary_helpers owned only `phase11_canary.py` and its tests; executor
reviewed/integrated them and retained all Git/remote/GPU/terminal actions.
Local targeted canary/batch tests4 and producer entry tests5 passed, exit0;
worker also ran existing analysis9 tests, exit0. Shell syntax/diff checks passed.

Source was deployed from authentic local Git objects/refs into the fresh stage
in `gate_c_setup.json`, then cloned with `git clone --no-hardlinks --branch
loopscope`. Remote clean HEAD/branch/fixed ancestor passed. Initial tar import
emitted Mac extended-header warnings; these are preserved in tool evidence.
An initial CPU test missed PYTHONPATH and failed before GPU; rerunning the
original command with source/src pinned passed2, exit0. No environment changes.

## Scheduler and membership

Live user association permits emergency_gpu. Its A800 partition has
PriorityTier/JobFactor300, RootOnly=NO, ALL accounts/groups, above the
100/200 alternatives. All user-eligible QoS priority values are0; use the
permitted emergency_gpu QoS. Rental/admin-only resources are excluded.
Nonformal preflight uses debug/A40,29min,1GPU,8CPU,64G host memory.

`gate_c_membership.json` is the safe local copy of remote write-once membership
at setup.inputs/membership.json, frozen before any formal forward. The full
pool lengths/counts produce indices ARC982/841/956/326 (876/934/993/1243 input
tokens), MMLU6011/11972/3485/10500 (1118/1305/1588/2860), and
GPQA105/261/356/344 (178/227/285/2819). Every18-arm set shares these identities.
Remote all/first/rest/debug worklists are write-once under setup.inputs.
Three formal singleton index/N dry-runs passed with count1 and no model loading.
Original full N1172/12032/448 and all formal manifests/pools remain unchanged.

## Submitted debug attempt

Job12947157 uses the same source/launcher/Python as planned formal, packing2
independent processes, three synthetic task producers plus attempt verifiers.
Requested QoSdebuglimit is rewritten by the cluster to actual QoSdebug;
submission confirmation shows RUNNING on gpu3-9,1A40,8CPU,64G,29min.
No formal generation, answer inspection or gold access has occurred.
Job accounting and monitor lifecycle will be added after terminal verification.

## Preflight closure and first formal batch

Job12947157 COMPLETED/exit0, allocated30 GPU-seconds (0.008333333GPUh).
Batch completes3 producer processes; every task has2 synthetic records and
ATTEMPT_RAW_VERIFIED/target_gold_loaded=false. End-to-end batch28.040s.
It finished before the stable-running60s threshold; no debug heartbeat needed.

Formal first job12947166 was submitted after the fresh closure, with
--no-requeue --partition=emergency_gpu --qos=emergency_gpu
--gres=gpu:a800:1 --cpus-per-task=8 --mem=64G --time=00:30:00,
worklist-first.json, FORMAL_TEST, packing1 and the same source/launcher/Python.
Immediate scontrol confirms real job ID, actual partition/QoS as requested,
PENDING/ReasonPriority, no allocation yet. Budget reserved0.5 GPUh plus the
already allocated0.008333333; no unbounded job submission.

Heartbeat phase11-gate-c-a800-canary-monitor was created ACTIVE with full60min
cadence as soon as the formal real ID was verified, including its PENDING phase.
Creation tool returned automationId/mode/status; receipt is gate_c_monitor.json.
The monitor must pause itself and actively trigger this exact executor on
attempt terminal. Next executor action remains same-Gate resource admission
and the other213 retainable canary records, never D/full-panel or gold access.
No Gate terminal event has been emitted; Gate C remains in progress.

## First A800 completion and packing2 admission

Heartbeat observed12947166 COMPLETED/exit0:0/268 allocated GPU-seconds,
then paused itself with visible tool confirmation and delivered
AUTOMATION_TERMINAL_RESUME to exact executor with tool threadId confirmation.
Executor independently re-read scheduler, source clean447f64b, all3 successful
singleton summaries and safe telemetry; no raw answers/gold were inspected.
Allocation total(debug30+formal268)/3600=0.082777778GPUh.

All3 use A800-SXM4-80GB (81920MiB). Device telemetry255 samples has peak9749MiB,
not an allocator-peak sum. MMLU-Pro longest2860 and GPQA2819 both actually reached
2048 generated tokens and2047 incremental decodes, final all-layer cache lengths
4907/4866, no decodeSVD, released per-question direction banks. Their question
seconds98.746706/95.312555, prefill5.608679/5.344842, decode90.997708/88.929870.
ARC longest1243 scores4 full candidates in7.973008 question seconds,
7.887060 CUDA scoring seconds, no truncation, no incremental generation.

Packing2 capacity estimate2*9749=19498MiB is below80% of81920MiB; the longest
formal generation realizes the full2048KV shape, rather than an early-EOS proxy.
Process allocator peaks are recorded separately in the raw summaries.
The capacity calculation is not a device-wide packing2 measurement or speedup.

Formal packing2 trial12947347 retains2 previously unrun longest MMLU/GPQA
K3/current_t/lambda0.5 singleton records. Submitted same447f64b launcher/Python,
emergency_gpu/emergency_gpu,1A800/8CPU/64G/30min,--no-requeue.
Fresh scheduler confirms partition priority300, legal QoSpriority0 and real
PENDING/Priority job. Its worklist is worklist-pack2.json; after success there
will be5 completed canary records and211 remaining, not a task/full-panel closure.
Requested reserve0.5GPUh+already0.082777778 remains below12GPUh.
Same heartbeat retargeted12947347 ACTIVE/full60min with tool confirmation;
next executor action is actual packing2 peak/throughput assessment, then
budget-bounded remaining canary. No new scientific configuration or full Gate D.

## Packing2 observed resources and remaining211 dispatch

12947347 COMPLETED/exit0:0/217 allocated GPU-seconds. Monitor paused itself and
visibly delivered AUTOMATION_TERMINAL_RESUME; executor independently re-read
clean source, scheduler and both successful singleton metadata. Both generated
2048tokens with full frozen bounds and released directions. Total allocation
(debug30+first268+trial217)/3600=0.143055556GPUh; successful canary count5.

Packing2 actual device telemetry203 samples has peak18521/81920MiB, no OOM.
Both question wall times are156.743350/156.742656s. Aggregate question-window
throughput4096/156.743350=26.131890token/s versus packing1's
4096/(98.746706+95.312555)=21.106955, a +23.807% resource point estimate.
The processes overlap and may have a small start offset; actual batch wall
211.633647s versus two packing1 child walls215.038964s is only1.584% shorter.
These are different-strength, single-node/time points, not a controlled repeated
speedup estimate; input identities/lengths and2048 output lengths match. Startup
and loading contention materially dilute the question-window gain. Selected
remaining packing2 based on direct capacity, improved question throughput and
preserved startup-inclusive point, without claiming confirmed/general speedup.

Remaining211 jobs are round-robin partitioned into3 nonoverlapping worklists
under original full manifests/pools, count71/70/70. Each contains24/24/23 ARC,
23/23/24 MMLU-Pro and24/23/23 GPQA respectively. No successful roots are included.
Same immutable447f64b launcher/Python/source/science and fresh batch roots.
Formal jobs12947685/12947686/12947687 use emergency_gpu/emergency_gpu,
1A800/8CPU/64G/2h,packing2,--no-requeue. Job12947687 is
--dependency=afterany:12947685, so these jobs cannot allocate more than2GPU
concurrently. Total requested reserve6GPUh+actual0.143055556<12GPUh.

The first combined scontrol show job comma-list query was rejected by this
cluster CLI (not job failure). Original three submissions returned real IDs;
one bounded correction used individual scontrol calls and comma-list sacct,
confirming first two PENDING/Priority, third PENDING/Dependency exactly as set.
No cancellation/requeue/repeat submission occurred. Same single heartbeat
retargeted these3 IDs ACTIVE/full60min, tool-confirmed. Monitor must pause and
resume exact executor on new terminal attempt; executor preserves successful
siblings and monitors remaining live work. All216 canary closure and cost/report
remain outstanding; no Gate C terminal and no D/full/gold authority exercised.

## Final terminal confirmation and canary-only closure (2026-10-09)

Planning relayed explicit same-C continuation after the delivered terminal self-resume
had not completed executor closure. Latest control/handoff allows at most8A800;
all216/12allocationGPUh/1GPUjob/packing<=2 boundaries remain. Compute history
was at most2 GPUs and is not repeated to fill the new allowance. Planning-owned
control/handoff edits are preserved outside the executor commit.

One bounded independent executor sacct/squeue read confirms all6 parent jobs
COMPLETED0:0. Last3 elapsed allocations2339/2279/2262seconds; total7395seconds
=2.054166667allocationGPUh, reserved0. All five formal batch summaries complete
3+2+71+70+70=216 singleton records, zero failure files/nonzero child exits.
Remote compute source remains clean loopscope/447f64b/fixed ancestor.

Pinned Python/source/src command:
`python scripts/loopscope/phase11_canary.py verify --membership inputs/membership.json
--worklist inputs/worklist-all.json --out runs/phase11-gate-c-20261008T113100Z-canary-closure-20261009.json`
(paths expand using gate_c_setup.json). Exit0,216,CANARY_CLOSED_GOLD_UNREAD;
full_panel_closed=false,target_gold_loaded=false. Structural raw validation is
internal to the authorized verifier; no answers/extraction/outcomes printed or
manually read. Safe metadata-only resource summary script was reviewed and
copied to a fresh CPU postprocessing path outside immutable GPU source;
`--closure <above> --ledger inputs/resource-summary-ledger-20261009.json
--out runs/phase11-gate-c-20261008T113100Z-resource-summary-20261009.json`, exit0.
All54 cells have4 frozen identities; safe metadata/source agrees across216.

Actual A800 sampled device peak23917/81920MiB; no sampling errors or OOM.
MMLU-Pro generated398/1416.71/2048 min/mean/max,truncated33/72;
GPQA302/1345.43/2048,truncated36/72. Generation total198874.
Process allocator peaks are separate. Full54 question-time proxy2648.22GPUh,
observed per-cell min/max sum761.52–4039.61 (not confidence interval/population
bounds); output256/512/2048 scenarios620.44/1067.06/3746.80GPUh.
Length-selected4identities/task, startup/queue/tail/retry exclusions and linear
decode approximation prohibit an unbiased prediction or promised finish date.
The Chinese resource report and54cost table are outside clone in assets,
indexed by gate_c_final_artifacts.json. D recommendation is max8×1GPU/packing2,
<=2hjobs,initial<=16GPUhreservedcalibration and planning-frozen rolling budget;
4000–4800GPUh is a planning reference, not authorized D spending.

The same heartbeat was explicitly reaffirmed PAUSED with automation_update
{automationId:phase11-gate-c-a800-canary-monitor,mode:update,status:PAUSED}.
Historical pause/resume delivery and planning relay are recorded in
 gate_c_monitor.json. No newmonitor, GPUsubmission, cancellation, scientific
change, goldunsealing, fulltaskclosure or GateD action. Low-risk operational
errors from earlier steps remain above; no failed GPU science retry was needed.
Only planning may returnPASS/PASS_WITH_FIXES/BLOCK.
