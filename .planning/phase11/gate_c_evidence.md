# Phase11 Gate C execution evidence (in progress)

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
