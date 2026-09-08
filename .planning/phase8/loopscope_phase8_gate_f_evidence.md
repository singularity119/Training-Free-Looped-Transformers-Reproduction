# Gate F execution evidence

Executor `01a08053-81a4-70b1-a306-a721189018a2`; planning `01a08024-3556-75c1-8209-64e89345b940`.
Current authority: F handoff superseding concurrent execution clause, control `ACTIVE_GATE=E,F`, user parallel exception. Initial observed clean local base `3d1cd8012ff4daafe5540c9cdb6beb140ee29dc8`; implementation starts after planning `208ab8284547beb672047e7fb102593e62202819`.

## Source isolation and implementation

E confirmed its queued launchers read mutable dedicated HPC source at `a818dbb61ac82879858c34eb74b7ee152ad17bdc`. Planning explicitly authorized F fixed source snapshots under dedicated W/staging; F never fast-forwards E source. Original environment remains `.venv-loopscope-cu121-20260711/bin/python` under dedicated source, with F snapshot/src explicitly first in PYTHONPATH. E owns its evidence edits; F stages only named F files. Ordinary push includes the two already-committed planning updates.

All implementation is new F modules/scripts/tests; original runtime, adapter, wrapper, cache, strategies and original t1 fitter remain unchanged. Three bounded helpers implemented basis checks, panel/statistics, and calibration runner; executor integrates, tests and owns all Git/remote/scheduler actions.

Four t0 bases use retained C K2 residuals, source scope/identities/native FP32 residual matrices and positions checked; original SVD arithmetic reused through a t0 matrix adapter. Explicit fit_t=0/source_k=2/applies_to_k=[2,3,4]; single basis file shared per model/window. Two q17 K3 t1 bases remain K-specific. Intervention still original runtime t>=1 only.

## Initial validation and remote read-only check

`PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase8_gate_f*.py'`: 16 tests, exit0. Tests cover shared basis provenance, actual tensor fit/save/load when torch available, 36 logical cells and separate12+12 Holm families, 16-new/224672 score closure, missing/duplicate scores and intervention timing. `git diff --check` and all three F launcher `bash -n` checks exit0. No full historical test suite rerun.

Strict key-only forwarding-disabled SSH reached mgmt-3; dedicated HPC source clean `a818dbb`. C cell-0 residuals and metadata exist with512 validation source. Current emergency_gpu is A800, PriorityTier300, RootOnly=NO; user association account root includes emergency_gpu/debug QoS (account name does not imply administrator privileges). Debug provides A40, 30min limit. F bounds48GPUh including debug/retries, <=2GPUs, <=6h perjob; no changes to E jobs.

Planned debug: two independent model jobs, each 1A40/8CPU/128G/29min, expected <10min from four preflight calibration identities and six test identities per cell. Both windows exercise exact t0 equality K2/3/4, shared loading and real intervention, saved-C fitting path, raw standard scores and verifier; q17 covers new K3 t1 too. Formal blocked until both pass. A800 canary uses retainable first512 before expanding packing from actual memory/throughput.

## Producer snapshot and submitted debug

Producer `e5b8c8f418d9947a27967da5d13454cc13858745`; ordinary `git push origin loopscope` exit0 (`3d1cd80..e5b8c8f`). E dirty evidence preserved and excluded. Fixed source exported with `git archive e5b8c8f418d9947a27967da5d13454cc13858745 | ssh ... 'mkdir W/staging/phase8-gate-f-20260908T093426Z-source-a1 && tar -xf - -C <same>'`, exit0. No remote dedicated-source Git mutation.

In snapshot: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <original-venv>/bin/python -m unittest discover -s tests -p 'test_phase8_gate_f*.py'`:16tests,3.316s,exit0. All F Python entrypoints compile and launcher shell syntax checks pass.

Submission exit0 for both, with `/opt/slurm/bin/sbatch --parsable --partition=debug --qos=debug --account=root --job-name=p8-f-debug-<mi> --output=W/staging/phase8-gate-f-20260908T093426Z-debug-a1/model-<mi>-%j.out --error=<same>.err <snapshot>/scripts/loopscope/phase8_gate_f_debug.sbatch <snapshot> W/inputs/phase8-gate-c-20260905T100000Z-a1/calibration_pool.json W/inputs/phase8-gate-d-20260905T142900Z-a1/test_pool.json W/runs/phase8-gate-f-20260908T093426Z-debug-m<mi>-a1 e5b8c8f418d9947a27967da5d13454cc13858745 <mi>`.

`mi=0` job12687439; `mi=1` job12687440. Initial real states PENDING/MaxJobsPerAccount, elapsed0; no failure inferred. Requested planning observation-only exception for queue-phase10min single monitor; no cancel/retry/resubmit or altered resource request. Formal remains blocked until the two actual preflights succeed.

Planning queue-observer exception committed `c12049c`. Single executor-owned automation `loopscope-phase8-gate-f-debug-observer` created ACTIVE, 10min smoke interval; creation tool confirmed. Prompt binds exact executor/planning, both debug jobs/source/run roots and terminal pause → active AUTOMATION_TERMINAL_RESUME routing; healthy wakes stay quiet. The first create request lacked thread destination and returned argument error without creating an automation; corrected request explicitly targets this thread, succeeded once. No duplicate monitor.

Critical reused producer paths compared directly with E producer a818dbb: original Phase8 runtime/adapter, wrapper/strategies/cache/config and model/window config unchanged. New F scope does not change retained D/E scoring behavior. Formal basis/scoring and gold analysis have not started.

## Manual attempt continuation: debug verified, formal calibration submitted

Planning relayed user observation of12687440; executor independently queried both jobs.12687439 COMPLETED0:0,119seconds;12687440 COMPLETED0:0,152seconds. Total debug271GPU-seconds=0.0752778GPUh. Direct four-window verification.json checks: t0 matrices exactly equal across K2/3/4, answer positions equal, tensor intervention checks all true, q17 K3 t1 fitted/loaded/intervened. Saved-C preflight subset path checked actual scorer positions and cross-K loads. Raw-score closure6q4+10q17 cells,6samples each,96scores,all goldfalse. Logs only expected library warnings, no traceback/OOM. A40 score reserved-memory maxima q4 11102322688bytes, q17 6064963584bytes; these do not alone size A800 packing.

Single observer paused with successful tool response before continuation. Submitted12687480 with `/opt/slurm/bin/sbatch --parsable --partition=emergency_gpu --qos=emergency_gpu --account=root --gres=gpu:a800:1 --time=01:00:00 --job-name=p8-f-calibration --output=W/staging/phase8-gate-f-20260908T094800Z-calibration-a1/slurm-%j.out --error=<same>.err <snapshot>/scripts/loopscope/phase8_gate_f_calibration.sbatch <snapshot> <original-calibration-pool> W/runs/phase8-gate-f-20260908T094800Z-calibration-a1 e5b8c8f418d9947a27967da5d13454cc13858745`; exit0. Existing launcher fixes8CPU/128G; sequential formal four saved t0 fits and two q17 K3 t1 acquisitions. Immutable producer/source/environment unchanged from debug. No E source mutation.

Same automation ID updated ACTIVE, name `LoopScope Phase8 Gate F calibration observer`, interval30min probe, job12687480 includingPENDING. Terminal pause and exact executor resume preserved; next action verifies4+2 calibration then first512 retainable test canary and packing. No new full-test scores or gold analysis yet.

## Formal calibration complete; first512 retained test canaries

User signaled12687480 completion; independent sacct confirms COMPLETED0:0/153seconds. Read formal verification.json:4t0+2t1,each512 exact identities,FORMAL_CALIBRATION,goldfalse; maximum spectral eigen residual3.588254276e-8 (<1e-4). No OOM/error beyond expected library warnings. Cumulative completed GPU usage424seconds=0.1177778GPUh. Original four t0 files shared acrossK2/3/4; no additional K-specific t0 fits.

Paused existing observer with confirmation. Built fixed16cell FORMAL_TEST manifest at W/inputs/phase8-gate-f-20260908T102000Z-test-a1/panel.json using original test pool and formal calibration-a1/{t0,t1}; builder exit0.

Submitted12687717(q4sixcells),12687718(q17tencells),both A800x1,CPU8/128G/01:00:00,emergency_gpu partition/QoS,ordinary account root,packing1 to measure F-specific A800 peaks before expansion. Exact launcher: `<snapshot>/scripts/loopscope/phase8_gate_f_accuracy.sbatch <snapshot> <F-panel> <original-test-pool> W/runs/phase8-gate-f-20260908T102000Z-canary-m<mi>-a1 e5b8c8f418d9947a27967da5d13454cc13858745 FORMAL_TEST 1 0 512 <model-cell-IDs from frozen manifest>`. Logs W/staging/phase8-gate-f-20260908T102000Z-canary-a1/model-<mi>-%j.{out,err}; sbatch exit0. The0:512 scores are retainable experiment work, not discarded smoke. Only512:14042 remains after closure; no gold/partialacc.

Same automation updated ACTIVE/60min full,name `LoopScope Phase8 Gate F test observer`, jobs12687717/12687718 includingPENDING. On terminal independently verify canary and actual gpu.csv/throughput, choose packing<=3 with>=15percent headroom, submit only remainder within total48GPUh/2GPUs. Source,environment,scientific recipe unchanged; E source/jobs untouched.

## User-directed immediate full submission

User directly instructed F: “直接提交全部实验吧”. This supersedes waiting for canary measurement before remaining submission; it does not alter science or information boundaries. Submitted all remaining work immediately at conservative packing1, using afterok dependencies to retain at most2 active GPUs and preserve the first512 scores.

12687743(q4sixcells) depends afterok12687717;12687744(q17tencells) depends afterok12687718. Each emergency_gpu/QoS emergency_gpu/account root/A800x1/CPU8/128G/6h. Same immutable launcher/producer/manifest/pool as successful debug and canary; args now `FORMAL_TEST 1 512 14042`. Roots W/runs/phase8-gate-f-20260908T103000Z-remaining-m{0,1}-a1; logs W/staging/phase8-gate-f-20260908T103000Z-remaining-a1/model-{mi}-%j.{out,err}. Both sbatch calls exit0. At submission existing canaries remainedPENDING; no cancellation/replacement or duplicate identity interval.

Single60min observer updated to all4jobs; canary completion alone no longer prompts duplicate remaining submission. All4 complete or any failed/DependencyNeverSatisfied/material blocker pauses observer and resumes exact executor. Maximum newly requested remainder allocation12GPUh, plus canary2GPUh and used0.118GPUh remains below48GPUh. Final raw-score closure and accepted E comparator requirement unchanged. Planning notified of user's operational sequence override.

## Full new panel closed; waiting for accepted E comparator event

Observer encountered hostname-resolution exit255 and paused itself with tool confirmation, then actively delivered AUTOMATION_TERMINAL_RESUME to exact executor. No job failure inferred. Executor inspected split DNS/routes only: current utun7=10.21.0.64 and both campus DNS routes on utun7, dig and strict alias SSH then succeeded. F made no network changes; E is the planning-designated network repair owner.

Independent sacct:12687717/18/43/44 allCOMPLETED0:0; elapsed307/334/6686/8087seconds. All original attempts preserved, no retries or duplicates. Fresh `verify_phase8_gate_f_accuracy.py --manifest <F-panel> --pool <original-test-pool> --cell-roots <all32 original shards> --full-panel --output W/artifacts/phase8-gate-f-20260908T143000Z-closure-a1/verification.json` exit0:FULL_PANEL_CLOSED,16cells,32disjoint shards,224672rows,57subjects,target_gold_loaded=false,single sourcee5b8c8f. stdout retained verification.log. Full outcome remains uncomputed.

resources.json in same closure root records15838GPU-seconds=4.39944444GPUh including both debug and calibration, below48GPUh. All scoringOOM0,packing1. Actual A800 GPU used maxima: q4canary11430MiB/q17canary6958MiB; q4remainder26919MiB/q17remainder22030MiB out81920MiB. Remainder perprocess reserved maxima27246198784/22556966912bytes. First512 was not treated as proof of full upper-tail peak. Summed acquisition: q4remainder6592.326s for81180scores; q17remainder7933.503s for135300scores.

Planning confirmed E has not yet delivered its terminal acceptance packet. F independent work is closed; await planning's accepted four E K3 comparator paths before36logical closure/one uniform analysis. No E polling, no wait-only monitor, no new gold/accuracy, no Gate-levelBLOCK or FINAL_AUDIT yet. Existing observer remainsPAUSED. All network/job/identity/resource evidence and nonterminal readiness delivered to planning with success confirmation.
