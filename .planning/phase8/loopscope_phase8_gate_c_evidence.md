# Phase 8 Gate C execution evidence

Executor `01a070f0-158e-72f2-9d06-5ba47c02e2bd`; planning recipient
`01a06fe9-c7bb-7d72-a906-234f301de317`. Status: EXECUTING, not a Gate decision.

Start clean `loopscope` at `4fff325c2273cdc1602745b05c2252b037a61c91`.
The diff from accepted B terminal `ef54d2cdafa92483ed65575d0c5788b7980d60f3`
contains only the expected seven planning/navigation documents. Exact Gate C
control, handoff and contract v1 read after READY; skill and protocol actually read.
Runtime permission directly received: danger-full-access, approval never.

Implementation producer `6d6214f6ad912b1984b1e0994015270acf627f59`, ordinary
push to existing origin/loopscope successful. Remote dedicated clean checkout
fast-forwarded from B producer c7e4e1f to this commit. No protected runtime changes.
Two bounded subagents owned pool export and verifier respectively; executor owns
runner/launcher, integration, commits, SSH, jobs and terminal delivery.

Target tests: `PYTHONPATH=src python3 -m unittest discover -s tests -p '*phase8_calibration*.py'`
9 tests, exit0; existing `test_loopscope_phase8_pool.py` 6 tests, exit0.
Pool dry-run, runner dry-run, verifier help, `bash -n scripts/loopscope/phase8_calibration.sbatch`
and `git diff --check` exit0. No broad suite or B GPU rerun.

First remote SSH connected strictly to mgmt-6; bare shell lacked squeue (exit127),
resolved by existing login shell `/opt/slurm/bin` PATH, without environment mutation.
No current user jobs. User association is account root (ordinary xhuang225 user),
with debug and emergency_gpua40 QOS allowed. A40 emergency partition has
PriorityTier=300, above regular100/extended200, RootOnly=NO, AllowAccounts/Qos=ALL.
No administrator authority inferred from account name. Debug admits A40/29min;
CPU request is 8/job to respect debug QOS total16CPU/2GPU.

Frozen input source is B retry6 calibration_identities.json. New pool path:
`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase8-gate-c-20260905T100000Z-a1/calibration_pool.json`.
Only exact validation safe fields/dev demos are loaded by unchanged renderer;
512 exported identities must match B dataset/seed/order exactly.

Pending: pool build, new-producer dual-model debug, measured packing, formal eight
cells, independent verifier, resource accounting, final delivery confirmation.
No test, target gold, accuracy or Gate D execution authorized or performed.

## Pool complete and debug queued

Remote builder exited0; 512 rows exported with exact B identity closure. Offline
cache warning uses the frozen revision directory; builder verifies that source.
Submitted at producer6d6214f using `phase8_calibration.sbatch`, account root,
partition/QOS debug, time00:29:00, GPU1, CPU8, mem128G per job, packing3,
cell indices0 1 2 3, scope PREFLIGHT_ONLY:
- job12647752 model0, run `runs/phase8-gate-c-20260905T100000Z-debug-q4-a1`.
- job12647753 model1, run `runs/phase8-gate-c-20260905T100000Z-debug-q17-a1`.
- scheduler logs `staging/phase8-gate-c-20260905T100000Z-a1/{q4,q17}-%j.log`.
Both verified PENDING(MaxJobsPerAccount); no alternate account or formal job.
Expected execution is minutes based on B four-row calibration; upper bound29min.

Unique heartbeat `loopscope-phase8-gatec-monitor` ACTIVE every10min, including
PENDING per handoff. Creation tool confirmed success. It pauses on attempt
terminal and actively sends AUTOMATION_TERMINAL_RESUME to this exact executor;
relay fallback goes to exact planning. No other C monitor. Initial create call
lacked targetThreadId/destination and was rejected without creation, then corrected.

Resume order: inspect sacct/logs/new-producer verifier; measure per-cell and
concurrent GPU peaks/throughput; choose safe packing; formal jobs under allowed
highest A40 emergency priority if still legally admitted; retarget same monitor
30min including PENDING; complete eight512 bases and final verifier, report,
commit/push, and confirmed terminal delivery. Formal work not yet submitted.

## New-producer debug passed; formal submitted

Monitor observed both debug jobs COMPLETED0:0 (batch/extern too),47s+23s,
gpu3-9; paused itself and successfully delivered AUTOMATION_TERMINAL_RESUME to
exact executor. Executor directly read both verification.json and8summaries/envs.
All8 PREFLIGHT_ONLY cells contain4identities (first2 + longest2 of512), max2954tokens,
unit norm errors below1.2e-7, eigen residual relative errors below3.4e-8.
A40 total50899648512bytes; 4B per-process reserved12685672448bytes, global telemetry
peak37383MiB (~23% free); 1.7B reserved7683964928–7721713664bytes, peak23145MiB.
OOM0. Three-process trajectories/s sums:4B3.372 vs solitary K4 cell2.650;
1.7B4.754 vs solitary K4 cell4.787 (roughly preserves throughput; small debug
sample and different window/K mix limit exact comparisons). Concurrent model load
also shortens elapsed work. Packing3 retained; identical8CPU/thread environment,
A40 hardware type, producer/launcher/runtime. No extra debug rerun.

Formal jobs12649531(model0) and12649532(model1) submitted successfully with
account root, emergency_gpua40 partition/QOS (ordinary-user permitted Tier300),
1A40 GPU,8CPU,128G,2h maximum each; same launcher producer6d6214f, packing3,
allcell0..3, FORMAL_CALIBRATION. Initial queue PENDING(Priority). Fresh roots:
- `runs/phase8-gate-c-20260905T100000Z-formal-q4-a1`
- `runs/phase8-gate-c-20260905T100000Z-formal-q17-a1`
under dedicated workspace. Logs staging/phase8-gate-c-20260905T100000Z-a1/formal-<model>-%j.log.
Even maximum allocation of both formal jobs plus debug remains below16GPUh.
Same unique monitor updated ACTIVE probe30min at verified job submission,
covering PENDING; pause/self-wake/final closure route retained and tool confirmed.
Pending formal completeness/verifier/resource accounting/terminal delivery.

## BLOCK: connectivity prevents formal closure (2026-09-05 13:13 UTC)

Last confirmed formal scheduler state at11:39UTC: both PENDING(Priority).
12:10 source-bound SSH returned No route to host;12:40/13:11 hostname resolution
failed. Current ifconfig shows VPN10.21.0.39 onutun7. Read-only diagnosis:
scutil DNS has HPC split resolver10.90.63.2/.3; route to10.90.63.3 and previously
observed HPC10.120.18.63 instead goes via198.18.0.1/utun4. Source-bound
`dig -b 10.21.0.39 +time=2 +tries=1 @10.90.63.3 hpc2login.hpc.hkust-gz.edu.cn`
timed out. Public resolver returned NXDOMAIN. No network/hosts/trust mutation.

Unique monitor paused with tool confirmation and successfully delivered the
visibility-blocker AUTOMATION_TERMINAL_RESUME to exact executor. Executor completed
bounded read-only diagnosis above. Existing Gate handoff forbids network changes;
current permitted alias/source-binding cannot reach DNS/HPC. Need restoration of
VPN/split routing by user or separately authorized network workflow, then resume
this same executor. Cannot infer either job failed/completed or safely redo work.

Preserve jobs12649531/12649532, all formal/debug/input roots, producer6d6214f,
contract v1 and test barrier. No retry/cancel or test/gold/accuracy/D action.
Formal artifacts and actual GPU-hour completion currently unknown; maximum
submitted allocation still4GPUh plus70seconds debug, below16GPUh.
