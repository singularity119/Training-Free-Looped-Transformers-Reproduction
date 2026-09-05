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
