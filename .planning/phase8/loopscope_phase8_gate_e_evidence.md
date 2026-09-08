# Gate E execution evidence — 4B K3 supplement

Executor `01a08031-a85e-7101-b150-60916abdc4ac`; planning `01a06fe9-c7bb-7d72-a906-234f301de317`.
Scope and scientific authority: current control, Gate E handoff and amendment. Original A-D evidence remains frozen. This is a post-outcome exploratory supplement, not a replacement primary analysis.

## Admission and environment

2026-09-08: local dedicated clone `loopscope`, clean starting HEAD `f97b4ae5c28e70f127bbcb9adb70e5b756cab95d`; fixed-base ancestry check exit 0. Read skill/protocol, AGENTS, project memory, planning index, contract_v1, original terminal audit, current control and Gate E handoff/amendment.

Strict forwarding-disabled SSH succeeded on mgmt-3; dedicated remote clone clean at `2dd3dfd59caf547a3abbc0a1516bc4a3207d9af0`; account squeue empty. Existing Phase8 B/C/D monitors all PAUSED. No old jobs or outputs were changed.

Existing environment: torch `2.3.1+cu121`, transformers `4.51.3`, lm_eval `0.4.11`. Original calibration/test pool paths exist. Normal-user association includes debug/emergency_gpu; emergency_gpu is A800, RootOnly=NO, PriorityTier=300. Debug GPU is A40. Frozen resource cap: 24 GPUh total, at most 2 GPUs, 6h per formal job; batch1 unchanged.

Initial planning acknowledgement failed because recipient was archived. Dispatching task restored it; one retry succeeded to the exact unchanged planning ID. No authority/recipient was silently changed.

## Implementation and validation

Bounded delegated engineering: calibration agent owns K3 runtime admission and E calibration producer/verifier/tests; accuracy agent owns E four-cell manifest, scorer, closure and two-item K3 statistics/tests. Executor owns launcher integration, all Git/remote mutation, scheduler actions and terminal delivery.

Preflight estimate: under 29 minutes for two small calibration cells (first two plus upper-tail rows from original512), then four upper-tail test score cells. Serial A40 processes avoid importing A800 packing assumptions. The same calibration/accuracy launchers are used for preflight and formal scopes. Formal A800 starts with packing2; first512 scores per configuration are retainable canaries before remaining work.

Targeted validation before debug: `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase8_gate_e*.py'` passed 8 tests at initial integration; historical `test_phase8_accuracy*.py` passed 17. Calibration worker additionally ran 8 runtime and 9 historical calibration tests successfully. E runner/verifier/analyzer CLI `--help`, all three E launchers `bash -n`, and `git diff --check` passed. Protected wrapper/strategies/cache/config and original phase8_model_windows.json have no diff. K3 uses explicit opt-in shared accuracy entrypoints, leaving default Gate D behavior unchanged.

## First debug attempt

Producer `a818dbb61ac82879858c34eb74b7ee152ad17bdc`, ordinary push succeeded; remote fast-forward clean at same revision. Final E suite 9 tests passed locally and in existing HPC environment. Remote pool read-only metadata confirms validation512/test14042, 57 subjects each, original revision, no target fields; longest 4B prompts 2954/3096 tokens respectively.

Submitted job `12687173`, `debug` / QoS `debug` / ordinary association `root`, A40×1, CPU8, memory128G, 00:29:00. Command: `/opt/slurm/bin/sbatch --parsable --partition=debug --qos=debug --account=root --job-name=p8-e-k3-debug --output=W/staging/phase8-gate-e-20260908T085637Z-debug-a1/slurm-%j.out --error=W/staging/phase8-gate-e-20260908T085637Z-debug-a1/slurm-%j.err scripts/loopscope/phase8_gate_e_debug.sbatch W/inputs/phase8-gate-c-20260905T100000Z-a1/calibration_pool.json W/inputs/phase8-gate-d-20260905T142900Z-a1/test_pool.json W/runs/phase8-gate-e-20260908T085637Z-debug-a1 a818dbb61ac82879858c34eb74b7ee152ad17bdc` (exit0). W is the dedicated workspace above.

Initial scheduler state PENDING (MaxJobsPerAccount), no log created yet. This is queue admission, not producer failure. Asked planning for the minimal queue-phase monitor observation exception; no resource/account/QoS change or resubmission.

Planning approved queue-phase observation in control on 2026-09-08. Created sole heartbeat `loopscope-phase8-gate-e-monitor`, ACTIVE, 10-minute smoke cadence, exact executor/job binding above; tool confirmed creation and planning acknowledgement delivered. Attempt terminal pauses monitor then actively sends AUTOMATION_TERMINAL_RESUME; later scopes update this same monitor. Original monitors remain PAUSED. This is a scheduling wait, not Gate completion.

## Planning correction and debug completion

User corrected current planning to `01a08024-3556-75c1-8209-64e89345b940`; planning control/handoff commit `f110233` records this. Earlier recipient in admission history is historical only. Monitor now uses current planning for every terminal/relay route; executor remains unchanged.

Debug12687173 COMPLETED0:0,154s=0.0427778GPUh,A40x1,OOM0. Saved separate-process calibration verifier checks 2 K3 preflight bases,4 identities each,t0/t1/t2 counts4. Score verifier closes4cells×6=24 finite records,target_gold_loaded=false. Longest input3096tokens,no truncation/multitoken continuation. Spectral actual callbacks per window t0/t1/t2=6/6/6, applied=0/6/6. Max per-process reserved calibration12626952192bytes,score11020533760bytes. Debug monitor paused after terminal discovery by active executor; no redundant self-wake required.

## Formal calibration

Submitted12687223 to emergency_gpu/QoS emergency_gpu/account root,A800x1,CPU8,128G,01:00:00,packing2. Run root W/runs/phase8-gate-e-20260908T091052Z-calibration-a1; logs matching W/staging directory/slurm-12687223.out,.err. Same immutable producer a818dbb; remote clean remains on that version (later local commits are documents only). Command is phase8_gate_e_calibration.sbatch with original512 calibration pool, this fresh root, producer, FORMAL_CALIBRATION 2 0 1. Submission exit0. Both actual memory and throughput will be checked before downstream test packing; no A40-to-A800 peak extrapolation used to expand beyond approved conservative packing2.

Sole monitor loopscope-phase8-gate-e-monitor updated ACTIVE probe30min immediately after verified jobID, includesPENDING; current planning confirmation delivered. Next action: verify two complete512 K3 bases,then4cell retained test512 canary. No new outcome read.

## Calibration closure and test canary

Calibration12687223 COMPLETED0:0,172GPU seconds; accumulated E326GPU seconds=0.0905556GPUh. Independent verifier VERIFIED_CALIBRATION/FORMAL_CALIBRATION,2cells each original512 identities,width2560,t0/t1/t2 each512,1536 residual rows/cell; maximum eigen residual2.918e-8 below1e-4. No targetgold,OOM0. Each process peak_reserved15858663424bytes,6.036 trajectories/s. Whole-GPU telemetry peaked64826/81920MiB,20.87%headroom; this whole-device reading is not attributed solely to E processes and is not used to increase packing.

Frozen formal E panel W/inputs/phase8-gate-e-20260908T092723Z-test-canary-a1/panel.json binds the two verified formal bases. Submitted12687405,A800/emergency_gpu QoS/accountroot,CPU8/128G/01:00:00,packing2,4cells[0,512),root W/runs/phase8-gate-e-20260908T092723Z-test-canary-a1,matching staging slurm logs. Same phase8_gate_e_accuracy.sbatch producer a818dbb,submissionexit0. Unique monitor updatedACTIVE full60min with this job and current planning ID immediately after submission. No partialaccuracy; retain valid canary before remaining[512,14042).

User/planning separately authorized parallel F. E source remains clean a818dbb because launcher resolves that shared checkout at execution time; F confirmed its own immutable W/staging source snapshot and no fast-forward/mutation of E source. E not editing shared runtime; local file ownership and short Git commit/push coordination communicated once. F will push pending planning documents together with its own implementation; E evidence only is currently uncommitted for next E documentation commit. E outputs reach F only through planning acceptance; no E rerun for F.

## Canary closure, connectivity and remainder

Original12687405 COMPLETED0:0/147GPU seconds,4cells×512=2048finite scores,SHARDS_CLOSED,target_gold_loaded=false. Peak reserved per process11.42GB; throughput8.605–8.654samples/s;whole-device telemetry22831/81920MiB (72.13%headroom),OOM0. Keep packing2 to preserve measured paired throughput; two remainderjobs minimize time to result, no numerical changes. E cumulative before remainder=154+172+147=473GPU seconds=0.131389GPUh.

Monitor encountered HPC DNS failure, paused itself and visibly self-woke E. Read-only diagnosis found campusDNS route throughClash utun4 whileVPN utun7 was present; thenVPN interface disappeared and later returned10.21.0.64. Planning authorized only minimal sudo-n host-route repair; no sudo or network mutation was performed because reconnection restored both DNS routes and resolved10.120.18.63 throughutun7. StrictSSH to mgmt-4 succeeded. The job was preserved throughout; no unknown-state resubmission. Current planning and F were notified of restored connectivity.

Submitted12688973(w12-15) and12688974(w13-16),A800/emergency_gpu/QoSemergency_gpu/accountroot,each1GPU/8CPU/128G/02:00:00,packing2. Roots W/runs/phase8-gate-e-20260908T143124Z-remaining-a1-w12-15 and -w13-16; logs W/staging/phase8-gate-e-20260908T143124Z-remaining-a1/slurm-%j.out,.err. Same source clean a818dbb, frozen formalpanel from test-canary input directory. Exact launcher args: phase8_gate_e_accuracy.sbatch <same panel> <same originaltestpool> <window root> a818dbb61ac82879858c34eb74b7ee152ad17bdc FORMAL_TEST 2 512 14042 q4-w<window>-k3-loop q4-w<window>-k3-spectral. Both submissions exit0; only missing interval[512,14042), no repeated canary. Unique monitor ACTIVE full60min updated immediately with both IDs and terminal self-wake/currentplanning relay. Next outcome barrier is8shards/4cells/56168scores complete, before any gold analysis.
