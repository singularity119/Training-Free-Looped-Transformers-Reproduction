# Phase10 Gate C execution evidence

Executor `01a0fba5-4ee0-7262-83d4-5bde515f3a94`; planning `01a0fae2-12df-74e0-a874-190518ef0501`.
Current state: shard preflight running; no formal test scoring or gold access yet.

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
