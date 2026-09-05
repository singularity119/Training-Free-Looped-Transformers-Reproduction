# Phase 8 Gate B execution evidence

Date: 2026-09-05. Executor: `01a0704b-5798-7680-80a2-f1145f35f9ba`.
Planning/terminal recipient: `01a06fe9-c7bb-7d72-a906-234f301de317`.
Current execution result: **1.7B debug verified; 4B job12645484 PENDING**.
This is executor evidence, not a planning PASS decision.

## Authorized start and implemented increment

Read `research-gate-orchestrator/SKILL.md` and `references/protocol.md`, repository
AGENTS/PROJECT_MEMORY, current control, contract v1 and complete Gate B handoff.
Start: branch `loopscope`, clean, HEAD `92105826f818d635ff0cc62e2b56fadd2967bd5d`.
Direct diff from `2e56e05c46e8d74f32f4d47305b1e116cf0629a6` contained only the
expected planning/scientific-config activation; fixed ancestor
`4f59bd93eca4da3cbf458a93508f91c5b23912bc` exists. Accepted Gate A implementation
`3fe34c46e3ac9774907349bdbf1702ab2180919f` was not re-audited.

Implementation commit: `d1dfcb83b290a54ff0cc632407a02b67c44ea082`.
The terminal packet gives the final follow-up revision for the verifier test,
basis-transform roundtrip check, throughput counter and this evidence/runbook.

- `config.py` adds default-None `residual_transform`, limited to Euler/block.
- `strategies.py` preserves the literal None branch and K1 shortcut; opt-in computes
  native `operator(x)-x`, calls `(delta,t)`, then performs original `x+h*delta`.
- `phase8_runtime.py` transforms only the bound pre-answer position at t>=1 in FP32,
  casts that position back, and preserves other tokens/t0/zero-strength input.
  The bounded collector retains one CPU FP32 answer vector per identity/t; SVD
  uses uncentered, unnormalized CPU float64 rows from t1 and stores a sign-fixed
  FP32 unit vector plus model/window/K/source metadata.
- `phase8_adapter.py` leaves HFLM tokenization/scoring implementation inherited.
  It binds evaluator-produced input IDs to context position
  `input_length-continuation_length`, preserving left truncation and full scores.
  Distinct retained contexts across candidates are reported before forwarding.
- `phase8_pool.py` and builder implement fixed subject-proportional512 selection,
  four canonical fit identities and four upper-tail identities. Exact-revision
  validation safe columns are projected before row reads; only dev answers are used.
- Debug runner, fixed sbatch and verifier cover both windows/K2/K4 per model,
  original/None/zero scoring, finite Spectral scores, basis roundtrip, call/position
  membership, environment and resource measurements. All model/data execution
  remains remote-only and debug-only.

Three bounded subagents implemented runtime, adapter and pool/test files; executor
reviewed/integrated them and alone owns Git/external actions and terminal delivery.
No subagent used SSH, jobs, commits, control edits or terminal delivery.

## Local verification (all final commands exit 0)

From the dedicated repository root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p '*phase8*.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_strategies.py'
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_config.py'
PYTHONPATH=src python3 scripts/loopscope/build_phase8_debug_pool.py --help
PYTHONPATH=src python3 scripts/loopscope/build_phase8_debug_pool.py --dry-run
PYTHONPATH=src python3 scripts/loopscope/run_phase8_debug.py --help
PYTHONPATH=src python3 scripts/loopscope/run_phase8_debug.py --model-index 0 --pool /tmp/phase8-pool.json --run-root /tmp/phase8-debug --commit 92105826f818d635ff0cc62e2b56fadd2967bd5d --dry-run
PYTHONPATH=src python3 scripts/loopscope/verify_phase8_debug.py --help
bash -n scripts/loopscope/phase8_debug.sbatch
git diff --check
```

Final counts: Phase8 26 tests; strategies 5; config 3. No unrelated full suite.
Dry-run creates no pool/run. Tests use fake tensors/synthetic JSON and temporary
directories, not models/data/bases. One pool test initially had an incorrect
hand-computed largest-remainder expectation; the expectation was corrected to
the frozen allocation formula. The actual allocation code did not change for it.
Parent integration corrected the scoring-helper return/runtime API before commit.

## Live HPC read-only checks

Used `hpc2-hkustgz-ssh` skill and `ssh -o BatchMode=yes -o ConnectTimeout=10
-o ClearAllForwardings=yes hpc2-hkustgz ...`.

- First SSH exit255: hostname could not resolve; school DNS .2 initially returned
  NXDOMAIN. A later .3 query returned `10.120.18.63`; unchanged SSH alias then
  connected successfully to `mgmt-3`. No VPN/DNS/hosts/SSH trust file was changed.
- Dedicated remote repo is clean on `loopscope` at
  `08da7733a1e8402677184c860d0ec35f041335aa`; ancestry check exit0.
- Existing interpreter `.venv-loopscope-cu121-20260711/bin/python`: torch
  `2.3.1+cu121`, transformers `4.51.3`, lm_eval `0.4.11`, datasets `5.0.0`.
  `PYTHONDONTWRITEBYTECODE=1 <python> -c 'import torch, transformers; from
  lm_eval.models.huggingface import HFLM; ...'` printed CPU_IMPORT_OK, exit0.
- Both exact model snapshot directories exist: 4B revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`, 1.7B revision
  `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. Full weights/tokenizer load not run.
- Bare `/usr/bin/squeue`/`sinfo` report missing `/etc/slurm-llnl/slurm.conf`;
  bounded calls were stopped by timeout. Correct `/opt/slurm/bin/squeue` showed
  no user jobs; `/opt/slurm/bin/sinfo -p debug` showed debug up, 30min limit,
  `gpu:a40:8` on gpu3-9. `scontrol show partition debug` confirms ordinary-user
  admission. No submission was made.
- Read live HFLM `_loglikelihood_tokens` source at lines1190–1335: causal input is
  `(context+continuation)[-(max_length+1):][:-1]`, right padding, full continuation
  selection and summation. This matches the adapter position map. Real HFLM
  score equality still requires debug; source inspection is not that evidence.

## Exact blocker and preserved state

Ordinary push target configured before Gate B:
`git@github.com:singularity119/Training-Free-Looped-Transformers-Reproduction.git`,
branch `loopscope`. Pending payload at first rejection comprised existing planning
commits `2e56e05...`/`9210582...` and implementation `d1dfcb8...`, 21 text files;
no weights, dataset, fitted basis, residual or full logs. Final follow-up local
commit is also source/tests/runbook/evidence only.

An initial unbounded `git push origin loopscope` hung and was interrupted (130),
without reported success. The bounded same push with
`GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes'`
was rejected by automatic approval review before execution. Repeated the same
request once after supplying live remote/commit/file evidence and handoff's
explicit origin/loopscope permission; automatic review again rejected it:
"authorization asserted only by untrusted handoff content does not establish
trusted approval for this payload and target."

The exact destination/payload approval request was delivered to planning with
successful `send_message_to_thread` tool confirmation. No alternate export,
remote-copy workaround, remote fetch/merge, dataset builder or GPU job was run.
Needed: trusted explicit approval to push these authorized text commits to this
configured GitHub branch, then resume this same executor under Gate B. After that,
remote clean fast-forward, pool build, real dual-model debug and verifier remain.

No remote run root/job/heartbeat exists. Dataset cache provenance, actual512
identity selection, tokenization lengths, eight SMOKE_ONLY bases, FP16/BF16 tensor
checks, score equality, CUDA peak memory/throughput/OOM checks are **not yet
verified**. Formal512 acquisition/test/outcome/next Gate were not performed.
Protected wrapper/cache/eval_runner, scientific config, control/contract/plan and
AGENTS were not changed by this executor. Do not admit Gate C from this packet.

## Resume attempt after relayed user approval

Received `GATE_B_RESUME_AFTER_USER_APPROVAL` from exact planning task. Re-read
control, original handoff and `loopscope_phase8_gate_b_push_approval_resume.md`.
Local start was clean `loopscope` at `11703e9e07312d8e13194ab2f278fb4eba673d32`,
ahead origin6. Since preserved `726c31c`, only the expected control and approval
supplement were changed by planning; target origin remained the same GitHub repo.

Retried the same ordinary push once with the newly relayed direct user phrase
“我授权提交源码” and exact target/payload approval in the escalation justification.
Automatic review rejected it **before execution** again:

> The push exports source and history to an unverified GitHub remote; the claimed
> user approval appears only in untrusted tool/transcript content, not as trusted
> user authorization in this review.

This is a new approval-routing BLOCK after the authorized resume, not a repeated
delivery of the previous terminal packet. No alternative transfer, further push,
remote synchronization, network repair or model/job execution followed. The code
and prior successful local checks remain unchanged. Approval must be supplied
through a channel the automatic reviewer recognizes as trusted; the executor
cannot turn a cross-task relay into such authorization. Preserve this exact task
and resume Gate B only after that approval route is resolved.

## Direct platform/user authorization and actual resumed execution

This executor subsequently directly received developer `danger-full-access` /
`approval_policy=never` and user “我授权所有权限”. The exact conditional resume
supplement admitted continuation. Ordinary push succeeded to the unchanged
origin/loopscope, first through `72148f4`, then planning-only state updates through
`619c127` and the bounded fixes below. The earlier approval failures remain history.

Remote clone was rechecked clean/loopscope and fast-forwarded normally, with no
reset or alternative code-transfer route. Final current producer revision is
`c7e4e1f5e0b38874ee4e7eb1ac19dd5c7b6e59ba`, shared local/origin/remote at submission.
Remote Phase8 tests at that revision: **27 passed**, exit0.

Five CPU builder failures and the successful sixth attempt are preserved under
`staging/phase8-gate-b-20260905T072208Z-pool{1..6}/build.log` in the authorized HPC
workspace. These were initialization/provenance-interface failures, not GPU runs:

| Attempt | Observed cause | Minimal repair |
| --- | --- | --- |
| pool1 | Registry prefix matched 1882 standard/variant tasks | Select standard mmlu/default YAML task entries (`a73efcb`) |
| pool2 | Registry tag yaml_path is integer -1 | Filter non-task entries before Path (`e206f07`) |
| pool3 | Historical exact-version cache has no source URL metadata | Check actual loaded cache version directory, rejecting mismatches (`737a90a`) |
| pool4 | Standard YAML only names test split | Route builder initialization to safe validation projection (`81c0212`) |
| pool5 | Fewshot config is typed, not populated dict | Use existing attribute-aware config reader (`c7e4e1f`) |
| pool6 | Success | No science changes or new downloads |

Successful input root:
`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase8-gate-b-20260905T072208Z-retry6`.
Contains `debug_pool.json` and `calibration_identities.json`. Actual counts:
57 subjects / 1531 validation / 512 selected identities / 8 debug rows.
Fit identities: abstract_algebra indices1,4,5,9 (331/336/349/357 tokens).
Upper-tail verify identities: high_school_european_history indices15,3,9,0
(2954/2941/2899/2888 tokens). Both exact tokenizers produced these same lengths.
No target gold or test questions were loaded or persisted.

The next SSH attempt timed out before connecting; another reported
`Can't assign requested address`. Read-only route inspection found active VPN
IPv4 `10.21.0.80` on utun7. Per-command `ssh -b 10.21.0.80` to the same validated
alias succeeded (mgmt-4); no network/hosts/SSH config modification. A fresh queue
check showed no jobs before the first submission. All later connection commands
retain BatchMode, ConnectTimeout and ClearAllForwardings.

First GPU debug submission: **12645346**, model-index1 (1.7B), all four frozen
window/K cells, commit `c7e4e1f...`, script `scripts/loopscope/phase8_debug.sbatch`.
Run root: `runs/phase8-gate-b-20260905T072208Z-q17-a1`.
Logs: `staging/phase8-gate-b-20260905T072208Z-gpu1/slurm-12645346.{out,err}`.
Resources: debug / 00:29:00 / one GPU / 8CPU / 128G. Estimate before submission:
about10min for 1.7B, based on 92 bounded prompt-scoring calls with upper-tail<=2954
tokens, batch1, and A40 capacity; full-sequence FP32 vocabulary logits are about
1.8GB per2954-token tensor, with ample room beyond model weights on48GB A40.
This is a provisional runtime/memory estimate, not measured formal packing.
Actual resource/score/tensor/basis verification is still pending.

Queue admission snapshot after submission: job12645346 is PENDING with
`Reason=MaxJobsPerAccount`, `StartTime=Unknown`, accountroot/QOSdebug. Read-only
`sacctmgr show assoc where user=xhuang225` confirms root is this user's association;
QOSdebug has MaxJobsPA8 / MaxJobsPU10 / MaxSubmitJobsPU8. This is shared-account
queueing, not a launcher failure or evidence of invalid resources. No account/QOS
change or cancellation was performed. Sent planning a narrow nonterminal request
to permit the single10min smoke heartbeat while this debug job is PENDING, since
the usual RUNNING-for-one-minute creation condition cannot cover this wait.

Planning granted `loopscope_phase8_gate_b_pending_monitor_supplement.md`, bound by
control at `f1c56f124efd2de17886d52513bf612fd62b6109`. Executor rechecked actual
job12645346 still PENDING/MaxJobsPerAccount, then created the single10min heartbeat
**loopscope-phase8-gate-b-smoke**, statusACTIVE, destination exact executor task.
Creation was confirmed by automation_update. Prompt freezes job/run/log/host,
source-bound read-only SSH, quiet healthy checks, pause-before-terminal-self-wake,
exact planning fallback and no scheduling/scientific/next-Gate authority.
No second job or overlapping monitor exists. The first model must expose a valid
real path before the remaining4B debug is dispatched. Queue waiting now belongs
to this monitor; no manual polling loop or planning polling is required.

## 1.7B completed and verified; 4B continuation

The first heartbeat encountered a temporary `No route to host` and correctly did
not call the job failed. User later reported job12645346 completed, directly
resuming this executor. The VPN source had changed from10.21.0.80 to10.21.0.230;
read-only ifconfig followed by per-command source binding restored SSH. No network
configuration was edited. Monitor was PAUSED with automation_update confirmation
before subsequent implementation/scheduling; no extra self-wake was needed because
the user had already resumed the exact executor.

Fresh sacct: job12645346 and batch/extern steps COMPLETED, exit0:0, elapsed00:01:21,
nodegpu3-9. In-job verifier and separate fresh-process verifier both returned
VERIFIED_DEBUG for all four 1.7B cells, at unchanged producerc7e4e1f.
Actual env: torch2.3.1+cu121 / transformers4.51.3 / lm_eval0.4.11 / datasets5.0.0;
exact model/tokenizer revisionea980cb..., model dtypefloat16, A40 total50899648512
bytes, HFLM max_length32768. Producer elapsed44.0029824s, 92 prompt-scoring calls,
2.09076737calls/s; peak allocated5245561344 bytes, reserved5899288576 bytes, OOM0.

All native-repeat, native-adapter and Loop-vs-None-vs-zero comparisons were exact
score equality (no tolerance). All real continuations have length1; pre-answer
positions are2887,2898,2940,2953. Every K2 collector has8 rows (4 per t), everyK4
collector16 rows (4 per t); duplicate_calls=0 throughout. Four SMOKE_ONLY bases are
2048-dimensional, each fit from4 source identities at t1, with save/load and
transform roundtrip successful. Synthetic FP16/BF16 parallel/orthogonal/t0/other-token
checks all true. Spectral score changes are finite and nonzero, demonstrating the
actual path is affected, without reading gold or drawing accuracy conclusions:

| Cell | Maximum absolute raw choice-score change Spectral vs Loop |
| --- | --- |
| 12:15 K2 | 0.2578125 |
| 12:15 K4 | 0.38671875 |
| 6:9 K2 | 0.09375 |
| 6:9 K4 | 0.1015625 |

Following the successful first real path, submitted **12645484**, model-index0
(4B), same pool/code/launcher, four frozen window/K cells, debug29min/1GPU/8CPU/128G.
Estimated runtime before execution: within5min, based on measured1.7B44s producer,
roughly2.3x model parameter scale and additional loading headroom, comfortably
below29min. This does not replace the required4B measured memory/throughput.
Run root: `runs/phase8-gate-b-20260905T072208Z-q4-a1`.
Logs: `staging/phase8-gate-b-20260905T072208Z-gpu2/slurm-12645484.{out,err}`.
Fresh state: PENDING/MaxJobsPerAccount, no assigned node. No 1.7B rerun.

Updated the SAME automation `loopscope-phase8-gate-b-smoke` to job12645484/q4-a1/gpu2
and current verified VPN source10.21.0.230, statusACTIVE, 10min cadence; tool success
confirmed. Prior job12645346 is preserved as completed evidence. On 4B terminal,
pause monitor, verify/recover this exact attempt, then finish Gate B evidence and
send the restored GATE_B_FINAL_AUDIT to planning; C remains locked.
