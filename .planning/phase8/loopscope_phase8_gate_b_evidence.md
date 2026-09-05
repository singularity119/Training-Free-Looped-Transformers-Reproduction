# Phase 8 Gate B execution evidence

Date: 2026-09-05. Executor: `01a0704b-5798-7680-80a2-f1145f35f9ba`.
Planning/terminal recipient: `01a06fe9-c7bb-7d72-a906-234f301de317`.
Current execution result: **BLOCKED_ON_EXTERNAL_PUSH_APPROVAL; real debug NOT RUN**.
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
