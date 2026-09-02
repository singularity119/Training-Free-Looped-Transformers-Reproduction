# GATE_K_HANDOFF

Project/phase: LoopScope Phase 6 — MMLU 5-shot Prefix Trajectory Follow-up  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor thread: `019fc119-30b4-7dd0-92bf-e6f2f994c8c8`  
Authorized title: `execute-LoopScope-MMLU5-Prefix-RBRV2-第6阶段-Gate K`  
Gate: K — 5-shot contract, fresh prompt projection, minimal producer/verifier adaptation

## Objective

Starting from the accepted Gate J implementation, freeze and implement the smallest outcome-blind
contract layer needed to change standard MMLU evaluation from 0-shot to 5-shot while preserving the
same Qwen3-4B-Instruct model, validation population, prefix probe, trajectory metrics, V2 selector,
TFLT identity, and no-outcome terminal. Produce a fresh hash-closed 5-shot validation prompt
projection using dev demonstrations and make Gate L's real debug smoke importable.

Non-objectives: no model-weight load, CUDA/GPU/Slurm, native forward, formal trajectory, selector
execution, loop execution, test split, validation target/gold, correctness, accuracy, gain, outcome,
plotting, Gate L, or Gate M.

## Authoritative admission

- policy: `loopscope-tflt/AGENTS.md`, SHA-256
  `2cf835f3ab2eb4c70d29536ba267f0ebe7546af10358c3339bfcd4b828217677`
- control: `.planning/loopscope_phase6_control.md`, SHA-256
  `a7ab33593b10b5862d3dcde92eb61342e4270611f300af96cbf4bc45984d927e`
- plan: `.planning/loopscope_phase6_mmlu5_prefix_plan.md`, SHA-256
  `b7c9714d797d12ce1886e9146b88cdc05b0b4e3278532d72bf13d3121a3c8ff0`
- orchestration skill: SHA-256
  `73261e636b1a6cca1c123718670ac7a930af2f56f8cdfbce507868491c1ca35f`

Read all four completely before mutation. Stop on any Gate/executor/hash disagreement.

## Starting provenance and protected state

- repository: `loopscope-tflt`
- local clone: `/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt`
- branch: `loopscope`
- exact base/local origin: `37b454bca209b97512325475587cc919c13a4040`
- required ancestor: `4f59bd93eca4da3cbf458a93508f91c5b23912bc`
- expected dirty state: exactly planning-owned unstaged `M AGENTS.md`; no staged paths
- protected `AGENTS.md` SHA-256: `2cf835f3ab2eb4c70d29536ba267f0ebe7546af10358c3339bfcd4b828217677`
- do not stage, commit, restore, overwrite, or otherwise mutate `AGENTS.md` or `.planning/`.

## Frozen scientific contract

- model/tokenizer: `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`, 36 layers, hidden size 2560, bfloat16;
- dataset: `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`;
- target population: validation 1,531 identities / 57 subjects in the same canonical identity order as Gate J;
- evaluator: lm-eval 0.4.11 standard MMLU choice-loglikelihood, `num_fewshot=5`, `fewshot_split=dev`, plain prompt, `apply_chat_template=false`, `fewshot_as_multiturn=false`, generation=false, no direct-letter fallback;
- continuations: exact ` A`, ` B`, ` C`, ` D`; retain the accepted tokenizer/token IDs and native LM-head closure;
- prompt projection: fresh 5-shot rendering required; every target has exactly five standard dev demonstrations. Dev demonstration answers are permitted renderer inputs; validation target/gold is forbidden;
- probe: last non-padding token of the rendered 5-shot prefix, one zero-loop native forward, `use_cache=false`, `output_hidden_states=true`;
- boundaries/metrics: exact Gate J B0...B36 choice entropy, KL-to-final, hidden RMS-L2/cosine/cosine-distance, and 36 adjacent angular distances; hidden/angular remain diagnostic-only;
- selector: exact Gate J `RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE`, blocks 11...24, widths 3/4/5/6, 42 candidates, subject-stratified 2,000 bootstrap draws, seed 20260801, frequency threshold 0.80, projection fields identity/category/H/D only;
- 0-shot records or selector results must not enter the 5-shot population, bootstrap, or decision;
- no full loop outcome is part of this follow-up.

## Allowed actions

- Modify only the smallest necessary tracked set under:
  - `configs/loopscope/phase6_mmlu5_*`
  - `src/tflt/loopscope/phase6_mmlu5_*`
  - `scripts/loopscope/*phase6_mmlu5*`
  - `tests/test_loopscope_phase6_mmlu5*`
  - `docs/loopscope_phase6.md`
  - `src/tflt/cli.py` only if needed to register the Gate K command.
- Reuse Gate H/I/J modules directly when science is identical; do not build a new framework or copy unrelated historical validators.
- Run targeted local tests, compile/help/dry-run/self-test, and one outcome-safe CPU prompt-projection closure.
- Connect to `hpc2-hkustgz` with bounded SSH to inspect the dedicated clone, pinned environment, task source, cached model/tokenizer metadata, and cached MMLU validation/dev renderer inputs.
- Read validation identity/question/choices and dev demonstrations including dev example answers solely to render the standard 5-shot prefix. Do not read validation targets.
- Create one fresh write-once Gate K CPU/projection root under
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-k-mmlu5-*`.
- After targeted verification, explicitly stage only authorized tracked files, commit one purpose, push `loopscope`, and ff-only sync the clean dedicated HPC2 clone when needed.
- Use up to three concurrent bounded subagents; parent alone owns Git, remote mutation, integration, and terminal delivery.

## Forbidden actions

- No GPU/CUDA/Slurm/scheduler query or submission, model-weight instantiation, forward, generation,
  replay, loop, formal selector, test split, validation gold/label, correctness, accuracy, gain, flip,
  outcome, raw logits/full-vocabulary persistence, or hidden-tensor persistence.
- No Gate L/M work, no full historical test suite unless a concrete broad regression makes it material,
  no changes to 0-shot immutable run artifacts/selector/report, and no deletion/overwrite of any run root.
- No stash/reset/rebase/force-push/merge, destructive cleanup, protected checkout mutation, or staging
  of planning-owned files.

## Required engineering verification and artifacts

Minimum expected tracked outputs:

1. machine-readable 5-shot card with exact model/data/evaluator/renderer/probe/metrics/selector contract;
2. closed schema or explicit audited reuse of the Gate J trajectory schema;
3. fresh validation-1531 5-shot prompt-projection manifest/receipt with 57 subjects, exact canonical identity order, exactly five dev demonstrations per target, renderer/source/version hashes, and prompt/tokenization hashes without raw prompt persistence;
4. minimal Gate L runtime dry-plan proving the real smoke path imports but does not execute;
5. focused tests for 5-shot versus 0-shot prompt separation, demonstration count/order, target exclusion, last-valid-token probe semantics, forbidden-field rejection, and 42-candidate selector compatibility;
6. independent Gate K verifier/self-test that fails on the material normal-path contract violations above.

Suggested commands, adapting exact module names if the implementation chooses an equally narrow name:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest \
  tests.test_loopscope_phase6_mmlu5_gate_k tests.test_cli -q
PYTHONPYCACHEPREFIX=/tmp/loopscope-p6-k-pyc python3 -m py_compile \
  src/tflt/loopscope/phase6_mmlu5_*.py \
  scripts/loopscope/*phase6_mmlu5*.py src/tflt/cli.py
python3 scripts/loopscope/run_qwen4_phase6_mmlu5_gate_k.py --help
python3 scripts/loopscope/run_qwen4_phase6_mmlu5_gate_k.py --dry-run
python3 scripts/loopscope/run_qwen4_phase6_mmlu5_gate_k.py --self-test
git diff --check
```

Must-pass acceptance:

- exact 5-shot/dev/plain/no-chat/no-generation contract and unchanged model/validation/probe/metrics/selector closure;
- fresh 1531/57 prompt projection with no missing/extra/duplicate identities and exactly five dev demonstrations per target;
- validation target/gold/test/outcome access flags all false; no model/CUDA/Slurm execution;
- targeted tests and independent verifier PASS; Gate L path importable;
- one clean pushed commit, remote ref exact, final local dirty state only protected `M AGENTS.md`.

Optional hardening must not delay terminal delivery once these criteria pass.

## Agility and repair budget

- Smallest increment: adapt the accepted Gate J renderer/projection contract from 0-shot to standard 5-shot and reuse every scientifically identical helper.
- Earliest real evidence: fresh CPU renderer projection on validation-1531; do this after the minimal targeted contract tests, not after a broad framework build.
- Minimum decisive checks: exact prompt membership/order/hash, forbidden target access, schema/selector compatibility, and dry-plan import.
- Stop rule: once those checks pass, commit/push and deliver; do not add outcome machinery, portability layers, generalized task abstractions, or exhaustive tamper matrices.
- Executor-owned low-risk engineering repair loops: unlimited within this Gate and frozen authority; preserve failed write-once attempts.
- Audit-returned repair cap: one ordinary bundled cycle, second only for the same unresolved material root cause or a regression directly introduced by the first.
- Escalate only for model/data/split/fewshot renderer/probe/metric/selector changes, validation target access, GPU/scheduler authority, destructive action, or inability to make material progress.

## Planning acceptance boundary

Planning will verify only one to three decisive facts: exact Git/protected state, the fresh 1531/57
5-shot projection/forbidden-access receipt, and the targeted verifier result. Cross-Gate synthesis is
deferred to the Gate M phase-end audit.

## Terminal requirement

Send exactly one `GATE_K_FINAL_AUDIT` or genuine `BLOCK` to planning thread
`019fb3de-2298-75f2-a083-0dca453ea79c`, request `PASS / PASS_WITH_FIXES / BLOCK`, and retain visible
delivery confirmation. If delivery is unavailable, output a paste-ready
`TERMINAL_DELIVERY_UNCONFIRMED` packet. Gate L and Gate M remain locked.

Long-job continuation: none; Gate K authorizes no scheduler job or heartbeat.

