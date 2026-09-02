# GATE_N_HANDOFF

Project/phase: LoopScope Phase 6 — MMLU 5-shot V3 trajectory rescore  
Planning/audit thread: `019fb3de-2298-75f2-a083-0dca453ea79c`  
Authorized executor thread: `019fc2f1-812c-73a3-b5e3-011222a2f539`  
Authorized title: `execute-LoopScope-MMLU5-Prefix-ACTV3-第6阶段-Gate N`  
Gate: N — immutable Gate M trajectory, one V3 analyze, one fresh-process verify

## Objective and terminal

Implement the exact latest shared V3 selector, apply it once to the immutable, fully verified Gate M
MMLU 5-shot validation trajectory, and independently recompute/verify the complete decision. Gate N
terminates with one valid V3 `SELECTED_WINDOW` or legal V3 `ABSTAIN`.

This is explicitly post-hoc method development on an already observed trajectory. It is not a new
trajectory experiment and must not be described as prospective validation of V3. Do not run model
forward, GPU, Slurm, dataset rendering, test/outcome analysis, or any loop experiment.

## Exact authority admission

Read all five files completely before mutation and require exact SHA-256:

- policy: `loopscope-tflt/AGENTS.md`  
  `2cf1da569a21e233940e1348b4bd851f1724ef83051e94fab2fc1ee47516f58f`;
- control: `.planning/phase6/loopscope_phase6_control.md`  
  `2bb33410998f753e271c62cd1b5e77cd6aaa338603560f01f5ecf6c6996754b5`;
- Gate N plan: `.planning/phase6/loopscope_phase6_mmlu5_v3_rescore_plan.md`  
  `bba6198301e1aaaa42980e0b1c78afaf1f6e059757e88fc70fb52161a61f3c68`;
- shared V3 method: `.planning/selector_rules/aggregate_common_turn_v3_absolute_rate.md`  
  `2a7aaac6bcd4e0d4995758c8e25a720469c1d6e2afd65e7a1143103f31556644`;
- `research-gate-orchestrator` skill:  
  `186d38ed283aa393ffd94158f5ef5854f959c4ac8e40a4f3451d03a7a16115cb`.

Stop before mutation on any Gate/executor/path/hash conflict. The shared rule is the canonical owner
of every V3 formula, tolerance, tie, bootstrap, ranking, output and terminal semantic; this handoff
does not override it.

## Repository admission and protected dirty state

- repo: `loopscope-tflt` dedicated local clone;
- branch: exact `loopscope`;
- authorized base/local origin: `9446f0e4940a7ae762c95625f8bd891ac4090f6c`;
- required historical ancestor: `4f59bd93eca4da3cbf458a93508f91c5b23912bc`;
- HPC2 dedicated clone initially clean at the same exact commit;
- the authorized local starting dirty state is exactly these seven pre-existing, unstaged paths:
  - `AGENTS.md` SHA-256 `2cf1da569a21e233940e1348b4bd851f1724ef83051e94fab2fc1ee47516f58f`;
  - `docs/loopscope_phase2.md` SHA-256 `021f4fe274dc4130e4287d4dfb26aece0ee49fd78e335fa93d6c0aa4b3488c0d`;
  - `docs/loopscope_phase4.md` SHA-256 `29609d2a9a2f8d539944c148e5e69b671556536522b1748508b6f682a8084879`;
  - `docs/loopscope_phase5.md` SHA-256 `a4faf9f95657db47e782b84907390de5681a7e6b958d20489e69e62c6b72ef1a`;
  - `docs/loopscope_phase6.md` SHA-256 `2f5c797b47c4e692a9e269a978bd859ceea6b25e68079e285156615e5e39bda8`;
  - `scripts/loopscope/run_qwen4_phase4_p4b.py` SHA-256 `7c5614a3845fe34f7a232b1038be36019f17d36f9c5006eb3df74aeb695f9e68`;
  - `src/tflt/loopscope/phase4_outcome.py` SHA-256 `5b315c64af1ac0ed5e9cf4d284c462cffb50a771f82ebc24f5a9f745af072df0`.

Do not stage, commit, restore, overwrite, reformat, or otherwise modify those seven paths. Do not
stash, reset, checkout, clean, rebase, merge, force-push, or disturb unrelated user/planning state.
Use explicit-path staging only.

## Immutable scientific input

Remote host: `hpc2-hkustgz`.  
Dedicated clone: `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`.  
Gate M root:
`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-m-mmlu5-formal-20260802T085444Z`.

Required input closure:

- `manifest/formal_manifest.json` SHA-256
  `5fec780360b38676bda4cbb9cd7b8ac43ffa9480bce3283acb4f267cb288c6be`;
- `manifest/freeze_receipt.json` SHA-256
  `d57de8faed5e66a0f6c36f1af43602bc008349bd8d68426cd85dbd3a695b1468`;
- `merge/merge_receipt.json` SHA-256
  `c1b3b5f2dc3330caaf22bc4f81ec5a6fa5b367495094f9a8d451bf6ee6fde6df`;
- `merge/merged_trajectory_records.jsonl` SHA-256
  `0c6989abac8fab56d1661ffdfad9e291e2591b5eb6c87738cfd92282631a1d73`,
  exact 1,531 records / 57 subjects;
- `merge/verifier_receipt.json` SHA-256
  `13b81088f15428161afdfcd97ec7583cce42cb9f043199ffb76d6c4ed87634ec`, status PASS;
- historical V2 `selector/selector_freeze.json` SHA-256
  `a05ab56e9d9c38c2d98034bda829a6930660184759f27f6c9af38c3fad8ba95f`;
- historical V2 selector verifier SHA-256
  `c93c736f2d37c615e4f34536008d3c7e4e2557fe9c09f2b40afe2c3abbe24010`.

Only the merged sanitized trajectory and outcome-safe provenance/receipts may feed V3. The V2
freeze may be read only for an explicitly labeled historical comparison; it must not determine V3
eligibility, ranking, ties or terminal state. Do not mutate any Gate M artifact.

## Frozen V3 contract

Implement the shared method byte-for-byte semantically, including:

- `METHOD_ID=AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE`, version `3.1.0`;
- selector projection exactly `identity/category/H/D`;
- equal-category-macro point and bootstrap estimand;
- central blocks 11...24, widths 3/4/5/6, exact 42 candidates;
- `EligibleV3 = Scorable AND NetPositive AND RateStable AND AggregateCommonTurn`;
- deterministic two-segment constant-rate turn calculation, unique tau/tolerance and `Q>0.5`
  semantics exactly as the shared rule;
- no direction-sign reversal requirement, no direction margins, no `BiphasicStable`, no
  bootstrap tau search/vote and no `tau_pair_frequency`;
- point ranking by `S_RATE_TURN=sqrt(G_H*G_K*Q_H*Q_K)`;
- bootstrap 2,000, seed 20260801, exact draw/index serialization and macro max-t RateStable;
- fixed point tau in bootstrap combined-rank scoring, anchored tie semantics and canonical
  width/start tie-break;
- combined-rank selection frequency threshold 0.80;
- legal terminal set exactly `SELECTED_WINDOW`, `ABSTAIN_NO_V3_ELIGIBLE`,
  `ABSTAIN_COMBINED_RANK_UNSTABLE`.

Hidden RMS-L2, hidden cosine, hidden cosine-distance, adjacent angular metrics and all legacy V1/V2
diagnostics remain outside selector projection, admission, turn, ranking and frequency.

## Authorized implementation paths

Use only the smallest needed new tracked paths:

- `configs/loopscope/phase6_mmlu5_v3_selector_freeze_schema.json`;
- `src/tflt/loopscope/phase6_mmlu5_v3_selector.py`;
- `src/tflt/loopscope/phase6_mmlu5_v3_verifier.py`;
- `scripts/loopscope/run_qwen4_phase6_mmlu5_gate_n.py`;
- `tests/test_loopscope_phase6_mmlu5_gate_n.py`.

You may omit an unnecessary path, but may not edit existing tracked files. If a material normal-path
reason requires an existing implementation file, stop and request a bounded planning supplement.
Do not build a general selector framework.

## Ordered execution and agility budget

1. Admit exact authority, repository state and immutable input hashes. Inspect only the V3 method,
   the minimum existing Phase 6 selector/runtime helpers needed for reuse, and the safe trajectory
   schema.
2. Implement the smallest pure-CPU V3 producer plus independent full-recomputation verifier.
3. Run focused synthetic tests covering the shared spec's decisive cases: 42-candidate closure,
   macro aggregation, constant-rate/degenerate rejection, same-sign fast-to-slow acceptance,
   asynchronous tau rejection, tau and numeric ties, `Q>0.5`, RateStable, combined score/rank
   frequency, deterministic tie-break, and both legal ABSTAIN branches.
4. Commit only the authorized new paths and push the exact commit to `origin/loopscope`. Fast-forward
   the clean HPC2 dedicated clone to that exact commit. Parent executor alone owns Git and remote
   mutation.
5. Create one fresh write-once root under
   `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-n-mmlu5-v3-rescore-*`.
   Run exactly one valid V3 analysis from the immutable merged trajectory. If an engineering defect
   prevents any valid decision, preserve that failed root, repair within the frozen method, and use
   a fresh sibling root; never tune science after a valid decision exists.
6. In a separate fresh Python process, independently recompute all inputs, 42 rows, bootstrap draws,
   eligibility, turn metrics, ranking, tie sets, frequencies, digests and terminal decision. Require
   exact semantic agreement and verifier PASS.
7. Write a concise Chinese report under `资产/报告/phase6/` with `POST_HOC_V3_RESCORE` prominently
   stated, the V3 terminal, selected half-open/inclusive window mapping if any, top candidates and
   why `15:18` did or did not enter/rank. Do not claim prospective validation or loop benefit.

Earliest real computation is the CPU analysis after focused tests. No debug/GPU/Slurm preflight and
no heartbeat are needed because this Gate is synchronous CPU-only. Stop immediately after one valid
freeze, one fresh verifier PASS and the report.

## Allowed authority and repair

- Baseline permission to use bounded SSH on `hpc2-hkustgz`, read the exact Gate M root, fast-forward
  the clean dedicated clone, and write only the fresh Gate N root.
- Explicit permission to push the single-purpose Gate N commit to the existing exact
  `origin/loopscope` branch after remote identity verification.
- Executor-owned low-risk engineering repair loops are unlimited while model/data/candidates/
  formulas/thresholds/seed/bootstrap/tie semantics/input hashes and information barriers remain
  exact. Preserve failed roots and commits as applicable.
- Up to three concurrent bounded subagents are allowed inside this Gate; they may not own Git,
  remote writes, selector execution, terminal delivery or another Gate.

## Forbidden actions

- No model/tokenizer/dataset load, forward, generation, CUDA, GPU, Slurm or scheduler action.
- No validation gold/target, test split, answer, prediction, correctness, accuracy, gain, flip,
  outcome, raw prompt, token IDs, logits, full-vocabulary probability or hidden tensor access.
- No trajectory reacquisition, no change to Gate M artifacts, no mixing with 0-shot or MMLU-Pro.
- No V3 tuning, threshold change, candidate change, selector retry after a valid decision, outcome-
  informed comparison or full loop experiment.
- No next Gate action.

## Must-pass evidence

- exact method/input/repository closure;
- focused tests and compile/dry-run PASS;
- input membership exactly 1,531/57 and selector projection only identity/category/H/D;
- exact 42 candidates and complete public plus verifier-critical V3 fields;
- one write-once V3 freeze and one fresh-process verifier PASS with identical decision, ranking,
  bootstrap draw digest and semantic digests;
- output is one legal V3 terminal and is clearly separated from the historical V2 ABSTAIN;
- information barriers all false and protected dirty paths remain byte-identical;
- local/origin/HPC2 commit exact and HPC2 clone clean after execution.

## Terminal delivery

Send exactly one `GATE_N_FINAL_AUDIT` or genuine `BLOCK` to planning thread
`019fb3de-2298-75f2-a083-0dca453ea79c`. Include executor identity, exact commits/status, tests,
remote root, artifact hashes, V3 terminal and top ranking summary, verifier result, deviations,
information-barrier flags and actions not taken. Request planning decision
`PASS / PASS_WITH_FIXES / BLOCK`. Do not self-approve or continue beyond Gate N.
