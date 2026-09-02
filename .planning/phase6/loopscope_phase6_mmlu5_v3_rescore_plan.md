# LoopScope Phase 6 — MMLU 5-shot V3 trajectory rescore

## 1. Objective

Reuse the immutable, fully verified Gate M MMLU 5-shot validation trajectory and apply the latest
shared selector method exactly once. This is a post-hoc method-development rescore, not a new
trajectory acquisition and not prospective validation of V3.

The terminal output is one independently verified `SELECTED_WINDOW` or legal V3 `ABSTAIN`. No
model forward, GPU job, dataset rendering, validation outcome, test split, or loop experiment is
authorized.

## 2. Frozen source cell

- model/task context: `Qwen/Qwen3-4B-Instruct-2507 × cais/mmlu`, standard 5-shot prefix trajectory;
- source population: exact validation 1,531 identities / 57 subjects;
- source Gate M commit: `9446f0e4940a7ae762c95625f8bd891ac4090f6c`;
- source root:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase6-gate-m-mmlu5-formal-20260802T085444Z`;
- merged trajectory:
  `merge/merged_trajectory_records.jsonl`, SHA-256
  `0c6989abac8fab56d1661ffdfad9e291e2591b5eb6c87738cfd92282631a1d73`, 1,531 lines;
- merge receipt SHA-256:
  `c1b3b5f2dc3330caaf22bc4f81ec5a6fa5b367495094f9a8d451bf6ee6fde6df`;
- full verifier receipt SHA-256:
  `13b81088f15428161afdfcd97ec7583cce42cb9f043199ffb76d6c4ed87634ec`;
- old V2 freeze SHA-256:
  `a05ab56e9d9c38c2d98034bda829a6930660184759f27f6c9af38c3fad8ba95f`, historical comparison only.

## 3. Frozen V3 method

- method: `AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE`;
- version: `3.1.0`;
- canonical source: `.planning/selector_rules/aggregate_common_turn_v3_absolute_rate.md`;
- exact source hash is bound by the Gate N handoff and control at dispatch;
- candidate domain: central blocks 11...24, widths 3/4/5/6, exactly 42 candidates;
- selector projection: identity/category/H/D only;
- category-macro point and bootstrap estimand;
- eligibility: `Scorable AND NetPositive AND RateStable AND AggregateCommonTurn`;
- ranking: `S_RATE_TURN=sqrt(G_H*G_K*Q_H*Q_K)`;
- bootstrap: 2,000 replicates, seed 20260801;
- combined-rank frequency threshold: 0.80;
- terminal: `SELECTED_WINDOW`, `ABSTAIN_NO_V3_ELIGIBLE`, or
  `ABSTAIN_COMBINED_RANK_UNSTABLE`.

Hidden diagnostics, old V1/V2 shape fields, outcome fields, gold, correctness, accuracy and loop
results do not enter V3.

## 4. Gate N execution

1. Admit exact policy/control/plan/method hashes and exact protected local dirty state.
2. Implement the smallest V3 selector, closed output schema and independent verifier by reusing
   existing Phase 6 selector utilities where their semantics are identical.
3. Run focused synthetic tests for all V3 formula, turn, tie, bootstrap and terminal rules.
4. Commit/push only the new Gate N implementation paths, fast-forward the clean HPC2 clone, and
   run one CPU-only analysis into a fresh write-once sibling root.
5. Run one separate fresh-process verifier that fully recomputes the 42 candidates, bootstrap,
   ranking, digests and decision from the immutable Gate M trajectory.
6. Write one concise Chinese report under `资产/报告/phase6/`, explicitly labeling the result as
   post-hoc V3 rescoring of the already observed MMLU 5-shot trajectory.

## 5. Stop condition

Stop after one valid V3 freeze and one independent verifier PASS. Do not tune V3, rerun the model,
submit Slurm, inspect outcome fields, or start a loop experiment. A valid V3 ABSTAIN is a terminal
scientific result.
