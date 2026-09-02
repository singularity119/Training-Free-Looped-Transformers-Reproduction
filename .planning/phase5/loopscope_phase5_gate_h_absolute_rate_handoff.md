# GATE_H_HANDOFF

```text
Project/phase=LoopScope Phase 5 post-terminal method-development extension
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019f8fe5-0880-7173-9650-7c4c76ee475d
Authorized executor title=execute-LoopScope-Relative-Biphasic-Absolute-Rate-第5阶段-Gate H
Gate=H
Method=RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE
State=AUTHORIZED
Delivery=CONFIRMED
```

## 1. Objective

严格执行
`.planning/loopscope_phase5_relative_biphasic_v2_absolute_rate_contract.md`：
复用 1.7B/4B 两份 canonical validation no-loop trajectory 和冻结 bootstrap，在不读取
任何 outcome 的前提下，完整复现 V1 admission，再以中心窗口自身的
`S_RATE=sqrt(G_H*G_K)` 对 `NewEligible ∧ RateStable` 候选离线重排，输出两个模型各自
selected 或明确 ABSTAIN。

这是一个完整但单一的 CPU-only Gate。executor 最终只能向 planning thread 发送一次
`GATE_H_FINAL_AUDIT` 或 `BLOCK`，不得自判 PASS，也不得进入后续 Gate。

## 2. Admission and starting provenance

- Gate G 已由 planning 判定 PASS；executor authority 已撤销。
- repo:
  `/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt`
- branch: `loopscope`
- authorized base commit:
  `118de08fe45844edf406e811c2ece91886d3f6ed`
- protected reference ancestor:
  `4f59bd93eca4da3cbf458a93508f91c5b23912bc`
- origin/loopscope at dispatch:
  `f33a37a4e5c0cfc461184a3471277897c77cf83b`
- expected dirty state: only pre-existing untracked `报告/` containing:
  - `报告/LoopScope_第五阶段多宽度Top4全量Outcome.md`
  - `报告/LoopScope_第五阶段相对双相反转双模型离线重选窗.md`
- preserve both files exactly; do not stage, overwrite, delete, rename, clean or reset them.
- HPC2 source checkout is clean at
  `f33a37a4e5c0cfc461184a3471277897c77cf83b`; it is a read-only runtime reference and
  must not be synchronized or mutated for this Gate.

Before mutation, verify repository root/branch/base/dirty state and reread:

- `loopscope-tflt/AGENTS.md`;
- `.planning/loopscope_phase5_control.md`;
- this handoff;
- the frozen V2 contract.

Any binding, base, policy or dirty-state conflict must produce `BLOCK`.

## 3. Authorized writes

Only these repository paths may be added:

- `configs/loopscope/phase5_relative_biphasic_v2_absolute_rate_card.json`;
- `src/tflt/loopscope/phase5_relative_biphasic_v2_absolute_rate.py`;
- `scripts/loopscope/run_qwen_phase5_gate_h.py`;
- `tests/test_loopscope_phase5_relative_biphasic_v2_absolute_rate.py`.

The V1 card/module/tests/launcher and all prior artifacts are protected read-only inputs.
Prefer importing narrow V1 primitives; do not copy/refactor a general framework.

One Chinese report may be created outside the repo at:

```text
/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/报告/LoopScope_第五阶段V2绝对变化率双模型离线重选窗.md
```

Do not stage this outer report. A single local purpose commit containing exactly the four
authorized repo paths is permitted after targeted tests pass. No push is authorized.

Remote writes are limited to one staged copy of those four files and result artifacts under
the exact fresh root:

```text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-h-relative-biphasic-v2-absolute-rate-20260723T164940Z
```

Do not overwrite or append to any prior run root.

## 4. Frozen science

The exact authoritative formula, candidates, inputs, hashes, bootstrap, tie rules, schema and
failure states are in the V2 contract. In particular:

- V1 `NewEligible` admission is unchanged and must reproduce before V2 ranking;
- 1.7B candidates=26, bootstrap R=2000 seed=20260716;
- 4B candidates=42, bootstrap R=2000 seed=20260722;
- `S_RATE=sqrt(G_H*G_K)` is defined only for finite strict-positive point rates;
- `RateStable` uses the frozen within-window H/K one-sided lower max-t bound with
  `M_b=max_m[(G_point-G_b)/SE_m]`;
- `C_RATE=NewEligible ∧ RateStable`;
- top tie rel/abs tolerance=`1e-12`;
- point top exact-window bootstrap frequency threshold=`0.80`, equality passes;
- V1 fields and flags must reproduce, floating atol=`1e-10`;
- raw score values are model-local and may not be compared across models.

Do not change formula, threshold, candidate, shape contract, seed, draw stream or claim after
seeing results.

## 5. Execution stages and agility budget

1. **Admission/preflight**: close local Git state and exact canonical input hashes. A cheap
   bounded read-only SSH probe is enough; do not re-audit passed Gates.
2. **Minimal implementation**: add only the V2 card/analyzer/launcher/focused tests listed
   above. Reuse V1 trajectory parsing, candidate generation, shape and bootstrap primitives.
3. **Focused synthetic verification**: run the ten decisive contract tests, V1 focused tests,
   `py_compile`, launcher help/dry-run and `git diff --check`. Do not run repeated broad full
   suites; at most one broad suite only if a concrete shared-code regression risk appears.
4. **Formal offline analysis**: stage exact files and run exactly one login-CPU analyze in the
   fresh root. No GPU/Slurm/model/data/outcome access.
5. **Independent verification**: in a fresh process, independently recompute V1 compatibility,
   rate statistics, ranking/frequency/final decisions and byte-compare primary artifacts.
6. **Report and provenance**: write the concise Chinese report, one Gate-level manifest/
   verifier receipt, and one purpose commit.

Stop rule: once targeted tests, exact input closure, one analyze and one verifier pass, report
the Gate. Do not add framework hardening, extra scores, widths, seeds or outcomes.

## 6. Must-pass acceptance

- exact 1,531 identities / 57 subjects for both trajectories;
- exact 26/42 candidate membership;
- exact input hashes from the frozen contract;
- V1 candidate identity/flags/failure reasons exact and all V1 numeric fields within
  `1e-10`;
- all V2 formulas, finite checks, max-t sign, tie and frequency denominator follow the
  contract;
- all ten focused tests pass;
- analyzer count=1, fresh-process verifier count=1;
- verifier result `PASS`;
- full machine-readable tables, cross-model summary, Chinese summary, manifest and verifier
  receipt exist under the fresh root;
- report clearly separates `NewEligible`, `RateStable`, `ranking candidate`, `rank-1`,
  `selected` and positive gain;
- `outcome_values_consumed=false`, `model_or_dataset_loaded=false`,
  `forward=false`, `gpu_cuda_slurm=false`;
- protected V1/current runtime paths remain unchanged.

The observed selected/ABSTAIN identity is a scientific result, not an acceptance target.

## 7. Repair and escalation

- executor-owned low-risk repair loops: at most 2 before formal analyze;
- planning-audit-returned repair cycles: at most 1;
- after formal output creation, no outcome-informed or result-informed formula/threshold/
  candidate repair is allowed;
- a reproducible implementation bug may be fixed only if it preserves frozen science and
  writes a fresh sibling root when provenance would otherwise be ambiguous;
- any scientific ambiguity, input mismatch, V1 compatibility mismatch, nonfinite analysis,
  information-barrier breach, required protected-path change or authority expansion must
  produce `BLOCK`.

## 8. Explicitly forbidden

- modifying V1 or prior Phase 1–5 canonical artifacts;
- loading model/tokenizer/dataset, any forward, CUDA/GPU/Slurm;
- opening or parsing any outcome/accuracy/test/gold/correctness/prediction/gain/flip artifact;
- running any loop/full accuracy cell;
- adding another score, threshold, seed, width/model/task or intervention;
- public push, force-push, rebase, merge, destructive cleanup;
- creating or executing the next Gate.

## 9. Terminal packet

Send exactly one structured event to planning thread
`019f8604-4717-7be2-8bf8-9d4a26a3d7f7`:

```text
GATE_H_FINAL_AUDIT

Project/phase:
Gate:
Execution thread:
Result: AUDIT_REQUESTED
Authorized and final branch/commit/dirty:
Files changed:
Exact commands/tests/exit codes:
HPC2 host/run root:
Input hashes/population/candidate closure:
V1 backward compatibility:
1.7B full V2 summary and selected/ABSTAIN:
4B full V2 summary and selected/ABSTAIN:
Decision-critical artifact paths/hashes:
Verifier result:
Deviations/repair loops:
Completed state:
Actions not taken:
Protected-state confirmation:
Decision requested: PASS / PASS_WITH_FIXES / BLOCK
```

If blocked, send `BLOCK` with the narrow blocker, evidence, preserved state and exact decision
or authority required. Retain visible delivery confirmation; if cross-thread delivery is
unavailable, output a paste-ready `TERMINAL_DELIVERY_UNCONFIRMED` packet instead of claiming
delivery.
