# GATE_F_HANDOFF

```text
Project/phase: LoopScope Phase 5 post-terminal extension
Planning/audit thread: 019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread: 019f8eb5-2f08-7c80-afc2-b4ae52464162
Executor title: execute-LoopScope-MultiWidth-Top4Outcome-第5阶段-Gate F
Gate: F — full MMLU outcome for Gate E global ranks 1–4
```

## Objective

对 Gate E 已冻结总榜前四个 variable-width windows 各运行一次完整 Qwen3-4B-Base ×
MMLU 5-shot test-14,042 outcome，在四个 cells 全部完整闭合后一次性计算并报告：

- accuracy；
- 相对 Phase 5 canonical no-loop baseline 的 paired gain 与 95% CI；
- wrong→correct / correct→wrong；
- exact McNemar。

Gate PASS 只表示四个 outcome 和统计闭合，不要求结果为正。

## Non-objectives and claim boundary

- 不重跑 baseline 或 `15:18`；只精确复用 Gate C/D canonical baseline artifact。
- 不运行 Gate E rank 5 以后窗口，不做 width/K/alpha/cache sweep。
- 不重新排序、调 selector、修改 eligibility 或根据 smoke/partial outcome 改候选。
- 不把任何正结果称为 prospective selection success。统一标签为
  `POST_TERMINAL_MULTIWIDTH_TOP4_OUTCOME_EXPLORATORY`。
- 不改变 Phase 5 原终态 `H5_ABSTAIN_WITH_RANKING_RESULT`。

## Authoritative sources

- policy: `loopscope-tflt/AGENTS.md`
- control: `.planning/loopscope_phase5_control.md` 第 19 节
- this handoff
- Gate E score artifact:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-e-multiwidth-mid40-20260723T100836Z/phase5_multiwidth_mid40_scores.json`
  with SHA256 `a4bd8b7be3b531b89961667b1cbfba9db0f4020e01bfa8efc97a3ccd43826e44`
- Gate E manifest receipt:
  same root `/phase5_multiwidth_mid40_manifest_receipt.json`
  with SHA256 `4ccf5e31093d280896675997287b18dc118d0ea8dbd1c128729977013134c78e`
- canonical Gate C/D root:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-c-repair-identity-20260722T072256Z`
- Gate C completion receipt file/internal SHA256:
  `4f15ae291afb2fcb00992cfd3bb4d216386874cfb1ff55bf1d3a9ad7c8e9c8d4` /
  `979b4e61fc32b3eeb2cebdd5d6a5662f9bdbc0c87eb296a67f7be97ad0ee0ac6`
- Gate C test ordered identity SHA256:
  `2c4557079fca1a7c071460f779ed714805fc96c3391560825cd34386d4051532`
- reusable implementation patterns:
  `src/tflt/loopscope/phase5_outcome.py` and
  `src/tflt/loopscope/phase5_outcome_analysis.py`

Existing Phase 5 artifacts are read-only inputs. Do not modify, move, append to, or re-seal them.

## Starting provenance

- local repository:
  `/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt`
- branch: `loopscope`
- local base commit: `8ce09c07fa494b3c9c2a2844ebd0e2c2698358d3`
- expected local dirty state: clean
- public `origin/loopscope`: `f33a37a4e5c0cfc461184a3471277897c77cf83b`
- HPC2 source checkout:
  `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`
- expected HPC2 checkout commit: `f33a37a4e5c0cfc461184a3471277897c77cf83b`, clean
- audited venv:
  `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711`
- protected reference base:
  `4f59bd93eca4da3cbf458a93508f91c5b23912bc`

The local branch being two commits ahead of public origin is intentional. Preserve both local-only
report/analyzer commits. This Gate has no public push authorization.

## Frozen four-cell order

| array index | Gate E rank | role | width | window | score |
|---:|---:|---|---:|:---:|---:|
| 0 | 1 | `gate_e_rank1` | 5 | `13:17` | 23.122176074886756 |
| 1 | 2 | `gate_e_rank2` | 6 | `15:20` | 20.459460381229054 |
| 2 | 3 | `gate_e_rank3` | 6 | `12:17` | 16.292884498978600 |
| 3 | 4 | `gate_e_rank4` | 3 | `15:17` | 11.683278605031116 |

The implementation must derive and verify these rows from the exact Gate E score JSON, not merely
hard-code the text above. Exact order/membership/width/window/score drift is a blocker.

## Frozen outcome recipe

```text
model_repo=Qwen/Qwen3-4B-Base
model_revision=tokenizer_revision=906bfd4b4dc7f14ee4320094d8b41684abff8539
dtype=bfloat16
dataset=cais/mmlu
dataset_revision=c30699e8356da336a370243923dbaf21066bb9fe
task_group=mmlu
split=test
num_fewshot=5
lm_eval=0.4.11
metric=acc,none
apply_chat_template=false
fewshot_as_multiturn=false
k=3
operator_body_calls_per_prefill=3
iteration_mode=block
strategy_cli=euler
strategy_kernel=damped_euler_alias
alpha=1.0
beta=0.0
step_size=1/3
total_horizon=1
cache_strategy=first
decode_mode=bypass
log_samples=true
batch_size=16
```

Only window start/end/width varies across the four cells.
The user explicitly confirmed that loop count must match the earlier Phase 5 full outcome:
`K=3`, three operator-body calls, Euler step `1/3`, total horizon `1`. No K sweep is authorized.

Formal population per cell:

- 14,042 exact test identities;
- 57 subjects;
- ordered identity SHA256
  `2c4557079fca1a7c071460f779ed714805fc96c3391560825cd34386d4051532`;
- missing/duplicate/extra all zero.

## Baseline reuse and final analysis

Before formal launch, bind by stat/SHA only—without parsing scientific values—the canonical Gate C
baseline cell:

`<GATE_C_ROOT>/formal/cell-00-baseline_no_loop/`

After all four new cells are scheduler-terminal success and a structural sealer proves complete
identity/artifact closure, perform exactly one outcome parse/analysis. Recompute baseline accuracy
and correctness from its original raw artifact; do not copy the previously reported `73.016664%`
as a constant.

Use subject-stratified joint paired bootstrap across baseline plus the four new cells:

```text
R=2000
seed=20260723
shared draw stream across all five columns
SE ddof=1
percentile 95% CI with linear q*(R-1)
```

For each window report:

- `accuracy_percent`;
- `gain_percentage_points` versus baseline;
- paired 95% CI;
- four transition counts;
- exact two-sided McNemar p;
- Gate E rank/score/width/window.

Existing exact `15:18` may appear as a clearly labeled read-only reference row, but must not be
mixed into the new four-cell bootstrap or rerun.

## Information barrier

Until all four formal cells are complete and the sealer has closed exact 4 × 14,042 membership:

- do not open/parse `results.json`, logged samples, evaluator metrics, predictions, correctness,
  accuracy, gain, flip, or window performance;
- monitor only scheduler state, exit code, file existence/size/SHA, producer completion receipts,
  identity sidecars and structural receipts;
- do not inspect one completed cell while others are running/failed;
- do not modify code, candidate set, recipe, scheduler membership or formal root after formal launch.

After structural closure, authorize exactly one unseal/analysis and one independent verifier pass.
No outcome-informed repair is allowed. If analysis/verifier has a deterministic pre-read defect,
report `BLOCK`; planning decides any replacement authority.

## Allowed implementation and paths

Prefer the smallest dedicated sidecar, without modifying the original width-4 Gate C/D modules:

- `configs/loopscope/phase5_multiwidth_top4_outcome_card.json`
- `src/tflt/loopscope/phase5_variable_width_outcome.py`
- `scripts/loopscope/run_qwen4base_phase5_gate_f.py`
- `tests/test_loopscope_phase5_gate_f.py`
- final report:
  `报告/LoopScope_第五阶段多宽度Top4全量Outcome.md`

Allowed remote actions:

- read/write only inside one new timestamped run root with prefix
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-f-multiwidth-top4-outcome-`;
- stage exact new files into that root with SHA binding; run with staged `PYTHONPATH` before the
  immutable HPC2 checkout source;
- use existing model/data caches and audited venv offline;
- run one four-cell limit smoke, then one four-task formal Slurm array;
- use `emergency_gpu`, A800-equivalent GRES, array `0-3%4`, one GPU, 8 CPU, 64G, 24h,
  `--no-requeue`, batch size 16;
- before job creation, the executor may choose an available equivalent A800 partition/QoS if
  `emergency_gpu` admission is unavailable, recording the exact choice;
- install one low-frequency executor-owned read-only monitor for the formal array, with terminal
  self-resume/relay behavior required by the research Gate protocol;
- create one local single-purpose commit after tests pass. No push.

## Smoke and formal sequence

1. Freeze a Gate-level card/manifest binding the exact Gate E top4 rows and existing baseline/test
   artifacts.
2. Run targeted local tests and remote CPU dry-run.
3. Run exactly one four-cell limit smoke using the same fixed four identities used by Phase 5 Gate C.
   Validate window width/start, `operator_body_calls=3`, restore closure, identity projection and
   non-empty evaluator artifacts. Smoke accuracy is non-scientific and must not be reported.
4. Seal smoke structurally. Only after smoke PASS, freeze and submit the formal array.
5. Monitor at low frequency. When all four tasks reach terminal success, stop/pause the monitor and
   resume this exact executor.
6. Seal formal completeness without parsing outcome values.
7. Perform exactly one unseal/analysis plus one verifier.
8. Write the concise Chinese result report and terminal packet.

## Repair and retry budget

- Up to two executor-owned low-risk implementation/smoke repair loops before formal launch,
  preserving failed roots/attempts when any forward occurred.
- After formal launch: no code/science/candidate/recipe mutation.
- A formal task may be retried once in a fresh attempt directory only if its failure receipt proves
  evaluator/model forward requests `=0` and scientific sample records `=0`; otherwise `BLOCK`.
- No automatic retry/requeue/resubmit. Retry eligibility must be mechanically proven and the
  executor must record the one-time action.
- Any partial scientific output, identity mismatch, OOM after forward, or mixed cell completion with
  a failed cell requires `BLOCK`; do not salvage or inspect successful cells.

## Explicitly forbidden

- No fifth candidate, no baseline/15:18 rerun, no widening beyond the exact four cells.
- No selector/eligibility/score changes, no Gate E reranking, no K/alpha/cache/decode/model/task
  sweep, no representation intervention.
- No modification of original Phase 5 card, Gate B trajectory/selector/panel, Gate C/D roots,
  Phase 5 final report, or Gate E canonical artifacts/report.
- No modification of `wrapper.py`, `strategies.py`, `cache.py`, `config.py`, `eval_runner.py`,
  `phase5_outcome.py`, or `phase5_outcome_analysis.py`.
- No rebase, force-push, merge-main, destructive cleanup, overwrite, deletion, or public `git push`.
- No partial outcome inspection, no outcome-informed repair, no automatic next Gate.

## Required artifacts

Under the fresh Gate F run root:

1. frozen card/launch manifest and staged-code SHA manifest;
2. smoke structural seal;
3. formal submission and scheduler closure receipt;
4. four identity sidecars and producer completion receipts;
5. formal structural completion seal;
6. one unseal marker;
7. `phase5_gate_f_analysis.json`;
8. `phase5_gate_f_window_metrics.csv`;
9. `phase5_gate_f_bootstrap_digest.json`;
10. `phase5_gate_f_verifier_receipt.json`;
11. `phase5_gate_f_summary_zh.md`;
12. `phase5_gate_f_terminal_receipt.json`.

Persist aggregate statistics only after analysis; do not add copied raw per-sample payloads beyond
the evaluator's original required outputs.

## Minimum decisive verification

Executor verification:

- targeted tests prove exact top4 derivation, arbitrary widths 3/5/6 accepted, four cells only,
  unchanged recipe, baseline reuse, identity closure, information barrier and one-shot analysis;
- limit smoke proves three body calls and restore closure for all four windows;
- formal completion proves four tasks terminal success and exact 4 × 14,042 / 57 identity closure;
- verifier independently recomputes each accuracy/gain/CI/flip/McNemar and confirms one unseal,
  one analysis, no forbidden candidate or prior-artifact mutation.

Planning acceptance will inspect only:

1. exact four-cell membership/order and scheduler terminal success;
2. one cell's raw-sample accuracy/gain recomputation against canonical baseline;
3. verifier/one-shot/information-barrier receipt.

## Agility budget

- Smallest increment: dedicated four-cell variable-width adapter reusing Gate C/D identity and
  analysis primitives; do not generalize the old width-4 panel framework.
- Earliest safe experiment: after targeted tests/dry-run, run the four-cell limit smoke immediately.
- Minimum decisive checks: exact candidates, structural loop activity, 4 × 14,042 closure, one
  analysis/verifier.
- Stop rule: once the four requested accuracies and paired baseline statistics close, do not add
  figures, extra windows, reruns or broader audits.

Estimated compute: approximately 1.1–1.3 A800 GPU-hours for the four formal cells, plus a small
four-cell smoke.

## Terminal requirement

Send exactly one `GATE_F_FINAL_AUDIT` or `BLOCK` to planning thread
`019f8604-4717-7be2-8bf8-9d4a26a3d7f7`, with visible delivery confirmation.

The terminal packet must include:

- executor identity/title, starting/final Git state and exact files changed;
- smoke/formal job IDs, exact 0–3 scheduler states and GPU accounting;
- run root, implementation/input/output hashes;
- 4 × 14,042 / 57 identity closure;
- four-window table with Gate E rank/score, accuracy, gain/CI, flips and McNemar;
- baseline recomputation and optional clearly separated `15:18` reference;
- unseal/analysis/verifier counts and result;
- deviations, retry/repair use, actions not taken and protected-state confirmation;
- requested decision `PASS / PASS_WITH_FIXES / BLOCK`.

Long-job monitor must freeze:

- executor `019f8eb5-2f08-7c80-afc2-b4ae52464162`;
- planning thread `019f8604-4717-7be2-8bf8-9d4a26a3d7f7`;
- exact job IDs/run root after submission;
- terminal condition: tasks 0–3 all `COMPLETED/0:0`, or any failure;
- on terminal: pause/delete monitor, send `AUTOMATION_TERMINAL_RESUME` to this executor;
- fallback: `AUTOMATION_RELAY_REQUIRED` to planning with paste-ready resume packet.
