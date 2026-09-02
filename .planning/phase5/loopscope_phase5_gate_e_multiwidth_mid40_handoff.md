# GATE_E_HANDOFF

```text
Project/phase: LoopScope Phase 5 post-terminal extension
Planning/audit thread: 019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread: 019f8e65-eb1d-7cf0-a438-f26a91468679
Executor title: execute-LoopScope-MultiWidth-Mid40-第5阶段-Gate E
Gate: E — width 3/4/5/6 central-40% score-only ranking
```

## Objective

复用 Phase 5 Gate B 已闭合的 MMLU validation 1,531 条、57 subjects、B0…B36 native
no-loop trajectories，在不做任何新 forward、GPU/Slurm 或 outcome 访问的条件下，对
width `3/4/5/6` 的中央 40% 候选窗口重新计算严格 entropy–KL CONSENSUS 分数，给出完整
42 行总排名、分 width 排名、eligibility 诊断和简洁中文总结。

本 Gate 只回答“既有 validation trajectory 在中央深度区域如何给多宽度窗口打分”。
它不是新的 prospective selector freeze，也不评价 accuracy/gain。

## Authoritative sources

- policy: `loopscope-tflt/AGENTS.md`
- control: `.planning/loopscope_phase5_control.md` 第 17 节
- this handoff
- Phase 5 frozen card: `loopscope-tflt/configs/loopscope/phase5_card.json`
- Phase 5 Gate B trajectory source:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-b-repair-b36-native-20260722T035452Z/formal/validation1531_phase5_trajectories.jsonl`
  with file SHA256
  `aa4894e632f557ddc4614e88d06ec1fc4c525463be7531aef69fa3c28671edf9`
- trajectory manifest file SHA256
  `343689e4d2c4d8bf40f2237233a5333f1a7d53cef219e226e56d017d5aad21c6`
- existing width4 selector report:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-b-repair-b36-native-20260722T035452Z/selector/selector_report.json`
  with file SHA256
  `a6e7eb396b1da90152c2dc56a73e1866b3e98a40f044cfcc29e28091ae16beb3`
- reusable mathematical pattern only:
  `loopscope-tflt/src/tflt/loopscope/phase3_variable_width.py`

Authority precedence remains user instruction > live Git/artifacts > AGENTS.md > current control >
runbook/history. Phase 3 artifacts are implementation references, not Phase 5 data.

## Starting provenance

- local repository:
  `/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt`
- branch: `loopscope`
- local base commit: `01d1c6906508985928b55b0e7bb84a97986b15b5`
- expected local dirty state: clean
- public `origin/loopscope`: `f33a37a4e5c0cfc461184a3471277897c77cf83b`
- HPC2 source checkout:
  `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`
- expected HPC2 source commit: `f33a37a4e5c0cfc461184a3471277897c77cf83b`
- protected reference base:
  `4f59bd93eca4da3cbf458a93508f91c5b23912bc`

The local report-only commit ahead of public origin is intentional and must be preserved.

## Frozen candidate and scoring contract

### Candidate set

Decoder block indices are `0...35`. “Exclude the first and last 30%” is frozen as full-window
containment inside blocks `11...24`.

| width | allowed starts | inclusive windows | count |
|---:|---|---|---:|
| 3 | 11…22 | 11:13 … 22:24 | 12 |
| 4 | 11…21 | 11:14 … 21:24 | 11 |
| 5 | 11…20 | 11:15 … 20:24 | 10 |
| 6 | 11…19 | 11:16 … 19:24 | 9 |

Published/ranked candidate count must equal 42 exactly. The strict contrast references
`s-1,s+1,s-w,s+w` may lie outside the central candidate zone but must be legal same-width
windows in the full `0...36-w` universe. Reference windows are never published as candidates.

### Metrics and score

For every identity and `W(s,w)=[s,s+w-1]`, corresponding to `B_s -> B_(s+w)`:

```text
E_raw(s,w)=mean_i[H_(i,s)-H_(i,s+w)]
K_raw(s,w)=mean_i[D_(i,s+w)-D_(i,s)]
E_rate=E_raw/w
K_rate=K_raw/w
CEdrop_raw=E_raw-K_raw
CEdrop_rate=E_rate-K_rate

O_H=E_rate(s,w)-0.5*(E_rate(s-1,w)+E_rate(s+1,w))
O_K=K_rate(s,w)-0.5*(K_rate(s-1,w)+K_rate(s+1,w))
F_H=E_rate(s,w)-0.5*(E_rate(s-w,w)+E_rate(s+w,w))
F_K=K_rate(s,w)-0.5*(K_rate(s-w,w)+K_rate(s+w,w))
z_Q=Q/max(SE_bootstrap(Q),1e-12)
score=min(z_OH,z_OK,z_FH,z_FK)
```

Bootstrap is the Phase 5 Gate B joint subject-stratified sample bootstrap:
`R=2000`, `seed=20260722`, shared draw stream across all widths/starts/metrics, `ddof=1`,
linear percentile 95% CI.

Point eligibility remains diagnostic and uses the frozen strict rule:

- center `E_rate>0` and `K_rate>0`;
- every `s±1,s±w` reference has `E_rate<0` and `K_rate<0`;
- lower 95% CI of all four contrasts is strictly positive.

Rank order:

- global 42: `score descending`, then `width ascending`, then `start ascending`;
- within width: `score descending`, then `start ascending`.

Do not emit `selected_window`. Use `ranking_top1`, `eligible_count`, and
`post_terminal_exploratory_no_selection=true`.

### Required backward compatibility

For width4 starts `11...21`, compare against the existing Phase 5 selector report. New values
must exactly reproduce, within absolute tolerance `1e-10`:

- `score`;
- `z_OH,z_OK,z_FH,z_FK`;
- `point_eligible` and exact failure reasons.

`E_rate/K_rate` are the old raw width4 E/K divided by four; the z scores are scale invariant.
Any mismatch is material and must `BLOCK`; do not reinterpret the old selector.

## Allowed actions

- Read applicable policy/control/handoff and the exact Gate B validation trajectory and selector
  artifacts named above.
- Add only the smallest Phase 5 multi-width analyzer/card/launcher/tests needed, preferably:
  - `configs/loopscope/phase5_multiwidth_mid40_card.json`
  - `src/tflt/loopscope/phase5_variable_width.py`
  - `scripts/loopscope/run_qwen4base_phase5_gate_e.py`
  - `tests/test_loopscope_phase5_variable_width.py`
- Write a new result report:
  `报告/LoopScope_第五阶段多宽度中央40%选窗评分.md`.
- Use read-only SSH to `hpc2-hkustgz`.
- Create one fresh write-once HPC2 result/staging root under
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/`
  with prefix `phase5-gate-e-multiwidth-mid40-`.
- Transfer exact new analyzer files to that staging root and run them on HPC2 login CPU with
  `PYTHONPATH=<staging>/src:<HPC2 checkout>/src`; record file SHA256 and runtime provenance.
- Create one local single-purpose commit after targeted tests pass.
- Use up to two executor-owned low-risk repair loops for ordinary implementation defects.

## Explicitly forbidden

- No MMLU test split, gold, correctness, prediction, accuracy, gain, flip, Gate C sealed payload,
  Gate D artifact, or any outcome path may be opened, parsed, joined, copied, or reported.
- No model/tokenizer/dataset/cache load; no forward; no GPU/CUDA; no Slurm job.
- No loop outcome for width 3/5/6 and no reuse of width4 outcomes in this analysis.
- No modification of Phase 5 original card, trajectory records/manifest, selector freeze/report,
  outcome panel, Gate C/D artifacts, original final report, or Phase 1–4 artifacts.
- No modification of `wrapper.py`, `strategies.py`, `cache.py`, `config.py`, `eval_runner.py`.
- No geometry-augmented score, selector-weight tuning, K sweep, dynamic gating, intervention,
  edge-aware alternative score, or outcome-informed candidate adjustment.
- No rebase, force-push, merge-main, destructive cleanup, overwrite, or deletion of prior artifacts.
- No `git push`: the public-origin authorization covered only exact `f33a37a4`; this Gate has no
  public publication authorization.
- No future Gate or additional width/outcome execution.

## Required outputs

In the fresh HPC2 Gate E root:

1. `phase5_multiwidth_mid40_scores.json`
2. `phase5_multiwidth_mid40_scores.csv`
3. `phase5_multiwidth_mid40_summary.json`
4. `phase5_multiwidth_mid40_summary_zh.md`
5. `phase5_multiwidth_mid40_verifier_receipt.json`
6. one Gate-level provenance/manifest receipt binding inputs, code hashes and all output hashes.

The CSV/JSON must contain all 42 candidates with at least:

`global_rank,width,width_rank,start,window,boundary_transition,E_raw,K_raw,CEdrop_raw,E_rate,`
`K_rate,CEdrop_rate,O_H,O_K,F_H,F_K,z_OH,z_OK,z_FH,z_FK,score,point_eligible,`
`eligibility_failures,comparison_starts`.

The summary must report:

- candidate counts and exact start ranges by width;
- global top10;
- top5 per width;
- eligible rows, if any, without calling one “selected”;
- width4 backward-compatibility result;
- bootstrap draw digest;
- explicit `outcome_values_consumed=false`;
- explicit `post_terminal_exploratory_no_selection=true`.

## Minimum decisive verification

Executor verification:

1. Candidate enumeration is exactly 12/11/10/9 and every published window is fully contained in
   blocks 11…24.
2. Synthetic formula/bootstrap/ranking tests pass for widths 3/4/5/6.
3. Input closes at 1,531 unique validation identities / 57 subjects / 37 boundaries, with the
   exact file hashes above and no forbidden fields.
4. Width4 starts 11…21 reproduce existing score/z/eligibility/failure reasons to `1e-10`.
5. Independent verifier recomputes all 42 rows and output hashes from the raw validation trajectory.

Do not run the repository-wide suite unless the targeted changes reveal a credible broad-regression
risk. `py_compile`, targeted unittest/pytest, `git diff --check`, and protected-path diff are enough.

Planning acceptance will inspect only:

- exact 42-row candidate closure;
- width4 backward compatibility;
- one independently recomputed non-width4 row plus outcome-blind receipt.

## Agility budget

- Smallest increment: adapt the existing Phase 3 variable-width mathematical pattern to the exact
  Phase 5 trajectory schema and central-40% candidate filter; do not build a general framework.
- Earliest real computation: after targeted synthetic and width4 fixture tests, run one login-CPU
  analysis against the existing 1,531 trajectories.
- Decisive evidence: 42 finite rows, exact width4 reproduction, verifier PASS, no outcome access.
- Stop rule: once these pass and the Chinese table/report exists, do not add hardening, figures,
  outcome comparison, or extra widths.

## Terminal requirement

Send exactly one `GATE_E_FINAL_AUDIT` or `BLOCK` to planning thread
`019f8604-4717-7be2-8bf8-9d4a26a3d7f7`, with visible delivery confirmation.

The terminal packet must include:

- executor thread ID and title;
- starting/final branch, local commit and dirty state;
- exact files changed;
- exact commands/tests and exit codes;
- HPC2 run root and code/input/output hashes;
- 42-row/count closure, global top10 and top5 by width;
- exact eligible rows or eligible_count=0;
- width4 backward-compatibility result;
- verifier result and outcome-blind confirmation;
- deviations/repair loops;
- actions not taken;
- requested decision `PASS / PASS_WITH_FIXES / BLOCK`.

If cross-thread delivery is unavailable or unconfirmed, emit the same paste-ready packet labeled
`TERMINAL_DELIVERY_UNCONFIRMED`. Do not claim `PASS`; only planning/audit decides.
