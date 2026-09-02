# GATE_D_HANDOFF

```text
Project/phase=LoopScope Phase 5
Hypothesis=H5_QWEN4B_BASE_MMLU_TRAJECTORY_SELECTION_V1
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019f8978-eb01-7ab3-b37e-9b797a40a262
Executor title=execute-LoopScope-Trajectory-Selection-第5阶段-Gate D
Gate=D
State=AUTHORIZED_AFTER_VISIBLE_DELIVERY
Terminal event=GATE_D_FINAL_AUDIT or BLOCK
Phase terminal=after planning audit of Gate D
```

## 1. Objective and admission

Gate C 已由 planning 裁决 `PASS`。唯一 canonical sealed source 为：

```text
repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope
base_commit=684bd0ccaf1e508a645d0ab4d1884ea5939fc9c0
remote_repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
run_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-c-repair-identity-20260722T072256Z
formal_job=10032961
completion_receipt_file_sha256=4f15ae291afb2fcb00992cfd3bb4d216386874cfb1ff55bf1d3a9ad7c8e9c8d4
completion_receipt_manifest_sha256=979b4e61fc32b3eeb2cebdd5d6a5662f9bdbc0c87eb296a67f7be97ad0ee0ac6
completion_status=SEALED_COMPLETE_READY_FOR_PLANNING_AUDIT
cell_count=8
record_count_per_cell=14042
subject_count_per_cell=57
missing_duplicate_extra=0
ordered_identity_sha256_all_cells=2c4557079fca1a7c071460f779ed714805fc96c3391560825cd34386d4051532
```

Gate D 的唯一目标是：对上述八个完整 cells 做一次性 unseal，按 Phase 5 card 已冻结的统计合同生成独立可复核分析、终端标签和中文总结。不得重新运行模型、修改 selector/panel，或为更有利结果重做任何旧 Gate。

## 2. Frozen inputs and cell identities

正式 panel 顺序和角色固定为：

1. `no-loop` baseline；
2. known fixed comparator `15:18`；
3. blind High3：`4:7`, `5:8`, `13:16`；
4. blind Low3：`8:11`, `9:12`, `6:9`。

Gate B selector 决策固定为：

```text
selected_window=null
window_decision=ABSTAIN_NO_POINT_ELIGIBLE
eligible_count=0
selection_frequency=null
raw_high3=4:7,5:8,13:16
blind_low3=8:11,9:12,6:9
```

必须重新核对 card、selector freeze、panel、test manifest 和 Gate C completion receipt 的 file/internal hashes 与 Gate C receipt 闭合后，才可创建 one-shot unseal marker。任何不一致先 `BLOCK`，不得打开 outcome payload。

## 3. Authorized implementation scope

允许最小新增/修改：

```text
loopscope-tflt/src/tflt/loopscope/phase5_outcome_analysis.py
loopscope-tflt/scripts/loopscope/run_qwen4base_phase5_gate_d.py
loopscope-tflt/tests/test_loopscope_phase5_gate_d.py
loopscope-tflt/docs/loopscope_phase5.md
报告/LoopScope_第五阶段实验结果与结论.md
```

远端只允许在 canonical run root 下新建 write-once `gate_d/` 分析目录、one-shot unseal marker、analysis artifacts 和 verifier receipt。不得修改 Gate C sealed 文件或 identity sidecar。

允许两个 purpose-scoped commits：

1. unseal 前的 analyzer/verifier/targeted tests；
2. unseal 后的 canonical 中文结果报告与 runbook terminal update。

普通 fast-forward push；本地、origin 和 HPC2 必须在 unseal 前 clean 并闭合到 implementation commit。不得 force-push、rebase、merge main 或修改 protected Phase 1-4/Phase 5 A-C artifacts。

## 4. One-shot statistical contract

所有八 cells 必须按 canonical identity 精确 join。primary metric=`acc,none`；内部 fraction，报告 percentage points。

冻结 paired bootstrap：

```text
method=57-subject-stratified joint paired bootstrap
replicates=2000
seed=20260723
rng=random.Random/randrange
same_joint_draw_stream_for_all_cells_and_contrasts=true
within_each_subject=resample_n_subject_identities_with_replacement
aggregate=all 14042 sampled identities equally
SE=ddof=1
CI=linear percentile at q*(R-1), q=0.025/0.975
```

必须输出：

- 每个 loop window 相对 baseline 的 paired Δaccuracy 与 95% CI；
- High3 mean absolute gain；
- High3−Low3 gain enrichment 及 paired CI；
- `selected_vs_baseline`, `selected_vs_15_18`, `selected_panel_local_regret`：因合法 ABSTAIN 必须为 `null/not_applicable`，不得把 raw rank-1 冒充 selected；
- 可另报 `raw_high3_top1` 或 panel-best regret，但必须明确是 ranking diagnostic，不是 selection success；
- 每窗 `wrong→correct`, `correct→wrong`, `correct→correct`, `wrong→wrong`；
- exact two-sided McNemar（discordant pair 的 exact binomial，双侧并 cap at 1）；
- 六个 blind High3/Low3 windows 的 selector score vs gain Spearman，fixed comparator 不进入 blind rank correlation；
- ABSTAIN coverage/eligible count/selection frequency；
- geometry/output 与 geometry/gain diagnostics，只作 secondary，不进入 selector 或 terminal success 判据。

必须把以下概念分开：ranking/enrichment、absolute gain、known comparator competitiveness、合法 ABSTAIN。公开论文 `73.27/73.61/+0.34 pp` 只作背景，不是硬验收阈值。

## 5. Terminal decision grammar

互斥 labels 只能从 card 列表中选择：

```text
H5_PROSPECTIVE_SELECTION_SUPPORTED
H5_RANKING_ENRICHMENT_SIGNAL_ONLY
H5_ABSOLUTE_BENEFIT_ONLY
H5_KNOWN_COMPARATOR_HIT_ONLY
H5_ABSTAIN_WITH_RANKING_RESULT
H5_NOT_SUPPORTED
H5_INCONCLUSIVE
```

本次冻结 selector 已合法 ABSTAIN，因此：

- `H5_PROSPECTIVE_SELECTION_SUPPORTED` 不可用；
- 若完整 panel 可解析、paired analysis 与 verifier 通过，primary terminal label 为 `H5_ABSTAIN_WITH_RANKING_RESULT`，并用独立字段报告 ranking enrichment 是否为正/CI 是否跨零、High3 absolute gain 是否为正、comparator 表现及各 window 结果；
- 只有 analysis/verifier 所需证据无法完整闭合时才用 `H5_INCONCLUSIVE`；
- 不得因 High3−Low3 为正而把 ABSTAIN 改称 selected success。

Phase 5 claim boundary 仍只是该 Qwen3-4B-Base × MMLU cell 的 prospective/cross-scale evidence；不得声称得到跨模型通用自动选窗器。

## 6. Independent verifier and artifacts

analyzer 与 verifier 必须在同一命令族中分离实现。verifier 从 raw sealed samples 重新构造：

- 8 × 14,042 canonical identity equality；
- 每 cell accuracy、每窗 paired delta、flip counts 和 exact McNemar；
- bootstrap draw-index digest、High3/Low3 contrasts 和 CIs；
- blind score/gain Spearman；
- ABSTAIN fields、terminal label 和 claim-boundary flags。

推荐 write-once artifacts：

```text
gate_d/phase5_gate_d_unseal_once.json
gate_d/phase5_gate_d_analysis.json
gate_d/phase5_gate_d_window_metrics.csv
gate_d/phase5_gate_d_bootstrap_digest.json
gate_d/phase5_gate_d_verifier_receipt.json
gate_d/phase5_gate_d_summary_zh.md
gate_d/phase5_gate_d_terminal_receipt.json
```

artifacts 必须记录 source file SHA256、implementation commit/hash、card/selector/panel/completion receipt hashes、unseal_count=1、analysis_count=1 和 verifier result。不得持久化新的完整 logits/hidden states；可保留现有逐样本 outcome 原件及必要的 joined correctness/identity audit summary，不复制出新的未受控 raw payload。

## 7. Execution order and agility budget

1. 重读 policy/control/handoff，核对 executor binding、Git 和 canonical source hashes。
2. 只实现 Phase 5 analyzer、verifier 与 targeted synthetic fixtures；复用 Phase 3/4 已审计统计 primitives，避免建设通用框架。
3. unseal 前运行 targeted tests、py_compile、Gate A verifier、dry-run/preflight、`git diff --check` 和 protected-path diff；不跑无关 full suite。
4. 提交并 push implementation commit；HPC2 clean fast-forward 同步；再次 fail-closed 核对 completion receipt。
5. 在 HPC2 login CPU 上运行一次 vectorized analyzer+verifier，默认 15 分钟有界；不得加载模型、CUDA 或 GPU。若正常路径明确超过该界限，可 `BLOCK` 请求 CPU Slurm 权限，不得自行改为 GPU 或拆成多次 unseal。
6. verifier PASS 后，把 canonical 中文 summary 同步到报告路径并做一个 result-report commit；不得为美化结果修改统计或重新 unseal。
7. 发送唯一 `GATE_D_FINAL_AUDIT`；若 pre-unseal failure 可窄修但仍受 repair budget，任何 post-unseal analysis/verifier failure 必须保留 evidence 并 `BLOCK`，不得第二次读取/重算来覆盖工件。

Stop rule：targeted synthetic contract、pre-unseal closure 和 implementation hash 一旦通过，立即执行唯一 unseal；不增加额外 hardening、统计口径或探索性分析。

## 8. Repair budget and forbidden actions

unseal 前最多两次 executor-owned 窄工程修复；unseal 后 repair budget=0。禁止：

- 重新提交/重跑 Gate C cells，重新加载模型或使用 GPU；
- 修改 selector、eligibility、ABSTAIN、High3/Low3、panel、window、K、recipe、metric、bootstrap seed/R、CI 或 terminal grammar；
- 查看部分结果后再调代码、阈值、报告标准或选择窗口；
- 覆盖/删除任何 Gate B/C artifact、旧 invalid root 或 unseal marker；
- 扩展 K、width、model、dataset、intervention 或创建 Phase 6；
- 让本线程或其 subagent 取代独立 Gate D executor。

## 9. Terminal packet

向 planning thread `019f8604-4717-7be2-8bf8-9d4a26a3d7f7` 发送恰好一次 `GATE_D_FINAL_AUDIT` 或 `BLOCK`，并保留可见 delivery confirmation。终态包必须列出：

- final commits/dirty/protected-path state；
- exact commands/tests/exit codes；
- one-shot marker、analysis、verifier、summary、terminal receipt paths 和 SHA256；
- 8-cell accuracy/gain/CI/flip/McNemar 表与 High3/Low3/rank correlation；
- ABSTAIN fields、唯一 terminal label、claim boundary；
- unseal_count/analysis_count/verifier result；
- 未执行的 rerun/GPU/selector change/扩展实验。

executor 不得自行宣布 Phase 5 PASS；planning 将对 Gate D 做一次终端验收和跨 Gate 简洁审计。
