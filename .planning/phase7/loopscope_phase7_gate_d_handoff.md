# GATE_D_HANDOFF

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=D
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a00522-11a9-70d1-80e4-5d4401deb68c
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-V3-Scoring-第7阶段-Gate D
AUTHORIZATION_STATE=AUTHORIZED
TERMINAL_EVENT=GATE_D_FINAL_AUDIT_OR_BLOCK
NEXT_GATE_AUTHORITY=NONE
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
```

本 handoff 的协作与生命周期必须完整遵守
`/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md` 及其
`references/protocol.md`：一 Gate 一独立 executor；task 标题必须保持
`execute-LoopScope-V3-Scoring-第7阶段-Gate D`；executor 可在确实缩短本 Gate 时使用最多三个
有界 subagents，但独占 external actions、结果整合和 terminal delivery；退出前必须向 planning task
`01a0013e-71c3-7c90-a547-4059b462dc7e` 投递并确认唯一 `GATE_D_FINAL_AUDIT` 或真正的 `BLOCK`。
无法确认投递时必须输出 `TERMINAL_DELIVERY_UNCONFIRMED` paste-ready packet，绝不静默终止或跨 Gate。

## 1. Objective 与 stop condition

只读 Gate C 三模型 sealed sanitized trajectories，以冻结的
`AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE@3.1.0` 对每模型独立完成一次正式 V3.1 scoring、完整
candidate/source-data 封存和 fresh-process independent verification。三个模型各获得一个合法
`SELECTED_WINDOW`、`ABSTAIN_NO_V3_ELIGIBLE` 或
`ABSTAIN_COMBINED_RANK_UNSTABLE`，且 verifier 全部 PASS 后发送终包并停止。

合法 `ABSTAIN` 是科学终态，不得触发调参、selector variant 或 Gate C rerun。输入、公式、candidate
membership 或 verifier 不闭合才是工程 `BLOCK`。不得自行绘图或进入 planning plotting checkpoint、
Gate E。

## 2. Authoritative sources 与 provenance

执行前完整重读并遵守：

- `research-gate-orchestrator` 的 `SKILL.md` 与 `references/protocol.md`；
- `hpc2-hkustgz-ssh` skill（若访问 HPC2）；
- `loopscope-tflt/AGENTS.md`；
- `.planning/phase7/loopscope_phase7_control.md`；
- `.planning/phase7/loopscope_phase7_scientific_contract.md`；
- `.planning/phase7/loopscope_phase7_plan.md`；
- `.planning/selector_rules/aggregate_common_turn_v3_absolute_rate.md`；
- `loopscope-tflt/docs/loopscope_phase7.md`；
- 本 handoff。

优先级为用户最新指令 > live Git/HPC facts > `AGENTS.md` > control > 本 handoff/plan/runbook >
历史。executor ID、title、Gate、authority 或 frozen provenance 不一致时，在 mutation 前 `BLOCK`。

```text
repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
required branch=loopscope
frozen commit=bf37259a71ef4220236f51e5b659141748653f0b
remote checkout=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
remote frozen commit=bf37259a71ef4220236f51e5b659141748653f0b
Gate C sealed root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase7-gate-c-20260815T091500Z-packed-formal-recovery7/sealed
qwen input=<sealed root>/qwen25_3b/trajectory.jsonl; L=36; expected=1531/57; candidates=42
llama input=<sealed root>/llama32_3b/trajectory.jsonl; L=28; expected=1531/57; candidates=26
gemma input=<sealed root>/gemma2_2b/trajectory.jsonl; L=26; expected=1531/57; candidates=26
```

Gate C receipts and merge receipts are admission evidence and remain immutable. Do not overwrite、move、
expand 或修补 Gate C roots。Local staged membership 必须保持为空；七个 protected tracked dirty
paths 不得 reset、checkout、stash、clean、覆盖、暂存或提交。

## 3. Frozen V3.1 science

```text
selector=AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE@3.1.0
input_projection=identity,subject,H[0...L],D[0...L]
point_estimand=57-subject equal-category macro trajectory
bootstrap=within-subject equal-category-macro joint bootstrap
replicates=2000
seed=20260801
draw_map=one canonical map shared across H/D/boundaries/windows/widths and all three models
candidate_rule=t(L)=ceil(0.30*L), central containment, widths 3/4/5/6
candidate_order=width ascending then start ascending
eligible=Scorable AND NetPositive AND RateStable AND AggregateCommonTurn
score=S_RATE_TURN=sqrt(G_H*G_K*Q_H*Q_K)
combined_rank_frequency_threshold=0.80
```

三模型只作 model-local aggregation、eligibility、ranking 与终态；不得跨模型平均 raw score 或制造
共同 selected window。V1/V2 gates、旧 shape fields、hidden diagnostics、test/outcome 均不得回流。

## 4. Shortest authorized execution path

1. 只读核对 control、exact executor/title、frozen commit、remote clean state、三模型 Gate C
   verifier receipts 和 sealed input sizes；不重复 Gate C 全审计。
2. 使用现有 `run_phase7_v3.py` / `verify_phase7_v3.py` 的正式路径做 cheap CLI/import admission。
   Gate D 是 CPU-only：不加载模型或数据集，不申请 GPU，不提交 Slurm。
3. 在新的 write-once Gate D sibling root 中，用同一次冻结 invocation set 对三模型分别运行
   `R=2000, seed=20260801, --formal`；三模型可并行，但必须有独立 output/log/receipt，且并行策略
   不得造成 nested-thread oversubscription 或改变 canonical result。
4. 每模型分析输出后，由 fresh process 从原始 sanitized projection 完整重算 candidate domain、macro
   curves、turn candidates、bootstrap、eligibility、rank/frequency、tie-break 与终态；不能只校验自报字段。
5. 三模型全部 PASS 后封存：full candidate table、boundary aggregate、angular/hidden diagnostic aggregate、
   subject count、selector summary、analysis provenance 和 fresh verifier receipt。只在全部闭合后生成一份
   安全的三模型 manifest/summary；不得输出 per-identity prompt/token/gold/outcome。
6. 发送唯一终包并停止。不得绘图、冻结 Gate E top-3 panel 或进入 Gate E。

若 Gate D 现有实现暴露普通、材料且低风险的工程缺陷，bound executor 可在 Phase 7 V3 专用路径内做
最小修复，必须先给出失败复现，补 targeted regression test，作 purpose-specific commit/push/remote
fast-forward，并在 fresh Gate D root 完整重跑三模型。不得修改数学合同、阈值、seed、replicates、
candidate rule 或合法终态。若修复会改变科学合同或需读取被禁信息，立即 `BLOCK` planning。

## 5. Must-pass 与信息屏障

- 三模型 candidate membership 精确为 `42/26/26`，无 missing/duplicate/extra；
- point macro、全部 tau/turn fields、RateStable、EligibleV3、`S_RATE_TURN`、point rank、tie set、
  combined-rank frequency、selected flag 与终态由 fresh process 独立复算一致；
- hidden diagnostics 的变化不能改变 selector fields；
- 每模型 subject count=57，input record/identity count=1531，非有限或缺失 boundary 必须 BLOCK；
- 合法 selection 或 ABSTAIN 均 PASS，不得因结果不理想调参；
- 不读取或持久化 validation gold/target/label/correctness、test split、accuracy/gain/flip、prompt、
  token IDs、hidden tensors、完整 logits/probabilities；不运行 model forward、loop、generation；
- 不绘图，不访问 Gate E outcome，不改 Gate C sealed evidence。

## 6. Agility、subagents 与 terminal delivery

```text
earliest_decisive_work=run frozen formal analyzer once on all three sealed inputs
minimum_checks=exact input receipts,candidate counts,formal fresh recomputation,terminal legality
deferred=plotting,presentation QA,top-3 panel freeze,outcome,extra selector variants,extra models
deletion_test=remove any check that cannot change Gate D acceptance or interpretation
```

最多三个 subagents 只能做当前 Gate 的 narrow code inspection、targeted tests 或 read-only evidence
extraction；不得独立做 Git integration、remote mutation、终态投递、绘图或跨 Gate。CPU formal analysis
若可在 executor 当前运行中有界完成，不创建 heartbeat；若出现真正的长时间外部等待，只能创建一个
匹配 cadence、唤醒 exact executor 的 heartbeat，并在终态后删除。

终包 `GATE_D_FINAL_AUDIT` 至少包含：exact branch/commit/dirty/staged；Gate C sealed input roots 与
receipt admission；formal invocation/seed/R；每模型 output/receipt/root、candidate count、subject/record
count、合法终态与（若 selected）selected key；fresh verifier results；任何 repair/retry；信息屏障与
protected-state confirmation；明确请求 planning `PASS` 或 `BLOCK`。终包不得包含 gold/outcome、prompt、
token IDs、per-identity敏感内容或 digest/hash evidence。
