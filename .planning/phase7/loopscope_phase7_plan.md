# LoopScope 第七阶段总体计划：三种 Base 架构的逐层轨迹、V3 选窗与全量 Outcome

## 1. 阶段目标

第七阶段对 `Qwen2.5-3B`、`Llama-3.2-3B` 和 `Gemma-2-2B` 三个 Base 模型，在同一
MMLU validation-1531、标准 5-shot prompt 的 `Answer:` 末端 token 上采集一次 native
no-loop `B0...BL` 轨迹，随后按最新 V3.1 规则分别评分，并在科学数据闭合后绘制与前序阶段
同口径的输出分布和 hidden-state 图。绘图与 QA 完成后，追加 Gate E，在 MMLU test-14042 上
运行 Qwen/Llama top-3、Gemma top-2 的 width-4 V3 score top-available 冻结全量 outcome panel。

阶段终点包括三份模型内 V3 `SELECTED_WINDOW/ABSTAIN`、完整候选表、逐层图、三模型共 35 个
test outcome cells 的完整结果，以及一份区分 trajectory、ranking 与 exploratory outcome 的跨架构
总结。Gemma 的合法 `ABSTAIN` 模型运行实际存在的 width-4 score top-2 panel，但不得称为 prospective selection success。

## 2. Gate 总览

| Gate | 核心问题 | 最早真实动作 | 决定性产物 | PASS 条件 |
| --- | --- | --- | --- | --- |
| A | 三种架构能否在同一科学合同下被准确采集和复算？ | 本地 targeted tests；HPC2 只读 config/tokenizer/cache admission | Phase 7 card/schema、model adapter、producer、verifier、三模型候选域表 | exact model/tokenizer/layer/final-norm/choice-surface 闭合；focused tests/CLI/dry-run 通过 |
| B | 三模型真实 CUDA 路径是否都产生正确逐层 scalar？ | 每模型四 identity、`debug` partition、`<30min` smoke | 三个独立 smoke root 与 verifier receipt | 每模型 4/4 identity、`L+1` boundaries、`L` transitions、one forward、zero loop、六指标有限 |
| C | 三模型 full validation trajectory 是否完整可信？ | Gate B 有效后提交三模型 formal acquisition | 每模型 1,531 records、merge、data seal、independent verifier | 三模型均 1,531/57、无 missing/duplicate/extra、同 renderer/probe、barrier clean |
| D | 最新 V3.1 对三个模型分别给出什么？ | CPU-only，一次 combined analysis + fresh-process verify | 三份完整 V3 candidate table、终态、aggregate/source-data | 每模型候选域、bootstrap、公式、rank 与终态独立复算一致；合法 selection 或 ABSTAIN |
| Planning plotting checkpoint（非 Gate） | 如何在 outcome 前忠实冻结轨迹与 V3 呈现？ | planning 只读 Gate D source-data 绘图 | 每模型 6 trajectory + score landscape、normalized-depth 对照、outcome-blind 中文报告 | 三格式 QA 通过；3/3/2 width-4 score top-available panel manifest 冻结；未读 test/outcome |
| E | 三模型冻结的 3/3/2 width-4 配置在独立 MMLU test 全量上表现如何？ | plotting checkpoint 后先做同路径 `debug` preflight，再提交 35-cell formal panel | Qwen/Llama/Gemma 分别 13/13/9 cells、test-14042 completeness、combined outcome analysis 与 fresh verifier | 35/35 cells 完整；同模型 baseline 配对；无 partial outcome-driven panel change；analysis/verifier PASS |
| Planning closeout（非 Gate） | 如何整合 trajectory、ranking 与 outcome 并闭合阶段？ | Gate E PASS 后更新报告 outcome 章节 | 全部图表、35-cell 结果表、中文综合报告与 consolidated audit | 声明边界准确，Phase 7 terminal；不创建 Gate F |

Gate A–D 各由一个新的独立、用户可见 executor 完成；同一阶段保持 Project 与 Phase，topic
描述当前 Gate 自己的目标：

```text
Gate A historical identity: execute-LoopScope-三模型Provenance与轨迹探测实现-第7阶段-Gate A
Gate B: execute-LoopScope-Debug-Smoke-第7阶段-Gate B
Gate C: execute-LoopScope-Full-Acquisition-第7阶段-Gate C
Gate C replacement executor uses the same canonical title; exact thread ID distinguishes the replacement.
Gate D: execute-LoopScope-V3-Scoring-第7阶段-Gate D
Gate E: execute-LoopScope-TopAvailable-Outcome-第7阶段-Gate E
```

Executor 只发送一次 `GATE_X_FINAL_AUDIT` 或真正的 `BLOCK`，planning 做轻量验收后才可能授权下一 Gate。
Gate D `PASS` 后先由本 planning thread 完成绘图、QA 与 outcome-blind 报告；只有该 checkpoint 完成，
才创建独立 Gate E executor。Gate E `PASS` 后由 planning 完成阶段终审，不创建 Gate F。

所有 executor 在执行前必须完整重读并遵守 `research-gate-orchestrator`，从最短可信实验主线
出发，只保留可能改变科学解释、成本、安全或下一 admission 的决定性检查。Executor 可在
实质缩短当前 Gate 时使用 subagent，任一时刻最多三个；subagent 不得跨 Gate、改合同、独立
操作 scheduler/Git integration 或发送 terminal event，绑定 executor 保留全部整合与终态责任。

每次新 Gate dispatch 必须先通过以下协作启动检查；这是 handoff 的一部分，不是可选说明：

1. 从上面的唯一 canonical mapping 生成 title；替代 executor 不改变 Gate 字段。
2. 创建独立用户可见 executor，继承 Codex config 默认模型配置；立即设置 title 并核对工具返回的
   exact thread ID/title。
3. handoff 首段写明 `COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR`，同时冻结
   planning recipient、问题上报路径、subagent 边界、heartbeat continuation 和终态 fallback。
4. 先绑定 control 的 exact executor/title/handoff，再投递完整 handoff；取得一次可见投递确认后，
   planning 才能向用户说明 Gate 已 dispatch，随后停止轮询。
5. Executor 的普通工程/权限问题只回报 planning；不得用直接向用户请求 ordinary operational
   permission、普通 final answer 或“ready to resume”替代 `GATE_X_FINAL_AUDIT`、真正 `BLOCK`、
   `AUTOMATION_RELAY_REQUIRED` 或 `TERMINAL_DELIVERY_UNCONFIRMED`。

## 3. Gate A：合同、provenance 与最小实现

### Objective

用最小实现同时解决三个正常路径风险：模型访问与 exact revision 是否可用、三种架构的
raw boundary/final norm 语义是否可统一、四个 MMLU continuation 是否都是单 token。

### Ordered work

1. 绑定 activation 时的 local branch/base/dirty state；保护所有既有修改。
2. 直接检查 HPC2 的 model/tokenizer cache、config、runtime 和 MMLU validation pool；不做
   GPU forward。
3. 为三个 model repo 冻结 revision、`L`、dtype、tokenizer、decoder/final norm/lm head
   路径，以及由 `ceil(0.30L)` 生成的候选域。
4. 验证 `" A"/" B"/" C"/" D"` 的单 token 闭合；失败即科学合同 BLOCK。
5. 实现一个窄的 Phase 7 sidecar：model-specific adapter 只负责定位 decoder/embedding/
   final norm/lm head，公共 producer 负责一次 forward、scalar reduction、write-once record。
6. 实现 V3.1 depth-general candidate enumeration、closed schema、formal verifier 和 launcher。
7. 运行 targeted synthetic tests、compile、CLI help/dry-run 与 `git diff --check`；不跑旧阶段
   full suite，不修改 `wrapper.py/strategies.py/cache.py/config.py`。

### Must-pass

- 三个 exact Base repo 可读，未替换为 Instruct/量化/其他规模；
- 三个 tokenizer choice surfaces 单 token 且互异；
- 三种架构的 `B0...BL`、raw final-boundary hook 和 exactly-once FinalNorm 规则可实现；
- 任意 `L` 的 central-domain/width 3–6 枚举与 V3 formulas 有 targeted tests；
- schema 拒绝 gold/outcome/logits/hidden tensors/token IDs；
- 只改 Phase 7 新路径及必要的最小 docs/CLI 注册。

### Agility budget

```text
smallest increment=one Phase 7 card/schema + one narrow adapter/producer + one V3 depth-general analyzer/verifier + one launcher/test module
earliest experiment=Gate B four-identity smokes immediately after focused tests
minimum checks=choice-token closure, raw B_L closure, arbitrary-L candidate enumeration, forbidden-field rejection
stop rule=focused tests and three-model CPU admission pass -> request Gate A audit; no general model framework refactor
```

Gate A 不授权 GPU/Slurm、真实 trajectory、selector execution、plotting 或 Gate B。

## 4. Gate B：三模型真实 debug smoke

### Objective

用每模型四个固定 validation identities 证明真实 tokenizer→model→boundary→scalar→verifier
路径成立，并作为 Gate C formal job 的 `<30min` debug preflight。

### Execution

- 所有 smoke 只使用 HPC2 `debug` partition；三模型可并行，但 root、job、logs、records 和
  verifier 必须分离。
- 使用 Gate A 同一 immutable commit、launcher、venv、renderer、model snapshot、producer
  和 verifier；每模型四个 exact identities，至少覆盖两个 subjects。
- Debug job 若未稳定 RUNNING 约一分钟便结束，不创建 heartbeat；若仍需长等，executor 只建
  一个 10-minute smoke heartbeat 观察整个 Gate B job set。

### Must-pass per model

```text
record_count=4
unique_identity_count=4
forward_count=4
loop_insertions=0
probe=last non-padding token of rendered Answer: prefix
boundary_count=L+1
transition_count=L
final_norm_prehook_calls_per_forward=1
FinalNorm(raw B_L) closes native final hidden
four choice logits finite and choice probabilities sum to one
all six metric arrays finite with exact lengths
forbidden persisted fields absent
fresh verifier=PASS
elapsed<30 minutes
```

若修改 decision-critical launcher/runtime/hook/serialization/producer，必须只重跑受影响模型的
fresh debug root；未受影响模型的通过证据可保留。三模型全部 PASS 才 admission Gate C。

## 5. Gate C：三模型正式 validation-1531 acquisition

### Objective

采集并封存三个彼此独立但口径相同的完整 trajectory population；不运行 selector、不画图。

### Execution

1. Gate B 的 preflight 对每模型仍有效后，按模型分别冻结 shard manifest。
2. 使用 normal user 可合法获得的最高优先级 partition/QoS；GPU 型号由 smoke memory/runtime、
   当前可用性与 time-to-result 决定，不预先锁死卡型，也不能改变科学配置。
3. 三模型可并行提交；每模型 records、shards、merge、verification 和 run root 完全隔离。
4. Formal job IDs 核实后，executor 立即创建一个覆盖整个 Gate C job set 的 60-minute full
   heartbeat，包括 PENDING；不得为每模型创建重叠 monitor。
5. 每模型完成后可做 membership/schema/finite-value merge 与 data verification，但 Gate C
   只有三模型全部闭合才终态；不得提前运行 V3 或查看部分排名。

### Must-pass per model

```text
1531 unique validation identities
57 subjects
1531 successful native forwards
exact equality to the common canonical validation membership
no missing / duplicate / extra / partial records
one probe and renderer contract
boundary_count=L+1 and transition_count=L
all scalar arrays finite
zero loop / generation / validation-gold / test / outcome access
independent data verifier=PASS
```

Job failure只是 attempt-level event。Executor 在同一合同内诊断 launcher、environment、OOM、
serialization 或 shard 问题，保留失败 root，并只用 fresh sibling retry invalid/missing work。
改变 model/data/prompt/probe/dtype/metric/candidate 或读取 outcome 必须 Gate-level `BLOCK`。

### Agility budget

```text
smallest formal increment=the three already-smoked model acquisitions with no new metrics
decisive checks=1531/57 membership, L+1/L shape, finite scalar schema, information barrier, fresh verifier
stop rule=all three data seals pass -> terminal packet; no selector, plotting, extra models, Kneedle or full suite
```

## 6. Gate D：一次 V3.1 评分与数据封存

### Objective

只读三模型 Gate C records，按同一 V3.1 规则分别形成完整候选表、model-local selection/
ABSTAIN，并在画图前封存全部科学 source-data。

### Execution

1. 三模型全部 Gate C PASS 后才开始；CPU-only，不加载模型/数据集，不用 GPU/Slurm。
2. 从 sanitized records 投影 `identity/subject/H/D`，使用共同 canonical bootstrap draw map。
3. 按每模型实际 `L` 生成 central domain 与 width 3/4/5/6 全候选。
4. 一次 combined analyzer 计算全部 point macro trajectories、V3 candidates、bootstrap、
   eligibility、`S_RATE_TURN`、ranking frequency 和终态。
5. Fresh-process independent verifier 从原始 sanitized projection 重新构造三模型全部结果。
6. 写出每模型 V3 full candidate table、boundary aggregate、angular aggregate、subject count 和
   selector summary；数据通过后 immutable。

### Must-pass

- 三模型 candidate membership 分别由公式完整生成，无 missing/duplicate/extra；
- `EligibleV3`、turn、RateStable、score、tie-break、combined frequency 和终态从原始 H/D
  独立重算一致；
- hidden metrics 的任何变化都不能改变 selector fields；
- 每模型输出一个合法 `SELECTED_WINDOW` 或 V3 `ABSTAIN`；
- 不跨模型平均原始 score，不制造“共同 selected window”，不读任何 outcome。

合法 ABSTAIN 是科学终态，不触发调参或 repair。达到三模型 verifier PASS 后立即停止科学
分析，不追加 selector variant 或 post-hoc rule。

## 7. Planning plotting checkpoint：绘图、报告与 panel freeze（非 Gate）

### Owner 与 Objective

Owner 固定为本 planning thread `01a0013e-71c3-7c90-a547-4059b462dc7e`。Gate D executor
在 planning `PASS` 后撤销全部权限。Planning 只读 Gate D 已封存的 aggregate/source-data，
亲自生成可复现图表和 outcome-blind 跨架构描述性总结；绘图完成后，从每模型 Gate D candidate
table 中过滤 `width=4`，按最终 `S_RATE_TURN` 降序及 frozen tie-break 冻结 3/3/2 top-available Gate E panel。
此时仍不读取 MMLU test、gold 或任何 outcome。

### Required outputs

每模型：

- 六张逐层 trajectory 图；
- 一张 V3 candidate score landscape；
- 对应 source-data CSV 与 QA；
- SVG、PDF、600 dpi PNG 三种格式。

跨模型：

- 六类指标各一张 normalized-depth (`boundary_index/L`) 对照图；
- 三模型 layer/domain/selection/ABSTAIN 汇总表；
- outcome-blind 中文阶段报告，明确 trajectory、selector 与尚未执行 outcome 的边界。

图表只读取 Gate D source-data，不读取 per-identity records。Figure bug 只能在 fresh
presentation sibling 中修复，不能重算 Gate D、改 selector、重跑 Gate C 或移动旧 evidence。

### Checkpoint PASS

Planning 只复核：三模型 exact membership、probe/model/domain 合同、三个 V3 终态、图表与
source-data 一致性、报告声明边界，以及 3/3/2 width-4 top-available panel 是否由冻结候选表确定。复用 Gate A–D
的已验收 evidence，不重跑 tests、模型或 bootstrap。全部成立后该 checkpoint `PASS`，授权 Gate E；
Phase 7 此时不 terminal。

## 8. Gate E：三模型 3/3/2 Top-Available Width-4 全量 Outcome

### Frozen panel

共同 outcome population：`cais/mmlu` test 全量 14,042 identities / 57 subjects、标准 plain
non-chat 5-shot、dev demonstrations。Qwen/Llama 各使用三个 width-4 V3 score top-ranked windows，
Gemma 使用实际全部两个 finite width-4 windows；每窗运行
`k={2,3} × cache={first,last} × mode=block` 四个配置，每模型另有一个 no-loop baseline。
因此三模型分别为 13/13/9 cells，合计 35 cells。

Panel 与 Gate D eligibility/终态解耦：仍按 width-4 候选的 finite `S_RATE_TURN` 降序和
frozen tie-break 取已授权的 3/3/2 top available，不改阈值、候选域、seed 或 score。Gemma
保持 `ABSTAIN_COMBINED_RANK_UNSTABLE`，其 outcome 只是 post-ranking exploratory evidence。

### Execution order

1. 在任何 outcome 前冻结三模型 panel、共同 test-14042 identity manifest、cell IDs 和 output roots。
2. 在 `debug` partition 对同 immutable commit/launcher/env/model/data/loop-cache path/verifier 做
   `<30min` representative end-to-end preflight；正式 submission 在通过前锁定。
3. 正式任务使用 normal user 合法最高优先级 partition/QoS；根据显存、实测 runtime、可用性、
   queue 与 time-to-result 选择 GPU，不预锁卡型。
4. 三模型/35 cells 可并行调度，但 model/cell/run root/log/result 必须明确分离，no-loop baseline
   每模型只计算一次并作为该模型 12 个 loop cells 的共同 paired reference。
5. 正式 job IDs 核实后创建唯一 60-minute full heartbeat，覆盖 PENDING/RUNNING/terminal；失败 attempt
   保留，合同内只 fresh retry invalid/missing cells。
6. 35/35 cells 与共同 membership 全部闭合前，不读取部分 accuracy/gain 排名，也不据 outcome 改 panel。
   完整后只做一次 combined analysis 与 fresh verifier。

### Must-pass

```text
models=3
cells_per_model=13,13,9
total_cells=35
test_identities_per_cell=14042
subjects=57
one no-loop baseline per model
width-4 V3 score top-ranked windows per model=3,3,2
k={2,3}; cache={first,last}; mode=block
same canonical test membership across every cell
no missing/duplicate/extra/partial cells
pre-outcome panel and completeness verifier=PASS
one combined outcome analysis + fresh verifier=PASS
```

报告标准 MMLU accuracy、相对同模型 no-loop 的 paired delta 与相应 per-subject/paired evidence。
这是 Qwen/Llama/Gemma 分别 12/12/8 个 loop cell 的 fixed exploratory panel；未冻结多重比较修正前，nominal 单 cell 改善
不宣称为 confirmatory general selection success。ABSTAIN 模型的结果明确标记为 post-ranking
exploratory outcome。

## 9. Planning closeout：Outcome 报告与 phase terminal（非 Gate）

Gate E 经 planning 判定 `PASS` 后，撤销 executor 权限。Planning 复用 Gate A–E 已验收 evidence，
在既有中文报告中追加 outcome 章节和 35-cell 结果表，核对 model/window/config/test membership、
完整 jobs/artifacts、paired baseline 与声明边界，完成一次 consolidated Phase 7 audit。随后将 Phase 7
记为 terminal；不自动创建 Gate F。

## 10. 当前未授权事项

本计划落盘不授权：

- 修改 `loopscope-tflt` 代码或 Git；
- 访问 HPC2、model/data cache 或网络；
- 下载 gated model、处理 credential/license；
- 运行测试、CPU projection、GPU/Slurm、模型 forward 或 selector；
- 当前只授权 control 所绑定的 Gate；不得预建 Gate C/D/E executor 或 heartbeat；
- Gate E handoff 之前不得读取 test/gold/outcome 或运行真实 loop；planning plotting checkpoint
  也不授予这些权限。

## 11. Gate F：V3 point-rank Top-2 追加全量 Outcome

Gate F 是用户在 Gate E terminal outcome 已解封后追加的 Phase 7 扩展。每模型从 Gate D V3.1
完整候选表取冻结 point-rank 前两名：Qwen `[14,17)`、`[12,15)`；Llama `[11,14)`、
`[10,13)`；Gemma `[13,16)`、`[10,13)`。每窗运行 `k={2,3}`、`cache={first,last}`、
`mode=block`，合计 24 个新 loop cells；复用 Gate E 三个已验证 no-loop baseline，不重新运行，
形成 27-cell 分析 panel。

执行主线固定为：冻结 panel 与 baseline source references → targeted tests/static verifier → 同一
decision-critical path 的 `<30min` functional debug → 24-cell formal acquisition → 24/24 新 cell 与
三个复用 baseline 的共同 completeness/pre-outcome verifier → 一次 combined analysis → 一次 fresh
independent verifier → `GATE_F_FINAL_AUDIT`。Qwen/Llama batch size 16，Gemma batch size 8；正式
资源继续以不 OOM 为硬边界最大化有效显存与总体吞吐，默认一进程独占一 GPU、concurrency=1，
只有直接资源证据才允许有界 packing。

Gate F 全程不得在 completeness barrier 前读取局部 accuracy/gain，不得根据结果更换窗口、扩充
panel、调 k/cache/mode/batch 或重算 V3。其解释标签固定为
`POST_TERMINAL_V3_POINT_RANK_TOP2_OUTCOME_EXPLORATORY`。Gate F planning PASS 后由 planning 更新
表格和综合报告并重新封存 Phase 7；不自动创建 Gate G。
