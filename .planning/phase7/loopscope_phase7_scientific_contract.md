# LoopScope Phase 7：Base 架构泛化逐层轨迹与 V3 选窗科学合同

```text
CONTRACT_ID=LOOPSCOPE_PHASE7_BASE_ARCHITECTURE_TRAJECTORY_V3_OUTCOME_V2
STATUS=FROZEN_ACTIVE_CONTRACT_AMENDED_2026-08-15
PHASE=7
CLAIM_SCOPE=THREE_BASE_MODEL_TRAJECTORY_MODEL_LOCAL_V3_AND_FIXED_TEST_OUTCOME_PANEL
PLANNING_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
METHOD=AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE@3.1.0
OUTCOME_ACCESS=FORBIDDEN_THROUGH_PLOTTING_CHECKPOINT;GATE_E_ONLY_AFTER_EXACT_HANDOFF
EXECUTION_AUTHORITY=GATE_SCOPED_PER_PHASE7_CONTROL
AMENDMENT=.planning/phase7/loopscope_phase7_gate_e_332_panel_amendment_20260815.md
```

## 1. 研究问题与声明边界

第七阶段首先检验：在相同 MMLU 5-shot validation population、相同 pre-answer prefix probe
位置和相同 V3.1 数学下，三个此前未进入 LoopScope 主实验矩阵的 Base 模型，是否分别形成
稳定的模型内候选窗排序、`SELECTED_WINDOW` 或合法 `ABSTAIN`，以及它们的逐层输出分布和
hidden geometry trajectory 呈现何种架构差异。Gate D 后由 planning 先完成 outcome-blind 绘图与
QA；随后 Gate E 在独立 MMLU test-14042 上运行冻结的 3/3/2 width-4 V3 final-score
top-available full outcome panel。

Gate A–D 与 planning plotting checkpoint 不运行任何 loop，不读取 MMLU test、gold、correctness、
accuracy、gain 或 flip，也不把 `selected` 解释为 loop 有益。只有 plotting checkpoint 完成、
三模型 35-cell panel 冻结并创建 exact Gate E executor 后，Gate E 才可读取 MMLU test/outcome。
Gate E 是 fixed exploratory panel：它可以报告每个配置相对同模型 no-loop 的 paired outcome，
但不能把 ABSTAIN 模型的 top-available 结果改写为 prospective selection success，也不能据 outcome 声称
跨模型通用最优窗口。

此前已完成的 `Qwen3-1.7B-Base` 与 `Qwen3-4B-Base` 只作外部历史背景。本阶段对它们不做
fresh acquisition、离线重评分、source-data 重制或重新绘图；旧规则终态也不得冒充本阶段
V3.1 新证据。

## 2. 三个冻结 model-task cells

| model key | model repo | Phase 7 角色 |
| --- | --- | --- |
| `qwen25_3b` | `Qwen/Qwen2.5-3B` | Qwen2 dense Base 架构 |
| `llama32_3b` | `meta-llama/Llama-3.2-3B` | Llama dense Base 架构 |
| `gemma2_2b` | `google/gemma-2-2b` | Gemma dense Base 架构 |

共同冻结：

```text
dataset=cais/mmlu
evaluator=lm-eval 0.4.11 compatible standard MMLU renderer
task_group=mmlu
split=validation
population=all 1531 exact identities / 57 subjects
sampling=none
num_fewshot=5
fewshot_split=dev
apply_chat_template=false
fewshot_as_multiturn=false
generation=false
prompt_terminal=Answer:
runtime_dtype=bfloat16
quantization=none
```

上述配置是 Gate A–D 的 validation trajectory/V3 population。Gate E 另冻结：

```text
dataset=cais/mmlu
split=test
population=all 14042 exact identities / 57 subjects
sampling=none
num_fewshot=5
fewshot_split=dev
apply_chat_template=false
fewshot_as_multiturn=false
mode=block
```

Gate A 必须对每个 model repo 单独冻结 exact resolved revision、tokenizer revision、真实
`num_hidden_layers=L`、decoder container、embedding、final norm 和 lm head。若 snapshot
缺失、访问受限或 license/credential 需要用户动作，返回明确 `BLOCK_MODEL_ACCESS`；不得换成
Instruct、量化版、其他参数规模或未指定镜像。

## 3. Probe 与 boundary 合同

### 3.1 Probe position

Probe 固定在当前 validation query 的完整 MMLU 5-shot rendered prefix 最后一个非 padding
token，也就是 `Answer:` 结尾对应的 tokenizer-final token，答案 continuation 尚未加入：

```text
five dev demonstrations
+ current validation question and A/B/C/D choices
+ Answer:▮
         ^ probe position
```

它不是 demonstration answer、目标答案字母、生成 token、EOS 前 token 或 padding token。
每条 identity 只执行一次 native、zero-loop、`use_cache=false`、
`output_hidden_states=true` forward；batch size 固定为 1，`loop_insertions=0`。

### 3.2 Choice surface admission

Gate A 必须在每个 tokenizer 上直接验证 exact continuation surfaces：

```text
" A", " B", " C", " D"
```

四者都必须各自编码为一个唯一 token，且四个 token ID 两两不同。若任一模型不满足，返回
`BLOCK_CHOICE_SURFACE_NOT_SINGLE_TOKEN`，由 planning/user 决定新的科学合同；不得自行改成
首 token 近似、多 token 求和、直接字母 fallback、chat template 或模型特定 prompt。

### 3.3 Raw boundaries

对真实 `L` 个 decoder blocks：

- `B0` 为 embedding output 的 raw residual stream；
- `Bj` 为第 `j` 个 decoder block 后、FinalNorm 前的 raw residual，`j=1...L`；
- boundary 数必须为 `L+1`，相邻 transition 数必须为 `L`；
- 若框架的 `hidden_states[-1]` 已经过 FinalNorm，必须通过临时 final-norm forward-prehook
  捕获 raw `B_L`；每次 forward hook call count 精确为 1，并在 `finally` 中移除；
- `FinalNorm(raw B_L)` 必须闭合到模型原生 final hidden，禁止对 `B_L` double norm。

不同模型只允许最小 architecture adapter；不得修改 TFLT wrapper、strategy、cache 或模型
参数来伪造统一接口。

## 4. 六类逐层指标

令 `z_l=FinalNorm(B_l)`，`p_l` 为 `z_l` 经原生 lm head 后在四个冻结 choice token 上做
stable float64 softmax 得到的分布。

### 4.1 V3 selector 唯一输入

```text
H_l = -sum_a p_l(a) log p_l(a)
D_l = KL(p_l || p_L)
```

只允许 `identity / subject / H[0...L] / D[0...L]` 进入 V3。Entropy 和 KL 使用 nats；
`D_L=0` 必须在数值容差内闭合。

### 4.2 Diagnostic-only hidden metrics

在同一个 probe position 计算：

```text
hidden_rms_l2_to_final_l = sqrt(mean_d((z_l-z_L)^2))
hidden_cosine_to_final_l = cosine(z_l,z_L)
hidden_cosine_distance_to_final_l = 1-hidden_cosine_to_final_l
adjacent_angular_distance_l = acos(clamp(cos(B_l,B_(l+1)),-1,1))/pi
```

前三项使用 final-normalized hidden；相邻角距离只使用 FinalNorm 前 raw residual，transition
索引为 `l=0...L-1`。这些指标只用于诊断和绘图，不得进入候选 eligibility、排名、
selection frequency、tie-break 或 ABSTAIN。

## 5. V3.1 模型内选窗合同

### 5.1 跨深度候选域

对每个模型从其 exact config 得到 `L` 后，冻结：

```text
t(L)=ceil(0.30*L)
central_blocks(L)=[t(L), L-t(L)-1]  # inclusive
widths={3,4,5,6}
W(s,w)=[s,s+w)
publish W(s,w) iff every block s...s+w-1 lies in central_blocks(L)
```

候选顺序固定为 `width ascending, start ascending`。Gate A 必须从公式生成并发布每模型的
`L`、central blocks、各 width start 范围和 candidate count；不得复制 36-layer 的固定
`11...24/42 candidates` 到其他深度，也不得根据已见轨迹调整中央域或 width。

### 5.2 Point、bootstrap、eligibility 与排名

每模型独立执行 V3.1：

```text
point estimand=57-subject equal-category macro trajectory
bootstrap=within-subject resample, equal-category-macro joint bootstrap
replicates=2000
seed=20260801
same canonical draw map across H/D/boundaries/windows/widths and all three models
EligibleV3=Scorable AND NetPositive AND RateStable AND AggregateCommonTurn
point score=S_RATE_TURN=sqrt(G_H*G_K*Q_H*Q_K)
combined-rank selection frequency threshold=0.80
```

同一 draw map 只减少三模型间 Monte Carlo 口径差异；每模型的 macro curves、候选、评分、
eligible set、winner 和终态仍独立计算，不跨模型汇总原始分数。

每模型科学终态只能是：

```text
SELECTED_WINDOW
ABSTAIN_NO_V3_ELIGIBLE
ABSTAIN_COMBINED_RANK_UNSTABLE
```

输入不完整、非有限值、candidate membership、公式复算或 independent verifier 不一致是
工程 `BLOCK`，不能伪装成科学 `ABSTAIN`。V1/V2 的 `SoftRelativeStable`、
`BiphasicStable`、direction margins、tau bootstrap frequency 和 `S_REL` 不得回流。

## 6. 数据封存与信息屏障

每模型必须写出 1,531 条 sanitized scalar records。允许字段仅覆盖：

- schema/model repo/resolved revision；
- canonical validation identity、subject、sequence length、probe index；
- boundary/transition count、zero-loop 标记；
- 六类指标的有限 scalar arrays；
- renderer、runtime、job、run-root 等可读 provenance。

Gate A–D 及 planning plotting checkpoint 禁止持久化或读取：

```text
raw prompt/question/choices
validation target/gold/label/correctness
test split or test identity
prediction/accuracy/gain/flip/outcome
raw/full logits or full probabilities
raw/final-normalized hidden tensors
token IDs/input IDs
loop residual or generated answer
```

Phase 7 新证据使用可读 model/data identifier、路径、配置、命令、job ID、直接 manifest
membership 和 verifier 重算；不新增内容摘要、摘要清单或基于摘要的 Gate 绑定。

### 6.1 Gate E outcome panel 与信息解封

Gate E panel 必须在任何 test outcome 读取前冻结。每模型从 Gate D full candidate table 中过滤
`width=4`，按 finite `S_RATE_TURN` 降序及 frozen tie-break 取 top available；Qwen/Llama
各取 top-3，Gemma 取实际全部 top-2。该规则忽略 `EligibleV3` 与最终
`SELECTED_WINDOW/ABSTAIN` 状态，但不得重新评分、改阈值或改候选域。冻结数量为：

```text
Qwen2.5-3B: 1 baseline + 3 windows * k{2,3} * cache{first,last} = 13 cells
Llama-3.2-3B: 1 baseline + 3 windows * k{2,3} * cache{first,last} = 13 cells
Gemma-2-2B: 1 baseline + 2 windows * k{2,3} * cache{first,last} = 9 cells
mode=block
```

三个模型合计 35 cells。每个 cell 必须覆盖同一 canonical test-14042 membership；no-loop baseline
每模型只运行一次。Gate E 先完成 panel/identity/pre-outcome verifier，再做同路径 `debug` preflight，
最后运行 formal jobs。35/35 cells 完整前不得查看部分 accuracy/gain 排名或据 outcome 改 panel；完整
后只做一次 combined analysis 与 fresh verifier。

Gate E 可持久化 canonical test identity、subject、cell configuration、prediction/correctness 和完成
standard accuracy/paired analysis 所需的最小 outcome 字段；仍禁止 raw prompt、完整 logits/probabilities、
hidden tensors、token/input IDs 和无关生成内容。formal outcome records 与 Gate A–D trajectory records
分离存储，不能回流 V3 ranking 或 plotting source-data。

## 7. 图表合同

只有三模型正式 records、aggregate source-data 和 V3 score tables 全部封存并通过 data
verifier 后才能绘图。每模型必须输出：

1. Choice Entropy trajectory；
2. KL-to-final trajectory；
3. Hidden RMS-L2-to-final trajectory；
4. Hidden cosine-to-final trajectory；
5. Hidden cosine-distance-to-final trajectory；
6. Adjacent raw-residual angular-distance trajectory；
7. V3 candidate score landscape。

每张 trajectory 上半部分为 unsmoothed point mean 与 95% subject-stratified bootstrap CI，
下半部分为相邻 boundary/transition mean change。不得平滑、插值、删层或按视觉效果调整数据。

若模型终态为 `SELECTED_WINDOW`，只标记其 own V3 selected window 的 entry/exit boundaries；
若为 ABSTAIN，不画“selected”窗口。不得把 Qwen3 历史窗口或 Phase 6 outcome-best `15:18`
画成新模型参照窗。

另输出 normalized-depth (`boundary_index/L`) 的三模型同指标对照图，只作形态比较；不在图中
合并 selector score，也不据 normalized alignment 发布跨模型统一窗口。

科学数据与绘图严格分离：Gate D executor 只负责封存和验证科学 source-data，不负责绘图。
Gate D 经 planning 判定 `PASS` 并撤销 executor 权限后，由 planning thread
`01a0013e-71c3-7c90-a547-4059b462dc7e` 亲自编写/运行 plotter、完成图表 QA 与中文综合报告；
plotting checkpoint 完成前不创建 Gate E executor。Plotter 只读取 Gate D 已验证的 aggregate/source-data，不读取
per-identity records。展示错误只能做 planning-owned presentation-only fresh repair，不能
重跑模型、重算 selector 或改动已封存数值。

## 8. 科学停止条件

Gate D 每个模型得到一个合法终态并通过独立重算后，选窗科学部分立即停止。不得为了获得
更多 `SELECTED_WINDOW` 而调阈值、candidate domain、width、probe、seed、metric 或 prompt。
Planning 先对 Gate D 做轻量验收并决定 `PASS`，再亲自完成图表、QA、outcome-blind 报告和
Gate E panel freeze；checkpoint `PASS` 后才授权独立 Gate E executor。Gate E 35/35 cells、共同
test membership、combined analysis 与 fresh verifier 全部通过 planning 验收后，planning 更新
outcome 报告并完成 consolidated phase-end audit，随后把 Phase 7 记为 terminal。Gate E 后不自动
创建 Gate F。

### 8.1 Gate F 用户追加的 post-terminal 科学例外

用户在 Gate E 已解封并完成后明确追加 Gate F，因此本小节取代上面的默认“Gate E 后不自动创建
Gate F”停止分支，但不回写或改变 Gate A–E 的既有结论。Gate F 对每个模型采用 Gate D V3.1
完整候选表中按冻结 point score/tie-break 得到的模型内前两名窗口，而不是 Gate E 的 width-4
子集：Qwen2.5-3B 为 `[14,17)`、`[12,15)`；Llama-3.2-3B 为 `[11,14)`、`[10,13)`；
Gemma-2-2B 为 `[13,16)`、`[10,13)`。每个窗口固定遍历
`k={2,3} × cache={first,last} × mode=block`，共 `3 × 2 × 2 × 2 = 24` 个新增 loop cells。

Gate F 不重跑 baseline；只复用 Gate E 已独立验证的三个同模型 no-loop baselines，最终分析 panel
为 27 cells。数据仍为标准 `cais/mmlu` test 全量 14,042 identities / 57 subjects、plain 5-shot、
BF16、choice-loglikelihood；Qwen/Llama batch size 16，Gemma batch size 8。24/24 新 cell 及三个
复用 baseline 的共同 membership/completeness verifier 闭合前不得读取部分 accuracy、prediction、
correctness、gain 或据 outcome 改变 panel；闭合后只做一次 combined analysis 和 fresh verifier。

由于 Gate F 的 panel 在 Gate E outcome 已解封后才由用户追加，所有结果必须标记为
`POST_TERMINAL_V3_POINT_RANK_TOP2_OUTCOME_EXPLORATORY`，不得称为 prospective、confirmatory 或
独立于既有 outcome 的选择验证。Gate F 经 planning 审计 PASS 后仅由 planning 重新封存 Phase 7；
不自动创建 Gate G。
