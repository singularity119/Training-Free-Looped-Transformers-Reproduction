# AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE

```text
METHOD_ID=AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE
METHOD_VERSION=3.1.0
METHOD_STATUS=FROZEN_METHOD_SPEC_NOT_PROSPECTIVELY_VALIDATED
SCOPE=CROSS_PHASE_RULE_SPEC_WITH_36_LAYER_REFERENCE_BINDING
SOURCE_THREAD=019fb3de-2298-75f2-a083-0dca453ea79c
OUTCOME_ACCESS=FORBIDDEN
EXECUTION_AUTHORITY=NONE
POINT_RANK_SCORE=S_RATE_TURN
V3_1_AMENDMENT=USER_AUTHORIZED_TURN_STRENGTH_FUSION_2026-08-02
```

## 1. 科学目标与声明边界

V3 寻找同时满足以下性质的 loop window：

1. 窗口总体上使 entropy 下降，即 `G_H>0`；
2. 窗口总体上使 KL-to-final 上升，即 `G_K>0`；
3. entropy 与 KL-to-final 的 category-macro 聚合轨迹在同一个内部 boundary 出现主导的
   变化速率转折；
4. 总体绝对平均变化率在冻结 bootstrap 下稳定为正；
5. 最终按窗口自身的净变化率与共同转折强度联合分数
   `S_RATE_TURN=sqrt(G_H*G_K*Q_H*Q_K)` 排名。

“转折”只表示变化速率发生明显改变，不要求变化率变号。快速下降变为缓慢下降、缓慢
上升变为快速上升、以及变号反转都可以构成转折。H 与 KL 的转折前后方向不要求相同，
只要求各自的主导转折位置相同。

V3 是在已观察旧 V1/V2 行为、并知道 `15:18` 曾被旧 biphasic/tau-frequency 门槛拒绝后
形成的方法修订。因此，任何用于形成 V3 的既有 cell 都只能作为 post-hoc method
development evidence；本文件不宣称 V3 已经验证成功，也不授权读取 outcome、运行真实
selector、模型 forward、GPU、Slurm 或任何实验。

## 2. V3 明确删除和禁止回流的条件

以下旧 V1/V2 shape 条件从 V3 的准入、排名、selection frequency 和终态中全部删除：

- orientation `q in {+1,-1}`；
- segment/core 的预设方向与 12 个 direction margins；
- simultaneous biphasic direction lower confidence bounds；
- `BiphasicStable`；
- bootstrap 重新搜索共同转折点；
- `tau_pair_frequency` 及其 `>=0.80` 门槛；
- 窗口外相邻 transition 的 shape 支持；
- `SoftRelativeStable`、`S_REL` 或 reference family 对 admission/ranking 的控制。

实现和 verifier 必须直接复算：

```text
EligibleV3 =
  Scorable
  AND NetPositive
  AND RateStable
  AND AggregateCommonTurn
```

不得从旧 `NewEligible` 派生 `EligibleV3`，不得要求旧 `BiphasicStable=true`，也不得通过
`SoftRelativeStable`、failure reason、旧 shape frequency 或任何 legacy 字段间接恢复
被删除的条件。旧字段若为兼容审计而保留，只能放入明确的 `legacy_diagnostics` namespace。

Hidden RMS-L2、hidden cosine、cosine-distance-to-final、adjacent angular distance 及其他
hidden geometry 指标仍仅作诊断，不进入 V3 的 trajectory、turn、rate、eligibility、
ranking、tie-break、selection frequency 或 panel membership。

## 3. 输入、聚合估计量和数值顺序

### 3.1 输入投影

V3 selector 只读取每条 frozen identity 的：

```text
identity
category
H[0...L]
D[0...L]
```

其中 `H_i,l` 是冻结输入分布的 entropy，`D_i,l` 是同一分布的 KL-to-final。choice-space
与 full-vocabulary-space 是不同输入合同，必须由 consuming Phase 明确绑定，且原始数值
不得跨 cell 比较。Gold、prediction、correctness、accuracy、gain、flip、outcome 与所有
hidden diagnostics 禁止进入 selector 投影。

Phase contract 必须冻结 identity 集、category 集、每条 identity 的唯一 category、boundary
数和输入文件 SHA-256。category 为空、identity/category 重复或不闭合、数组长度错误、
缺失值或非有限值均触发 `BLOCK_INVALID_V3_INPUT`，不得作为科学 `ABSTAIN`。

### 3.2 Category-macro 聚合曲线

设冻结 category 集为 `CATEGORIES={1,...,C}`，category `c` 的样本集合为 `I_c`，且
`n_c=|I_c|>0`。V3 的 point trajectory 是等权 category-macro mean：

```text
H_c,l = mean_{i in I_c} H_i,l
D_c,l = mean_{i in I_c} D_i,l

Hbar_l = (1/C) * sum_c H_c,l
Dbar_l = (1/C) * sum_c D_c,l
```

这不是把全部 identity 直接平均的 micro mean。每个 category 对 `Hbar/Dbar` 的权重严格
相同，避免 category 样本数不均衡改变 turn 和 rate estimand。

为保证 byte-level 复算一致：category label 按 Unicode code-point 升序；每个 category 内
按 canonical identity string 升序；所有均值用 binary64 `math.fsum(values)/count`；禁止
先四舍五入 trajectory 再计算 turn/rate。

## 4. Boundary、窗口和固定候选域

对 `L=36` 个 decoder blocks，boundary 为 `B_0...B_36`。半开窗口：

```text
W(s,w) = [s,s+w)
```

表示循环 decoder blocks `s,s+1,...,s+w-1`，对应 boundary path：

```text
B_s -> B_(s+w)
```

为兼容历史图表，必须同时输出：

```text
window_half_open       = "s:s+w"
window_layers_inclusive = "s:s+w-1"
boundary_entry         = "B_s"
boundary_exit          = "B_s+w"
```

例如视觉窗 `15:18` 的 selector 语义是 `[15,19)`、`width=4`、entry `B15`、exit
`B19`；不得误写为 `[15,18)`。

当前 36-layer reference binding 冻结中央 blocks `11...24`、widths `{3,4,5,6}`：

| width | start `s` | count |
| ---: | --- | ---: |
| 3 | `11...22` | 12 |
| 4 | `11...21` | 11 |
| 5 | `11...20` | 10 |
| 6 | `11...19` | 9 |
| total |  | 42 |

候选枚举顺序固定为 `width ascending, start ascending`。任何缺失、重复、额外 candidate，
或不完整落在 blocks `11...24` 的窗口都触发 `BLOCK_CANDIDATE_DOMAIN_MISMATCH`。

其他模型深度若复用 V3 的 turn/rate 方法，必须由新 Phase 在 outcome 前单独冻结其 central
domain、width 和 candidate count；不得静默把本文件的 42-candidate reference binding
外推成已经验证的通用域。

## 5. NetPositive 与绝对平均变化率

在 category-macro trajectory 上定义期望方向的逐 transition rate：

```text
rH_j = Hbar_j - Hbar_(j+1)      # 正值表示 entropy 下降
rK_j = Dbar_(j+1) - Dbar_j      # 正值表示 KL-to-final 上升
```

对 `W(s,w)=[s,s+w)`：

```text
G_H(s,w) = (1/w) * sum_{j=s}^{s+w-1} rH_j
           = (Hbar_s-Hbar_(s+w))/w

G_K(s,w) = (1/w) * sum_{j=s}^{s+w-1} rK_j
           = (Dbar_(s+w)-Dbar_s)/w

NetPositive = finite(G_H,G_K) AND G_H>0 AND G_K>0
```

等于零失败。只有 `NetPositive=true` 时定义绝对速率基项：

```text
S_RATE(s,w) = sqrt(G_H(s,w)*G_K(s,w))
```

单位为 `nats/boundary`。`ABSOLUTE_RATE` 表示窗口自身的 reference-free rate，不是对数值
取绝对值。`S_RATE` 是 V3.1 联合主分数的 rate factor，不再单独作为最终排名主键。不得改为
total change、算术平均、调和平均、最小值、带权和，或混入 relative score 与 hidden
metrics。

## 6. AggregateCommonTurn

### 6.1 内部候选转折点

对指标 `m in {H,K}`，令窗口内部 rate vector 为：

```text
x_j^H = rH_j
x_j^K = rK_j
j = s,...,s+w-1
```

候选转折 boundary 仅为严格内部位置：

```text
TAU(s,w) = {s+1,...,s+w-1}
```

`tau` 表示 transition `tau-1` 与 transition `tau` 之间的 boundary `B_tau`。`tau=s` 和
`tau=s+w` 不可选。`w=3` 时允许 `1+2` 或 `2+1` transition 分割；V3 不借用窗口外
transition，也不要求每侧至少两个 transition。

### 6.2 两段常速拟合与转折强度

对每个 `tau in TAU(s,w)`：

```text
n_L = tau-s
n_R = s+w-tau

mu_L^m(tau) = mean_{j=s}^{tau-1} x_j^m
mu_R^m(tau) = mean_{j=tau}^{s+w-1} x_j^m
mu_all^m    = mean_{j=s}^{s+w-1} x_j^m

delta_m(tau) = mu_R^m(tau)-mu_L^m(tau)
T_m(tau)     = abs(delta_m(tau))

SST_m = sum_{j=s}^{s+w-1} (x_j^m-mu_all^m)^2

BSS_m(tau) = (n_L*n_R/w) * (mu_L^m(tau)-mu_R^m(tau))^2

WSS_m(tau) =
  sum_{j=s}^{tau-1} (x_j^m-mu_L^m(tau))^2
  + sum_{j=tau}^{s+w-1} (x_j^m-mu_R^m(tau))^2

Q_m(tau) = BSS_m(tau)/SST_m
```

在精确算术中 `SST=BSS+WSS`，且 `Q in [0,1]`。`T_m` 是原单位的变化率转折幅度；
`Q_m` 是用一个共同 split 对该指标窗口内 rate 变化解释的比例，也是 one-change
two-segment constant-rate fit 相对 one-constant-rate fit 的 `R^2_turn`。

`delta_m` 的正负只作诊断：正值表示 split 后期望方向 rate 增大，负值表示 rate 减小。
V3 不要求 `mu_L/mu_R` 变号，不要求 H/K 的 `delta` 同号，也不要求任何预设 orientation。

### 6.3 可测转折、唯一 tau 与最小强度

对每个指标独立执行：

```text
scale_m = max(1.0, max_j abs(x_j^m))
rate_tol_m = 1e-12 * scale_m
sst_tol_m  = w * rate_tol_m^2
```

若 `SST_m<=sst_tol_m`，或全部 `T_m(tau)<=rate_tol_m`，则该指标为
`NO_MEASURABLE_TURN_m`，不定义 `tau_m`。

否则令 `Qmax_m=max_tau Q_m(tau)`。使用：

```text
math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
```

形成所有与 `Qmax_m` 相等的 tau 集。集合大小不是 1 时，记
`TURN_TAU_TIE_m`，不定义 `tau_m`；禁止用较早/较晚 tau 打破 shape tie。

唯一最大值存在时：

```text
tau_m = unique argmax_tau Q_m(tau)

TurnDominant_m =
  Q_m(tau_m)>0.5
  AND NOT isclose(Q_m(tau_m),0.5,rel_tol=1e-12,abs_tol=1e-12)
```

`Q>0.5` 等价于 `BSS>WSS`：由转折 split 解释的 rate 变化严格大于 split 后剩余的段内
变化。这是 V3 唯一的非数值性 turn-strength 门槛。它是 scale-free、无方向约束、无需
bootstrap tau 投票，并能阻止每条非恒定有限曲线仅因总有一个 argmax 就自动通过。

该 `0.5` 不是根据已有窗口 outcome 调出的阈值，而是“主导转折”定义本身。若未来改用
其他阈值、runner-up prominence、固定-tau CI、BIC/AIC 或 bootstrap turn stability，必须
建立新版本；不得仍称本 V3。

### 6.4 共同转折准入

```text
AggregateCommonTurn =
  measurable_H
  AND measurable_K
  AND unique_tau_H
  AND unique_tau_K
  AND TurnDominant_H
  AND TurnDominant_K
  AND tau_H == tau_K
```

位置比较是整数 boundary ID 的精确相等，不设置 `±1` 容忍。以下是普通 scientific
ineligibility，而不是 analysis BLOCK：

- `NO_MEASURABLE_TURN_H/K`；
- `TURN_TAU_TIE_H/K`；
- `TURN_NOT_DOMINANT_H/K`；
- `TURN_LOCATION_MISMATCH`。

任何缺失或非有限 trajectory/rate/turn scalar、负到超出数值容差的 BSS/WSS、
`|SST-(BSS+WSS)|` 超出 `1e-10*max(SST,BSS+WSS,1e-24)`，或 `Q` 超出 `[0,1]` 且偏差大于 `1e-12`
都触发 `BLOCK_INVALID_V3_TURN_ANALYSIS`。只允许把距 0 或 1 不超过 `1e-12` 的 Q clamp
到端点；其他异常禁止吞掉。

### 6.5 速率—转折联合点分数

仅对 `NetPositive=true` 且 `AggregateCommonTurn=true` 的窗口，使用两个指标在各自唯一主导
转折点上的 point `Q` 定义：

```text
S_TURN(s,w) = sqrt(Q_H(tau_H)*Q_K(tau_K))

S_RATE_TURN(s,w)
  = S_RATE(s,w)*S_TURN(s,w)
  = sqrt(G_H(s,w)*G_K(s,w)*Q_H(tau_H)*Q_K(tau_K))
```

由于 `AggregateCommonTurn` 已要求 `tau_H=tau_K`，两个 `Q` 实际在同一个 boundary 上取值。
`S_TURN` 无量纲，故 `S_RATE_TURN` 仍为 `nats/boundary`。该乘法分数对四个正因子分别严格
单调，不设置可调权重，并要求 rate 与 H/K turn strength 都不能由另一项完全补偿。

主版本直接使用原始解释比例 `Q_H,Q_K`。不得改用 `2Q-1`、turn amplitude `T_H/T_K`、
`min(Q_H,Q_K)`、加权和或其他 outcome-informed 变换。`T_H/T_K` 保持诊断字段；把它们加入
主分数会与 `G_H/G_K` 的原单位幅度发生重复计量。`Q>0.5` 继续承担 point admission，
`Q` 的连续值同时承担 eligible 集内部的转折清晰度排序，这是 V3.1 的明确设计选择。

## 7. Category-macro bootstrap 与 RateStable

V3 保留 V2 的 within-window H/K one-sided fixed-SE max-t lower confidence bound，但
bootstrap replicate 必须估计与 point 一致的 category-macro estimand。

当前 reference binding 冻结：

```text
method=within-category resample, equal-category-macro joint bootstrap
replicates=2000
seed=20260801
prng=random.Random
draw_api=randrange(n_c)
standard_deviation=ddof=1
epsilon=1e-12
quantile=linear at q*(R-1)
```

每个 replicate `b`：

1. 按冻结 category 顺序遍历；
2. 在 category `c` 的 `n_c` 个 canonical identities 中有放回抽取 `n_c` 次；
3. 对抽中样本计算 `H_c,l^(b),D_c,l^(b)`；
4. 对 C 个 category mean 等权平均得到 `Hbar_l^(b),Dbar_l^(b)`；
5. 从同一组 macro curves 联合重算全部 boundaries、H/K、42 windows 和 4 widths 的
   `G_H^(b),G_K^(b)`。

同一 replicate index map 必须联合复用于全部 boundary、metric、window 和 width。禁止把
所有 identity 直接 micro-average，禁止按 category 样本数加权，也禁止为不同 window 或
H/K 重新抽样。

规范 draw digest：先按 category、再按 identity 形成 global integer index；每个 replicate
按 category 顺序连接抽中的 global indices；全部 replicate 组成二维整数数组，以 UTF-8
JSON、`ensure_ascii=false`、`allow_nan=false`、`separators=(",",":")` 编码，末尾无换行，
其 SHA-256 记为 `bootstrap_draw_index_sha256`。实现与独立 verifier 必须复算一致。

对每个窗口和 `m in {H,K}`：

```text
SE_m = sd({G_m^(b)},ddof=1)

M_b = max_m(
  (G_m(point)-G_m^(b))/max(SE_m,1e-12)
)

c95 = max(0,linear_quantile({M_b},0.95))
LCB_m = G_m(point)-c95*max(SE_m,1e-12)

RateStable = finite(SE_H,SE_K,c95,LCB_H,LCB_K)
             AND LCB_H>0
             AND LCB_K>0
```

`LCB=0` 失败。任一 bootstrap rate、SE、`M_b`、`c95` 或 LCB 非有限触发
`BLOCK_NONFINITE_RATE_ANALYSIS`。该 max-t 只对同一窗口内 H/K 两维同时覆盖，不声称
across-window 或 across-model simultaneous coverage。

Bootstrap 不搜索 `tau_H/tau_K`，不计算 tau frequency，也不对 AggregateCommonTurn 投票。
turn location 和 point eligibility 只在完整样本的 point macro curves 上确定一次。第 8.3 节
仅为联合分数排名稳定性，在冻结的 point tau 上重算 replicate `Q`；这不构成 tau 重搜索或
shape admission 投票。

## 8. 最终 Eligibility、排名和终态

### 8.1 Eligibility

对 42 个候选分别计算：

```text
EligibleV3 =
  Scorable
  AND NetPositive
  AND RateStable
  AND AggregateCommonTurn
```

`Scorable` 仅表示固定候选的 point macro H/D 和所需内部 boundary/transition 完整、有限。
零 turn variance、无唯一 tau 或 turn strength 不足属于 AggregateCommonTurn 的普通科学
失败，不改变 `Scorable`。它不要求 overlap/flank reference、`SoftRelativeStable`、旧
shape support、hidden diagnostics 或 outcome registry。

### 8.2 Point ranking 与确定性 tie-break

只对 `EligibleV3=true` 的窗口排序。主键始终是 point `S_RATE_TURN` 降序；`S_RATE` 和
`S_TURN` 必须同时报告但不得单独决定排名。完整确定性排名采用
anchored tie-group 算法，禁止把非传递的 `math.isclose` 直接作为 sort comparator：

1. 在尚未排名的候选中取精确最大 finite `S_RATE_TURN` 为当前 anchor；
2. 收集所有与 anchor 满足 `math.isclose(rel_tol=1e-12,abs_tol=1e-12)` 的候选；
3. 该 tie group 内按 width/start canonical order 排列并从候选池移除；
4. 重复直到候选池为空。

第一组即 `point_top_tie_set`，其第一项为唯一 operational winner。canonical order 为：

```text
1. width ascending
2. start ascending
```

该 tie-break 只保证复现性，不代表较窄或较早窗口具有更强科学证据。必须输出完整 tie
set、`tie_break_applied` 和选择理由；V3 不使用 `ABSTAIN_NO_UNIQUE_TOP1`。

### 8.3 Combined-rank selection frequency

V3.1 将 V2 的 rate-rank stability 扩展为速率—转折联合排名稳定性。它与已删除的 tau
frequency 完全不同。固定 point `EligibleV3` 集及每个候选的 point `tau_H=tau_K` 后，
每个 bootstrap replicate：

1. 重算每个固定候选的 `G_H^(b),G_K^(b)`；
2. 任一 rate `<=0` 时，该候选在该 replicate 不可胜；非有限则 BLOCK；
3. 从同一 replicate macro curves 构造窗内 `rH^(b),rK^(b)`，只在冻结的 point tau 上按
   第 6.2 节重算 `Q_H^(b),Q_K^(b)`；不得搜索其他 tau，也不要求 replicate `Q>0.5`；
4. 在冻结 point tau 上，若 replicate `SST_m<=sst_tol_m`、
   `T_m(tau_point)<=rate_tol_m` 或任一 `Q_m(tau_point)<=0`，该候选在该 replicate 不可胜，
   但不改变其 point eligibility；其他非有限值或超出容差的分解恒等式/`Q` 范围错误触发
   `BLOCK_INVALID_V3_BOOTSTRAP_TURN_ANALYSIS`；
5. 对其余候选计算：

```text
S_RATE_TURN^(b) =
  sqrt(G_H^(b)*G_K^(b)*Q_H^(b)(tau_H_point)*Q_K^(b)(tau_K_point))
```

6. 取精确最大 replicate score 为 anchor，收集与其 isclose 的 top tie set，并用与 point
   相同的 `width/start` canonical tie-break 产生唯一 winner；
7. 无可用候选仍计入总分母 2000，但该 replicate 无 winner。

```text
combined_rank_selection_frequency =
  count_b(point operational winner is replicate winner)/2000
```

频率 `>=0.80` 才发布，等于 `0.80` 通过。该门槛评估最终联合分数排名对 category
重采样的稳定性；replicate `Q` 只在 point tau 上评估，不重新估计、搜索或投票 turn
location，也不构成第二个 `BiphasicStable` 或 shape-frequency eligibility gate。

### 8.4 终态

科学终态只有：

```text
ABSTAIN_NO_V3_ELIGIBLE
ABSTAIN_COMBINED_RANK_UNSTABLE
SELECTED_WINDOW
```

- `EligibleV3` 集为空：`ABSTAIN_NO_V3_ELIGIBLE`；
- 集合非空但 point operational winner 的 combined-rank selection frequency `<0.80`：
  `ABSTAIN_COMBINED_RANK_UNSTABLE`；
- 其余情况：`SELECTED_WINDOW`。

`SELECTED_WINDOW` 必须同时发布 half-open window、inclusive layer label、entry/exit boundary、
`S_RATE_TURN`、`S_RATE/S_TURN`、`G_H/G_K`、LCB、`tau_H=tau_K`、turn strengths、tie set 和
selection frequency。
任何输入闭合、非有限、candidate membership、bootstrap digest、公式复算或 verifier mismatch
都是 `BLOCK_*`，不得伪装成 `ABSTAIN`。

## 9. 必需输出和 freeze schema

### 9.1 全局 metadata

至少包含：

```text
method_id, method_version, method_file_sha256
input_distribution_contract, probe_contract
trajectory_file_sha256, identity_manifest_sha256
record_count, category_count, ordered_category_sha256
layer_count=36, boundary_count=37
central_blocks=[11,24], widths=[3,4,5,6], candidate_count=42
aggregation=equal_category_macro
bootstrap_method, replicates=2000, seed=20260801
prng=random.Random, draw_api=randrange
ddof=1, epsilon=1e-12, quantile_method=linear_q_times_R_minus_1
bootstrap_draw_index_sha256
turn_strength_Q_threshold=0.5_strict, numeric_tolerances
point_rank_score=S_RATE_TURN
combined_rank_frequency_threshold=0.80
selector_decision, selected_key, point_top_tie_set
```

### 9.2 每个 candidate row

至少包含：

```text
width, start, stop_exclusive, end_inclusive
window_half_open, window_layers_inclusive
boundary_entry, boundary_exit
Scorable, NetPositive, RateStable, AggregateCommonTurn, EligibleV3
G_H, G_K, S_RATE
S_TURN, S_RATE_TURN
rate_SE_H, rate_SE_K, rate_c95, rate_LCB_H, rate_LCB_K
rH_window[], rK_window[]
tau_candidates[]
turn_H_by_tau[], turn_K_by_tau[]
tau_H, tau_K, common_tau
turn_T_H, turn_T_K
turn_delta_H, turn_delta_K
turn_BSS_H, turn_BSS_K
turn_WSS_H, turn_WSS_K
turn_Q_H, turn_Q_K
turn_tau_tie_H, turn_tau_tie_K
turn_strength_pass_H, turn_strength_pass_K
point_rank, point_top_tie, tie_break_applied
combined_rank_selection_frequency, selected
v3_failure_reasons[]
legacy_diagnostics{}   # optional and non-gating
hidden_diagnostics{}   # optional and non-gating
```

`turn_*_by_tau` 必须包含每个内部 tau 的 `n_L,n_R,mu_L,mu_R,delta,T,BSS,WSS,Q`，不能只
输出获胜 tau。独立 verifier 必须从原始 sanitized H/D projection 重新构造 macro curves、
全部 turn candidates、bootstrap、RateStable、EligibleV3、排名和终态，不能只检查 producer
自报字段。

## 10. 必需单元测试和独立 verifier 检查

1. 36-layer candidate domain 精确产生 42 个唯一半开窗口，顺序为 width/start 升序；
2. `15:18` inclusive label 精确映射 `[15,19)`、`B15->B19` 和四个内部 rates；
3. category 大小极不均衡时，macro point/rate 与 micro mean 不同且复算为 equal-category；
4. category/identity 重排不改变 point 结果；canonical normalization 后 bootstrap digest 固定；
5. 常速 H/K rate 和纯数值噪声失败 `NO_MEASURABLE_TURN`；
6. H/K 都是“快速正 rate -> 缓慢正 rate”、同 tau、`Q>0.5` 时通过，不要求变号；
7. 一条或两条发生变号也可通过，不存在 orientation/q gate；
8. H/K 的唯一 tau 不同则失败 `TURN_LOCATION_MISMATCH`；
9. 两个 tau 在 tolerance 内并列则失败对应 `TURN_TAU_TIE`；
10. `Q<0.5` 和 `Q` 与 `0.5` isclose 均失败，`Q` 明确大于 0.5 才通过；
11. 修改 `SoftRelativeStable/BiphasicStable/tau_pair_frequency/S_REL/S_SHAPE` 不改变
    `EligibleV3`、排名和终态；
12. 修改任一 hidden diagnostic 不改变任何 selector 字段；
13. macro bootstrap 同 draw 联合复用 H/K、boundaries、windows、widths，seed/digest 可复算；
14. 任一 rate LCB 不严格为正时 `RateStable=false`；
15. point `S_RATE_TURN` 对 `G_H,G_K,Q_H,Q_K` 分别严格单调；修改 `T_H/T_K` 不改变主分数；
16. bootstrap 只在 point tau 重算 Q，不搜索 tau；replicate `Q<=0.5` 不触发 point
    ineligibility，而是通过联合分数连续影响 replicate 排名；
17. top combined score tolerance tie 使用 width/start canonical tie-break，并完整报告 tie set；
18. 无 eligible 输出 `ABSTAIN_NO_V3_ELIGIBLE`；combined-rank frequency 不足输出
    `ABSTAIN_COMBINED_RANK_UNSTABLE`；达到 0.80 输出 `SELECTED_WINDOW`；
19. non-finite、缺失 boundary、candidate count 漂移、bootstrap digest 或 verifier mismatch
    必须 BLOCK，不得 ABSTAIN。

## 11. 与 V2 的逐项差异

| 组件 | V2 | V3 |
| --- | --- | --- |
| Point aggregation | identity micro mean | equal-category macro mean |
| Candidate domain | 36-layer central `11...24`, widths 3/4/5/6, 42 | 原样保留 |
| Net direction | `G_H>0,G_K>0` | 保留，但基于 macro curve |
| Relative admission | `SoftRelativeStable` 为硬门槛 | 仅可作诊断，不 gating |
| Shape concept | shared biphasic reversal with orientation `q` | shared aggregate rate change，无 orientation |
| Direction sign | 左右方向必须反向并满足 12 margins | 不要求变号，也不要求 H/K delta 同号 |
| Shape uncertainty | simultaneous direction LCB | 删除 |
| Turn search | point pair + bootstrap re-search | 仅 full-sample macro curve 各搜索一次 |
| Turn frequency | `tau_pair_frequency>=0.80` | 删除 |
| Turn strength | `S_SHAPE`/direction margins | `Q=BSS/SST>0.5` 的主导两段速率变化 |
| External shape support | 可借窗口外一步 | 删除，仅用 window-internal rates |
| Rate factor | `S_RATE=sqrt(G_H*G_K)` | 保留为联合分数的 rate factor |
| Turn factor in ranking | 不进入排名 | `S_TURN=sqrt(Q_H*Q_K)` |
| Point ranking | `S_RATE` | `S_RATE_TURN=sqrt(G_H*G_K*Q_H*Q_K)` |
| RateStable | V2 within-window H/K max-t | 公式保留，estimand/bootstrap 改为 macro |
| Point tie | scientific tie 导致 ABSTAIN | canonical width/start tie-break，完整披露 tie set |
| Rank frequency | rate-rank `>=0.80` | combined-rank `>=0.80`；固定 point tau 重算 replicate Q |
| Empty eligible | `ABSTAIN_NO_RATE_STABLE_ELIGIBLE` | `ABSTAIN_NO_V3_ELIGIBLE` |
| Rank unstable | `ABSTAIN_RATE_RANK_UNSTABLE` | `ABSTAIN_COMBINED_RANK_UNSTABLE` |
| Hidden metrics | diagnostic only | 保留 diagnostic-only 边界 |

V3 与 V2 的 trajectory input schema 可以兼容，但 point estimand、bootstrap aggregate、
eligibility、shape fields 和 tie semantics 不兼容；不得把旧 V2 row 仅改字段名后冒充 V3。

## 12. 已知开发例与不得越界的结论

MMLU 5-shot 图中的视觉窗 `15:18` 对应：

```text
W(15,4)=[15,19)
entry=B15
exit=B19
```

已确认的开发数值为：

```text
G_H=0.12706225709344093
G_K=0.15649257081076318
S_RATE=0.14101169903795466
NetPositive=true
RateStable=true
```

这组数值满足 `S_RATE=sqrt(G_H*G_K)`。V3 不再因旧 `BiphasicStable`、direction LCB 或
`tau_pair_frequency` 拒绝它；它是否进入 `EligibleV3` 只取决于从完整 category-macro
trajectory 按本文件复算的 `AggregateCommonTurn`。其 `S_RATE_TURN` 还需要完整轨迹复算得到
`Q_H/Q_K` 后才能定义。即使通过，也必须对全部 42 个候选使用同一规则重算后才能报告
point rank、combined-rank frequency 或最终 `SELECTED_WINDOW`。

本例是 V3 的 post-hoc development illustration，不是 acceptance fixture，不允许为了让
它通过而修改 `Q>0.5`、tie、macro aggregation、candidate domain 或任何阈值。

## 13. 尚存风险与版本边界

- `Q>0.5` 冻结了“主导转折”的最小解释强度，但尚未经过 prospective validation；
- width 3–6 很短，平滑加速/减速可能被两段常速模型近似成转折；
- macro aggregation 防止大 category 主导，但可能提高小 category 噪声的权重；
- `tau_H==tau_K` 是严格位置一致，可能对相邻 boundary 的近同步现象偏保守；
- endpoint rate 与两段拟合不使用 hidden geometry，也无法证明 loop 的因果机制；
- `Q` 同时参与准入和连续排名，可能使 point score 对 `Q=0.5` 附近的估计变化更敏感；
- 原始 `Q` 的 eligible 范围为 `(0.5,1]`，因此 turn factor 对 point 分数的乘法惩罚有界；
- point tie 的 width/start 决胜只用于可复现发布，不代表科学优越性。

这些都是冻结 V3 的可证伪风险，不是本文件内可继续调节的自由参数。改变 turn threshold、
允许 `tau_H/tau_K±1`、增加最短 segment、引入 fixed-tau CI、删除 combined-rank frequency、
把 relative/hidden 指标重新加入 gating，或把原始 Q 改为 threshold-centered/turn-amplitude
变换，都必须建立新版本。
