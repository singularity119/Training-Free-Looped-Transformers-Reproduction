# RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE

```text
METHOD_ID=RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE
METHOD_STATUS=CURRENT_PRIMARY_METHOD
SCOPE=CROSS_PHASE_RULE_SPEC
```

## 1. V2 的唯一方法变化

V2 完整继承 V1 的候选语义和准入：

```text
NewEligible =
  Scorable
  AND NetPositive
  AND SoftRelativeStable
  AND BiphasicStable
```

V2 不删除 `SoftRelativeStable`，所以 reference windows 仍会影响 admission。V2 只消除
reference family 对最终排名的直接影响：`S_REL` 与 `S_SHAPE` 均降为诊断，最终排名改用
中心窗口自身的绝对平均变化率。

## 2. Absolute Rate Score

对 `W(s,w)` 对应的 `B_s -> B_(s+w)`：

```text
G_H(s,w) = mean_i[(H_i,s-H_i,s+w)/w]
G_K(s,w) = mean_i[(D_i,s+w-D_i,s)/w]
```

只有 `G_H>0` 且 `G_K>0` 时定义：

```text
S_RATE(s,w) = sqrt(G_H(s,w)*G_K(s,w))
```

几何平均对两项分别严格单调，且不允许一项特别大完全补偿另一项接近零。不得改为带
可调权重的线性和，也不得在看到 outcome 后改按 total change 排名。`S_RATE` 只允许在
同一 model-task cell 内比较。

## 3. RateStable

复用该 Phase 冻结的 sample-stratified joint bootstrap；每个 replicate 必须同时重算
全部层、H/K、窗口和 width。对同一窗口的 `G_H/G_K` 构造 one-sided 95% simultaneous
lower bounds：

```text
RateStable = LCB_H>0 AND LCB_K>0
C_RATE = NewEligible AND RateStable
```

等于零失败。不确定性不混入 point `S_RATE`。

历史冻结实现采用 within-window H/K fixed-SE max-t：

```text
SE_m = sd({G_m^(b)}, ddof=1)
e_b,m = (G_m_hat-G_m^(b))/max(SE_m,1e-12)
M_b = max(e_b,H,e_b,K)
c_0.95 = max(0, linear empirical quantile({M_b},0.95))
LCB_m = G_m_hat-c_0.95*max(SE_m,1e-12)
```

标准差使用 `ddof=1`；linear quantile 在排序后的位置 `q*(R-1)` 做相邻 order
statistics 线性插值。rate、SE、`c_0.95`、LCB 或 replicate rate 任一非 finite 都是
`BLOCK_NONFINITE_RATE_ANALYSIS`，不得降格为 `ABSTAIN`。

未来 Phase 若改变 CI 合同，必须在 outcome 前显式绑定并视为新的实验实例；不得静默
称为完全复用历史 V2。

## 4. Ranking 与状态机

只在固定 point `C_RATE` 内按 `S_RATE` 降序。width/start 只用于表格展示，不得作为
科学 tie-break。若多个窗口与最高分满足 `rel_tol=abs_tol=1e-12`，返回
`ABSTAIN_NO_UNIQUE_TOP1`。

对固定 point eligible set，每个 bootstrap replicate 重算 `G_H^(b),G_K^(b)` 和
`S_RATE^(b)`；任一 rate 非正时，该 replicate 中该窗口不可胜，并重新选择 replicate
top-1：

```text
selection_frequency =
  count_b(point top exact window is replicate unique top1)/R
```

最终状态：

- 无 `C_RATE`：`ABSTAIN_NO_RATE_STABLE_ELIGIBLE`；
- point top 不唯一：`ABSTAIN_NO_UNIQUE_TOP1`；
- point top selection frequency `<0.80`：`ABSTAIN_RATE_RANK_UNSTABLE`；
- 否则发布 `selected_window`。

`rank-1`、`eligible`、`selected` 与 `positive gain` 仍必须分别报告。balance ratio、total
change、旧 `S_REL`、`S_SHAPE`、`q/tau`、turn strengths、pair frequency 与 `pi` 都只作
诊断，不进入 V2 排名。

## 5. 科学边界

V2 是在看过 Qwen3-1.7B/4B MMLU outcome 后形成的；这两个 cell 只能承担 retrospective
method-development。真正支持必须来自新 model-task cell，或尚未读取 outcome 且在评测
前完整冻结的候选集合。合法 `ABSTAIN` 是方法输出，不得为获得窗口而调规则。

历史完整冻结合同：
`../phase5/loopscope_phase5_relative_biphasic_v2_absolute_rate_contract.md`。
