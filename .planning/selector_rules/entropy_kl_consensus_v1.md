# ENTROPY_KL_CONSENSUS_V1

```text
METHOD_ID=ENTROPY_KL_CONSENSUS_V1
METHOD_STATUS=HISTORICAL_BASELINE
SCOPE=CROSS_PHASE_RULE_SPEC
```

## 1. 输入与窗口语义

输入是同一 model-task cell 中，每个样本 `i` 在 native no-loop forward 下形成的逐层
entropy `H_i,j` 与 KL-to-final `D_i,j`。输入分布可以是冻结的 choice distribution，
也可以是冻结的 full-vocabulary distribution，但必须由 Phase contract 明确，且两种
输入的分数不能混合比较。

窗口 `W(s,w)=[s,s+w-1]` 表示 decoder blocks `s...s+w-1`，对应 boundary 变化
`B_s -> B_(s+w)`。定义期望方向的逐层变化：

```text
gH_i,j = H_i,j - H_i,j+1      # 正值：entropy 下降
gK_i,j = D_i,j+1 - D_i,j      # 正值：KL-to-final 上升

G_m(s,w) = mean_i[(1/w) * sum_{j=s}^{s+w-1} g_i,j^m], m in {H,K}
```

## 2. 相对邻域证据与排名

同宽 overlap references 为 `W(s-1,w),W(s+1,w)`；flank references 为
`W(s-w,w),W(s+w,w)`。只有所有必需参照都合法时才可评分：

```text
O_m = G_m(s,w) - 0.5*(G_m(s-1,w)+G_m(s+1,w))
F_m = G_m(s,w) - 0.5*(G_m(s-w,w)+G_m(s+w,w))

z_X = X / max(SE_boot(X), 1e-12)
S_REL = min(z_OH,z_OK,z_FH,z_FK)
```

主排名按 `S_REL` 降序。若需要稳定的展示次序，可用 width、start 升序作为表格排序，
但它们不得打破科学上的 point tie。`S_REL` 仅在同一 model-task cell 内解释。

## 3. Strict Eligibility

历史严格准入同时要求：

```text
Scorable
AND G_H>0 AND G_K>0
AND 四个 reference window 的 G_H<0 AND G_K<0
AND LCB(O_H)>0 AND LCB(O_K)>0 AND LCB(F_H)>0 AND LCB(F_K)>0
```

等于零一律失败。CI/bootstrap 的精确定义、replicate 数、seed、样本分层和
selection-frequency threshold 必须由 Phase contract 绑定。

## 4. 发布边界

- `rank-1`、`eligible`、`selected` 和 `positive gain` 是四种不同状态。
- 无 eligible window、point top 不唯一或 bootstrap 排名不稳定时必须返回 Phase contract
  定义的 `ABSTAIN`，不得降低阈值以强行选窗。
- 本规则不检查窗口内部是否存在持续的共享双相反转，因此小锯齿和 reference-range
  artifact 是已知风险。

历史来源：Phase 3/4/5 的 CONSENSUS selector 与 Phase 5 Gate E 多宽度扩展。

