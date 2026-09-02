# RELATIVE_BIPHASIC_REVERSAL_V1

```text
METHOD_ID=RELATIVE_BIPHASIC_REVERSAL_V1
METHOD_STATUS=RETROSPECTIVE_METHOD_DEVELOPMENT
SCOPE=CROSS_PHASE_RULE_SPEC
```

## 1. 继承关系与排名

本版本继承 `ENTROPY_KL_CONSENSUS_V1` 的 `gH/gK`、`G_H/G_K`、overlap/flank
references、`O/F/z` 与：

```text
S_REL = min(z_OH,z_OK,z_FH,z_FK)
```

V1 仍只按 `S_REL` 在最终 eligible 集中排名。它删除 Strict Eligibility 中“四个
reference 本身必须同时反向”的绝对符号否决，改为相对稳定性加窗口内形状准入。

## 2. Soft Relative Eligibility

```text
NetPositive = G_H>0 AND G_K>0

SoftRelativeStable =
  LCB(O_H)>0 AND LCB(O_K)>0
  AND LCB(F_H)>0 AND LCB(F_K)>0
```

这里的 LCB 沿用该 Phase 对旧 `O/F` percentile bootstrap 的冻结合同；等于零失败。

## 3. Shared Biphasic Reversal

对窗口内共同候选分割点 `tau=s+k, k=1...w-1` 和方向 `q in {+1,-1}`，先逐样本算：

```text
L_i^m(tau) = mean_{j=s}^{tau-1} g_i,j^m
R_i^m(tau) = mean_{j=tau}^{s+w-1} g_i,j^m
```

同一个 `(q,tau)` 必须支持 H/K 同步、左右相反的两段方向。其 12 个单侧 margin 为：

```text
segment:
  q*L_H, q*L_K, -q*R_H, -q*R_K

two-transition core:
  q*gH_(tau-2), q*gK_(tau-2)
  q*gH_(tau-1), q*gK_(tau-1)
 -q*gH_tau,     -q*gK_tau
 -q*gH_(tau+1), -q*gK_(tau+1)
```

每个 scalar 必须先逐样本计算，再按冻结 strata 聚合；禁止只检查总体平均曲线。允许的
边界支持至多借用窗口外紧邻的一步 transition，固定为：

- `w=3`：`k=1` 借一步左侧外部支持，`k=2` 借一步右侧外部支持；
- `w=4`：`k=2` 全内部，`k=1/3` 分别借对应一侧的一步外部支持；
- `w=5`：`k=2/3` 全内部，`k=1/4` 分别借对应一侧的一步外部支持；
- `w=6`：`k=2/3/4` 全内部，`k=1/5` 分别借对应一侧的一步外部支持。

若支持 transition 超出模型 boundary，则该 pair 不可评分；不得 pad、clamp、插值或删除
margin。新的 width 若没有预先冻结的两侧两步支持合同，不属于本 V1。

对每个 pair：

```text
z_shape(a,r) = theta_hat(a,r)/max(SE_boot(a,r),1e-12)
S_SHAPE(a)   = min_r z_shape(a,r)
```

`S_SHAPE` 只用于定位与诊断共享 change point，不进入最终窗口排名。

## 4. BiphasicStable

每个窗口在所有可评分 `(q,tau)` 与 12 margins 上构造 one-sided 95% within-window
simultaneous max-t lower bounds。point `(q*,tau*)` 必须是 `S_SHAPE` 的唯一最大值；
tie tolerance 固定为 `rel_tol=abs_tol=1e-12`。

```text
e_b(a,r) = (theta_hat(a,r)-theta_b(a,r))/max(SE_boot(a,r),1e-12)
M_b      = max over all scorable (a,r) in this window of e_b(a,r)
c_0.95   = linear empirical quantile({M_b},0.95)
LCB_sim(a,r) = theta_hat(a,r)-c_0.95*max(SE_boot(a,r),1e-12)
```

```text
BiphasicStable =
  unique point (q*,tau*)
  AND all 12 simultaneous LCB(q*,tau*,r)>0
  AND tau_pair_frequency>=0.80
```

`tau_pair_frequency` 要求每个 bootstrap replicate 重新计算全部 pair 并重新搜索唯一
argmax；replicate tie 记为不匹配。等于 `0.80` 通过。

## 5. 最终准入与发布

```text
NewEligible =
  Scorable
  AND NetPositive
  AND SoftRelativeStable
  AND BiphasicStable
```

只在 `NewEligible` 内按 point `S_REL` 排名。窗口 selection frequency 使用固定的 point
eligible set；每个 replicate 以冻结的 full-sample SE 为 z 分母重新计算 `S_REL` 并重选
top-1。无 eligible、point top tie 或 top-window bootstrap selection frequency `<0.80` 时
分别返回 `ABSTAIN_NO_NEW_ELIGIBLE`、`ABSTAIN_NO_UNIQUE_TOP1`、
`ABSTAIN_WINDOW_UNSTABLE`。

本规则受到已观察的 Qwen3-1.7B/4B MMLU outcome 启发，因此历史两 cell 仅是
retrospective method-development，不能作为 prospective validation。

历史完整实例：`../phase5/loopscope_phase5_gate_g_relative_biphasic_handoff.md`。
