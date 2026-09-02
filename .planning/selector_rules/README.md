# LoopScope 跨阶段选窗规则库

本目录只保存可跨实验阶段复用的 selector 方法定义。某个 Phase 的模型、数据集、
probe 位置、候选范围、bootstrap seed、样本分层、运行权限和 outcome 信息屏障，仍由
该 Phase 的 control、contract 和 handoff 绑定。

## 规则版本

| 顺序 | METHOD_ID | 文件 | 核心变化 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | `ENTROPY_KL_CONSENSUS_V1` | `entropy_kl_consensus_v1.md` | 用 overlap/flank 相对反转证据排名，并采用严格参照符号准入 | 历史基础规则 |
| 2 | `RELATIVE_BIPHASIC_REVERSAL_V1` | `relative_biphasic_reversal_v1.md` | 删除绝对参照符号否决，增加共享双相反转形状准入，仍按 `S_REL` 排名 | 方法开发版本 |
| 3 | `RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE` | `relative_biphasic_reversal_v2_absolute_rate.md` | 完整继承 V1 准入，改用窗口自身的 `S_RATE` 排名 | 历史主版本 |
| 4 | `AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE` V3.1.0 | `aggregate_common_turn_v3_absolute_rate.md` | 删除双相方向和 tau-bootstrap 硬门槛，采用 category-macro 共同主导速率转折；保留 RateStable，并用 `S_RATE_TURN=sqrt(G_H*G_K*Q_H*Q_K)` 联合排名 | 当前主版本，待前瞻验证 |

## 使用约定

1. 新 Phase 必须在读取 outcome 前，把选用的 `METHOD_ID` 和本文件的具体版本写入
   Phase contract；只写“沿用旧规则”不构成冻结。
2. 复用规则不等于复用窗口。不同模型深度或任务必须用其自身 outcome-blind trajectory
   重新形成候选、准入和排名，除非实验问题明确是固定窗口迁移。
3. Phase contract 必须显式绑定：输入分布、probe token、boundary 定义、候选域、width、
   strata、bootstrap 次数与 seed、CI、tie tolerance、selection-frequency threshold 和
   `ABSTAIN` 状态。
4. 规则文件不保存模型运行权限、executor、Slurm、run root 或已经观察的 accuracy。
5. 已进入 Phase 冻结或执行的规则若修改公式、准入、阈值或状态机，必须新增版本且不得
   追溯覆盖历史实例。尚未进入前瞻 Gate 的当前开发主规则，仅在用户明确授权时允许原文件
   升 minor version；必须同步更新 `METHOD_VERSION`、SHA、规则索引与人类可读报告。
6. 历史 Phase 文件仍是相应已执行 Gate 的 provenance。共享规则文件不会追溯改变旧
   Gate 的声明边界。

`PV-EK-TRS` 是 `ENTROPY_KL_CONSENSUS_V1` 在 prefix 最后一个 token 的 full-vocabulary
entropy/KL trajectory 上的 Phase 4 实例；choice entropy 与 full-vocabulary entropy
属于不同输入合同，不能把二者的原始分数跨 cell 直接比较。
