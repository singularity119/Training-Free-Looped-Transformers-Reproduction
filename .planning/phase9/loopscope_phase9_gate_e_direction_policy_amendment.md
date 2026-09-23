# Gate E 方向策略可切换模块补充（2026-09-23）

用户新增明确要求：逐轮更新方向必须作为可切换的增量模块，能够选择“t0方向用于所有后续轮次”或“每轮使用前一轮残差重估方向”。本补充优先于Gate E原contract/handoff中只描述Lag1的实现措辞；不改变已确认的模型/窗口/K/λ、12个新增正式配置、对照、预算或探索性统计。

## 对外配置与实现

新增Gate E模块必须暴露**显式** `direction_policy`，仅允许 `fixed_t0` 和 `lag1` 两值；与 `intervention_mode`（`spectral`、`matched_norm`）独立组合。四种组合走同一个Gate E scoring/adapter、有效prefix掩码、原生残差与FP32精确SVD、答案位置更新路径。不要复制两套SVD或为两个方向策略分别维护完整评分器，也不覆盖原Phase9 Online-t0/Matched模块。配置、CLI/manifest、run metadata、verifier、debug输出均记录policy与mode；缺失/未知policy报错，不靠隐式默认猜测实验臂。

`fixed_t0`：t0从D0[S,:]拟合v0，t>=1始终用v0衰减当前答案位置；对应原Phase9 Online-t0或原Matched-norm。`lag1`：t0拟合v0，t>=1使用上一轮v_(t-1)处理当前答案位置，同时从当前**干预前**原生D_t[S,:]拟合供下一轮使用的v_t；最后一轮不做无用SVD。每轮来源要在有界日志里记录 `applied_t` 与 `direction_fit_t`，验证两策略t0/t1一致、t2起分化；两个mode都支持两策略，Matched在自身轨迹内按当前所选方向的定向处理范数比例缩放。方向只限同题exact retained prefix，换题释放，不能由当前单答案向量自投影。

debug必须对两模型、两mode、K3/4验证切换无状态串扰、K2两policy及原Phase9对应臂分数/残差在同配方下数值等价，并针对lag1证明t2来源D1、K4 t3来源D2。等价容差按实际dtype和已用Phase9真实debug尺度事前明确，不以acc巧合代替分数级验证。

正式新增全量仍只运行 `direction_policy=lag1` 的12个配置；`fixed_t0` 的全量逐题原始分数使用既有Phase9通过核验的历史结果，不重算。可切换性是模块能力和debug验收，不另增正式panel或改变一次冻结分析。第九阶段原主线报告与代码保持历史版本。
