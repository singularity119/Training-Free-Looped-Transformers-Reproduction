# LoopScope planning 索引

这里保存随 Git 同步的阶段控制、冻结合同、Gate handoff、修订补充与审计记录。
新任务先读 [PROJECT_MEMORY.md](../PROJECT_MEMORY.md)，再按问题检索这里的原始依据。
新设备操作见 [接续说明](../docs/new_device_handoff.md)。

## 当前控制入口

最新阶段的唯一动态控制入口是 [Phase 7 control](phase7/loopscope_phase7_control.md)。
2026-09-02 迁移核对时，其状态为 `PHASE7_TERMINAL_PASS_AFTER_GATE_F`，
`ACTIVE_GATE=NONE`，`AUTHORIZED_EXECUTOR=NONE`。后续若有更新，以 control 顶部状态区块为准。
旧文件中“当前”“已激活”“允许执行”都是写入当时的语境，不能单独视为今天的授权。

## 阶段和方法索引

| 目录 | 记录范围 | 控制文件 | 执行说明 |
| --- | --- | --- | --- |
| `phase1/` | 初始 Qwen3-1.7B × MMLU grid 与工作区治理 | [control](phase1/loopscope_phase1_control.md) | [runbook](../docs/loopscope_phase1.md) |
| `phase2/` | refinement/NCA 机制与 fixed-horizon 补充 | [control](phase2/loopscope_phase2_control.md) | [runbook](../docs/loopscope_phase2.md) |
| `phase3/` | 选窗回收、blind completion、variable-width 与补充实验 | [control](phase3/loopscope_phase3_control.md) | [runbook](../docs/loopscope_phase3.md) |
| `phase4/` | MMLU-Pro prefix full-vocabulary selector 与 CoT outcome | [control](phase4/loopscope_phase4_control.md) | [runbook](../docs/loopscope_phase4.md) |
| `phase5/` | Qwen3-4B-Base 尺度迁移、选窗规则和几何诊断 | [control](phase5/loopscope_phase5_control.md) | [runbook](../docs/loopscope_phase5.md) |
| `phase6/` | 已终止 MMLU-Pro 分支、MMLU 0/5-shot、V3 与五-cell outcome | [control](phase6/loopscope_phase6_control.md) | [runbook](../docs/loopscope_phase6.md) |
| `phase7/` | 三种 Base 架构、V3.1 与 Gate E/F outcome | [control](phase7/loopscope_phase7_control.md) | [runbook](../docs/loopscope_phase7.md) |
| `selector_rules/` | 跨阶段版本化方法定义；不授予实验权限 | [规则索引](selector_rules/README.md) | [V3.1.0](selector_rules/aggregate_common_turn_v3_absolute_rate.md) |

最近的终态依据：

- [Gate D V3.1 审计](phase7/loopscope_phase7_gate_d_audit_decision_pass_20260815.md)
- [Gate E 审计](phase7/loopscope_phase7_gate_e_and_phase_audit_pass_20260816.md)
- [Gate F 与阶段重新终审](phase7/loopscope_phase7_gate_f_and_phase_reclose_audit_pass_20260816.md)

## 原文保留和路径解释

2026-09-02 将外层 `.planning` 目录迁入 Git 根，保留各阶段原始文件名与原文。
唯一真实副本位于 `<repo>/.planning/`；旧设备 `<repo>/../.planning` 是指向它的兼容链接。
原 `AGENTS.md` 未修改，所以新设备也需要按接续说明建立该链接。

根目录中的 50 个 `loopscope_phase*` 条目是指向 `phaseN/` 文件的相对符号链接，
用于旧记录的短路径兼容，不是重复文件。新引用使用 `phaseN/` 下的规范路径。

历史正文中的路径以当时工作区为基准，不能一律按当前 Markdown 所在目录拼接：

| 原文写法 | 迁移后的查找方式 |
| --- | --- |
| `.planning/phaseN/...` | 相对 Git 根查找 `.planning/phaseN/...` |
| `.planning/loopscope_phaseN_...` | 使用仓库 `.planning` 根中的兼容链接，或按文件名在 `phaseN/` 中查找 |
| `../loopscope-tflt/AGENTS.md`、`loopscope-tflt/docs/...` | 对应当前 Git 根的 `AGENTS.md`、`docs/...` |
| 历史 control 中 `./loopscope_phaseN_control.md` | 查找对应 `phaseN/loopscope_phaseN_control.md` |
| `/Users/huangxutao/...` | 旧设备 provenance；本地仓库路径映射到当前 checkout，不能照抄为新设备命令 |
| `资产/...` | Git 根的外层独立人类交付目录；未纳入本仓库 |
| `/hpc2hdd/...` | 保留原始远端工件路径；使用前核对现场，不为目录整齐而迁移 |

为保留已冻结历史，迁移没有批量重写 control/contract/handoff 内的路径、线程 ID、命令、方法或结果。
若旧文档标题/段落的状态与当前 control 顶部不同，按阶段时间线和最后的终态审计理解。

## 更新规则

- 只在 Gate 决策、执行身份、实验配方或权限变化时更新相应 control。
- 新阶段使用新的阶段目录和明确合同，不把旧 Gate 重新标成活动。
- 方法版本已冻结时，科学变更新增版本，不追溯覆盖原实例。
- `PROJECT_MEMORY.md` 保留关键结论和证据链接；这里保留可检索的详细历史。
- 不追加逐工具调用日志，不提交凭据、原始聊天、模型、数据或完整运行输出。
