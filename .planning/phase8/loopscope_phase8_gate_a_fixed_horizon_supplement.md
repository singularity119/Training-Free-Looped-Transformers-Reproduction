# Gate A 科学配置补充：所有 K 固定 alpha=1

日期：2026-09-05。用户最新明确指令是“和以往实验相同的固定 alpha=1，即循环步数不论怎么变，总时长不变”。本补充仅更新这一科学配置，不扩大执行权限。

- Planning: 01a06fe9-c7bb-7d72-a906-234f301de317
- Executor: 01a07030-6413-7660-b712-c18d9e93d26e
- Active handoff: [Gate A handoff](loopscope_phase8_gate_a_handoff.md)
- 固定 damped_euler / alpha=1 / h=1/K / total horizon=1，适用于两个模型、全部窗口、原版与 SFA 版本。
- K=2/3/4 对应 h=1/2、1/3、1/4；不授权任何 alpha 随 K 增大或 Lower-alpha 的独立分支。
- K 集合、评测 split、SFA 提案确认仍待回复；当前明确 alpha 不等于用户已批准全部 K。
- Gate A 的模型窗口配置须把 alpha=1 标为已冻结；数学函数仍可用通用 lambda 参数测试，不据此扩大实验搜索。
- 跨 K 状态和残差前缀不保证相同，不能沿用 v0.1 固定 h 的前缀复用论证。Gate B 前需写清 basis 拟合 K 与应用 K。Gate A 不采集真实 residual。

原 handoff 的文件范围、protected-code 只读、HPC 只读、无模型/GPU/Slurm 权限、一个终态包与送达确认要求不变。验收只需补看配置中的固定 alpha 字段，无须为本补充增加独立测试框架或重跑历史结果。
