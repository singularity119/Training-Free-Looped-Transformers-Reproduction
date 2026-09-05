# GATE_A_AUDIT_DECISION

日期：2026-09-05。Decision: **PASS**。
Planning: 01a06fe9-c7bb-7d72-a906-234f301de317
Audited executor: 01a07030-6413-7660-b712-c18d9e93d26e
Audited commit: 3fe34c46e3ac9774907349bdbf1702ab2180919f
Evidence: [执行证据](loopscope_phase8_gate_a_evidence.md)

## 直接核对的三个决定性事实

1. 来源消息与 control 中 Gate A executor 身份一致；现场 HEAD 为上述提交，loopscope 工作树 clean，提交只新增获准的数学模块、测试、模型窗口 JSON 和 Gate 证据四文件。protected runtime 无变更。
2. 直接读取数学实现和四窗口配置，并运行 `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_loopscope_phase8_spectral_core.py'`：6/6 通过、exit 0。投影、lambda=0/0.5/1、方向符号不变性、同范数控制与层边界映射均支持 Gate A 目标；alpha=1 为 FROZEN_USER_ALL_K。提交 diff check 通过。
3. 最小接入方案能解决实际 normal-path 问题：audit callback 返回值被忽略，故 B 应 opt-in 接入真实 Euler residual；默认 None 保持原路径，pre-answer 位置须与 HFLM continuation tokenization/right padding 对齐。原生 residual 与 FP32 测量差分分开。这些是 B 的必要实现/验证项，不要求 A 提前实现。

## 证据边界与接受的偏差

HPC 模型 config、包版本和 debug 资源是 executor 本 Gate 的只读现场记录；本次 planning 未重复 SSH。A 未加载权重或运行模型，因此这不是 GPU 路径通过证明。B 应同步专用 clone，并在 debug 验证两模型权重加载/评分/干预路径。

1.7B 的历史 FP16 runtime 来自仓库历史文档，checkpoint config 为 BF16；二者已明确区分，B 前必须显式冻结 runtime dtype。executor 曾在枚举只读根之外，对给定历史 1.7B root 作目录存在/文件名检查，未读取内容或 outcome，也未写入。记录为一次非材料性范围偏差，不追认未来该根访问，不要求重跑或返修。

此前详细通知据 executor 报告被自动审批拒绝；当前最小 GATE_A_FINAL_AUDIT 已实际到达规划任务，交付恢复，不需要重复发送终态。

## 决策与下一步

Gate A 所需工程证据充分，PASS；不等于 SFA 有效或正式实验已获授权。无返修要求。Gate A executor 权限撤销，任务保留只读。

Gate B admission 暂未满足：用户尚未明确评测 split、K 集合、rank1/lambda0.5 提案。alpha=1 的用户确认适用于任何获准 K，但不自动批准 K=4。planning 保持阶段连续推进职责；收到这些科学选择后，冻结 B 合同及 basis 拟合/应用 K、运行 dtype、评分和最小预算，再创建新独立执行任务。无模型/GPU/Slurm/outcome 权限提前授予。
