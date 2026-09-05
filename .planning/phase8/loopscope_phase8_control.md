# LoopScope Phase 8 Control

```text
CONTROL_ID=LOOPSCOPE_PHASE8_CONTROL_V1
STATUS=GATE_A_PASS_WAITING_SCIENTIFIC_CHOICES
PLANNING_THREAD=01a06fe9-c7bb-7d72-a906-234f301de317
ACTIVE_GATE=NONE
AUTHORIZED_EXECUTOR=NONE
LAST_GATE_A_EXECUTOR=01a07030-6413-7660-b712-c18d9e93d26e
AUTHORIZED_EXECUTOR_TITLE=NONE
CURRENT_DECISION=GATE_A_PASS
GATE_A_STATE=PASS
GATE_B_STATE=LOCKED
GATE_C_STATE=LOCKED
GATE_D_STATE=LOCKED
STANDING_PHASE_CONTINUATION=USER_AUTHORIZED_ADVANCE_SERIAL_GATES_UNTIL_PHASE8_TERMINAL
OPERATIONAL_PERMISSION_DECISIONS=PLANNING_GATE_SCOPED_PER_PROJECT_DELEGATION
EXECUTOR_MODEL=INHERIT_USER_CONFIGURATION
BASE_COMMIT=b53ff067992e41b66a8df01794dfb106c61b29d0
AUDITED_COMMIT=3fe34c46e3ac9774907349bdbf1702ab2180919f
LATEST_ACCEPTANCE=.planning/phase8/loopscope_phase8_gate_a_acceptance.md
ACTIVE_HANDOFF=NONE
ACTIVE_SUPPLEMENT=NONE
FROZEN_USER_CHOICES=QWEN3_4B_BASE_12_15_13_16_CACHE_FIRST;QWEN3_1_7B_BASE_12_15_6_9_CACHE_LAST;MMLU_5SHOT;DAMPED_EULER;ALPHA_1_ALL_K_FIXED_HORIZON
PENDING_USER_CHOICES=EVALUATION_SPLIT;K_SCOPE_AND_SFA_PROPOSAL_CONFIRMATION
NEXT_ADMISSION=SCIENTIFIC_CHOICES_RESOLVED_AND_GATE_B_CONTRACT_FROZEN
PHASE_TERMINAL=COMPLETE_FROZEN_PAIRED_PANEL_AND_PLANNING_PHASE_AUDIT_OR_DECLARED_MATERIAL_STOP
```

## 权限与事实来源

用户 2026-09-05 明确授权本任务规划、逐 Gate 创建执行任务、验收并持续到第八阶段完成；后续回复确认普通循环为原计划 Euler 步。用户进一步确认所有 K 固定 alpha=1，h=1/K，总时长为 1；固定步长而随 K 增大 alpha 的方案不进入本阶段。四个窗口按层编号解释，12:15=B12→B16。详细方案见 [总体计划](loopscope_phase8_plan.md)。Gate A 允许在科学选择未完成时推进独立工程准备；不授权真实模型实验。

本轮起点是 loopscope 分支 b53ff067，前序规划任务留下的 `.planning/README.md`、`.planning/phase8/` 和 `docs/loopscope_phase8.md` 文档修改由本规划任务所有。本次激活另外修改 AGENTS.md 与 PROJECT_MEMORY.md 的入口。这些文档可纳入单一规划提交；executor 不得把它们误当作自己的实现或回滚。

Gate A 具体本地写路径、Git、SSH、CPU 与禁止项以 handoff 为准。无 GPU/Slurm/model forward/outcome 权限；B/C/D 无 executor。默认 protected runtime 文件仍只读，后续需要变更时由规划基于 Gate A 接入点证据授权。

## 轻量验收和连续推进

A 验收只检查四窗口与实际模型 provenance、纯数学测试输出、最小接入方案能否支持 B。B 前必须解决待定 split/K/SFA，冻结 runtime/scoring 和预算。PASS 后撤销当前 executor 权限，立即给已满足 admission 的下一 Gate 创建新任务。用户未回复的科学选择不靠超时默认通过。

规划只拥有当前控制状态及科学合同的写权限；executor 只写自己的 Gate 实现与证据文件。通过的历史 Gate 不重审。负结果允许阶段完成，不要求反复调参得到正收益。

## Gate A 已验收

[验收决定](loopscope_phase8_gate_a_acceptance.md)：PASS，四文件实现与 6 项目标测试已核对。执行任务权限撤销并保留只读；原 handoff/supplement 现在仅为 A 的历史依据。尚无 Gate B executor 或运行授权；等待现有待定科学选择，之后继续阶段推进。
