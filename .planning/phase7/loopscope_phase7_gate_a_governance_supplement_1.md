# Gate A Governance Supplement 1

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=A
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a00159-6c55-72b3-bbf8-5ac798868258
CORRECT_EXECUTOR_TITLE=execute-LoopScope-三模型Provenance与轨迹探测实现-第7阶段-Gate A
SUPPLEMENTS=.planning/phase7/loopscope_phase7_gate_a_handoff.md
SCIENCE_CHANGE=NONE
CURRENT_GATE_ACTION_ENVELOPE_CHANGE=NONE
```

## 1. 标题纠正

Gate A 的规范标题以本文件中的 `CORRECT_EXECUTOR_TITLE` 为准。用户最终选择 per-Gate topic，
并已要求同步更新 `research-gate-orchestrator`：同一阶段保持 Project 与 Phase，
`<Card-or-topic>` 概括当前 Gate 自己的目标。Phase 7 预定映射为：

```text
Gate A: execute-LoopScope-三模型Provenance与轨迹探测实现-第7阶段-Gate A
Gate B: execute-LoopScope-三模型四样本CUDA轨迹Smoke-第7阶段-Gate B
Gate C: execute-LoopScope-三模型MMLU1531逐层轨迹采集-第7阶段-Gate C
Gate D: execute-LoopScope-三模型V3选窗评分与数据封存-第7阶段-Gate D
```

线程 ID 仍是唯一执行身份；本次改名不创建新 executor、不重启工作，也不改变 Gate A 权限。

## 2. 协作协议与敏捷主线

Executor 必须完整重读并持续遵守 `research-gate-orchestrator/SKILL.md` 及
`references/protocol.md`：

- 优先最短可信实验主线，只做会影响科学解释、安全、成本、provenance 或下一 admission 的
  决定性检查；
- 对拟增加的 must-pass 项应用 deletion test；删除只服务通用框架、重复全套审计、假想攻击
  或非正常路径防御的工作；
- 当前 Gate 内低风险工程修复可持续推进；一次 job/test 失败是 attempt-level 事件，不自动
  等于 Gate-level `BLOCK`；
- Executor 只在 Gate 完成或真正无法继续时发送一次 `GATE_A_FINAL_AUDIT` 或 `BLOCK`，且
  terminal delivery 必须有可见确认。

## 3. Subagent 权限

Gate A executor 被明确允许按 skill 使用 subagent：

- 仅当并行委派能实质缩短当前 Gate；
- 任一时刻最多三个 active subagents；完成或退出后可以启用新的窄任务；
- 任务必须完全落在 Gate A handoff 的既有科学、路径和权限 envelope 内；
- subagent 不得跨 Gate、修改合同、独立 Git integration、远端写入、GPU/Slurm、发送
  terminal event 或再委派；
- executor 必须复核、整合并对所有结果、commit/push 与 terminal packet 独立负责。

## 4. Operational permission 路由

用户已把 Phase 7 普通 operational-permission 决策委托给 planning thread。若 Gate A 需要
当前 handoff 之外的普通工程权限，executor 先给 planning thread 发送窄请求，说明：

```text
OPERATIONAL_PERMISSION_REQUEST
blocked Gate outcome:
observed normal-path evidence:
smallest permission required:
exact host/path/action/duration:
why existing authority is insufficient:
state that remains forbidden:
```

Planning 核实材料性后可直接发最小充分 supplement，不要求用户逐项批准。Executor 不得
自授权。科学合同变更、提前解封、阶段/实验矩阵扩张、不可逆破坏、凭证/秘密、外部人员或
法律承诺仍必须停下并请求用户决定。

本 supplement 不扩大 Gate A 当前 action envelope；Gate B–D 继续锁定。
