# LoopScope Phase 7 Gate F Fresh-Verifier Lifecycle Supplement 4

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=F
AUTHORIZED_EXECUTOR_THREAD=01a00965-fa7f-7411-b060-0c8315a94711
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
SCIENCE_CHANGE=NONE
SECOND_VERIFIER_INVOCATION_AUTHORITY=NONE
```

Gate F combined analysis 已完成并产生 write-once
`analysis/combined_analysis.json`。随后唯一获授权的 fresh verifier invocation 已启动；PID
`2258047` 当前仍活跃，`fresh_verifier_receipt.json` 尚未出现。普通 SSH/tool window 早于程序
结束返回，不是 verifier failure。原 analysis heartbeat 已在 analysis terminal 后删除，不再活跃。

授权 exact executor 创建一个新的、唯一的 10-minute metadata-only heartbeat，专门监控 PID
`2258047` 与精确 formal root：

```text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase7-gate-f-20260816T073546Z-formal
```

每次唤醒只允许一次有界、只读检查：PID state/elapsed，以及
`analysis/combined_analysis.json` 和 `analysis/fresh_verifier_receipt.json` 的 existence/size/mtime。
不得读取 combined analysis 内容、partial outcome、accuracy/gain 或重复调用 verifier。

终态路由：

- PID 仍活跃：保持安静并结束该次唤醒；
- PID 结束且 fresh receipt 存在：先删除 heartbeat，再向 exact executor 投递
  `AUTOMATION_TERMINAL_RESUME`；executor 可读取并审计 fresh verifier receipt、CLI/session exit evidence，
  然后准备唯一 `GATE_F_FINAL_AUDIT`；
- PID 结束但 receipt 不存在，或出现异常 process state：先删除 heartbeat，再投递终态 evidence；
  executor 只恢复 session/exit/error 证据并报告 planning，不得自行第二次调用 verifier、覆盖工件、
  重跑 analysis/formal 或修改 science。

Heartbeat 必须保留可见 creation/deletion/terminal-self-wake 证据；无法确认 executor continuation 时，
按 research-gate-orchestrator 向 planning thread 发送完整 `AUTOMATION_RELAY_REQUIRED`。
