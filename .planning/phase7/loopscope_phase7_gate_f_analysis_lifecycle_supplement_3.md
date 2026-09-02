# LoopScope Phase 7 Gate F Analysis Lifecycle Supplement 3

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=F
AUTHORIZED_EXECUTOR_THREAD=01a00965-fa7f-7411-b060-0c8315a94711
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
SCIENCE_CHANGE=NONE
FORMAL_RERUN_AUTHORITY=NONE
ANALYSIS_CONTENT_INSPECTION=FORBIDDEN_DURING_DIAGNOSTIC
```

Gate F pre-outcome verifier 已 PASS：27 cells、24 个新执行 cells、3 个 Gate E baseline references，
每 cell 14,042 records / 57 subjects，`outcome_aggregates_computed=false`。第一次 analysis invocation
未产生 CLI 输出，随后 fresh verifier 仅确认 `analysis/combined_analysis.json` 不存在。

只读代码审计表明，Gate F wrapper 调用 Gate E CLI main；`analyze` 成功路径必定先以 write-once
方式写入 `analysis/combined_analysis.json` 再打印 JSON，受控 `Phase7OutcomeError` 也必定打印 JSON。
不存在合法的“已完成但无输出且无 artifact”代码分支。分析包含 24 个 contrasts 的 2,000 次
subject-stratified paired bootstrap，可能超过普通 SSH/tool yield 窗口。因此先区分仍在运行与会话中断，
不得直接重跑或修改分析代码。

授权 executor 执行一次有界、只读 lifecycle check，且不得打开 outcome rows、分析内容或局部结果：

1. 在 HPC2 检查是否存在 command line 同时包含
   `run_phase7_gate_f_panel.py analyze` 和精确 formal root 的活跃进程，仅记录 PID/elapsed/state；
2. 仅以 `test -e`/`stat` 检查 `analysis/combined_analysis.json` 与
   `analysis/fresh_verifier_receipt.json` 是否存在、大小和 mtime，不读取内容；
3. 核对原调用可获得的 tool/session/exit 状态；不得把“无即时 stdout”等同于 exit 0。

条件处置：

- 若原 analysis 进程仍活跃：禁止第二次启动。使用原 session 等待；若必须跨 turn，则创建唯一
  10-minute bounded heartbeat，只监控该 PID terminal 与 artifact existence，不读取内容。进程完成且
  artifact 出现后，只运行一次 fresh verifier。
- 若进程已消失且 artifact 仍不存在：将原调用记为 invocation/session lifecycle failure，授权用同一
  commit、同一 run root、同一 audited Python、同一 arguments 重试 analysis 一次。必须使用可持续的
  unified exec/SSH session；首次 yield 后保存 session ID，并以不超过 60 秒的 wait/poll 继续到明确
  exit code，不得因暂时无 stdout 启动 verifier 或第二个 analysis。不得改 bootstrap、seed、panel、
  input、analysis path 或实现。
- 若 artifact 已存在：不得重跑 analysis；直接用 audited Python 运行一次 fresh verifier。

任何条件下均不授权重跑 24 个 formal cells、修改 science/panel、删除工件、覆盖已存在 artifact、
查看 partial outcome 或创建 Gate G。若受控重试得到非零 exit，保存完整但不含 outcome payload 的
错误与 lifecycle evidence，并回报 planning。
