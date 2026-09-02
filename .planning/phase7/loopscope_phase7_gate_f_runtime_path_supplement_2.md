# LoopScope Phase 7 Gate F Runtime Path Supplement 2

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=F
AUTHORIZED_EXECUTOR_THREAD=01a00965-fa7f-7411-b060-0c8315a94711
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
SCIENCE_CHANGE=NONE
FORMAL_RERUN_AUTHORITY=NONE
```

Gate F formal array job `10239586` 的 24 个任务均已 `COMPLETED 0:0`。第一次 pre-outcome
verifier 调用因不存在的相对路径 `.venv/bin/python` 在 Bash 启动阶段失败；Python 与 verifier
均未启动，未读取 sealed outcome/result rows，未产生 analysis artifact。因此该事件是
invocation-only operational repair，不要求也不授权重跑 formal cells。

HPC2 dedicated checkout 的已审计 Phase 7 Python 精确路径为：

```text
/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
```

授权 executor 先对该精确路径执行一次有界、只读的 executable/version check，然后在 dedicated
checkout 根目录以 `PYTHONPATH=src` 使用该解释器重试同一个 pre-outcome verifier：

```text
PYTHONPATH=src /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python scripts/loopscope/verify_phase7_gate_f_panel.py preoutcome --run-root /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase7-gate-f-20260816T073546Z-formal
```

除解释器路径与显式 `PYTHONPATH=src` 外不得改变 run root、arguments、panel、membership、
verifier logic 或 science。若 pre-outcome PASS，继续原 handoff 的一次 analysis 与 fresh verifier；
若失败，保留证据并向 planning 报告材料性错误，不得擅自重跑 formal、绕过 completeness barrier
或读取局部 outcome。
