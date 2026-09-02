# LoopScope Phase 7 Gate F 与阶段重新终审：PASS

## 裁决

Planning/audit thread `01a0013e-71c3-7c90-a547-4059b462dc7e` 对 Gate F executor
`01a00965-fa7f-7411-b060-0c8315a94711` 的唯一 `GATE_F_FINAL_AUDIT` 做轻量独立核验后，判定：

```text
GATE_F=PASS
PHASE7=TERMINAL_PASS_AFTER_GATE_F
AUTHORIZED_EXECUTOR=NONE
NEXT_GATE=NONE
```

## 独立核验依据

- 本地 `loopscope`、`origin/loopscope` 与 HPC2 dedicated checkout 均为
  `08da7733a1e8402677184c860d0ec35f041335aa`；远端 checkout clean，本地 staged membership 为空。
- 本地只保留七个既有 protected tracked dirty paths；Gate F 未覆盖、暂存或提交它们。
- Slurm formal array `10239586` 对应的 24 个执行任务全部为 `COMPLETED`、`ExitCode 0:0`。
- Canonical root：
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase7-gate-f-20260816T073546Z-formal`。
- Pre-outcome receipt：`PASS`；27 total cells、24 execution cells、3 retained Gate E baselines，
  每 cell 14,042 records / 57 subjects，解封前 `outcome_aggregates_computed=false`。
- Combined analysis：`COMPLETED`；27-cell population 为 14,042 / 57，multiple-comparison 状态为
  `exploratory_nominal_single_cell_results_non_confirmatory`。
- Fresh verifier：`PASS`；`panel_cell_count=27`、`independent_recomputation=true`。
- Planning 直接读取 canonical safe aggregate artifact 后，逐 cell accuracy、delta 与 paired CI 均与
  executor terminal packet 一致。

## 科学结论边界

Gate F 是 Gate E outcome 已解封后由用户追加的 V3.1 point-rank top-2 panel，因此所有结果仅标记
`POST_TERMINAL_V3_POINT_RANK_TOP2_OUTCOME_EXPLORATORY`。Qwen 的 8 个 contrasts 全部为负且 CI
均低于零；Llama 的 8 个 contrasts 全部为负，其中 `[11,14), k=3` 的 CI 低于零，其余窗口/k
组合的 CI 跨零或上界极接近零；Gemma 仅 `[10,13), k=2` 有 `+0.0142 pp` 点估计且 CI 明显跨零，
其余为负。没有任何可信正增益，不能宣称 prospective selector success 或 confirmatory 改善。

本次 `mode=block` 下，同一模型/窗口/k 的 `cache=first` 与 `cache=last` 结果完全相同；该现象只作
本次 acquisition 的描述，不外推为一般 cache 机制结论。

## 生命周期偏差裁决

不存在 science/panel repair。错误 `.venv/bin/python` 调用在 Python 启动前失败；随后只替换为已审计
解释器。Analysis 与 fresh verifier 各只启动一次，均因完整 bootstrap 长于普通 SSH/tool yield 而
由 metadata-only heartbeat 等待原 PID，未重复调用。历史异常根与生命周期证据均保留，不影响
Gate F 科学结果，故无需 `PASS_WITH_FIXES`。

## 终态

Gate F executor 的 Git、HPC、GPU、Slurm、analysis、repair 与后续 Gate 权限全部撤销并保留为只读
provenance。Planning 已更新结果表、最终报告与 control；Phase 7 重新终态，不创建 Gate G。
