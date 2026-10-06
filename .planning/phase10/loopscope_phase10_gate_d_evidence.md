# Phase10 Gate D 执行接续：等待 debug

2026-10-07，executor `01a111f8-7ba3-7862-b0d6-78c0a5a2f767`。这不是 Gate 终态或验收；D 保持授权，E 锁定。用户暂不定时监控，不创建/恢复 automation，也不人工轮询。下一次显式接续先核实已有作业，不能重复提交。

## 当前证据

起点 `ecaa1367eace0a1abb3268095057f4b22b616f36`、branch `loopscope`、祖先检查 exit0。初始 dirty 仅 planning 的 C acceptance/control/D handoff；原样代提交并正常 push 为 `2a0351ca115daf6296f6d9bceafe888a27c86b2a`。没有修改评分实现、protected runtime 或 Phase8/9。

统一 HPC workspace：`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope`。

- source：`staging/phase10-gate-d-20261006T161109Z/source-2a0351c`，完整 `git archive HEAD` 部署，之后没有原地修源码。
- inputs：`inputs/phase10-gate-d-20261006T161109Z/prepared-attempt2`。
- run：`runs/phase10-gate-d-20261006T161109Z`。
- runtime：`/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python`。
- accepted pools：`inputs/phase10-gate-a-20261002T062100Z/prepared-attempt1/ARC-test-pool.json` 与 `ARC-validation-pool.json`，直接只读引用，没有重建、下载或改输入。

`gate_d_plan.json`、`arc-manifest.json`、`arc-preflight-manifest.json`、`arc-canonical-identities.json` 已生成：68 独立配置、86 展示行、18 别名、1172 test identities、79696 正式记录。C 的 `mmlu-analysis-attempt1/transfer_lambdas.json` 原文复制到 D inputs，并按 `strategy_id/strength` 核对七参数全部一致。Native2 + 七迁移 Online 为首批9，剩余59。

debug 固定 validation indices `[0,32,35,85,89,210]`，九配置预期54条。使用普通 `phase10_accuracy.sbatch`、同 immutable source/runtime/producer/verifier、max_length32768，逐配置独立进程，batch1/packing1。此次按完整 cell 计划，不采用分片；不重复 B 已接受的评分等价/方向检查。估计15分钟，依据 B 代表性6题作业单个最长65秒及加载/验证余量；29分钟硬上限。

CPU 准备与两个生成 launcher 的 `bash -n`、一个 producer `--dry-run` 均 exit0；无代码改动，不重复全套测试。`CPU_PREPARATION_COMPLETE.json` 记录这些结果。两个只读子任务分别检查 ARC producer/launcher 和 verifier/analysis，父 executor 整合并执行全部 Git/remote/Slurm 动作；无嵌套或独立调度。

CPU 两项准备错误已保留：第一次误用不存在的 `lambda_star` 字段而停止，canonical 字段是 `strength`；第二次 `scontrol show partition` 逗号列表调用被拒绝。前者在新的 `prepared-attempt2` 重做；后者保留已成功的输入/launcher，只重新读取全 partition 到 `partitions-attempt2.txt`。均发生在 GPU 提交前，消耗0GPUh，没有改科学、丢题或覆写失败证据。

现场 SSH strict/BatchMode/ConnectTimeout10/ClearAllForwardings/UpdateHostKeys=no 成功，login mgmt-6；提交前队列无 D 作业且无既有 D run。snapshot `partitions-attempt2.txt`、`association-attempt2.txt`、`qos-attempt2.txt` 确认 debug/A40、29min 可准入；emergency_gpu/合法同名QoS、A800、PriorityTier/JobFactor300、RootOnlyNO 为当前最高合法正式选项。未修改 SSH/hosts/VPN/Clash、旧远端 checkout、venv 或 shared caches。

## 已提交的唯一作业

```bash
/opt/slurm/bin/sbatch --parsable --partition=debug --qos=debug \
  --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=00:29:00 \
  --job-name=p10-d-arc-debug \
  --output=<run>/logs/debug-%j.out --error=<run>/logs/debug-%j.err \
  <run>/debug-preflight.sbatch
```

submission exit0，job `12932349`。提交后唯一确认 `/opt/slurm/bin/scontrol show job 12932349` exit0，状态 `PENDING`。完整命令/stdout/stderr 在 `debug-submission-attempt1.json`；状态在 `debug-initial-status.txt`。预留上界29/60=`0.48333333333333334 GPUh`，实际分配待下一次接续核实。没有正式 job、重试/取消/重投、target gold/correctness/accuracy/ARC outcome 或分析。

## 显式接续后的动作

远端接续入口 `executor-resume-after-debug-submission.json`，本地资源附件 `loopscope_phase10_gate_d_resources.json`。先只读 accounting/日志/`debug-attempt1/closure.json`；核实 job 终态、ExitCode、9cells/54records、gold=false、同 source/runtime 和资源。失败是 attempt 事件，在现有科学/预算内保留证据后诊断和低风险恢复。

debug 完整通过后再现场核对最高合法 A800，使用已准备 `formal-initial.sbatch`、array `0-8%2`、每job1A800/packing1/batch1/time4h 跑首批9个完整1172题 cell。这一批总预留36GPUh，连同 debug 上界36.483333GPUh，低于初始40GPUh；提交前须核实实际已消耗和全部运行/待运行上界。不能仅因提交前CPU通过而启动正式任务。

首批完整原分数闭合后测两模型/策略耗时与显存，只有已消耗+1.2×剩余保守预测<=100GPUh才继续其余59。每次批次已消耗+所有运行/待运行GPU×time上界<=100GPUh；全程max2GPU。全68×1172/79696共同闭合、jobs终态和预算核实后才写ARC_FULL_PANEL_CLOSED并读取固定ARC test gold做一次分析。仍只执行D。

这一等待点未发送 GATE_D_FINAL_AUDIT/BLOCK，也没有创建监控或宣称无人值守推进。control/contract/handoff保持只读。
