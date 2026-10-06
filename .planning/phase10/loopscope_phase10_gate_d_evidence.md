# Phase10 Gate D 执行接续：等待双 GPU bundle debug

2026-10-07 最新接续：新 debug job **12932434**，首次确认 **RUNNING**，A40×2/time29min；正式尚未提交。当前 source `0f7f608a8687ac195f21fb10a68946b72124646e`。恢复步骤以文末“单 allocation 当前接续”为准；原分批 array 方案仅作被修订取代的历史。

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

## 单 allocation 当前接续

收到用户单job要求及 planning 的 `loopscope_phase10_gate_d_single_job_amendment.md` 后继续。治理原文代提交e750a191f3a7f320fd1d7878781d4199286f4de5。新增 runner/launcher、定向测试/runbook 提交并推送0f7f608a8687ac195f21fb10a68946b72124646e；没有修改科学或 protected runtime/Phase8/9。

`run_phase10_arc_bundle.py` 与 `phase10_arc_bundle.sbatch` 在一个 allocation 内每GPU一个worker、每cell独立进程，使用既有 producer/verifier，先9后自动预算准入余59。预算按从真实 Slurm StartTime 起的全部GPU分配墙钟时间（含空闲）计量。余下采用同模型/window/K/policy的首批实测，Loop用同model/window/K Online最大值，最长先调度；首批40GPUh、总100GPUh、剩余walltime及清理余量均受限。预算不足显式BUDGET_STOP，保留成功/失败/未启动配置，不读outcome。

5项针对预算/空闲计费/剩余时限的 tests、无第三方依赖CLI help、launcher bash-n及diff--check均exit0。一个有界worker仅拥有两个新增scripts，父executor审阅整合、tests/docs、Git、远端与Slurm，无嵌套/独立外部动作。

提交新job前核实旧12932349 COMPLETED/0:0、ElapsedRaw234s、1A40=0.065GPUh，closure9cells/54records/goldfalse。`old-debug-verified-before-bundle.json` 保留证据；旧结果不替代新双worker预检，没有取消/重跑。

完整archive源码：`staging/phase10-gate-d-20261006T161109Z/source-0f7f608`。计划：`inputs/phase10-gate-d-20261006T161109Z/bundle-attempt1/gate_d_bundle_plan.json`；七迁移清单原文固定并核对。manifest和accepted pools仍只读引用。远端新launcher DRY_RUN exit0，记录`bundle-cpu-preflight.json`。

唯一新增submission exit0/job12932434：debug/qosdebug/nodes1/ntasks1/A40×2/cpus16/mem128G/time29min，用新bundle launcher、PREFLIGHT_ONLY、priorGPUh0.065、walltime_hours29/60、固定六validation indices。完整绝对argv/stdout/stderr在`bundle-debug-submission-attempt1.json`；唯一提交后scontrol确认RUNNING/exit0在`bundle-debug-initial-status.txt`。预留0.966667GPUh；旧job已终态，没有3GPU重叠。新输出根`bundle-debug-attempt1`，之后未继续人工查询。

下次明确接续：先读job12932434终态accounting、`bundle-debug-attempt1/bundle-summary.json`（必须PREFLIGHT_COMPLETE）、`preflight-closure.json`（9/54/goldfalse）及两个不同GPU worker receipts；核实source/runtime/每cell闭合。失败在原Gate科学/预算内保留attempt后诊断恢复。

新预检通过后再现场重查最高合法A800和全部实际已消耗，优先提交一个2A800、49小时allocation，用同bundle launcher/source/runtime/producer，FORMAL_TEST、test manifest/pool、indices参数`-`。首批9完成后在同job自动预算准入余59，不退出allocation等人工。正式预留98GPUh，旧0.065+新debug上界0.966667+正式98=99.031667GPUh，低于100；实际提交时重新核实。原每cell job、array0-8%2及8h限制已被修订取代。

远端最新接续包`executor-resume-after-bundle-debug-submission.json`，本地资源附件已更新。正式68/79696原分数及job终态/资源共同闭合前，ARC gold/accuracy/analysis仍禁止。无automation、无Gate终态包，E仍锁定。
