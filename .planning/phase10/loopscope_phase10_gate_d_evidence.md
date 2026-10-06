# Phase10 Gate D 执行接续：等待 ARC 正式单作业

2026-10-07 最新接续：12932763 已完成并通过双 GPU 预检核实；正式单作业 **12932810** 已提交，首次确认 **PENDING**，emergency_gpu/QoS emergency_gpu、A800×2/time49h，全部68配置。当前 source `ee84dd1226ed036479496862bf624df30639fc40`。恢复步骤以文末“正式单作业当前接续”为准；此前等待点仅作历史。

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

## 启动器修复与当前接续

用户明确报告“12932434 退出”后继续。只读 accounting 确认12932434 FAILED/1:0、ElapsedRaw0s、2A40/debuggpu3-9，Start=End=2026-10-07T00:29:38；按 Slurm 记录秒粒度计费0GPUh。stderr显示启动器获取 allocation StartTime 时，计算节点 `/opt/slurm/bin/scontrol show job -o 12932434` 返回127。失败发生在 bundle runner/model scoring 之前，无 bundle-summary/score root；旧日志与输出根保持原样，失败事件保存为 `<run>/bundle-debug-attempt1-failure.json`。12932349仍COMPLETED/0:0、234s/1A40/0.065GPUh，确认累计终态成本0.065GPUh。

低风险工程修复仅限 bundle launcher 与定向测试：用 Slurm 直接提供的 `SLURM_JOB_START_TIME` 作为真实 allocation 开始 epoch，保留全部GPU分配墙钟计费；缺失或非法值明确失败，不用晚于开始的当前时间替代。已核对官方 sbatch 文档及集群23.02.3对应官方 env.c 实现。6项定向测试通过，包括在无 Slurm CLI 的 PATH 下执行真实 launcher 并核对准确 epoch、缺失 epoch 在 runner 前失败；bash-n、diff--check通过。修复提交并推送 `ee84dd1226ed036479496862bf624df30639fc40`，完整 Git archive 部署到新的 `staging/phase10-gate-d-20261006T161109Z/source-ee84dd1`；不原地修改旧source，不改科学/protected runtime/评分器/环境/缓存。

新计划 `<inputs>/bundle-attempt2/gate_d_bundle_plan.json`，保留68/9/59配置及固定七迁移参数，priorGPUh0.065。新的CPU DRY_RUN exit0在 `<run>/bundle-cpu-preflight-attempt2.json`。debug沿用 accepted validation pool、旧预检manifest、固定六indices；新输出根 `<run>/bundle-debug-attempt2`。

唯一新增submission exit0/job **12932763**：debug/qosdebug/nodes1/ntasks1/A40×2/cpus16/mem128G/time29min；使用修复source/同runtime/bundle launcher、PREFLIGHT_ONLY、priorGPUh0.065。完整绝对argv/stdout/stderr保存在 `<run>/bundle-debug-submission-attempt2.json`；提交后唯一状态确认 **PENDING**，保存在 `<run>/bundle-debug-attempt2-initial-status.txt`。未重复查询新job，无定时监控或automation。旧两个job均终态，无GPU重叠。当前debug预留上界0.966667GPUh；正式49h×2GPU预留98GPUh，总已消耗与预留上界99.031667GPUh<100。

最新远端接续包 `<run>/executor-resume-after-bundle-debug-attempt2.json`，本地资源附件同步。下一次显式接续先检查12932763终态accounting、日志中的 allocation epoch来源、`bundle-debug-attempt2/bundle-summary.json` 的 PREFLIGHT_COMPLETE、`preflight-closure.json` 的9cells/54records/goldfalse，以及两个不同GPU worker receipts/同source/runtime。失败则保留attempt并在原科学/预算内诊断恢复；通过后才重新核实最高合法A800、实际总成本与49h资格并提交单个正式2A800 allocation，首9后自动预算准入余59。正式提交数0，ARC gold/accuracy/analysis未读取；68/79696及终态资源共同闭合前保持outcome barrier。此为真实等待点，无Gate终态包，E仍锁定。

## 正式单作业当前接续

用户报告“12932763完成”后显式接续，并要求权限范围内尽量多用A800；当前D max2GPU/100GPUh边界保持，正式使用上限2A800。12932763 accounting已核实COMPLETED/0:0、ElapsedRaw107s、2A40/debuggpu3-9，实际0.059444444444444446GPUh；累计三个终态debug jobs为0.12444444444444444GPUh。bundle-summary为PREFLIGHT_COMPLETE、9个配置CELL_VERIFIED，producer/verifier全部exit0；preflight-closure为FULL_PHASE10_SCORE_CELLS_CLOSED/PREFLIGHT_ONLY、9cells/54records、每cell固定六validation indices、goldfalse，同source/runtime/max_length32768。两个worker visibleGPU分别0/1均成功，stderr真实epoch1791308271来源SLURM_JOB_START_TIME。工程核实回执 `<run>/bundle-debug-attempt2-verified-for-formal.json`；不把debug评分解读为科学结论。

现场全partition/association/QoS/queue/A800 nodes快照 `<run>/formal-admission-site-snapshot-attempt2.json`：emergency_gpu PriorityTier/JobFactor300最高合法、RootOnlyNO、AllowAccounts/AllowQos ALL、StateUP、MaxTime14天；用户association允许emergency_gpu QoS，全部现场QoS Priority0，对应QoS单用户gpu:a800/gpu上限8、cpu64，无更短MaxWall。申请2GPU/cpu16/128G/time49h合法，用户队列提交前为空。CPU现场查询曾使用本Slurm不支持的MaxTRESPJ字段；改用支持的MaxTRES重新取得完整快照，保存formal-admission-cpu-query-failure.json。CPU核对producer argv时一次误用输出路径index8来核对commit，改为正确index9并与closure交叉核对，保存formal-admission-cpu-argv-check-failure.json。两次均发生在正式提交前，0GPUh，未改producer/源码/科学。

正式新inputs `inputs/phase10-gate-d-20261006T161109Z/formal-bundle-attempt1`，plan将priorGPUh更新为实际累计0.12444444444444444，保留9/59配置划分和七迁移参数原文复制；accepted ARC test pool与原formal manifest只读引用。新输出根 `<run>/formal-bundle-attempt1`，同immutable source-ee84dd1/runtime/launcher/producer。CPU formal launcher DRY_RUN exit0记录 `<run>/formal-bundle-cpu-preflight-attempt1.json`。无代码变更，不重复已通过定向tests。

唯一新增正式submission exit0/job **12932810**：partition/QoS emergency_gpu、nodes1/ntasks1/A800×2/cpus16/mem128G/time2-01:00:00、job-name p10-d-arc-full-bundle；FORMAL_TEST、indices参数`-`。完整argv/stdout/stderr为 `<run>/formal-bundle-submission-attempt1.json`；提交后唯一scontrol确认PENDING/exit0为 `<run>/formal-bundle-initial-status-attempt1.json`。正式预留98GPUh，累计已消耗+全部预留上界98.12444444444445GPUh<100。一个allocation内每GPU一个顺序worker，先七迁移Online+两Native，首批40GPUh内闭合后按实际完整cell耗时、1.2余量/清理时间/剩余walltime自动准入余59；不另起array或每配置排队job，不等待人工批准余59。

最新远端接续包 `<run>/executor-resume-after-formal-bundle-submission.json`，本地资源附件同步。按用户无监控要求，不继续查询新job、不创建automation/人工轮询。下一次明确接续先核实12932810已有终态/资源与bundle状态、预算准入和每cell结果；预算停止/失败保留有效结果与原attempt，必要时只补缺失，不重复成功配置。全68/79696、identity/有限值/科学来源/full-panel closure及所有jobs终态预算共同闭合后，才写ARC_FULL_PANEL_CLOSED并按handoff做一次固定分析；当前target gold/ARC accuracy未读，Gate D未终态，E锁定。
