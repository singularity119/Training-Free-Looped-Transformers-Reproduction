# Phase10 Gate D：ARC全量核验分析完成，提交规划验收

2026-10-08 最新交付：用户直接授权核验分析。68正式配置／79696条无标签raw scores由原source-0fa5ea7的official verifier全量闭合；all74 attempts accounting核实终态，先写ARC_FULL_PANEL_CLOSED。随后按id/index/question/choice文本与标签顺序精确join1172 frozen test gold，official analysis fresh verification后仅运行一次固定分析。另用独立NumPy预测、SciPy二项McNemar、独立Holm与10000次bootstrap重算全部68配置／185比较，全部相符，CI最大误差2.22e-16pp。机器分析根arc-analysis-attempt1-20261008；human报告外层资产/报告/phase10/gate-d-arc-20261008。14迁移／126扫描／45策略均无显著正或负差异；总成本40.991944GPUh、正式40.805833、peak7/OOM0。原始逐题分数和gold只留HPC，未下载本地。详见本目录analysis_summary与final_audit。Gate D为AUDIT_REQUESTED，未自判PASS、未进入E。


2026-10-08 当前接续：用户在当前聊天明确要求提交剩余9配置。只读现场核实原59个正式job全部COMPLETED/0:0，59个score root均SCORES_COMPLETE、各1172条，共69148记录，target_gold_loaded=false；当前用户队列为空。原先因MaxSubmitJobsPU=50暂缓的9配置均无已有score root，未重复成功配置。本次仅统计原分数完整条数和summary状态，未重跑full-panel verifier或读取outcome。

复核emergency_gpu的PriorityTier/JobFactor300、普通用户关联、同名QoS提交上限50及GPU8/CPU64现场限制。复用已通过debug且未修改的source-0fa5ea7/原runtime/phase10_accuracy.sbatch及冻结manifest/pool；9项CPU DRY_RUN均PASS，每项1172/gold=false。不改源码、环境、科学参数或七迁移lambda；用户已取消GPU小时硬上限，保留用量记录。

新增write-once根：`formal-multijob-deferred9-20261008T025036Z`，完整argv/stdout/stderr和首次状态分别在`submission-0.json`至`submission-8.json`及`initial-confirmation.json`。9次sbatch均exit0，job12942735–12942743；emergency_gpu/QoS同名，每job1A800、8CPU/64G、30min、packing1/batch1。前7项无依赖；12942742 afterany12942735、12942743 afterany12942736，峰值<=7。首次一次性squeue确认9项均PENDING。

配置映射：12942735=q4K2 fixed_t0 lambda0.9；12942736–12942743=q17K2 fixed_t0 lambda0.1–0.8。剩余未提交0，68配置均已有正式job。旧输出与失败receipt原样保留。本地完整接续附件为`loopscope_phase10_gate_d_deferred9_submission_20261008.json`；远端`resume-after-submission.json`在上述新根。下一次明确接续先核实末9job及每cell完整1172条，再结合原59根闭合68/79696；闭合前不读ARC gold/accuracy。未创建或恢复monitor，未发送Gate终态或进入E。

---
以下为此前接续历史，最新状态以上文及新增接续附件为准。

2026-10-07 最新接续：首批12935369–12935377全部COMPLETED/0:0，9cells/10548记录无标签验证闭合，累计实际4.863333GPUh。剩余59准入后已提交50项 **12938926–12938975**，首次全PENDING；另9项因QoS MaxSubmitJobsPU=50保留未提交。7依赖链峰值<=7，当前已消耗+活动预留60.613333GPUh。GPU小时数现仅作资源记录，最新用户授权见文末“取消GPU小时数上限”。

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

## 最大合法卡数核验与当前接续

planning依据用户追加授权更新live control与single_job_amendment，覆盖旧max2GPU，保留100GPUh/初始40GPUh/科学/无监控。父executor现场先读12932810：PENDING/Resources、RunTime0、AllocTRESnull、无正式输出根。以 `scancel --state=PENDING 12932810` 仅取消待运行状态，避免竞态取消运行作业；随后accounting CANCELLED by204599/ElapsedRaw0/无分配，0GPUh。保留完整前态/取消命令/后态在 `<run>/formal-two-gpu-pending-replacement-receipt.json`。没有正式评分成功数据需要重跑。

现场association与root父关联无额外GrpTRES/MaxTRES/MaxWall限制；合法QoS含emergency_gpu、gpu16/gpu32、专用8ka等，QoS Priority全部0。仅看emergency_gpu单用户GPU8上限不足以确认合法job8：job_submit/lua实际拒绝emergency_gpu的8GPU请求，错误明确要求8GPU jobs只在i64m1tga800u8ka/i64m1tga40u8ka。sbatch --test-only无allocation验证：最高优先级emergency_gpu（Tier300）单节点7与4卡通过；专用A8008ka（Tier1，3节点×8A800，MaxTime7天）单节点8、两节点16、三节点24请求通过（24用账户合法gpu32）；emergency_gpu四节点×7=28/gpu32也通过。8节点×7/gputest因现场节点配置不可用被拒绝；8卡/节点在emergency_gpu均被提交规则拒绝。test-only输出中的模拟job编号不是已提交作业，不计GPUh。直接回执 `<run>/max-a800-admission-test-only.json`、`...-attempt2.json`、`...-attempt3.json`。

数量、最高优先级和预算存在需要明确的选择：单节点8卡必须用低优先级专用分区；单节点最高优先级最多7；跨节点可以更多，需分布式launcher与初始9/尾部空闲可行性核实。已向exact planning线程发送现场证据和最小资源决定请求（工具确认送达），未擅自把用户记忆8或单QoS上限当作全局最大，也未扩大100GPUh/削减科学。等待planning对单节点8/单节点7/多节点的明确决定。

仅工程增量：将原runner硬编码1/2 worker改为正整数GPU数，仍严格要求本节点CUDA_VISIBLE_DEVICES枚举恰好实际分配数、独立cell进程、每GPU一个worker、同预算/预测/全部墙钟计费；当前入口仍为单节点，多节点不声称已支持。9项定向tests exit0，新增8worker运行9+59完整清单、每设备无重叠/无重复配置、8卡尾部空闲与100GPUh拒绝、非法卡数检查；bash-n/diff--check exit0。源码提交 `0fa5ea760eae859fe069db7a84d9777a608e55f6`，完整archive新immutable `staging/phase10-gate-d-20261006T161109Z/source-0fa5ea7`，不改科学/评分器/protected runtime/环境/cache。planning修改的control/supplement保持原样，由planning拥有。

新计划 `inputs/phase10-gate-d-20261006T161109Z/multiworker-attempt1/gate_d_bundle_plan.json`，实际priorGPUh0.12444444444444444、固定9/59/七迁移复制、formal卡数/时限留待决定。debug现场合法上限2GPU/cpu16，按补充允许用2GPU同新runner/runtime/producer/accepted validation、固定六indices；CPU已覆盖8正式worker差异。CPU DRY_RUN exit0为 `<run>/bundle-cpu-preflight-attempt3.json`；准入核实回执multiworker-debug-admission.json确认旧正式取消、队列空、debug合法。

唯一新增debug submission exit0/job **12932910**：debug/QoSdebug/A40×2/nodes1/ntasks1/cpus16/mem128G/time29min、PREFLIGHT_ONLY；输出 `<run>/bundle-debug-attempt3`。完整argv/stdout/stderr在bundle-debug-submission-attempt3.json；提交后唯一scontrol状态PENDING/exit0在bundle-debug-attempt3-initial-status.json。当前累计终态0.124444GPUh+唯一待运行debug预留0.966667=1.091111GPUh；活动正式jobs0。原2卡正式预留98已释放，不能继续计为活动预留。

最新接续 `<run>/executor-resume-after-multiworker-debug-submission.json`。下一次明确接续核实12932910终态、9/54/goldfalse闭合、两设备worker同source/runtime与实际预算，再按planning资源决定及新现场资格安排唯一正式allocation。无监控/automation，不继续人工查询新job。target gold/ARC accuracy未读取，Gate D未终态，E锁定。

## 7卡正式作业当前接续

planning转达用户批准“单节点7张A800、最高优先级、一个作业”，并在control/single_job_amendment最终决定中明确采用emergency_gpu单节点7卡，取代最大数量/多节点歧义，不授权8卡低优先级或多节点方案。父executor已实际读回最新决定，无需再次问用户。

该明确消息接续后，12932910 accounting核实COMPLETED/0:0、ElapsedRaw111s、2A40/debuggpu3-9，实际0.06166666666666667GPUh。PREFLIGHT_COMPLETE、9个CELL_VERIFIED，全部producer/verifier exit0；FULL_PHASE10_SCORE_CELLS_CLOSED/PREFLIGHT_ONLY、9/54、固定六validation indices、goldfalse，同source0fa5ea7/runtime/lm_eval0.4.11/max_length32768、两设备0/1成功。真实epoch1791309879来源SLURM_JOB_START_TIME。累计终态D消耗0.18611111111111112GPUh，含失败/全部debug；取消旧正式12932810仍0GPUh。

CPU在immutable同source真实Bundle.stage上额外验证7worker完整first9+remaining59：首7worker并发设备独立、同设备无重叠、68配置无缺失/重复、7卡整allocation预算拒绝超100GPUh。检查脚本保存在新inputs/formal-seven-gpu-attempt1/check_seven_worker_schedule.py，完整运行回执 `<run>/seven-worker-cpu-verification.json` exit0，无model forward/GPU/gold。此前9项定向tests包含8worker检查仍有效；此次无新的源码变更，不重复GPU评分等价检查。按补充记录差异：debug2A40/cpu16/mem128G，正式7A800/cpu56/mem448G，固定同每cell执行源码/runtime/producer。

新现场全partition/association/QoS/queue快照 `<run>/seven-gpu-formal-site-admission.json` 确认emergency_gpu Tier/JobFactor300、普通用户合法、MaxTime14天，对应emergency_gpu QoS Priority0/单用户8A800和64CPU上限、队列空。正式精确submit参数的sbatch --test-only exit0，回执formal-seven-gpu-exact-submit-test-only.json；该test-only模拟编号不属于已提交作业。工程闭合/资源准入回执bundle-debug-attempt3-verified-for-seven-gpu-formal.json。保留9/59划分/七迁移原文；新inputs `inputs/phase10-gate-d-20261006T161109Z/formal-seven-gpu-attempt1`，新输出 `<run>/formal-seven-gpu-attempt1`。CPU完整formal launcher DRY_RUN exit0为formal-seven-gpu-cpu-preflight.json。

唯一新增正式submission exit0/job **12932927**：emergency_gpu/QoSemergency_gpu、nodes1/ntasks1/A800×7/cpus56/mem448G/time14:00:00、job-name p10-d-arc-seven-gpu；FORMAL_TEST、indices`-`、priorGPUh0.18611111111111112。完整argv/stdout/stderr在formal-seven-gpu-submission-attempt1.json；提交后唯一scontrol确认PENDING/exit0在formal-seven-gpu-initial-status.json。预留7×14=98GPUh，已消耗+全部活动预留98.18611111111112<100。首9自动计时/预算准入余59同job调度；首40/总100与loading/空闲/尾部计费保持。活动正式job只有12932927，无重叠/重复评分。

最新远端接续 `<run>/executor-resume-after-seven-gpu-formal-submission.json`，本地资源附件同步。按用户无监控要求提交后不继续查新job、不创建automation。下一次明确接续先核实12932927终态、预算准入/每cell/全68×1172=79696 gold-free full-panel身份闭合及所有终态资源；成功后才ARC_FULL_PANEL_CLOSED/一次固定分析，失败/预算停止保留原attempt和有效结果，只补缺失。当前ARC gold/accuracy未读，D未终态/E锁定。Git只提交本executor的源码/docs/tests与D evidence/resources，planning拥有的control/single_job_amendment最新改动保持原样未代提交。

## 多任务首批当前接续

planning转达用户最新明确要求“还是采取以往的方案，多个任务提交”，实际读回live control/single_job_amendment最新节：每job1A800/总并发最多7、最高合法优先级、100GPUh/首批40GPUh、首9完整实测后准入余59、无monitor/科学标签隔离保持。旧七卡job12932927现场PENDING/Resources、RunTime0、AllocTRESnull、无formal-seven-gpu-attempt1输出根。仅scancel --state=PENDING取消，再accounting确认CANCELLED by204599/ElapsedRaw0/无分配/0GPUh，完整前态/条件取消/后态保存formal-seven-gpu-pending-replacement-receipt.json，确认后才释放原98GPUh预留并提交替代。没有已完成正式cell被重跑。

新现场快照multijob-initial-site-admission.json核实emergency_gpu仍最高合法Tier/JobFactor300、RootOnlyNO、MaxTime14天，association允许对应QoS、QoS单用户GPU8/CPU64；取消后用户队列为空。计划峰值7×1A800/7×8CPU=7GPU/56CPU符合现场更低限制。累计实际终态D成本仍0.18611111111111112GPUh。

直接复用immutable source-0fa5ea7的phase10_accuracy.sbatch/runtime/producer/原verifier，debug12932910在完全相同每cell关键路径上9/54/两GPU闭合。此次没有代码/序列化/科学变化，只将父bundle内的独立子进程改为独立sbatch资源申请，按用户补充不重做GPU科学验证。9个formal producer CPU DRY_RUN全部exit0，确认每cell1172、goldfalse，保存在multijob-initial-cpu-dry-runs.json。新inputs `inputs/phase10-gate-d-20261006T161109Z/multijob-initial-attempt1/gate_d_multijob_plan.json`，保留68配置/首9/余59/七迁移清单原文；新输出 `<run>/formal-multijob-initial-attempt1/cells/<cell_id>/scores`。

时限按此前debug的native6题完整进程约18–20s（含约13s loading）与Online约15–35s、完整1172题/加载验证余量设置，Native2各1h、Online7各4h（原D首批边界），不把七卡14h复制到每job。首批预留2×1+7×4=30GPUh，已消耗+全部活动/待运行上界30.18611111111111<40<100，记录multijob-initial-budget-admission.json。

独立提交9项exit0，每job emergency_gpu/QoSemergency_gpu/nodes1/ntasks1/A800×1/cpus8/mem64G、packing1/batch1、FORMAL_TEST/max_length32768。job映射：12935369 q4-native/1h；12935370 q17-native/1h；12935371 q17K2current_t.2/4h；12935372 q17K2fixed_t0.9/4h；12935373 q4K2current_t.9/4h；12935374 q4K2fixed_t0.6/4h；12935375 q4K3current_t.9/4h；12935376 q4K3fixed_t0.1/4h、afterany12935369；12935377 q4K3lag1.9/4h、afterany12935370。前7自由调度，后两各替代对应已终态Native的并发槽，总峰值<=7。afterany仅控制资源，不以成功/失败或outcome筛选配置；每cell独立保存。完整逐次argv/stdout/stderr在multijob-initial-submission-0至8.json，集合multijob-initial-submissions.json。全部提交后的唯一首态squeue exit0/九项全PENDING在multijob-initial-first-status.json，之后不继续轮询新job。

最新接续 `<run>/executor-resume-after-multijob-initial-submission.json`，本地资源同步。下一次明确接续核实9个既有job终态/ExitCode/实际分配与每cell1172完整score closure，共10548首批记录；按同model/window/K/policy完整cell实测耗时/显存评估余59，只有实际已消耗+1.2×保守预测<=100且未来批次全部活动/待运行时限预留<=100、峰值<=7才继续。仅补缺失、不重跑成功。完整68/79696、alias/身份/配方/有限值/来源与全部job终态资源闭合后才ARC_FULL_PANEL_CLOSED/一次冻结分析。当前ARC gold/accuracy未读，无automation/人工轮询，D未终态，E锁定。planning control/supplement改动仍保持原样未代提交。

## 首批闭合与剩余面板当前接续

用户“现在呢，任务什么情况”显式接续后一次状态/accounting读取：12935369–12935377九项均COMPLETED/0:0，原队列为空，每cell SCORES_COMPLETE/1172条score记录/goldfalse。实际allocation秒分别366、198、2961、619、3203、822、4625、855、3189，首批合计16838s/4.677222222222222GPUh，含debug/失败的D累计4.863333333333333GPUh。两Native约6.1/3.3min；Online约10.3至77.1min。使用原verifier在同immutable source/runtime、完整test pool/manifest执行--full-cell九root检查，exit0，FULL_PHASE10_SCORE_CELLS_CLOSED/9cells/10548records/complete_cells9/goldfalse，严格identity/配方/候选有限值/方向时序检查通过；回执multijob-initial-verification-attempt1/verification-receipt.json与initial-nine-closure.json。未读取target gold/accuracy。

实际首批peak reserved显存约9.0至13.77GB，packing1/batch1原样。余59用既有stdlib forecast：同model/window/K/policy已测完整allocation秒数，Loop用同model/window/K Online最大值；估计39.16138888888889GPUh，已消耗+1.2×预测51.857GPUh<100，满足扩展条件。每job time=1.2×匹配实测+600s加载/尾部余量，向上取整15min，形成30/75/105min三档；全部59预留60.25GPUh，已消耗+全部计划预留65.11333333333333<100。首批4.863333<40。现场最高合法emergency_gpu Tier/JobFactor300、用户关联与GPU8/CPU64限额、原队列空均复核；无科学/源码/环境改动。

新inputs `inputs/phase10-gate-d-20261006T161109Z/multijob-remaining-attempt1/gate_d_multijob_remaining_plan.json`，新输出 `<run>/formal-multijob-remaining-attempt1/cells/<cell>/scores`，source仍0fa5ea7。固定七迁移原文、68/59清单不变。59个producer CPU DRY_RUN exit0、每1172/goldfalse，multijob-remaining-cpu-dry-runs.json；实际首批accounting/显存/预算预测见multijob-remaining-budget-admission.json，现场快照multijob-remaining-site-admission.json。

按最长预测耗时优先、7条依赖链当前估计负载最小者分配lane；每job1A800/cpu8/mem64G/emergency_gpu/QoS同名，每条lane仅首项无依赖、后项afterany前一项，保证峰值<=7、各cell独立结果。前50项submission exit0/job12938926–12938975；逐次完整argv/stdout/stderr在multijob-remaining-submission-0至49.json。第51项（index50，q4K2fixed_t0.9）提交被QOSMaxSubmitJobPerUserLimit拒绝，exit1/无jobID/0新增GPUh；原回执multijob-remaining-submission-50.json保留，停止继续提交，不取消/重复成功50项。此次准入漏查QoS提交数量字段，随后实际sacctmgr核实emergency_gpu MaxSubmitJobsPU/MaxSubmitJobsPerUser=50、MaxJobsPU空；回执multijob-remaining-submit-limit-query-*.json。该限制补入后续接续，不通过更换QoS绕开。

提交成功50项后的唯一首态squeue exit0/50项PENDING在multijob-remaining-first-status-partial.json；之后不人工轮询。集合回执multijob-remaining-submissions-partial-attempt1.json含job-cell/lane/dependency/time映射和9个未提交cell。活动50时限预留55.75GPUh，累计实际+当前预留60.61333333333333<100；未提交9均30min、未来预留4.5GPUh，尚未计作活动预留，不丢任何配置。9项清单：q4K2fixed_t0.9，以及q17K2fixed_t0.1至.8。首批9成功不重跑。

最新远端接续executor-resume-after-multijob-remaining-partial-submission.json；本地资源同步完整50映射、9 deferred、7 lane heads、提交数量上限。下一次明确接续先核实已有50 jobs及实际成本/成功cell、现场QoS上限与运行/待运行数量，在释放的名额内仅提交9个缺失配置，采用fresh attempt receipt、保留原失败回执与lane边界，已终态lane按实际释放状态处理，不盲依赖被清理的旧job。所有阶段累计实际+活动/待运行预留<=100，峰值<=7并受账户更低可用额度约束。全68/79696、18alias/身份/配方/来源/有限值与全部终态资源共同闭合前，gold/accuracy/analysis保持封闭；之后一次固定分析。无monitor/automation，D未终态/E锁定。此为实质进展后的真实等待点，不将临时提交名额限制判为Gate/科学失败。

## 取消GPU小时数上限

2026-10-07 exact executor内用户明确指示“不要设置GPU小时数上限”（原文：不需要设置gpu小时数上限）。此最新人类授权立即取代D历史首批40GPUh、总100GPUh及基于它们的提交/扩展/停止门槛；上文数值比较保留为当时准入证据，不继续施加。实际allocation GPU×walltime、预测与时限预留仍记录，作为accounting而非预算封顶。现有50作业及单作业30/75/105min时限保留；time是Slurm作业运行时限，按实测与余量设置。现场association/QoS/partition/提交数量50/当前并发7等限制仍遵守，冻结68配置、packing1/batch1、全量闭合前gold/outcome隔离、无monitor与E锁定均不变。

新远端write-once资源授权回执executor-resource-policy-no-gpuh-cap-20261007.json；新接续executor-resume-no-gpuh-cap-20261007.json引用此前50映射/9 deferred/7 lane heads，不改写旧receipt，不查询/取消/重提交作业。planning-owned control/supplement未由executor修改；向planning交付最新直接授权，请其同步当前控制面。后续明确接续只按实时Slurm名额补9项，并保持用量记录，不以GPU小时数拒绝已授权工作。
