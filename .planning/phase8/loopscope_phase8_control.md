# LoopScope Phase 8 Control

```text
CONTROL_ID=LOOPSCOPE_PHASE8_CONTROL_V1
STATUS=ACTIVE_PARALLEL_GATE_E_F_USER_AUTHORIZED
PLANNING_THREAD=01a08024-3556-75c1-8209-64e89345b940
HISTORICAL_PLANNING_THREAD=01a06fe9-c7bb-7d72-a906-234f301de317
ACTIVE_GATE=E,F
AUTHORIZED_EXECUTOR=01a08031-a85e-7101-b150-60916abdc4ac
LAST_GATE_D_EXECUTOR=01a071e9-d133-7742-8826-e0cf5993a1f1
LAST_GATE_C_EXECUTOR=01a070f0-158e-72f2-9d06-5ba47c02e2bd
LAST_GATE_B_EXECUTOR=01a0704b-5798-7680-80a2-f1145f35f9ba
LAST_GATE_A_EXECUTOR=01a07030-6413-7660-b712-c18d9e93d26e
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-4B-K3-Spectral-Supplement-第8阶段-Gate E
CURRENT_DECISION=USER_AUTHORIZED_4B_K3_GATE_E
SCIENTIFIC_RESULT=NO_RELIABLE_SPECTRAL_GAIN_IN_FROZEN_PANEL
GATE_A_STATE=PASS
GATE_B_STATE=PASS
GATE_C_STATE=PASS
GATE_D_STATE=PASS
GATE_E_STATE=AUTHORIZED
STANDING_PHASE_CONTINUATION=USER_AUTHORIZED_ADVANCE_SERIAL_GATES_UNTIL_PHASE8_TERMINAL
OPERATIONAL_PERMISSION_DECISIONS=PLANNING_GATE_SCOPED_PER_PROJECT_DELEGATION
EXECUTOR_MODEL=INHERIT_USER_CONFIGURATION
BASE_COMMIT=b53ff067992e41b66a8df01794dfb106c61b29d0
AUDITED_COMMIT=2dd3dfd59caf547a3abbc0a1516bc4a3207d9af0
AUDITED_GATE_D_FINAL_COMMIT=8bf0748c7d4576da4d1708c72a645dc5574a9148
AUDITED_GATE_C_FINAL_COMMIT=8c8ef434a4be2025f9603ce7690b90d4abb2f330
AUDITED_GATE_B_FINAL_COMMIT=ef54d2cdafa92483ed65575d0c5788b7980d60f3
LATEST_ACCEPTANCE=.planning/phase8/loopscope_phase8_gate_d_and_phase_final_audit.md
ACTIVE_HANDOFF=.planning/phase8/loopscope_phase8_gate_e_handoff.md
SCIENTIFIC_CONTRACT=.planning/phase8/loopscope_phase8_contract_v1.md
ACTIVE_SUPPLEMENT=.planning/phase8/loopscope_phase8_gate_e_amendment.md
PRIOR_OPERATIONAL_SUPPLEMENTS=loopscope_phase8_gate_b_push_approval_resume.md;loopscope_phase8_gate_b_platform_permission_resume.md
FROZEN_USER_CHOICES=QWEN3_4B_BASE_12_15_13_16_CACHE_FIRST;QWEN3_1_7B_BASE_12_15_6_9_CACHE_LAST;MMLU_5SHOT;DAMPED_EULER;ALPHA_1_ALL_K_FIXED_HORIZON;VALIDATION512_FIT_TEST14042_EVAL;K2_PRIMARY_K4_SECONDARY;RANK1_LAMBDA0_5
PENDING_USER_CHOICES=NONE
BLOCKER=NONE
PRESERVED_GATE_C_COMMIT=51e48d8c33fbadb985783804100bfbbd82f66fd6
PRESERVED_GATE_C_JOBS=12649531,12649532
PRESERVED_GATE_B_COMMIT=e6da2f8790e5ef7612104c41c68a2a7d79138692
NEXT_ADMISSION=E_AND_F_ACCEPTANCE_AND_SUPPLEMENT_RECLOSURE
PHASE_TERMINAL=COMPLETE_FROZEN_PAIRED_PANEL_AND_PLANNING_PHASE_AUDIT_OR_DECLARED_MATERIAL_STOP
```

## 当前授权：Gate E追加4B K3

2026-09-08用户明确新增4B K3实验，其他配置不变。按[科学追加](loopscope_phase8_gate_e_amendment.md)和[Gate E handoff](loopscope_phase8_gate_e_handoff.md)授权新独立executor完成2个K3方向拟合、4个完整test配置和探索性配对统计。alpha1/h1/3、两窗/cachefirst/BF16/rank1/lambda0.5保持。正式优先最高合法A800，24GPUh预算、同时≤2GPU。

原A-D终审/18配置结论保留，旧executor不恢复。新结果属于已知旧结果后的探索性追加，不改原family或主结论；该历史终点被下述用户新增F计划取代；E本身不获F权限。下文历史操作权限均已过期。

## Gate E 排队观察补充（2026-09-08）

授权唯一 E executor `01a08031-a85e-7101-b150-60916abdc4ac` 为已提交 debug job `12687173`（run root `W/runs/phase8-gate-e-20260908T085637Z-debug-a1`）从真实 PENDING 状态启动唯一 10 分钟 smoke monitor，RUNNING 后沿用。此条仅替换 handoff 中该 job 必须稳定 RUNNING 约一分钟才启动监控的条件。观察权限、终态暂停及主动触发 exact executor 的约定不变；不改变 partition、QoS、账户、资源、科学、重试或取消权限，不重提原 job。转换到后续 probe/full 时更新同一个 monitor；planning 不例行轮询。

## Gate D 调度更新

用户2026-09-06要求切换A800、最高合法优先级。授权同一D executor按[A800补充](loopscope_phase8_gate_d_a800_supplement.md)核实并迁移未完成工作，取消仍排队的本Gate A40旧jobs12651219/12651220，保留第一512有效canary，更新唯一monitor。科学合同、完整性/解封、总资源上限不变；新A800卡型先可保留小批实测，再完成余量。

## 已结束的 Gate B 运行历史

## Gate B 暂停决定：GitHub 推送审批

收到绑定executor的唯一BLOCK终态包，核对其身份、local clean HEAD726c31c、origin实际地址与ahead4列表，以及 [Gate B证据](loopscope_phase8_gate_b_evidence.md)。规划决定BLOCK，范围仅GitHub推送的外部审批；不判定实现失败，也不判定B通过。HPC连接已恢复；未同步、未运行真实debug，无job/heartbeat。

自动审批两次拒绝普通 `git push origin loopscope`，executor报告的理由是handoff本身不能建立可信用户授权来向GitHub导出源码/历史。目标为 `git@github.com:singularity119/Training-Free-Looped-Transformers-Reproduction.git` 的 `loopscope` 分支。现有实现/规划提交为2e56e05、9210582、d1dfcb8、726c31c，仅项目代码、测试、配置与文档；无模型、数据、残差、basis或完整运行日志。本条规划状态提交同属文档范围。

请求用户明确批准该项目/分支的第八阶段源码、测试、配置、规划/运行文档普通推送。未批准前不重试、不由planning代推、不用其他传输渠道绕过审批。Gate B绑定保留，暂停变更和外部执行；批准后给同一executor发恢复补充，继续原debug任务，C仍锁住。当前尚未做B最终工程验收。

## 推送阻塞已解除

用户最新明确回复“我授权提交源码”，已绑定到上述已说明的GitHub仓库/loopscope分支/第八阶段文本源码与文档范围。[恢复补充](loopscope_phase8_gate_b_push_approval_resume.md)授予同一executor继续普通push及原handoff内HPC debug。此前BLOCK仅保留为历史，不再是当前暂停状态；C仍锁住。

## 最新条件恢复

恢复后push仍被自动审批拒绝，用户授权没有撤销。planning收到平台直接下发的无审批权限更新，executor是否同样更新尚待其实际指令核对。仅在executor自身平台更新成立时恢复原工作，否则保持BLOCK、不重试被拒push。[当前补充](loopscope_phase8_gate_b_platform_permission_resume.md)。

## 平台核对结果

绑定executor已回传：其当前实际developer权限仍为workspace-write/auto_review，未收到无审批更新。条件恢复未满足，Gate B保持BLOCK；本次未push/写入/远程执行。需要用户直接在该executor任务发送已明确目标与范围的推送授权，或该任务实际收到平台权限更新；不能用规划转述代替。

## 直接授权和平台更新已核实

用户在规划任务确认已在B手动授权；planning有界读取B最新回合，看到直接userMessage“我授权所有权限”及executor收到自身平台danger-full-access/never更新的回传。当前补充的恢复条件已满足，Gate B恢复原handoff范围内普通push、HPC快进同步和debug；不扩大科学/资源范围，不进入C。历史BLOCK记录保持。

## Gate B 排队观察补充

为真实job12645346的未知PENDING等待，授权exactexecutor唯一10分钟smoke monitor从排队期启动，运行后沿用。仅观察启动时机例外，不改变账户/partition/QOS/资源/取消/重试权限。详见当前supplement；planning不例行轮询。

## Gate B 已验收关闭

[Gate B验收](loopscope_phase8_gate_b_acceptance.md)为PASS，两个debug作业及8cell证据经planning直接核对。B权限撤销，历史正文中的B运行权限已过期。C admission成立，接下来绑定新的独立C executor。

## Gate C 连接阻塞（不代表实验失败）

收到exact C executor的BLOCK终态。代码及证据已push，两个debug完成，formal jobs12649531/12649532已提交；当前终态/工件完整性尚不明，不能PASS或进入D。唯一C monitor已由executor暂停。

planning现场只读确认VPN utun7/10.21.0.39仍存在，但10.90.63.2/.3及历史HPC地址10.120.18.63路由指向198.18.0.1/utun4；split resolver仍存在。尝试只对校园DNS .3添加临时utun7 host route，命令 `sudo -n /sbin/route -n add -host 10.90.63.3 -interface utun7` 在执行修改前退出1，提示sudo需要密码；没有路由/hosts/Clash配置变更。

请求用户重新连接EasyConnect以恢复VPN路由（如需管理员验证由用户本机完成，不向agent提供密码）。恢复后planning作一次有界连接核对并唤醒同一C executor，检查已有jobs/artifacts再继续verifier/closure或原范围修复；不能重提、取消或替换未知状态作业。冻结科学与D锁定保持。

## Gate C 连接恢复与原作业续接

独立诊断任务报告用户授权重登EasyConnect后路由恢复；planning随后实际执行严格alias的只读hostname探针，exit0返回mgmt-4，确认SSH可达。连接BLOCK解除，恢复同一C executor按既有handoff检查jobs12649531/12649532与原工件。作业是否完成尚由executor查询，不把网络恢复当作实验PASS。

先查原job状态与有效输出：若terminal则直接完成verifier/资源记账/8basis完整性与证据闭合；若仍等待则恢复同一个已暂停probe monitor。不得因为断网而重提/取消/替换作业。真实无效attempt按原Gate C修复授权处理，不扩大资源/科学范围，D仍锁定。完成后主动向planning送恢复后的GATE_C_FINAL_AUDIT及送达确认。

## Gate E规划身份纠正（2026-09-08）

用户直接向E执行任务纠正当前规划身份为`01a08024-3556-75c1-8209-64e89345b940`。本次同步control及E handoff终态recipient；旧规划ID仅保留历史，不再作为E终态或monitor relay目标。executor、debug12687173及既有科学/资源权限不变，monitor由exactexecutor同步。

## Gate F 用户新增与提前准备特例（2026-09-08）
用户明确要求现在创建新线程：双模型各两窗K2/3/4，t0拟合四份方向跨K复用。绑定 `01a08053-81a4-70b1-a306-a721189018a2`，title `execute-LoopScope-T0-Shared-Direction-第8阶段-Gate F`；[完整handoff](loopscope_phase8_gate_f_handoff.md)。
GATE_F_STATE=PREPARATION_AUTHORIZED；当前E执行权保持，F仅独立新文件/只读准备，无提交、共享代码写入、远程写入或模型/GPU权限。用户明确创建请求作为提前绑定特例；不是并发正式实验授权。E PASS后planning主动发送GATE_F_RUN_RELEASE并变更当前executor，F不得自解锁。F准备阶段可发送一次F_PREPARATION_READY，不是terminal。
F冻结36逻辑cell（12组×Loop/t1/t0），20旧cell复用，16新增；4t0共享basis与补齐1.7B K3的2t1 basis。其他原配方保持；详细科学/48GPUh待释放运行包由F handoff唯一持有。终点改为E及F验收完成后的追加综合结案，无Phase9授权。

## Current superseding user authorization: parallel E/F
User explicitly permits concurrent E/F without waiting for E acceptance. This overrides the preceding F preparation restriction. GATE_F_STATE=AUTHORIZED; F_EXECUTOR=01a08053-81a4-70b1-a306-a721189018a2; F_ACTIVE_HANDOFF=loopscope_phase8_gate_f_handoff.md. Existing singular AUTHORIZED_EXECUTOR/ACTIVE_HANDOFF fields retain E binding; this paragraph binds F independently. Full F implementation/Git/HPC authority is effective now, with 48 GPUh and at most 2 GPUs; E retains 24 GPUh and at most 2 GPUs. Coordinate shared-source/Git writes and preserve E immutable runtime. F waits only for valid E comparator artifacts at final paired closure, not for execution admission. No Phase9 authority.

F concurrent runtime isolation: approved fixed committed source snapshot under its own W/staging/phase8-gate-f-* root; E shared source remains at a818dbb. See F handoff source isolation supplement.

## Gate F debug queue observer exception
Planning authorizes exact F executor 01a08053-81a4-70b1-a306-a721189018a2 to start its single 10-minute smoke observer while verified debug jobs 12687439 and 12687440 are PENDING. This replaces only the stable-RUNNING startup condition for these jobs. Preserve jobs/resources/science; no cancellation or resubmission is granted by this exception. Use the same monitor after RUNNING; terminal detection pauses it and actively resumes the exact F executor, with relay to planning 01a08024-3556-75c1-8209-64e89345b940 if delivery fails. Subsequent probe/full waits update the same monitor to 30/60 minutes.

## Gate F full remainder submission order update
The bound F executor reports the user's direct instruction to submit all remaining experiments immediately. Record the updated ordering: canaries 12687717 (q4) and 12687718 (q17), followed by already submitted remainders 12687743 and 12687744 with corresponding afterok dependencies. Preserve disjoint ranges 0:512 and 512:14042; packing stays 1, each job 1 A800 / 8 CPU / 128G, remainder walltime <=6h, concurrent GPUs <=2, total F budget 48 GPUh. No packing expansion from unverified canary measurements. The single full60min observer covers all four jobs; canary success must not trigger duplicate remainder submission. Executor still verifies completeness and numerical validity, repairs only invalid/missing work within authority, and keeps new gold/outcomes sealed until full panel closure. This records operational order, not job success or Gate acceptance.
