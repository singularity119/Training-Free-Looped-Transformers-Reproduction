# LoopScope Phase 8 Control

```text
CONTROL_ID=LOOPSCOPE_PHASE8_CONTROL_V1
STATUS=GATE_C_BLOCKED_ON_VPN_ROUTING
PLANNING_THREAD=01a06fe9-c7bb-7d72-a906-234f301de317
ACTIVE_GATE=C
AUTHORIZED_EXECUTOR=01a070f0-158e-72f2-9d06-5ba47c02e2bd
LAST_GATE_B_EXECUTOR=01a0704b-5798-7680-80a2-f1145f35f9ba
LAST_GATE_A_EXECUTOR=01a07030-6413-7660-b712-c18d9e93d26e
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-Residual-Calibration-第8阶段-Gate C
CURRENT_DECISION=GATE_C_BLOCK_CONNECTIVITY_RESTORATION_REQUIRED
GATE_A_STATE=PASS
GATE_B_STATE=PASS
GATE_C_STATE=BLOCK
GATE_D_STATE=LOCKED
STANDING_PHASE_CONTINUATION=USER_AUTHORIZED_ADVANCE_SERIAL_GATES_UNTIL_PHASE8_TERMINAL
OPERATIONAL_PERMISSION_DECISIONS=PLANNING_GATE_SCOPED_PER_PROJECT_DELEGATION
EXECUTOR_MODEL=INHERIT_USER_CONFIGURATION
BASE_COMMIT=b53ff067992e41b66a8df01794dfb106c61b29d0
AUDITED_COMMIT=c7e4e1f5e0b38874ee4e7eb1ac19dd5c7b6e59ba
AUDITED_GATE_B_FINAL_COMMIT=ef54d2cdafa92483ed65575d0c5788b7980d60f3
LATEST_ACCEPTANCE=.planning/phase8/loopscope_phase8_gate_b_acceptance.md
ACTIVE_HANDOFF=.planning/phase8/loopscope_phase8_gate_c_handoff.md
SCIENTIFIC_CONTRACT=.planning/phase8/loopscope_phase8_contract_v1.md
ACTIVE_SUPPLEMENT=NONE
PRIOR_OPERATIONAL_SUPPLEMENTS=loopscope_phase8_gate_b_push_approval_resume.md;loopscope_phase8_gate_b_platform_permission_resume.md
FROZEN_USER_CHOICES=QWEN3_4B_BASE_12_15_13_16_CACHE_FIRST;QWEN3_1_7B_BASE_12_15_6_9_CACHE_LAST;MMLU_5SHOT;DAMPED_EULER;ALPHA_1_ALL_K_FIXED_HORIZON;VALIDATION512_FIT_TEST14042_EVAL;K2_PRIMARY_K4_SECONDARY;RANK1_LAMBDA0_5
PENDING_USER_CHOICES=NONE
BLOCKER=CAMPUS_DNS_AND_HPC_ROUTED_TO_CLASH_TUN
PRESERVED_GATE_C_COMMIT=51e48d8c33fbadb985783804100bfbbd82f66fd6
PRESERVED_GATE_C_JOBS=12649531,12649532
PRESERVED_GATE_B_COMMIT=e6da2f8790e5ef7612104c41c68a2a7d79138692
NEXT_ADMISSION=GATE_C_PASS_8_FORMAL_BASES_AND_512_IDENTITY_CLOSURE
PHASE_TERMINAL=COMPLETE_FROZEN_PAIRED_PANEL_AND_PLANNING_PHASE_AUDIT_OR_DECLARED_MATERIAL_STOP
```

## 当前授权

Gate B已[PASS](loopscope_phase8_gate_b_acceptance.md)，两个debug jobs/8cell经planning直接验收。当前授权独立Gate C：复用合同v1及B固定512 identity，按8配置采集正式无干预answer residual、拟合8basis及无标签诊断。实现基线ef54d2c（B producer c7e4e1f），允许本轮planning的纯文档提交前置。

具体路径、科学和资源以[Gate C handoff](loopscope_phase8_gate_c_handoff.md)为准。新producer先debug，再按普通用户合法资源正式校准；上限同时2GPU、每GPU1–3独立进程、每job2h、总16GPU-hours，packing须测量。仅同一512验证集，不读test/gold/accuracy，无Gate D权限。用户源码提交授权继续用于当前项目loopscope分支。

B executor已撤权；下文历史B授权/阻塞/恢复记录不再授予B执行权限。每新任务首条消息/handoff强调实际读取skill；C终态由唯一monitor主动唤醒executor，executor继续闭合并发送planning，不能仅宣布恢复后结束。

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
