# Phase11 D7 跨设备接续 — 2026-10-18

## 当前状态

D4、D5、D6已PASS；D7仍ACTIVE，尚未最终验收。Native/current_t K2 λ0.5/fixed_t0 K3 λ0.9各12032题，共36096条已独立raw闭合，first array13115919_0..7全部COMPLETED0:0。正确数分别8342/8273/8290，准确率69.3318%/68.7583%/68.8996%。这些分数来自用户授权的提前评分，不是五组最终结果。

最后普通Loop K2/K3共24064条尚无成功提交、无正式输出。首次提交于2026-10-17T22:45:41.939750Z被AssocGrpSubmitJobsLimit拒绝，无jobID。日志显示root于北京时间2026-10-16 17:38:47将association2544的grp_submit_jobs设0，未记录业务原因，不能认定欠费。后来本机VPN退出，SSH不可达；当前GrpSubmit是否恢复UNKNOWN，0只是最后成功观察值。不是已证实的主机密钥异常。没有修改账户/权限/SSH信任。

## 先读取与绑定

正常clone/pull origin的loopscope分支，操作步骤见[new_device_handoff.md](new_device_handoff.md)。本地根以git rev-parse --show-toplevel为准，外层.planning兼容链接按该文档建立，不复制两份control。读取AGENTS.md，再读以下当前文件，旧阶段“当前”措辞不能覆盖它们：

- `.planning/phase11/loopscope_phase11_control.md`
- `.planning/phase11/loopscope_phase11_gate_d7_handoff.md`（包括用户例外与KV补充）
- `gate_d7_execution_binding.json`、`gate_d7_analysis_binding.json`
- `gate_d7_first_closure_receipt.json`、`gate_d7_last_admission_state.json`
- `gate_d7_last_submission_execution.json`、`gate_d7_ssh_vpn_path_diagnosis.json`
- `gate_d7_interim_evidence.json`、`gate_d7_k3_interim_aggregate.json`、`gate_d7_monitor_lifecycle.json`

planning为01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local；D7唯一executor为01a142a7-562a-7e83-b085-70999fbbc780/local。迁移Git代码不自动迁移Codex聊天/工具host或automation。新设备先核实实际线程与host；若换executor/host，须由planning明确重绑定control/handoff并关闭旧调度权，不能让新聊天凭本说明自行抢占。唯一automation phase11-d7当前PAUSED；不得在新旧设备同时建立监控或提交。

## 恢复剩余Loop

1. 连接正常校园VPN，按hpc2-hkustgz-ssh技能严格验证SSH；不得自动接受变更host key，必要时走恢复技能。
2. 只读核实xhuang225原association正常提交资格及exact已有jobs/无重复。若仍GrpSubmit=0，联系管理员恢复正常权限，不能换账户或改权限绕过。
3. 保留成功first与失败submission receipt，读取已存在last-plan。不要直接重跑要求新plan目录的旧submit_last_remote脚本；仅为原保留计划创建新的write-once submission attempt。
4. 原科学、source、packing不变：max8 A800、正常用户合法最高优先级emergency_gpu/QoS（现场核对）、8CPU/64G每卡、合法最长14天、packing4独立进程、batch1。first实测峰值45.634GiB，最小headroom42.9565%。不重跑36096成功记录。
5. 获取真实新jobID后才恢复唯一full60min监控；terminal先pause再self-wake exact executor并确认投递。完成全60160 raw/S2后统一固定统计，再唯一GATE_D7_FINAL_AUDIT投递planning。不得把first闭合当D7 PASS。

## 冻结科学与远端位置

MMLU-Pro完整12032、同类别5-shot CoT，模型Qwen3-4B-Instruct-2507，8192/greedy/nativeEOS，runtime seed20261016。窗口inclusive15:18，block Euler α1 β0、步长1/K，cachefirst、decodefull。两方法prefill＋decode衰减；KV沿用K轮临时回退、原始入窗表示额外stash一次、每token持久追加一次，方向decode冻结。S2-MMLU按实际选项范围解析；不调整prompt/parser/方法。

HPC workspace根：`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope`。
本次tag：`phase11-gate-d7-20261016T031135Z`。
- immutable producer：`staging/<tag>/source`，commit3866f6aa0d775f0b6e721f90a88fff0af5455abc。
- CPU gold analysis：analysis-source-4b28daa，exact路径见analysis_binding。
- 剩余计划：`staging/<tag>/plans/last/last-plan.json`。
- first闭合：`artifacts/<tag>/first-closure-20261017T224332Z`。
- 两组快照：`artifacts/<tag>/interim-20261017T173526Z`。
- K3快照：`artifacts/<tag>/interim-k3-20261017T222743Z`。

首次gold实际读取2026-10-17T17:35:52.986717Z，K3复用gold。最终不得声称首次gold；observer的target_gold_loaded=false仅指observer本身未读，不代表Gate全程未读。

## 同步边界

Git包含代码、测试、runbook、规划合同/验收与聚合机器证据。原题、生成、逐题预测、gold、模型/cache/venv和凭据保留原位置，不入Git。中文报告按项目约定外置于repo父目录资产/报告/phase11/，Git clone不会自动带走它们；本说明和机器聚合证据足够接续实验，Notion报告另行保存。此同步不启动新实验或迁移监控。
