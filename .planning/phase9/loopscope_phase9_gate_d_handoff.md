# Gate D：第九阶段结果报告与交付

按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，必须实际读取skill和references/protocol.md。使用HPC只读能力时实际读取hpc2-hkustgz-ssh。
Planning=01a07ff5-79b1-79f0-809a-4d0971475686；Executor=01a0ca78-3cd8-7c92-9d1a-fbc29e1795d7；gpt-5.6-luna/max。
先读AGENTS.md、PROJECT_MEMORY.md、.planning/README.md、phase9 control、contract_v2、C acceptance、monitor_routing_amendment。B和C已PASS，当前只允许D。

## 任务

用C既有冻结结果制作中文第九阶段报告、便于导师阅读的摘要和完整结果表。主报告放外层 资产/报告/phase9/LoopScope_第九阶段实验结果报告.md；完整47cell和81contrast可为同目录CSV（直接提取既有分析，不重新统计）。更新同目录LoopScope_第九阶段总体目标与Gate计划.md顶部当前结论并明确保留历史计划，避免旧计划被误读为当前状态。报告需有可核验原始产物路径。无需为了形式重画图或新建通用报告工具；表格足够可不作图。

来源：HPC /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase9-gate-c-20260922T161600Z-exec，closure两JSON与analysis-contract-v2-a1/analysis.json、pre_outcome_verification.json。从该既有分析读所有数字，不重复访问gold或再次运行统计。允许把现有分析及闭合证据拷贝到本地外层资产供报告和审计使用，但不把完整raw score/model/data/log提交Git。

报告至少包括：
- 第八阶段跨512题答案前残差拟合共享方向；第九阶段每题首次窗口各有效prefix token残差拟合自己的方向，随后仅衰减答案前位置。说清残差不是hidden state，prefix包含demo和当前题/选项/Answer:，排除padding/special/continuation，t0不干预，lambda0.5、h1/K。Phase8实现保留。
- 两模型、三窗口、全部K2/3/4、batch1及原配方，18新配置+29历史复用=47、每配置14042；排除窗口来自用户在Phase9 outcome前依据历史选择，验证有条件效果潜力，不叫严格上界。
- 9行主展示表：模型/window/K，Native/Loop/Shared-t0/Shared-t1/Online/Matched acc%，Online−Loop和Online−Matched pp及净纠错；清晰区分K2主/K3、4次。CI/Holm另表或附表完整列出，不把acc当pp。关键正向点估计先述，随后解释限制。
- 唯一校正显著次要结果4B13:16 K3：Online−Loop +41题/+0.291981pp、95CI[+0.099701,+0.477140]、Holm p0.015258；Online−Matched +25题/+0.178037pp、Holm p0.410956，未通过。K2全部未通过、K4及Shared探索全部未通过；没有cell同时通过两比较，所以未确证方向处理优于范数缩放。非显著不是等效/无效，K>2自身轨迹matched不是跨臂逐轮严格等范数；不隐藏负结果。说明相对Native表现，不能只用相对Loop提升暗示绝对提升。
- B诊断仅四组完整512、九debug通过，其余因debug时限停止且用户允许不补，不把部分几何当全panel机制证明。
- 资源：A800、batch1、实际packing、总23.936667GPUh，4B packing2 OOM后packing1成功；区分进程reserved与整卡峰值。保留三GPU重叠超过授权2GPU、远端分支偏离及source标签需映射真实a2c9e58的偏差。若C随后补充现成证据可纳入，不编造OOM原因，不以总预算合规掩盖并发越界，不宣称显存优化充分。
- 给导师一段可直接说的insight和结论边界；后续设想仅明确未授权/未实验，不开启新实验。

## 权限与交付

仅CPU文档整理/既有结果读写与HPC strict只读，无GPU、Slurm提交/取消、远端Git切分支/同步/写入、实验、重算分析权限，无需任何monitor。本地专用clone loopscope分支，保护祖先4f59bd93eca4da3cbf458a93508f91c5b23912bc；每次写入/提交先核root/branch/HEAD/dirty。你不独占代码库，不回退他人修改。

可更新docs/loopscope_phase9.md、PROJECT_MEMORY.md中已过时的Phase9状态（注明A/B/C已PASS、D待planning验收），自己的 .planning/phase9/loopscope_phase9_gate_d_evidence.md；不得改control/contract/handoff/acceptance或Phase8。允许普通本地commit及push origin loopscope，将现有未推送的Phase9实现按正常FF提交链备份；先inspect远端，非FF只报告不擅自merge/rebase/force。不要为了报告修改producer代码。可最多2个bounded只读/报告子agent，明确所有权，无scheduler/终态权限。

验收以数字与原分析一致、9行展示及完整附表、科学解释边界与偏差准确、路径可读为准。对生成报告直接读回核查头尾/表格/关键值，不跑无关代码测试。完成后主动向planning发送唯一 GATE_D_FINAL_AUDIT并确认送达，附报告绝对路径、Git状态/commit/push结果、原始证据与核查、未解决项。你不能宣布Phase9最终完成，planning负责阶段末综合审计。禁止新Phase10或后续Gate。
