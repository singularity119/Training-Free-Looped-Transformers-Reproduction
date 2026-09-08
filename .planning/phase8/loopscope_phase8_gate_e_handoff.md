# GATE_E_HANDOFF — 4B K3 Spectral Supplement

Planning：01a08024-3556-75c1-8209-64e89345b940。
唯一executor：01a08031-a85e-7101-b150-60916abdc4ac。
标题：execute-LoopScope-4B-K3-Spectral-Supplement-第8阶段-Gate E。
必须按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，实际读取skill和references/protocol.md；HPC操作读hpc2-hkustgz-ssh。AGENTS是稳定边界，control绑定当前权限；科学采用contract_v1加[Gate E追加](loopscope_phase8_gate_e_amendment.md)，后者仅替换明确K3/4B/4cell部分。

## 准入、基线和目标

用户新增授权；A-D已PASS，无活动旧executor。起点本地专用loopscope-tflt、loopscope clean d9df62a2f6c8a48d8920d8fe20376aabf80cf9df，加本planning纯文档提交。每次变更前核对root/branch/HEAD/dirty和祖先4f59bd93eca4da3cbf458a93508f91c5b23912bc；保护他人改动。目标最短路径：K3最小支持→同版debug→2方向校准→4配置全test→一次统计和追加报告。不能续用D身份或self-PASS。

本地repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt。
HPC alias=hpc2-hkustgz，source=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope，venv=source/.venv-loopscope-cu121-20260711/bin/python。
W=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope。
只读复用C pool：W/inputs/phase8-gate-c-20260905T100000Z-a1/calibration_pool.json；D pool：W/inputs/phase8-gate-d-20260905T142900Z-a1/test_pool.json。原panel不修改，E新manifest只4cell。
D closure：W/artifacts/phase8-gate-d-20260906T060000Z-closure-a1/verification.json；旧analysis：W/artifacts/phase8-gate-d-20260906T131600Z-analysis-a1/analysis.json，可读作已知对照背景。所有A-D原始工件、旧报告/终审只读。

## 实施及权限

允许最小修改/新增src/tflt/loopscope/phase8*、scripts/loopscope/*phase8*、新E专属configs/loopscope/*phase8*、针对性tests；可写本Gate evidence和docs/loopscope_phase8.md的E运行说明。control/amendment/acceptance由planning持有。原contract_v1、旧配置manifest与旧结果语义保持。

已发现正常K3路径会受phase8_runtime collector/load_basis的(2,4)限制，允许最小扩展为支持K3，保持旧K2/K4行为；允许K3专用校准runner/verifier/panel/分析入口。现有accuracy_stats将非K2全部归K4，必须为E显式2项K3探索family，不得直接落入旧4项K4统计。可复用底层统计函数和评分，不做泛化框架。protected wrapper/strategies/cache/config不需改变且仍只读；必要变更报planning。

允许本项目单目的源码/测试/配置/文档commit与普通push到既有GitHub origin loopscope，远端专用clone快进同步；不force/rebase/改main。用户历史push授权延续。模型/数据/basis/完整日志不入Git。只读shared hf_home、datasets及既有venv，普通CPU/import/CLI检查允许；不得静默更换torch/transformers/lm_eval或模型数据revision。严格SSH，不绕过hostkey，不操作凭据。

所有新远端工件使用W/{inputs,runs,artifacts,staging}/phase8-gate-e-<timestamp>-<purpose>-a<N>，保留失败attempt，新路径重试，仅补无效/缺失输出；不覆盖有效分片。只保存512答案位置残差、basis及test四选项分数，不能保存完整token hidden tensor。身份从原pool直接复用，不重新抽样；同一test完整14042，不删长题。

校准不读gold；test采集不算partialacc。新增4配置全闭合后允许一次gold读取和统一分析。历史结果已知如实标注；分析真实bug可确定性修复重算并记录，不outcome驱动重采集/调参。报告写外层资产/报告/phase8/gate_e/，此为明确本地写入例外；独立中文K3追加报告和紧凑表格，与旧主报告分开。不得由executor改Notion旧报告，planning验收后负责追加导航。

## HPC资源与最早实验

先只读确认源版本、已有jobs/monitor和缓存，不接管旧任务。正式优先A800普通用户可用最高优先级（现场候选emergency_gpu/Tier300），保留用户此前A800偏好，实际资格/账户资源提交前查询。debug/smoke必须debugpartition，单job<30min，建议00:29:00。允许1GPU/8CPU/128G；正式每job≤6h，最多同时2GPU，Gate E累计≤24GPUh含debug/retry，超出报planning。使用/opt/slurm/bin工具，root若为普通association名称不代表管理员权限。

新增producer须同不可变commit/launcher/env/runtime的passing端到端debug再正式：4B K3校准微样本拟合/保存/加载basis，原Loop和Spectral评分、干预t1/t2、有限输出/verifier。debug预估小于30min，覆盖实际上尾长度；debug方向只能PREFLIGHT，不得用于正式test。正式校准和正式test两种路径均需对应preflight证据，可一次debug连续覆盖。关键代码/launcher/env改变后重做受影响最小debug。

A800先用已知保守packing2，在1–3独立进程范围按实际峰值/吞吐决定，至少15%瞬时余量。batch1/BF16不改；不共享模型tensor以填显存。小debug不足以估计长期allocator时，先保留正式test前512/配置作为canary，验证后再补余量。记录GPU/packing/峰/吞吐/OOM/GPUhours；只允许executor取消本Gate有明确失败/超预算/OOM风险job，不动无关或未知状态job。连接异常先查旧job，不能盲重提。

## Agility budget与验证

最小实现仅2basis+4cell K3，不扩成新平台。针对性验证K3 collector三个步骤、h1/3和两次干预、2basis映射、4cell完整membership、K3两项Holm family，旧K2/K4小范围回归保证行为不变。实际命令和测试数写evidence，例如PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase8_gate_e*.py'；CLIhelp、bash -n、git diff --check。禁止0tests冒充通过，不要求复跑A-D全套或SVD。

必须证据：2×512同序身份basis、每t512有限native-derivedFP32残差、K3正确加载；4×14042唯一完整有限scores；全闭合早于新增outcome；正确数/翻转/pp等式和两项统计；terminaljobs及资源记录。执行者做工程验证和结果报告，不做额外自审平台。通过关键检查即运行实验。

planning接受时只独立核对1–3决定性事实：K3配方/两basis与原identity；完整4cell和配对统计；运行与报告限制。追加结束只做一次结合原终审的简洁重结案，不重放A-D。

## 修复、监控、交付

同一Gate内低风险工程修复次数不限但遵守24GPUh预算，保留失败attempt，只重试缺失/无效工作；真实科学/信息/外部权限或protected边界报BLOCK。planning退回最多一次集中修复，第二次仅相同材料性根因未解或修复引入回归。负/不确定结果不BLOCK。

允许至多3个并发子agent承担明确当前Gate实现/只读子任务；告知非独占代码、不回退他人编辑，无嵌套/无子agent调度/Gate权限。exactexecutor负责整合、Git、Slurm和终态。

真实formal job返回ID后即创建唯一E executor-owned monitor（含PENDING），probe30min、full60min；debug稳定RUNNING约1min才建smoke10min，短作业无需monitor。转换类型更新同一个monitor，旧D monitor保持PAUSED。健康无变化安静；attempt terminal先暂停monitor，再用send_message_to_thread触发exact executor01a08031-a85e-7101-b150-60916abdc4ac，AUTOMATION_TERMINAL_RESUME含job/root/状态/下一授权动作；executor必须继续验证或有权修复/提交余量，不只说ready便结束。送达失败发AUTOMATION_RELAY_REQUIRED给planning。

完成后暂停monitor，向planning01a08024-3556-75c1-8209-64e89345b940发送唯一GATE_E_FINAL_AUDIT，Result=AUDIT_REQUESTED，含起点/finalcommit/dirty/push、变更/实际测试、jobs/roots、2basis/4cell闭合、新增两项K3结果/限制、资源/失败/报告路径，保留送达成功响应。本地final只作副本；无法送达明确TERMINAL_DELIVERY_UNCONFIRMED。不得self-PASS，不进入Gate F/Phase9。
