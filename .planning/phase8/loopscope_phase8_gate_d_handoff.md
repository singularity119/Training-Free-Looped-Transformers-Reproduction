# GATE_D_HANDOFF — Paired Accuracy

2026-09-05。Planning：`01a06fe9-c7bb-7d72-a906-234f301de317`。
唯一 executor：`01a071e9-d133-7742-8826-e0cf5993a1f1`。
任务名：`execute-LoopScope-Paired-Accuracy-第8阶段-Gate D`。

必须按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，实际读取该skill及`references/protocol.md`。HPC操作读取hpc2-hkustgz-ssh skill。AGENTS.md是稳定规则，control是当前授权，contract_v1是科学参数唯一来源。本handoff不授权下一阶段。

## 目标与准入

Gate C已PASS：使用8个正式校准basis，完成双模型MMLU plain5-shot全test14042的18配置，统一解封、配对统计和中文结果报告。不要重拟合方向，不扩充窗口/K/方法arms，不实施逐prompt在线方向，不训练。正、负、不确定均为合法终态。

起点为本地专用clone `loopscope-tflt`、branch `loopscope`、clean HEAD `8c8ef434a4be2025f9603ce7690b90d4abb2f330`加本轮planning纯文档提交；保护祖先`4f59bd93eca4da3cbf458a93508f91c5b23912bc`。实际本地绝对路径见AGENTS.md，不得在外层误认Git根。C producer `6d6214f6ad912b1984b1e0994015270acf627f59`；C acceptance是依赖准入，不重复SVD或全套C审计。

HPC workspace `W=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope`。
C固定pool：`W/inputs/phase8-gate-c-20260905T100000Z-a1/calibration_pool.json`。
C basis根：`W/runs/phase8-gate-c-20260905T100000Z-formal-q4-a1`与`W/runs/phase8-gate-c-20260905T100000Z-formal-q17-a1`；每根cell-0..3/basis.json依次为第一窗口K2、第一窗口K4、第二窗口K2、第二窗口K4。load_basis使用对应FORMAL_CALIBRATION元数据，保留其validation来源；test运行记录另记test来源，不能把basis provenance改成test。
C closure：`W/artifacts/phase8-gate-c-20260905T100000Z-closure-a1/verification.json`。

## 冻结科学与信息边界

完整合同为[contract_v1](loopscope_phase8_contract_v1.md)。4B BF16窗口12:15/13:16 cache first；1.7B FP16窗口12:15/6:9 cache last。窗口12:15=B12→B16。K2主、K4次，两者alpha1、h1/K；block、batch1、decodebypass、原HFLM原序A/B/C/D完整continuation求和。Spectral只在t>=1 pre-answer residual减0.5v(vᵀdelta)，方向固定；原Loop使用None原数值路径。模型/tokenizer/dataset revision及截断/位置语义全部沿用合同。

正式panel是2 native + 16 Loop/Spectral，每配置14042题/57subjects，252756样本评测。native每模型新跑一次共享用于上下文比较，不用旧阶段结果替代。配置membership和8basis路径在提交正式作业前冻结于可读manifest。test identity必须带split=test；不得误用phase8_pool中validation专用identity helper。

允许CPU从指定版本test读取question/choices/subject/doc_index以构建gold-free pool、token长度和canonical顺序，不得在采集时持久化/消费目标answer/gold、正确性或聚合acc。dev示例答案是授权的5shot上下文。renderer可复用已有safe做法，保持标准prompt。target answer仅因加载原始数据对象存在不等于允许读取用于结果；不得把target gold放进prompt或producer。debug同样不计算test accuracy。

所有18配置闭合后，授权Gate D一次解封test gold，运行完整冻结分析；保存解封前全panel验证记录。此后不得调参、换方向、筛样本或选最佳K替代主分析。若分析实现的真实错误需修复，可从已保存分数确定性重算并记录原因，不重新采集或利用结果扩充实验。

统计：四个K2 Spectral−Loop为主family、四个K4为独立次要family。报告micro acc、correct/N、差值pp、wrong→right/right→wrong、subjectmacro；57subject内paired bootstrap10000次seed20260905、nominal95%CI；exact双侧McNemar+Holm分别校正四项。Native差为探索性上下文。所有cell均呈现。没有matched-norm/Random正式对照，不归因为方向特异性优于同范数阻尼；历史开发域，不宣称untouched test。

## 最小实现与允许修改

可新增/修改`src/tflt/loopscope/phase8_*`、`scripts/loopscope/*phase8*`、`configs/loopscope/*phase8*`、对应定向tests、docs/loopscope_phase8.md和本Gate evidence；不修改contract/control/acceptance，这些由planning持有。优先复用B adapter/runtime和C basis loader，实现gold-free testpool、18cell runner/分片、launcher、完整性verifier、统计分析器。只有明确正常路径需要时最小调整phase8模块，不能自行修改wrapper/strategies/cache/config等protected runtime。不要构建泛化调度/审计平台。

用户已授权本项目源码提交与普通push到`git@github.com:singularity119/Training-Free-Looped-Transformers-Reproduction.git`的loopscope分支。允许单目的commit、普通push、远端专用clone快进同步；不rebase/force/main合并，保留他人未提交改动。无模型/数据/完整日志/basis/residual提交Git。

远端SSH仅hpc2-hkustgz、严格host key、不关闭信任校验。source=`/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`；runtime为其中`.venv-loopscope-cu121-20260711/bin/python`。只读指定shared HF/dataset caches和C历史工件；允许远端专用source快进、定向CPU/import/CLI检查。默认复用环境与cache；普通项目所需小依赖修复可记录后执行，不替换torch/transformers/lm_eval或模型/dataset版本，不操作凭据。原复现checkout/workspace只读。

新数据工件仅`W/{inputs,runs,artifacts,staging}/phase8-gate-d-<timestamp>-<purpose>-a<N>`，每attempt独立路径，不覆盖既有输出。允许保存4原始choice scores/identity、必要配置/有限标量计数/时间/显存，不保存完整hidden/residual。定长canonical分片可作为恢复单元；已有效输出保留，重试仅缺失/无效分片，最终合并必须唯一且无缺漏。分片不改变任何样本/评分，CPU重算统计不是额外科学arm。

人类中文结果报告和紧凑结果CSV允许写本地外层`资产/报告/phase8/`，这是明确的本地写入例外；原始逐样本结果留HPC。脚本/runbook/证据索引进Git，人类报告不纳入Git。报告必须解释离线512方向如何用于test、每窗口配对变化、K2/K4固定时长差异、负结果及方法限制。

## HPC资源与debug

所有engineering debug/smoke在debug partition，预计每job<30min，授权上限00:29:00。新producer先双模型同不可变commit/launcher/env/runtime/model/CUDA/实际干预/输出/verifier端到端通过，再提交正式；关键路径改变后重做对应最小debug。上尾长度来自实际test安全文本，不用C校准最大长度代替。至少覆盖每模型native/Loop/Spectral及K4目标位置/序列截断路径；8个basis映射CPU检查即可，不要求每debug全量panel。

总资源上限：最多同时2GPU，每job1GPU、CPU≤16、RAM≤192G、正式walltime≤6h，Gate D累计≤64 GPU-hours（含失败debug），超预算前报planning。普通用户合法最高优先级partition/QoS按现场选择，不限定A40或管理员资源。使用`/opt/slurm/bin`工具；实测root是普通用户account名称，不等于获得系统root权限。固定batch1，不共享模型tensor或改变dtype来填显存。

4B Gate C三进程长期峰46346MiB，仅约4.5%余量，因此D在A40/约48GiB GPU先最多packing2，1.7B最多packing3；其它卡和混合packing按实测可用显存重新估算，在初始上限3独立进程内选择更低值。短debug不能直接推断长程allocator峰；需多种长度和上尾输入的持续运行证据。若debug无法实现正式长程形状，先运行一个小的可保留正式canary（建议每活动cell前512 canonical test身份），并包含独立debug上尾测量；无标签完整性通过后计入正式panel。canary闭合及稳定显存峰确认后才展开其余分片；维持实测至少15%瞬时余量，若达不到则降低packing。不要仅依据空闲显存或显示利用率提升并发。

记录GPU型号/总量/packing/实测峰/吞吐/OOM和累计GPU时间。允许仅对本Gate失败/将OOM的作业取消和新路径重试，保留证据、降低packing；不得取消无关或状态未知的作业。连接失败先只读核实已有job，禁止盲目重提。首个launcher修复先验证单child，再扩大发送。

## 验证与agility budget

最小增量就是pool→18cell保存choice scores→闭合→一次统计。定向测试覆盖test身份/18cell-basis映射、gold隔离/正确位置与原评分保留、分片唯一闭合，以及配对计数/pp、bootstrap和McNemar-Holm的小例子。无需重复B/C全套回归或SVD，不增加内容摘要/manifest hash/假想攻击测试。

本地命令由新增文件确定并在evidence记录，例如`python3 -m unittest discover -s tests -p 'test_phase8_accuracy*.py'`；使用同模式匹配实际文件，不能以0 tests为通过。远端`PYTHONPATH=src`指定venv运行runner/verifier --help、bash -n launcher，实际debug验证serializer/loader及有限scores。通过这些关键检查即推进实验，不添加非必要hardening。

解封前verifier必须检查18/18 membership、每cell canonical14042/57subjects、4有限rawscores、model/revision/window/K/cache/basis匹配、无缺漏/重复及无partial outcome。不要求重复跑模型。解封后fresh CPU分析验证从保存score和gold重新计算acc/配对计数与声明统计，确认pp=(wrong_to_right-right_to_wrong)/14042*100。保留一份集中验证结果，不构造多层receipt。

planning只独立核查1–3项：全panel闭合及解封时序、保存结果与核心配对算术、运行/资源/报告证据。跨A-D一次综合终审在阶段末，不逐Gate重复。PASS取决于完整可信结果，非正收益。

## 恢复、监控与终态

允许当前范围内无限次低风险工程修复循环，但遵守64GPUh总预算、保留失败attempt且修好根因；两次planning退回修复后仍有实质性相同问题需报告no-progress并决策。科学/信息边界、protected源码变更、无法debug准入、超资源或无权限需要BLOCK回planning，不自授权。

只有exact executor提交Slurm、重试/取消和最终整合；可用最多3个并发子agent做当前Gate明确子任务，告知不独占代码、不回退他人改动，不能让子agent提交作业、改科学或发终态，不嵌套委派。

真实job产生后维护唯一executor-owned heartbeat，从PENDING开始可观察；smoke10分钟、formal full60分钟，阶段切换更新同一monitor。健康无变化保持安静。attempt terminal后先暂停monitor并通过send_message_to_thread主动唤醒exact executor `01a071e9-d133-7742-8826-e0cf5993a1f1`，发送AUTOMATION_TERMINAL_RESUME及job/root/状态/后续动作；不能只说ready后结束。executor收到唤醒应完成验证/有授权修复或继续剩余采集，再进入终态包；需要等待真实后续job时再恢复同一monitor。无法送达则向planning发AUTOMATION_RELAY_REQUIRED。planning不例行轮询。

全部闭合后暂停monitor，向planning `01a06fe9-c7bb-7d72-a906-234f301de317`发送唯一`GATE_D_FINAL_AUDIT`，Result=AUDIT_REQUESTED，含起点/final clean commit/push、变更、测试、debug/formal jobs、各root、全18闭合和分析结果、资源/失败、报告路径、限制及未越界确认，保留工具送达成功响应。真正BLOCK用同一路由报告最小阻塞及恢复条件，本地final只是副本。executor不能self-PASS，不进入Phase9或扩展panel。
