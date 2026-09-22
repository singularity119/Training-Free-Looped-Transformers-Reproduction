# Gate C：全量评分与冻结统计

按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，实际读取 skill 与 references/protocol.md。HPC操作读取 hpc2-hkustgz-ssh。监控路由服从 loopscope_phase9_monitor_routing_amendment.md：只报 exact executor 自身，不向planning转发监控或fallback。

Planning=01a07ff5-79b1-79f0-809a-4d0971475686；Executor=01a0c811-fbd3-7330-8898-72ce8d13e4ec；model=gpt-5.6-luna / max。
Admission：B PASS见 gate_b_acceptance。实际producer970ef52；新launcher/producer变更需在debug做短preflight。无需补齐B的512诊断。

## 科学与面板

唯一科学规范为 contract_v2，必须实际读取。4B13:16/15:18、1.7B12:15，K2/3/4，Online-t0及Matched-norm各9，共18新配置×14042 test题=252756评分。合并经核验的29历史配置形成47逻辑配置，共659974配置—样本。保留所有冻结K；不得加窗、调lambda/rank、近似SVD、全token干预或按结果筛选。

每题方向来自其自身t0实际保留prefix的残差矩阵，不跨题拟合。精确FP32 reduced SVD/gesvd、lambda0.5、alpha1、h1/K、原dtype求残差、仅后续轮答案前位置处理。batch1、dtype/cache/model/tokenizer/data revision、plain5shot dev first_n5、原lm_eval0.4.11完整continuation评分全部保持。不改变logits_cache来方便调度。第八阶段实现与basis只读。

## 顺序与信息边界

1. 先核对仓库/冻结配置/原始test identity manifest，构建无目标gold的评分输入。允许读取test prompt/choices用于本方法在线SVD及分数采集，不把gold传入producer或中途计算acc。dev示例答案属于合法prompt。只检查历史文件路径/schema/配置；历史correctness/outcome在新评分闭合前不接入分析。
2. 增量补齐全量runner、可恢复的分片输出和验证器。优先复用原HFLM与Phase9 adapter，不重写评分器。CPU定向检查后，使用短debug验证同一正式launcher/environment/runtime/producer/serialization路径，覆盖两模型两arm；B已验证的数值语义不重复全套审计。正式代码更改后对受影响路径重新短preflight，先验证一个child再扩展array。
3. A800代表性资源canary与全量：从固定身份中按prefix长度选常规与上尾样本，只看长度不看gold/acc；正式canary正确结果可并入面板，不能重复算作新样本。每模型实测显存峰值、SVD开销、吞吐后扩展。优先完整独立评分分片，成功分片持续保留，失败仅补无效/缺失身份。
4. 先验证18新配置各14042唯一身份、57subjects、候选分数完整finite、无缺失重复、revision/config一致，写 NEW_SCORE_PANEL_CLOSED 证据。闭合之前不得读取目标gold/correctness或中途统计acc。
5. 达到上述客观闭合条件后，本handoff直接授权接入gold和29历史逐题分数/结果，不需再次等planning。直接核对身份、模型、数据、配方、评分语义和原分数；不能仅复用acc。历史不可核验则报具体缺失给planning，不擅自补跑29配置。
6. 47逻辑面板验证闭合后按contract_v2执行一次冻结分析：K2六项Holm主family，K3/K4各六项次family，Online对两Shared共18探索family；exact双侧McNemar，subject内paired bootstrap10000 seed20260922，nominal95CI，micro/macro、pp、净纠错与两向翻转。不以结果扩展实验或多次改分析。实现错误可有证据地纠正同一冻结分析并保留失败，不数据驱动换分析。

## A800、资源与修复授权

用户要求尽量吃满A800显存。优先申请普通用户可合法使用的A800正式partition/QoS，先核对实际型号/容量及合法CPU:G​​PU约束，使用合法最高优先级；debug/smoke仍debug且<30min。若debug非A800，其结果只验证功能，正式A800可保留canary验证资源形状。A800不可用/排队导致实质阻塞时向planning提供具体候选与时间估计，不擅自降科学配方。

本C累计上限240 GPUh（含preflight、canary、失败重试），同时最多2GPU；单job至多24h并服从partition更短限制。先在8 GPUh以内完成最小工程和资源canary；提交其余全量前记录实测吞吐推算的剩余GPUh，预测含合理余量在剩余预算内即可自行扩展，无需再等planning。超预算报告实测瓶颈，planning可给运营补充。预算是上限，不是消耗目标。

每GPU初始8CPU/128GiB主存，允许按实测packing需求和节点/队列合法约束扩大至24CPU/256GiB以内，不碰其他任务资源。允许同卡1–8个独立进程的有界packing或worker pool，具体上限取实测长prefix安全峰值与吞吐；跨模型/卡型不可无证据外推。数值batch始终1，不共享模型张量、不改dtype。逐步试到吞吐不再改善或瞬态余量不足即止，不为显示100%而增加并发；记录GPU总显存、峰值、并发、题/秒、OOM次数。OOM保留证据、降低packing，只重做无效缺失分片，不更改SVD算法或删长题。允许取消本C明确故障/超预算/有OOM风险的作业，禁动历史和其他任务，禁单因PENDING循环取消重提。

## 路径与工程所有权

本地专用clone为外层下 loopscope-tflt，branch精确loopscope，保护祖先4f59bd93eca4da3cbf458a93508f91c5b23912bc；每次写入/提交/远端运行先检查root、branch、HEAD、dirty、ancestor。不得回退他人改动。允许src/tflt/loopscope/phase9_*、scripts/loopscope/*phase9*、tests/test_phase9_*、configs/loopscope/phase9_*、docs/loopscope_phase9.md，以及自己的gate_c_evidence；禁止改control/contract/handoff。Phase8及通用wrapper/strategy/cache受保护，必要改动先报告具体理由。

SSH strict BatchMode/StrictHostKeyChecking=yes/UpdateHostKeys=no/ClearAllForwardings=yes/ConnectTimeout10，alias hpc2-hkustgz。远端专用clone /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope；新输入/产物仅 workspace /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope 的inputs/runs/staging/artifacts 下 phase9-gate-c-<timestamp>。缓存 /hpc2hdd/home/xhuang225/shared 与既有固定环境，不擅自升级依赖。

允许本地普通commit/push origin loopscope，远端clean安全FF同步或项目内独立源码快照；禁止force/rebase/reset。低风险launcher、序列化、恢复、资源错误在预算内自主修复并保留尝试，不静默吞错。实现/测试以正常路径材料性为准，不搭通用审计平台。

允许最多3个bounded子agent，不嵌套；明确文件所有权、你不独占代码库、不回退别人编辑。子agent不得调度GPU、远端写入或发Gate终态，集成/Git/Slurm由你负责。

## 验证、监控和终态

最低git diff --check、受影响Phase9定向测试、launcher bash -n、CLI/import以及真实debug。全量验证身份和分数完整性，不重复旧Gate完整测试。

同一executor仅一个heartbeat，debug稳定运行无启动错误后10min；全量真实job返回即可60min从PENDING观察，切换更新同一个。健康无变化安静；终态先暂停再向本executor发AUTOMATION_TERMINAL_RESUME并确认送达。无法自唤醒则在本执行任务留TERMINAL_DELIVERY_UNCONFIRMED，不向planning发监控fallback。你负责故障修复和继续；最终你本人向planning确认送达唯一 GATE_C_FINAL_AUDIT 或真正BLOCK。

验收：18新配置完整、29历史核验和47面板闭合、一次冻结统计输出、资源/失败偏差记录。产物不要求正向显著。机器证据在HPC，中文人类表/报告放外层资产/报告/phase9，报告完整版留D。终态暂停monitor，记录分支/HEAD/dirty、路径/jobs、命令测试、NEW_SCORE_PANEL_CLOSED和gold开放时点、完整性与统计、预算/packing、保护状态。不得进入D，不自行PASS。
