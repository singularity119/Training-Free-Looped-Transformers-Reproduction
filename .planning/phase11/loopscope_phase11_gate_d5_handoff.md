# Gate D-5 — Phase10/11 4B 最优配置固定输入两次新运行种子重复测评

按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行。实际读取 skill 与 references/protocol.md。

PLANNING_THREAD=01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f
PLANNING_HOST=local
EXECUTOR_THREAD=01a1385a-fdfa-7d43-a6b8-1405a585b8b9
EXECUTOR_HOST=local
GATE=D-5
MODEL=gpt-6.1-sol
REASONING=high

用户2026-10-14明确授权新独立执行线程，对已列出的最优配置及Native/同K普通Loop新跑两次；并明确选择固定题集、顺序、提示、选项排列、种子与评分，检验运行复现性。本Gate属于Phase11追加复现Gate，包含Phase10已冻结Base配方的限定重跑；不重新开放Phase10旧executor，不重启D4或MMLU-Pro。D4已PASS、关闭。

## 2026-10-14用户最终种子修订（最高优先级，覆盖旧种子条款）

用户已明确确认：repeat1全部14配置统一运行seed=20261014，repeat2全部14配置统一运行seed=20261015；输入保持原样。只改变producer运行随机种子，Python/NumPy/torch/CUDA中实际使用的运行RNG在模型/计算开始前显式设为对应repeat seed，并记录有效值；不得让历史producer硬编码seed随后覆盖。每个配置/新进程按本轮seed初始化，不引入cell-specific或sample-specific偏移。

canonical题集、题目/候选顺序、原few-shot演示身份与顺序、prompt/chat template、GPQA原选项排列及其输入构建seed20261008均保持；不要用新运行seed重新构建/抽样输入。各family的bootstrap分析seed仍按原handoff固定，不改为运行seed。

允许针对Phase8/8G/10/11实际producer新增显式运行seed参数或D5限定入口，以小幅opt-in改动替换运行seed硬编码；旧默认值与历史source/manifest/结果不修改，其他科学路径不变，protected runtime仍只读。需验证运行seed实际生效且输入身份/演示/选项未变，修改后的关键producer/launcher按原规则完成最小debug后正式提交。保持fresh两轮28cells/102796条、全部对照、8A800及原资源权限，恢复此前暂缓的新增GPU/正式提交权限。

本次准确表述为“固定输入下两次新运行种子的重复测评”，不再称所有seed均与reference0完全相同；确定性评分/greedy下不同运行seed仍可能得到相同输出，不等于独立题集验证。若已有旧seed D5正式输出，不计入这两轮且不覆盖；先回报exact jobs/状态，保留证据，必要处置由planning另定。旧的逐cell seed澄清仅为reference0 provenance历史。

## 冻结面板（每轮；repeat1、repeat2都fresh）

| family | 模型 | 题数 | 独立配置 | 每轮记录 |
|---|---|---:|---|---:|
| p10-base-mmlu | Qwen/Qwen3-4B-Base | 14042 | Native；Loop K2；current_t K2 lambda0.9 | 42126 |
| p10-base-arc | Qwen/Qwen3-4B-Base | 1172 | Native；Loop K2；fixed_t0 K2 lambda0.4 | 3516 |
| p11-instruct-arc | Qwen/Qwen3-4B-Instruct-2507 | 1172 | Native；Loop K2；fixed_t0 K2 lambda0.9 | 3516 |
| p11-instruct-gpqa | Qwen/Qwen3-4B-Instruct-2507 | 448 | Native；Loop K2；Loop K3；current_t K2 lambda0.5；fixed_t0 K3 lambda0.9 | 2240 |

每轮14cells、51398条；两轮28cells、102796条。K2 fixed_t0=lag1只跑一次、别名可展示。Base ARC普通Loop与衰减并列最优，其Loop也只运行一次/轮。GPQA Native每轮仅一份，与两种K共享；严禁跨轮复用旧结果或缓存预测/方向bank。fresh指新进程/独立run roots、重新模型前向；允许只读共享模型权重/输入缓存。旧实验仅为reference0，不计入两次新跑。

## 科学配方与可读来源

完整沿用各来源实验自身producer、已固定输入与模型/tokenizer revision、软件环境、数值路径、样本/候选顺序、prompt、few-shot演示与种子。不要为了统一实现而把Base迁到Phase11路径。所有4B干预窗口inclusive15:18、BF16、batch1、alpha1、beta0、block damped_euler、cachefirst、FP32 exact reduced SVD rank1；三种方向时序保持来源语义，Native不loop。
- Phase10：contract_v2及已接受的A输入binding，MMLU 5shot test14042/57subjects、原完整字母continuation评分；ARC固定revision210d026faf9955653af8916fad021475a3f00453、25shot test1172、完整候选文本，主acc_norm、附acc；decode=bypass；producer运行seed按逐cell实际来源保留，不能统一设置。Base MMLU Native沿用Phase8 producer2dd3dfd、Loop K2沿用Phase8G producercffc8d3，两者torch.manual_seed=20260905；current_t K2 lambda0.9沿用Phase10 producer及20261002。Base ARC沿用Phase10原20261002。复用原演示身份/顺序而非重新抽样。Base model revision906bfd4b4dc7f14ee4320094d8b41684abff8539。
- Phase11：contract_v1、A输入binding、gate_d_arc_full_analysis.json及D4冻结源/最终分析。ARC teacher-forced完整候选评分，主acc_norm，不能将其误作生成CoT或额外干预候选位置。GPQA Main作者GitHub canonical448，0shot原生chat template、原选项打乱seed20261008、greedy/nativeEOS、max_new_tokens8192；decode=full，prefill在本题本臂自身轨迹拟合，decode冻结复用、decode_fit_count=0，first KV路径不变；主指标为D4最终S2整行语法及重复一致归并，冲突/无匹配计错，不语义救答案、不修parser。
- 先定位原manifest/producer命令与运行环境，保存一份可读14cell recipe映射及来源。HEAD462539bfde14d0bc66d175c7f3d92280c65902a3为派发观察点，不代表所有历史producer同一版本；原Base/Phase11路径若当前代码产生材料性漂移，用原immutable来源实现与新输出root，或向planning报告最小差异，不擅自换科学实现。
- 原结果本地：外层资产/报告/phase10/gate-c-mmlu-20261006/{independent_cells,contrasts}.csv；gate-d-arc-20261008同名文件；资产/报告/phase11/notion_20261013/ARC_18cells.csv；资产/报告/phase11/notion_full_20261014/GPQA_8192_S2_full_{cells,paired,aggregate}文件。D4远端final root见gate_d4_final_execution_evidence_20261013T141800Z.json，分析source f40cd50247dae1d4c196e9252acb43c6119e70e9。

## 执行顺序、资源与工程授权

1. 核对exact绑定、git路径/loopscope分支/基点祖先、现有dirty与历史来源。只做对当前依赖的廉价漂移核对，不重审旧Gate。读hpc2-hkustgz-ssh skill再访问HPC；信任异常走hpc2-recover-host-key，不自动接受key、不改VPN/hosts/SSH信任。读取remote相关source、inputs、env、cache、jobs获授权。
2. 最小increment为新14cell×2清单、独立repeat输出路由、现有producer launcher及闭合/复现报告。允许scripts/configs及Phase10/11专属增量工具、定向tests/docs/runbook；protected wrapper/strategies/cache/config/eval_runner与冻结输入/历史工件只读，需要材料变更报planning。正常commit/push origin/loopscope，仅stage自身文件，保留他人dirty；不force/reset/rebase或动main。
3. 复用同关键路径既有预检；新launcher/repeat路由/环境/producer发生变化时，做最小同路径<30min端到端debug，覆盖Base似然和Instruct生成/保存/校验。此Gate明确沿用项目用户规则：所有非正式debug/preflight只用debug分区，正式用当前普通用户合法最高优先级A800分区及最高合法QoS（预计emergency_gpu，须现场核实）。debug无合适准入则报告，不能将工程测试改名为正式绕过。不新增长度pilot或重新选8192。debug通过即推进正式，不加全套防御测试。
4. 正式总最多8A800，单GPU至多8CPU（遵守现场更紧限制）；无累计GPUh硬上限，记录预计/实际/失败/预留成本。沿用最近用户少量长任务、最大合法TimeLimit策略；兼顾并行，不强制两小时分片。两轮可并行但独立进程/输出/随机状态，任何交错不得改变单次科学路径。按各workload同硬件实测packing留实际峰值余量，GPQA8192原packing5仅在环境/工作负载一致时可复用，不外推到MMLU/ARC。禁止改batch或共享可变model/tensor状态填显存。先用可保留的正式首批补足必要资源证据，再展开，零OOM为目标；意外OOM保留证据降packing、仅重跑缺失/无效项。
5. 每条/最小可恢复单位落盘，repeat/family/cell/identity独立绑定；两轮内部成功记录不得重复/覆盖。允许同科学/资源范围内低风险launcher、环境、序列化、scheduler修复与新路径恢复，不靠准确率筛样本/调配置。仅允许取消executor自身确证错误/重复/失控的D5 exact jobs，不能为排队盲取消，不操作其他Gate。

HPC= hpc2-hkustgz。只读原source /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope 与原 pinned venv；可写 workspace /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope 下 inputs/staging/runs/artifacts 的 phase11-gate-d5-<timestamp> 新root（各repeat独立），源码以已提交immutable staging部署；旧run/复现checkout及shared模型/数据cache只读，不升级/下载依赖或模型。确需新环境报planning。

## 闭合与预先冻结分析

先完成两轮全部102796条raw scores/GPQA outputs、同一canonical identities与配置无缺重、合法数值及GPQA提取预测保存；之后允许读取已有canonical gold一次统一分析。不新增中途gold例外，不按某轮结果改变另一轮。历史已见gold无需假称全项目blind；此处隔离的是本次未完成重复的outcome。

主问题为运行复现性：逐family/cell报告reference0、repeat1、repeat2的正确数、acc、相对该轮Native及同K Loop的delta；逐题预测一致率、改变数、旧正确→新错误/旧错误→新正确、raw score差异（合理浮点容差需在读新outcome前固定）及GPQA token序列/长度/EOS/S2状态一致率；不新增digest。严格逐题相同与仅汇总同分必须区分。差异不直接当工程失败，先记录硬件/数值/进程/输入证据，解释可接受非确定性，不能择优选重复。

每轮5个衰减候选×[Native,同KLoop]=10个主配对比较；两轮共20项作为一个新的Holm family，双侧exact McNemar、alpha.05；普通Loop-Native对照按family计共5/轮、两轮10项另一个次family。不沿用旧全网格调整p替代此次family，也不把两轮拼成2N独立题目。MMLU沿用subject内paired bootstrap10000 seed20261002；Base ARC题级同seed；Phase11 ARC/GPQA题级10000 seed20261008，名义95%CI。同时保留旧探索全网格p作为历史旁列，不能混为同一检验。

重复同题/同配置是运行复现性证据，不是独立数据验证。原最优为事后选择：即便重跑p显著也不升级为确认性效果。报告每轮delta的符号/大小、是否与原值一致及相对Loop额外收益；Base ARC原deltaLoop=0不得声称复现了额外增益。不设“必须正增益”作为Gate工程PASS条件；负结果/漂移如实交付。

## Agility budget / 终态

最小配置/launcher改动→关键路径定向测试及必要debug→正式两轮→一次闭合分析。不得重审旧54/68/18全网格、增加新lambda/模型/任务、重跑成功条目或读MMLU-Pro。5候选、14cells每轮之外均未授权。executor可按skill最多3个有界subagent做实现/只读证据，不能独立调度/Git整合/终态；不必为用满名额而delegation。正常低风险修复不限轮次，审计退修通常一次合并。

正式提交获得真实jobID后立即创建唯一executor-owned full60min heartbeat，包括PENDING；短debug运行稳定约1min才需要smoke10min，结束快则无需。monitor每次一次有界只读，健康不变静默；终态先pause后通过send_message_to_thread恢复exact executor并保留工具确认，fallback向planning发送AUTOMATION_RELAY_REQUIRED。新Gate采用Phase11现行监控授权，不继承旧Phase10关闭监控的历史条款。用户已授权本Gate派发/修复/终态路由。executor提交后回报真实job IDs和packing/规模/预估，不能把创建聊天说成已提交GPU。

完成后唯一GATE_D5_FINAL_AUDIT（或真正BLOCK）投递planning exact thread/local，工具成功才算交付；否则写TERMINAL_DELIVERY_UNCONFIRMED。报告源码/命令/定向tests退出码/jobs与root/102796闭合/两轮对比/统计/资源/偏离/未做事项。机器证据.planning/phase11，中文结果外层资产/报告/phase11/gate-d5-repeat-<date>，原始题目/输出不上Git不下载Mac。planning用1–3决定性事实验收，不自行PASS；此Gate完成后等待用户讨论，不进入下一Gate或MMLU-Pro。

## 2026-10-14来源种子澄清（planning已冻结）

用户选择原样重跑，因此以每个reference0 cell的真实producer/seed为准。已直接查看2dd3dfd run_phase8_accuracy.py及cffc8d3 run_phase8_gate_g_accuracy.py，均torch.manual_seed(20260905)。两轮MMLU Native/Loop保留20260905，current_t.9保留20261002；演示/样本/选项顺序仍按原输入，不因运行seed澄清重新抽样。分析bootstrap种子保持本handoff既定20261002，与producer RNG区分。允许用逐cell历史immutable producer运行至D5新root，最小新launcher负责调度/输出/汇总；不得修改旧manifest的new_in_phase10或绕过当前producer拒绝来冒充原路径。无需把历史reuse cells强塞Phase10 producer。资源、两轮fresh及闭合规则不变。

## D5历史producer最小debug输入许可

2026-10-14：executor确认尚无任何D5 GPU/Slurm提交。允许P8 Native/Loop历史producer最小debug各使用其原gold-free canonical pool中doc_index=0的第一条canonical记录（若索引按subject分组，取冻结canonical顺序第一条，保存完整identity），保留原prompt/候选，不读gold、不计算correctness、不按结果选题，不纳入正式面板。在debug分区<30min验证模型加载、评分、seed生效、逐题checkpoint；不能截prompt/候选以凑时长，不能扩为长度pilot。P10已有工程池、P11既有synthetic短生成路径继续沿用；此许可只适配旧producer输入schema，无新增科学配置。
