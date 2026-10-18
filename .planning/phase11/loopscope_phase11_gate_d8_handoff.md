# Gate D-8 — GPQA 原版 Qwen3 thinking 三组迁移检验

日期：2026-10-18。用户已明确授权独立执行；本Gate不是此前只读评分审计的继续。
PLANNING_THREAD=01a14da8-5e0b-7550-9874-5bd47416bc6f
PLANNING_HOST=local
EXECUTOR_THREAD=01a14eeb-0aae-7733-a86e-62b55c8dde54
EXECUTOR_HOST=local
CANONICAL_TITLE=execute-LoopScope-GPQA-Thinking-第11阶段-Gate D-8
MODEL=gpt-6-luna
REASONING=max

按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行；实际读取该skill及references/protocol.md。唯一独立executor负责实现、定向验证、GPU调度、分析和终态交付；本planning负责验收。绑定与START前只可阅读，不提交GPU。用户授权执行线程与本planning就本Gate沟通及终态交付；不得绕过planning扩大实验。

## 已冻结配方

- 模型严格为Qwen/Qwen3-4B（原版后训练模型），enable_thinking=True；不是Qwen3-4B-Base，不是Instruct-2507，也不是Thinking-2507。先核对缓存并绑定一个可访问的exact模型/tokenizer revision，三组一致；如果本地无缓存，允许仅获取该官方模型及tokenizer所需文件到新/既有共享缓存，不替换其他模型、不升级共享环境。模型配置、模板、生成参数可读保存；完整prompt＋32768必须落在模型已支持的上下文容量内，不截prompt、不私自启用YaRN或变更RoPE，容量不够报告planning。
- GPQA Main全448题，沿用D4/D5/D6作者源idavidrein/gpqa revision56686c06f5e19865c153de0fdb11be3890014df7及已冻结canonical identity/原题/选项顺序（input seed20261008）。从已有gold-free pool重建单user消息，0-shot，保留原CoT题面与Final answer指令，使用新模型原生chat template、add_generation_prompt=True、enable_thinking=True，无新增system。不把旧模型渲染后的chat token直接套入新模型，不读取目标gold构造prompt。
- 三组fresh：Native无任何loop；current_t K2 lambda0.5；fixed_t0 K3 lambda0.9。共3×448=1344条，单次采样，不增加普通Loop组、其他lambda/窗口、其他任务或第二seed。不重新选最优。
- 两方法沿用inclusive15:18、block damped_euler、alpha1、beta0、步长1/K、cachefirst、decodefull，prefill＋decode全程残差衰减。FP32 exact reduced SVD rank1、自身prefill拟合；decode沿用冻结方向，不重新拟合。维度/层数核对后使用同一实现语义，不因新模型改方法。
- BF16、每worker batch1、native EOS；max_new_tokens=32768（含thinking＋最终回答）；do_sample=True、temperature0.6、top_p0.95、top_k20、min_p0、repetition_penalty1。不强制提前结束thinking，不添加Question:停止串，无答案补救、best-of、重采样或中途更改预算。
- 固定基础seed20261018；每题实际seed=20261018+canonical_index，各组同一题同seed，生成前重置Python/NumPy/torch/CUDA RNG，记录actual seed，避免worker分片/重试改变采样流。一次样本是探索性结果，不声称采样方差已经估计。

## 评分和统计（在新outcome前冻结）

保存全部生成token IDs和完整解码（含thinking标记）、原生end reason、生成长度、thinking/final token范围；原题/生成/gold/逐题预测只留HPC，不提交Git或下载Mac。
主评分：沿用GPQA冻结S2字母A-D语法及一致重复归并，仅作用于thinking结束后最终回答区间。用该tokenizer的实际</think> special token ID确定首个标准thinking终止边界（prompt含开启标记、生成中的闭合标记必须正确处理），只解析其后文本。缺闭合标记、未输出final、无S2匹配、多个不同答案均计错，不从thinking中挽救答案；多个异常thinking边界记录诊断，不静默选择有利片段。工程阶段用合成token样例冻结边界处理；如果原生模板与这个契约不兼容则报告planning，不自改评分。旧S2实现保持原样，新Gate只加独立区间选择适配器。不得把思考中的临时Final answer行计入最终评分。
所有1344成功raw、完整identity和预测固化后统一读取既有独立gold并评分；失败/缺失的工程运行按证据保留后补齐，不用部分accuracy决定停跑。保留全部448为分母，截断不排除。报告三组correct/acc、两方法DeltaNative、exact双侧McNemar、两项组成一个Holm family；题级paired bootstrap10000、seed20261018的名义95%CI。配置由同GPQA旧结果选择，本Gate是跨checkpoint/thinking配方的探索性迁移，不是独立题集确认，也不能将变化单独归因为thinking或checkpoint。
同时报告EOS/length_limit、未闭合thinking、final为空、S2 no_match/conflict、一致重复、完整生成及thinking/final长度均值/中位数/p90、walltime/GPUh和peak memory。原历史实验/评分与本Gate不可覆盖。

## 工程、资源与保护边界

工作目录为本project的loopscope-tflt，branch loopscope，起点4ccd8c18d674cf8e33b6e7ffee8e035d8881cdb4。先检查保护dirty，fetch后仅允许fast-forward，禁止reset/force/stash他人改动。允许本Gate最小独立plan/launcher/producer/analysis、定向测试和报告，以及必要向后兼容opt-in model/template/sampling/answer-span参数；所有旧默认保持不变。protected wrapper/strategies/cache数值核心只读，确需更改先报告planning。正常stage自身文件、commit/push origin/loopscope；本planning新D8 handoff和plan/control条目属于可携带的授权配置，其他人的变更不纳入。source staging必须固定到已提交源码。禁止通用框架/无关重构/新增digest校验层。
HPC2按hpc2-hkustgz-ssh skill、StrictHostKeyChecking=yes、UpdateHostKeys=no、ClearAllForwardings=yes连接。workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope；只写inputs/staging/runs/artifacts下phase11-gate-d8-<timestamp>新root。历史source、inputs、raw、gold、score、D7控制与监控均保护。允许读scheduler确认资源和已有job，不接管D7，不提交D7剩余Loop，不取消/重跑任何其他Gate。
普通用户合法最高优先级A800分区/QoS；整个Phase11与既有任务合计最多8张A800，不因新Gate把并发翻倍。非正式工程debug必须debug分区且<30分钟；必须先有同模型/模板/采样/runtime的最小真实路径smoke，再提交可保留正式首批。32768不能沿用旧8192的packing5假设；保守首批测显存，约20%余量，容量证据充分后才提高packing。尊重每GPU≤8CPU及现场更紧限制，使用少量可恢复长任务及合法TimeLimit。无累计GPUh硬上限，记录预计/实际/失败成本。只允许自身D8确证错误/重复/失控job的限定取消和缺失/无效项fresh retry，成功题不覆盖不重跑。
Agility budget：复用D6/D7机制，最小增量仅新checkpoint的thinking模板、采样seed和最终回答区间；定向检查身份/选项、thinking生效、sampling参数、S2区间、loop/cache语义与seed恢复。检查通过即推进1344条，不新增长度扫描、lambda扫描、多轮预检或大套审计。低风险工程失败可同科学修复并保留失败证据；科学配方/模型容量/权限不符报planning。

## 监控和交付

D7既有监控保持原状态，不迁移、不唤醒。D8真实job存在后按协议仅一个executor-owned heartbeat（debug10min，正式60min，健康无变化静默），只观察自身job并恢复本executor处理终态；本planning不建重复monitor、不常规轮询。终态先暂停monitor，再向本planning发送唯一GATE_D8_FINAL_AUDIT或真正BLOCK，保留投递确认；无法发送时给TERMINAL_DELIVERY_UNCONFIRMED的可粘贴包，不声称已送达。此消息路由按用户独立执行授权及技能协议执行。
交付：exact模型/tokenizer/source版本、三组1344闭合、定向测试、真实job及cost、评分/配对统计/长度诊断、偏离、monitor暂停证据。聚合中文报告放外层资产/报告/phase11/gate-d8-gpqa-thinking-<date>；机器绑定到.planning/phase11。planning只验1–3项决定性事实；负结果可PASS。D8完成后等待用户，不自动进入后续实验。

## 2026-10-18 Temporary model-download proxy supplement

Planning authorizes the bound D8 executor to use one temporary loopback-only SSH reverse proxy, remote 127.0.0.1:58018 to local 127.0.0.1:7897, solely to download Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c and its tokenizer. Use StrictHostKeyChecking=yes, UpdateHostKeys=no, ExitOnForwardFailure=yes. For this dedicated tunnel use isolated SSH configuration without inherited RemoteForward entries, retaining the verified host/port/user/identity/known_hosts checks; ordinary SSH still uses ClearAllForwardings=yes. Do not terminate unknown listeners on port conflict. Restrict proxy environment to the download process. Record ownership and close only this tunnel immediately after download completion or abandonment, verifying release. No persistent SSH/VPN/hosts/routes/proxy/trust changes, credential output, unrelated downloads, environment upgrades or D7 operations. This is a transport repair within the existing model-download authority.
