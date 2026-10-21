# Gate D-8 — GPQA 原版 Qwen3 thinking 三组迁移检验

日期：2026-10-18。用户已明确授权独立执行；本Gate不是此前只读评分审计的继续。
PLANNING_THREAD=01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f
PLANNING_HOST=local
EXECUTOR_THREAD=01a15760-371e-7e61-af51-d8375a503bdc
EXECUTOR_HOST=local
CANONICAL_TITLE=execute-LoopScope-GPQA-Thinking-第11阶段-Gate D-8
MODEL=gpt-6.1-sol
REASONING=high

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

## 2026-10-20 规划接回及恢复条件

用户授权planning接回01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local；旧planning为01a14da8-5e0b-7550-9874-5bd47416bc6f。原executor及科学合同、模型档位不变。原EXECUTOR_HOST=local为来源设备记录，本机不可访问，须核对实际host并确认新终态路由；未确认前保持执行暂停。今天11:26北京时间debug与emergency_gpu 1GPU的test-only均通过，GrpSubmit为空、队列为空，无真实提交。收到路由确认后由planning明确恢复同executor，沿原补权重、真实debug、1344采集、全量闭合后评分顺序执行；正式exact资源/QoS与Phase11共享8A800仍由executor提交前验证。D7保持原绑定，禁止重复调度或监控。

## 2026-10-20 用户明确批准替代及恢复（覆盖旧线程路由等待）

旧executor 01a14eeb-0aae-7733-a86e-62b55c8dde54 的D8执行权限撤销；唯一新executor 01a15760-371e-7e61-af51-d8375a503bdc/local，gpt-6.1-sol/high，planning仍01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local。用户授权本次替换、恢复及终态消息路由。先确认本绑定和消息路由，再按GO接续原source-r2/debug-plan-r2。先检查远端现存D8 job/raw/下载进程及monitor，成功记录不重跑；若旧执行仍活跃先向planning报告，不盲取消未知任务。原科学/统计/下载代理/资源边界不变；当前设备7897代理需实测。真实debug通过后才正式1344，Phase11与D7共享最多8A800。唯一executor-owned heartbeat仅真实job后建立，debug10min/full60min，健康无变化静默，terminal pause+self-wake及投递确认。旧不可访问线程未确认收到撤权，不能声称已经停止其进程。新executor可携带本次planning control/handoff授权配置提交，不纳入其他已有staged/unstaged修改；既有runtime source可原样复用，无需为了路由变化重跑CPU全套。

## 2026-10-20 用户批准国内 HF 镜像下载

用户经本planning明确授权可尝试国内Hugging Face镜像，仅下载固定 `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` 及相同tokenizer所需文件。先验证镜像可用及固定revision文件，不能退回main或更换权重；镜像endpoint只设置于下载子进程，不改变全局配置或升级依赖。公开模型不向镜像发送HF token或其他凭据。仅停止或切换自有下载进程，避免重复下载并保留断点，记录实际来源，结束或弃用代理后关闭自有临时隧道。此补充仅改变模型传输来源，不扩大科学、GPU、评分或D7权限。

## 2026-10-20 planning 澄清 debug 排队监控启动

planning明确允许真实debug job已存在且PENDING时即建立本Gate唯一10分钟heartbeat，不要求等RUNNING稳定一分钟。健康排队静默，不盲取消或重投；终态先暂停monitor并保留证据，再唤醒exact executor。`MaxJobsPerAccount`按现场排队事实处理，不视作历史`GrpSubmit=0`复发或Gate失败。正式仍只在真实debug通过后提交，并将同一monitor更新为60分钟。

## 2026-10-20 用户直接授权提交D8全量

用户向本executor明确要求“直接把d8全量也提交了吧”。真实debug已通过后，允许依据已有正式canary显存证据与保守完整KV预算直接提交其余1341条，不再以三条canary全部终态作为全量提交的前置条件。三组各自canonical344的成功/在途canary保留，剩余计划均排除该题；最终仍三组各448，共1344。全量与既有canary合计最多8A800，packing须保留实际容量余量，batch1与独立进程/RNG语义不变。原模型、输入、采样、方法、评分和gold屏障不变；既有低风险缺失/无效项恢复权限不变。只更新同一monitor覆盖全部exact D8 jobs并保持full60min，D7不接管。
