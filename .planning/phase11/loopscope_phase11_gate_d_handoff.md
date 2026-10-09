# Gate D — ARC and GPQA parallel / AUTHORIZED

2026-10-09用户最新明确“正式任务不设置gpuh上限”：取消正式累计128GPUh及首轮16GPUh预留准入限制，GPUh继续记账但不构成正式续批/同范围低风险重试停止条件。此修订只覆盖当前已授权ARC/GPQA完整面板；max8A800、每正式job1GPU/<=2h、debug<30min、packing约20%显存余量与科学信息屏障不变。不授权MMLU-Pro或下一Gate。

2026-10-09用户明确授权ARC与GPQA并行，取代下文所有先ARC完成再GPQA的顺序限制。两任务仍分别通过同路径debug与packing准入，统一调度总max8A800。可按现场队列、显存与有效吞吐弹性分配GPU，不要求固定4+4；既有运行/成功工件保留，不为改顺序盲取消重跑。每任务全18配置闭合后可独立解封并分析，不受另一任务完成状态约束；结果不得改变另一任务配方。MMLU-Pro新计算/解封仍禁止。当前D身份不变。

2026-10-09。Executor01a11e77-9550-7092-877d-1ecb30c531de/local，gpt-6.1-sol/high；planning01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local。用户明确先ARC→GPQA、尽量提高显存利用率、MMLU-Pro两者完成再商议、最多8A800。按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，实际读skill/protocol、AGENTS、contract_v1、GPQA source binding、control。

## 范围与Agility budget

C PASS，起点949a7aabff9d2a835170ae8bc0a5fe971a948002，原正式source447f64be3172dae403f7cd0c07ae17399a6197b4。先ARC1172×18=21096（复用72、新21024），再GPQA448×18=8064（复用72、新7992）。MMLU禁止新forward、资源实验、gold/outcome；旧72保留。科学不改：BF16/batch1/15:18/K2K3/.1 .5 .9/first cache/full decode/prefill frozen exact FP32 gesvd方向，GPQA greedy2048/nativeEOS/strict extraction，ARC完整候选似然。

最小实现为原index非重叠批处理和同GPU独立worker队列，一次模型加载处理多题。复用C144映射，不能重新编号或重复成功记录。新批次接口仍校验完整pool/manifest/原N。针对性CPU检查排除、无重叠、逐题方向/cache reset及完整闭合，不建设泛化框架、不重跑历史全套。verifier可在明确兼容审查后支持approved source set，不绕过来源检查；commit变动若仅调度/观测，用diff和关键数值/身份等价证据接纳C复用，数值变化先回planning。

当前任务同commit/launcher/environment必须先debug synthetic producer→attempt verifier，<30min，批次及多worker关键路径一致。ARC完成转GPQA时先GPQA路径debug。通过即正式可留存资源首批，再完整面板，不拿纯synthetic任务冒充正式canary。

## 任务独立显存与吞吐准入

每任务按packing2→4→6递增，容量/吞吐支持才到8；不要求用满8进程。每worker独立模型/cache/direction、batch1、唯一输出，无tensor共享。每GPU<=8CPU合理分配，记录线程设置与CPU/SVD瓶颈。显存有余量但有效题吞吐下降时选较低最佳packing，不追求百分比。

A800首批只用未完成、可计入最终面板的正式记录；选择依据冻结输入长度/arm成本而非答案。包含长输入及K3current .9。GPQA更高packing必须将2048KV、同时加载、SVD和临时张量计入容量，C已有A800完整2048两进程事实可参考，不能凭早EOS峰值盲扩。设备实测峰值/采样间隔、进程allocator分别报告，留约20%容量余量；8模型如已不足余量则不试8。记录整批题吞吐/token吞吐/启动时间/OOM。不同批题长和arm成本需匹配或归一化，不凭不同工作量单点宣称加速，不为测性能重复正式样本。异任务/异卡最佳packing不能直接外推。

## 预算与调度

正式任务没有累计GPUh或首轮GPUh预留硬上限；持续记录actual allocation GPUh和运行/排队walltime×GPU预留，用于成本/吞吐报告而非正式停止条件。按实测滚动发有限批；正式每job<=2h、1GPU、最多8A800，现场更低配额优先。仍只完成ARC/GPQA，不进入MMLU。出现科学、信息屏障、资源安全或无材料进展的真实阻断时报告planning；不因累计正式GPUh达到旧128而停止。

非正式任务debug/<30min；真实可保留正式首批与全量用普通用户最高合法A800 partition/QoS，现场核对资格/优先级/提交和运行限制。只可取消自己明确失败/无进展D作业。OOM/launcher等低风险故障保留尝试、降低packing、新路径重试缺失/无效部分，不重复成功、不操作其他作业。

## 路径与权限

真实nested loopscope-tflt/loopscope，写入/提交/远端执行前检查root/HEAD/dirty/固定基点祖先。可改Phase11旁路source/scripts/config调度元数据、tests/docs和D证据；protected wrapper/strategies/cache/config与冻结科学字段只读，必要修改先最小复现和diff回planning。正常单目的commit/push允许，当前planning-owned控制/C验收/max8修订/D文件可一并提交，勿覆盖或stage无关文件。

HPC2 hpc2-hkustgz/xhuang225按SSH skill。旧source/environment/cache只读，pinnedPython沿C；独立真实Git source staging/phase11-gate-d-*，新写入仅专属workspace/{inputs,staging,runs,logs,artifacts}/phase11-gate-d-*。A inputs/C raw/closure只读。无本地模型/原始数据下载，安全metadata可本地；人类报告在外层资产/报告/phase11/。

每任务全部18配置完整identity raw闭合前不读生成正文、gold、correctness、accuracy；资源只看metadata。闭合后授权对应任务隔离gold提取（ARC固定官方test标签、GPQA既有sealed mapping）、exact join及冻结30/12/2 families/McNemar/bootstrap10000分析。提取失败/截断与完整分母按合同，不救答案，不公开GPQA原题/生成。ARC闭合分析后再GPQA，前任务结果不改后任务科学。

## 协作、监控与终态

最多3个有明确文件责任subagent，不覆盖他人修改；仅executor负责Git/远端write/Slurm/整合/终态。低风险自修不限次数但预算和科学不扩；单job失败不是Gate BLOCK。

正式真实ID（含PENDING）取得即唯一executor-owned full60min heartbeat，未变静默；短debug稳定RUNNING约1分钟可smoke10min，后同monitor改full，不重叠。终态暂停/删除monitor并AUTOMATION_TERMINAL_RESUME唤醒exact D；executor必须恢复闭合/修复/续批，不能停在observer后idle。恢复失败relay给planning。用户已授权这些跨线程消息，planning不轮询。

两个任务全完成后唯一GATE_D_FINAL_AUDIT给planning：commands/exit/source/dirty/job、完整closure/统计/资源、C144兼容复用、成本/失败/monitor暂停、MMLU未动声明。可发送任务级非终态进度；预算/权限真正无低风险路径才BLOCK。未确认送达输出TERMINAL_DELIVERY_UNCONFIRMED。规划仅核对1至3项决定性事实；普通材料性返修通常一轮。D不自行PASS，完成后不进入MMLU，等用户商议。
