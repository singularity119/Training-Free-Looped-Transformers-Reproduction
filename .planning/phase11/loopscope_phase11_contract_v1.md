# Phase 11 科学合同 v1

GPQA来源最新绑定见loopscope_phase11_gpqa_source_binding.md：已接纳作者GitHub固定版本Main归档，取代下文待绑定HF分发来源；不声称与HF候选版本逐题等同。

2026-10-08。用户已确认主面板、三档强度、full decode、prefill方向冻结复用及GPQA生成提取，并允许继续推进。本文替代配置草案中的待确认建议；以下评测细节为规划在该授权内冻结的实施选择。精确数据绑定由Gate A补齐，正式运行必须等待规划接纳。

## 面板与方法

Qwen/Qwen3-4B-Instruct-2507，BF16，inclusive15:18（B15到B19），K2/K3，batch1，block damped_euler，alpha1、步长1/K、beta0、cachefirst、decodefull。模型/tokenizer候选revision为既有Phase6的cdbee75f17c01a7cc42f958dc650907174af0554，A核对可访问性和模型元数据后绑定，不静默换版本。

Native1、普通Loop2、Online15：K2的fixed_t0/lag1共享臂与current_t各三档，K3三策略各三档。lambda={0.1,0.5,0.9}。每任务18，合计54独立配置；K2别名须真实实现等价验证。无Matched-norm、无新窗口选择、无跨配置方向共享。Native不安装Loop。

各Online配置自身prefill轨迹的干预前残差D_t用于拟合，S为实际保留prompt全部非padding、非special位置。沿用Phase10的原dtype残差、FP32未中心化未行归一化exact reduced SVD top1/CUDA gesvd。仅prefill最后有效位置p衰减delta'=delta-lambda*v*(v^T delta)，cast回原dtype再Euler更新。

fixed_t0：prefill拟合v0，t0不干预，t>=1用v0。lag1：t0拟合不干预，t>=1用自身上一轮干预前残差方向。current_t：每轮拟合当轮干预前方向并立即干预，含t0。保存所需prefill方向后冻结；每个decode步重置迭代编号t，fixed_t0的t>=1用prefill v0、lag1的t>=1用prefill v_(t-1)、current_t每轮用prefill v_t。decode只修改当前token残差，禁止重新SVD、全序列重放替代incremental cache或滚动历史方向。方向在每题结束释放，不跨题污染。current_t比较同时改变干预起点与方向时序，不单独归因。

## 数据与评测

- ARC-Challenge：25-shot、完整test1172，沿用Phase10 revision210d026faf9955653af8916fad021475a3f00453、train示例seed20261002及plain原生候选文本评分。主acc_norm、附描述性acc。不截题/缩短候选。完整候选teacher-forced forward中，方向只拟合共同prompt，干预位置仍限prompt末端，不扩展为所有候选位置干预；此任务不生成CoT，decodefull不意味着实际发生增量生成。正式报告明确这一差异。如候选长度处理破坏该语义则停止报告，不静默换评分。
- MMLU-Pro：TIGER-Lab/MMLU-Pro候选revision b189ec765aa7ed75c8acfea42df31fdae71f97be、完整test12032，5个validation同类别CoT示例，沿用lm_eval0.4.11既有renderer的示例选择。目标cot_content/答案不进入prompt；示例答案允许。A固定实际示例identity、类别及渲染文本规则。
- GPQA-Main：Idavidrein/gpqa，gpqa_main完整448题，0-shot CoT；具体revision、物理split和顺序由A核对固定。正确/错误答案字段仅由隔离输入构建器用于构造四选项，随机打乱用独立固定seed20261008，映射写入封存gold侧文件，不将字段角色送入模型或日志。不可把物理train split误当可用于调参的训练集，不替换Diamond。仅使用已有合法访问；如缺访问权限报告，不代替用户接受条件。原题与生成内容不提交Git或公开发布。

MMLU-Pro/GPQA统一为原生tokenizer chat template、单user消息装载示例/问题，无额外system提示。结尾固定英文指令：Reason step by step, then end your response with exactly one line in the form Final answer: (X), where X is the letter of the correct option. 示例CoT保留并将其末尾答案格式统一为Final answer: (X)。选项按实际数量渲染，不假设MMLU-Pro恒有十项。

生成greedy/do_sample=false、单次生成、max_new_tokens2048，原生EOS/EOT停止，不增加Question:字符串停止。无重采样/答案补救。提取规范：仅接受独立行Final answer: (大写合法选项字母)，允许行首尾空白；全输出必须恰有一个有效匹配行，否则提取失败，计错。不使用宽松字母兜底。命中长度上限且未EOS记truncated；若仍有合法唯一答案正常计分，同时计入截断率。准确率以完整题数为分母；另报提取失败率、截断率、生成长度。以上为Phase11配方，不宣称原论文精确复现。

总正式题级记录：18*(1172+12032+448)=245736，其中生成记录224640。最多460062720生成token，为上界而非成本预测。输入不静默截断；A测token长度并绑定足够上下文，不能以资源紧张为由删题或减少shot。

## 信息与统计

冻结网格后完整采集；gold/correctness/accuracy只能在同一任务18臂原始工件完整、identity闭合后解封。不同任务独立闭合解封，不根据前一任务结果改后一任务。GPQA构建器必要标签接触是有限输入构造权限，不授权查看模型正确率。合成/非测试工程prompt允许debug；GPQA无独立开发集，不能用正式题调配方。

主报告各任务准确率及配对差值。每任务15Online各对Native和对应K的Loop，30比较为一个探索Holm family；三任务不作未校正整体显著结论。策略比较每lambda共4项（K2 current对共享臂1项，K3三策略两两3项），三档共12比较/任务另一个次family。Loop2对Native2比较/任务作为基础family。exact双侧McNemar、alpha0.05；paired bootstrap10000/seed20261008、nominal95%CI，MMLU-Pro按类别分层，其余题级。不重复检验K2别名。报告全网格与负结果，当地最优仅探索、不称迁移验证；无Matched-norm不能证明方向特异性。
