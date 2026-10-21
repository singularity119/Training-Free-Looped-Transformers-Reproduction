# 第11阶段：公司集群接续 prompt 与 Git 结果索引

截至2026-10-21；整理基点 `987fa85fc84d2d35ade54a2be3f0a782954ea441`，本文件随接续包提交。获取后记录实际 HEAD，勿把基点当成要求回退的版本。

## 可直接复制到新设备的 prompt

你接续我的 LoopScope 第11阶段研究。我这台设备不能访问学校 HPC2，但可以访问公司集群；希望将后续适合的重计算放在公司集群。先从 Git 恢复上下文、阅读已有结果、检查公司集群环境并给出具体实验分工。不要尝试修复本机的校园VPN，也不要默认重跑已完成实验。

仓库：`git@github.com:singularity119/Training-Free-Looped-Transformers-Reproduction.git`，工作分支 `loopscope`，`main` 为只读复现参照。首次可用 `git clone --branch loopscope --single-branch <上述地址> loopscope-tflt`；已有仓库先保留本地变更再pull，不能reset或force-push。本机路径以 `git rev-parse --show-toplevel` 为准；旧Mac/HPC2绝对路径只是来源定位，不是新设备的写入目录。

依次阅读：
1. `AGENTS.md`、`docs/new_device_handoff.md`（其中旧动态状态已过时）、本文件。
2. `.planning/phase11/loopscope_phase11_control.md`（顶部最新决定优先），`loopscope_phase11_contract_v1.md`、`loopscope_phase11_followup_plan.md`。初版合同的2048、S0、54配置并不是当前全部授权，须结合后续handoff与用户修订。
3. `loopscope_phase11_gate_d4_acceptance.md`、`loopscope_phase11_gate_d5_acceptance.md`、`loopscope_phase11_gate_d6_acceptance.md`、`loopscope_phase11_gate_d8_acceptance.md`，均在同一planning目录。
4. D7 handoff、`docs/loopscope_phase11_d7.md`、`docs/loopscope_phase11_d7_s2.md`；D8 handoff及`docs/loopscope_phase11_d8.md`仅为已完成配方和证据。
5. `.planning/phase11/portable_results_20261021/manifest.json`及本文件下方结果索引。无需HPC2就能读取这些JSON/CSV；每个文件有来源，不能只挑最优行。

研究问题与方法：我们研究免训练循环Transformer，在固定窗口中重复block并对残差的首要方向做衰减。第11阶段主模型为`Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`，BF16、batch1；窗口inclusive15:18（边界B15到B19）、block damped_euler、alpha1、beta0、步长1/K，K2或K3，cache_strategy=first、decode_mode=full。方向用本题本配置prefill的干预前残差，FP32未中心化、未行归一化exact reduced SVD top1，prompt非padding/non-special位置参与拟合；仅最后有效prompt位置施加衰减。fixed_t0在t0拟合不干预、t>=1用v0；lag1用前轮方向；current_t含t0用当轮方向。decode只复用已冻结prefill方向，对当前token干预，不能每token重拟合SVD，不跨题共享方向。方向衰减为delta' = delta - lambda v(v^T delta)。保护现有KV逻辑：K轮临时KV回退、原始入窗表示额外stash一次、decode每token持久追加一次。不要为了换集群改成cachelast、全序列重算或另一套近似实现。

已有结果和边界：
- ARC-Challenge：完整1172×18，25-shot、原生候选文本评分，主acc_norm；不是生成CoT任务。结果在原始聚合及portable arc表中。
- D4 GPQA Main：448×18、8192＋S2，已PASS。最高两配置244/448=54.46%，44项Holm均未拒绝。S2是看过格式诊断后的事后修订；多条合法答案行字母一致可合并，不一致计错，无匹配计错。不能把历史2048/S0或S1分数混作当前主结果，也不能声称方向特异性或独立确认。
- D5：14配置各fresh重复两次，共102796条。repeat1/2的14个cell预测完全一致；与reference0的差异只在历史Base MMLU Native A40子集发现，属于披露的跨硬件边界。固定输入重复不是新题独立确认，不能把family变化导致的显著性变化当新证据。
- D6：GPQA CoT/direct及prefill-only对照，12×448=5376，已PASS。K3提示交互Holm p约0.00788为探索结果；full与prefill-only的decode增量两项均不显著，不能证明decode增益或等价。direct保留历史推理尾句，不能说彻底去除CoT。
- D7：MMLU-Pro 12032题、同类别5-shot CoT、8192、greedy、seed20261016，S2按每题实际3–10个选项范围处理。Native/current_t K2 lambda0.5/fixed_t0 K3 lambda0.9各12032条已完成，正确数8342/8273/8290，准确率69.3318%/68.7583%/68.8996%。独立CPU复算36096条未发现预测、gold对齐或token解码差异。D7尚未最终PASS：普通Loop K2/K3共24064条在已交付证据中未完成。不能把前三组当五组闭合。已有提前gold访问，首次2026-10-17T17:35:52.986717Z，不能再称D7全程gold-unread。
- D7敏感性：旧8192轨迹截取2048 token后S2为63.7217%/64.1705%/64.1955%；4096为66.6639%/66.6057%/66.7553%；完整为69.3318%/68.7583%/68.8996%。这是已有生成前缀的事后分析，不是新跑三种预算。官方API提取与adapted-lm-eval规则不同，分别保留，不能混表。原冻结S2仍主评分。
- D8：不同checkpoint `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` 的原生thinking，GPQA448×3，32768，temperature0.6/top_p0.95/top_k20/min_p0/repetition1，每题seed20261018+canonical_index。S2只读首个生成`</think>`后的final区间，不从thinking挽救答案。已PASS并关闭：Native183/448=40.8482%，K2 current_t .5为164=36.6071%，K3 fixed_t0 .9为154=34.375%；Holm p为.061386/.017470，仅K3显著下降。格式失败增加，不能全解释为推理受损。单次采样、同题选过配置，不能分离checkpoint和thinking因果效应。正式66.99 A800 GPUh，实际双进程整卡峰值81031/81920MiB，只余1.09%，无OOM不等于packing2安全。

接续方式：本次请求授权你在新设备恢复上下文、读取Git结果、检查公司集群并规划重计算分工；还没有指定具体的新面板、GPU数量、集群路径或新Gate executor。因此先用已连接环境查清GPU型号/显存、CUDA/PyTorch/Transformers、调度器、合法配额、存储/缓存、联网及固定模型数据可用性，给出最小可运行方案。缺失的入口和预算向我一次性问清，不猜SSH账户或沿用学校分区名。不要将HPC2的8CPU/GPU和max8A800当公司集群实际额度。长序列先测真实显存和吞吐，保留合理余量；不能继承D8 packing2的错误准入估计。

候选重计算优先讨论：D7普通Loop两组补齐最直接，但必须先协调其已有executor和任务状态；其他预算扩展、采样重复、机制消融须明确问题与配方，不能因为读到历史计划就自动启动。若公司GPU不同，不应只在公司跑Loop后把与HPC2历史Native的差异全归因于方法；先制定同硬件对照或明确跨硬件桥接设计，并冻结统计解释。已完成D4–D8不自动重开。

按research-gate-orchestrator协作约定执行（新设备有此skill时实际读取；没有时遵循仓库的责任分离规则）：规划负责合同、授权和轻量PASS；每Gate一个独立执行聊天，新线程默认gpt-6.1-sol/high。本机规划线程ID为01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f；D7原executor为01a142a7-562a-7e83-b085-70999fbbc780，host=local是来源设备视角，不能在新设备照抄路由。在control/handoff明确任务、executor与主机分工之前，不抢占D7、不恢复其monitor、不重复提交。如果用户指定迁移D7，先明确撤销旧调度权和接收方绑定。一次最小debug、一次轻量验收，不反复全量审计已PASS结果。

迁移输入：Git内只有代码、配方、汇总和来源，未包含原题、gold、逐题生成/预测、模型或缓存。公司集群从合法来源按冻结revision重建数据，使用仓库renderer、题序/选项seed/示例规则；检查可读sample identity和配置一致性。不要把远端路径存在误当文件在公司集群已存在，也不要用缺失逐题文件的汇总伪造配对统计。必要输入需要另行受控传输或重建，不提交Git。新运行写fresh root，保留失败尝试。预测先闭合再评分；如果用已看过结果的同题集，应如实标注探索性。每Gate一个monitor，终态暂停并唤醒exact executor，executor回传FINAL_AUDIT或BLOCK，规划判定PASS。

你第一轮的交付请包含：当前理解的阶段结论；公司环境实测与缺失信息；你建议承接的一个具体重计算任务及预算/对照设计；需要本设备协调的D7分工；明确下一步。先完成可做的只读工作，不止回复“可以”。

## Git结果索引

所有路径相对仓库根；portable根为 `.planning/phase11/portable_results_20261021/`。

|内容|直接读取|
|---|---|
|ARC完整18配置及配对统计|portable根 `arc/cells.csv`、`arc/contrasts.csv`；完整原结构 `.planning/phase11/gate_d_arc_full_analysis.json`|
|D4 GPQA8192 S2全18配置及44对比|`d4/cells.csv`、`d4/contrasts.csv`；原结构 `gate_d4_final_analysis_20261013T141800Z.json` 的 `s2`|
|D5两次fresh复现|`d5/analysis.json`、`cells.csv`、`contrasts.csv`、`reproduction.csv`、`analysis_spec.json`|
|D6提示与decode范围对照|`d6/analysis.json`、`cells.csv`、`primary_contrasts.csv`、`auxiliary_contrasts.csv`、`cross_prompt_descriptive.csv`、`length_contrasts.csv`|
|D7前三组主结果|`d7/three_cells_aggregate.json`，及 `d7/hpc2/interim-*/aggregate.json`、`interim-k3-*/aggregate.json`|
|D7独立评分审计|`d7/hpc2/independent-score-audit-*/` 三个聚合JSON|
|D7长度×评分规则六组表|`d7/hpc2/prefix4096-dual-sensitivity-*/combined_lengths.json`|
|D7官方API/local与adapted harness|`d7/hpc2/official-local-rescore-*/aggregate.json`、`harness0411-*/aggregate.json`（规则分开）|
|D7失败诊断|`d7/hpc2/failure-diagnosis-*/` 聚合JSON；不将子集人工诊断加分当正式全量评分|
|D8最终thinking结果|`d8/analysis.json`、`cells.csv`、`contrasts.csv`、`diagnostic.json`、`resources.json`|
|来源、下载时间与范围|`manifest.json`；D7与D4本次直接从HPC2取回；D5/D6/D8 analysis已与HPC2逐结构比对一致；ARC为既有聚合证据|

便捷列举：`git ls-files .planning/phase11/portable_results_20261021`。JSON内HPC2路径仅作来源，离线读取聚合时不需要访问它们。此包不是原始记录备份，不能独立重评分或重算新的逐题配对检验。全部结论需保持各自parser、长度、统计family与探索性边界。

## 本次只读现场观察

2026-10-21经严格SSH验证连接mgmt-6；xhuang225的squeue成功返回空。D7绑定run root仅见logs/canary/first/debug，未见last正式目录；不等价于搜遍所有新run root或确认原executor已撤权。本次没有提交/取消作业、读取新outcome或改动监控。D7的现有分析属于此前已授权评分，本次只取汇总。
