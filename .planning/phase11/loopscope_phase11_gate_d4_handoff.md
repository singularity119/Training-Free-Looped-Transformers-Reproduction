# Gate D-4 — GPQA 8192 full acquisition and analysis

2026-10-13最新用户资源收敛：用户在D4执行聊天明确“就使用最高优先级分区，最多八卡吧”，覆盖此前max-legally-available扩展授权。仅emergency_gpu及其最高合法QoS，总最多8A800；不扩其它分区。剩余2565条已合并array12997851_0..7，TimeLimit14天，全部8064已提交；逐题落盘与断点恢复保持。

2026-10-13用户最新授权：允许使用正常权限范围内尽可能多的A800，覆盖此前max8A800及最多8个长任务的人为限制。D4须现场核对用户/账户association、partition、QoS的GPU/CPU/submit/节点限制及可用性，使用合法可获资源扩大并行；不申请管理员专属资源、不修改权限。长任务数按实际可用并行GPU数与剩余工作量确定，优先最少必要长任务/array且不牺牲吞吐。实测packing5、batch1、科学合同、逐题落盘/断点续跑及成功记录去重不变，已提交任务不盲取消；MMLU继续暂停。

2026-10-13用户最新调度修订（覆盖下文每job2h/旧20条批次约束）：仅尚未提交exact pairs合并为尽可能少的长任务，已提交/运行/排队不取消。D4先现场ledger划定剩余范围；保持max8A800及实测packing5/batch1，优先单job-array最多8长任务，在并行不下降前提下减少作业。取消人为单job2h及累计GPUh硬上限；申请集群正常权限允许最大时长，系统MaxTime仍必须遵守。每条完成立即完整落盘flush并保存可恢复状态；以已验证成功输出重建done集合，重启只做未完成/无效记录，半条隔离重算，不覆盖成功结果。断点默认题间恢复，非token级KV恢复。新增恢复层先做CPU模拟中断/恢复及必要最小同路径debug；不新增长度预检。完整8064闭合后gold及MMLU暂停保持。

2026-10-12用户明确立即派发全量，不再长度预检。Executor01a12f40-cbc9-7790-ac6e-f3fe1c454ab0/local；planning01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local；gpt-6.1-sol/high。按照research-gate-orchestrator skill/protocol及AGENTS执行。

## 冻结范围

GPQA Main全448 canonical identities×原18独立cell=8064，max_new_tokens8192、native EOS，逐条记录generated token IDs/count/end_reason及S1/S0状态。原模型revision/BF16、0shot prompt/选项seed、greedy、batch1、window15:18、K2/K3、lambda0.1/.5/.9、first缓存/full decode/prefill方向冻结完全保持。不优化KV、不强制答题、不改prompt、不新增配置。保留D2048及D2事后结果，不覆盖；本次属于同已见数据后续探索。

S1严格沿用D2冻结regex与唯一整行匹配，主评分；原S0辅助。失败计错、每cell448分母。先全8064原始输出identity/cell无缺重闭合并保存gold-unread状态，再读取原GPQA gold分析。沿用30/12/2对比族、bootstrap10000 seed20261008、exact McNemar/Holm，准确率/CI/长度/截断/提取率并列。禁止中途按准确率调参或提前筛cell。无MMLU读取/运行/解封。

## 立即执行路径

D3 compressed98已由planning验收满足长度选择，8192已由用户确定。禁止追加pilot、补齐336或等待新的长度选择。复用D3未变commit/launcher/环境/producer同关键路径debug通过证据；若正式启动器确需改动，做最小<30min debug端到端，不重新扫描样本/配置。正式A800显存并行准入用可保留的正式首批，先packing1，长序列实测后提高并行、留约20%余量；不能把A40 packing3或2048 A800 packing6直接当8192准入。每正式job1A800/8CPU/<=2h，普通用户合法最高优先级A800分区/QoS，提交前现场核定，最多8GPU总并发。正式无累计GPUh硬上限，记录actual/reserved/预估与失败成本；资源安全仍为硬边界。无需再问用户批准正式提交。

D3无在途任务，旧monitor暂停；若现场变化先核对exact ledger，D4唯一调度者。保留D3全部98结果作为pilot，默认正式面板独立A800采集，不混入A40 pilot以避免增加硬件兼容验证拖延。允许一次全量8064及仅invalid/missing保留证据的新路径恢复；禁止重复成功正式条目。未知低风险工程问题原Gate解决，科学变更才BLOCK。

## 路径/交付

nested loopscope-tflt分支loopscope，当前观察e8d31e201d77f4d475128308f1d1fbe9df345b57，D3仍可能提交收尾文档；保护他人dirty文件，仅stage自身D4文件和planning明确D4 control/handoff，不回退D3。读取最新真实Git版本后绑定runtime。HPC hpc2-hkustgz原workspaces/training_free_looped_transformers_loopscope，新write-once phase11-gate-d4-* staging/runs；固定复现checkout只读。原pinned环境，完整prompt+8192原生context准入，不截输入/改RoPE。SSH异常按对应skill安全恢复，不改host信任。GPU正式作业实际ID返回后立即建立唯一full60min monitor，终态暂停并恢复exact executor继续闭合，最后GATE_D4_FINAL_AUDIT回planning。允许用户既有授权派发/修复/回传；不自行PASS。

新增独立8192配置与必要运行/闭合脚本；原2048工件/parser/runtime不修改。最小targeted验证，正常commit/push，原始数据/输出不上Git不下载本地。中文报告外层资产/报告/phase11，机器证据.planning/phase11。首次提交后尽快回报真实job IDs/partition/packing/预留；执行线程持续推进全量，不停在计划。验收三项：8064闭合及gold顺序、冻结方法/统计、调度结算/科学局限。MMLU持续暂停。
