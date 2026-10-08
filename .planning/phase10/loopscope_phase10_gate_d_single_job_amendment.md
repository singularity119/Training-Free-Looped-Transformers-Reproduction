# Gate D 单次正式allocation补充

2026-10-07用户要求ARC尽量合在一个任务提交，减少各任务重复排队。此补充覆盖D handoff原先首批9job/余59job分批和正式单job最多8h规定，其余科学、权限、预算、无监控要求不变。exact executor=01a111f8-7ba3-7862-b0d6-78c0a5a2f767。

目前executor报告仅debug12932349提交，正式未提交。先查现场，勿重复提交或盲取消已有debug。

优先一次sbatch申请同一allocation的2张A800（若现场时间/资源可行性更适合1张，可用1张并记录依据），每GPU一个worker、packing1/batch1；68cell工作清单由父runner在allocation内调度，worker顺序跑各cell，不为各配置另起排队job/array。各cell独立保存结果/退出状态，配置/题/candidate状态彻底释放，避免跨cell缓存污染。可沿现有每cell独立进程runner，减少改动。

首批7迁移Online+2Native在同job内先执行，计时/预算准入后直接跑余59cell，不需退出allocation等人工批准。首批累计40GPUh的边界和总100GPUh保留；总预算按实际allocation GPU数×walltime计费，包含worker空闲、debug、失败和重试，不能只累计有效forward时长。runner在首批完成时自动核实已消耗+1.2×剩余保守预测<=100GPUh、剩余walltime可容纳；否则保留成功cell并正常停止后报planning资源缺口，不读outcome/不丢配置。首批40GPUh内无法闭合也停止报告。提交前要求既有消耗+该job GPU数×time限制<=100GPUh，并满足现场分区/QoS合法时限；优先选能容纳全部68cell的walltime，不能保留旧8h人为限制。分区仍最高合法A800。

新增整体runner属于关键launcher变化：必须先在debug对同runner/commit/env/两GPU worker调度（若正式2GPU）/producer/每cell验证跑最小<30min端到端预检。已提交旧debug结果可保留为评分证据，但不能声称覆盖新runner。不得为此取消已成功或将工程测试改投A800正式分区。

如果现场合法walltime/GPU资源/硬预算不能容纳单job全量，提供实测/现场依据，只拆成尽量少的必要长job，而非每配置单独提交；不擅自扩大100GPUh/max2GPU。不为了强求一次运行修改科学配置。失败保存成功结果和原attempt，仅后续补缺失，不重跑成功cell。正式job结束后沿D原合同全量闭合再一次分析。用户暂不定时监控继续有效。

## 用户追加：单任务最大合法A800数

2026-10-07用户明确允许执行线程单任务使用集群对该账户授予的最高合法A800数量，用户记忆为8张。此条覆盖本文与原D handoff/control的max2GPU限制。executor提交前现场核验用户association、分区、QoS、单job/单用户GPU限制及现有占用，确定有效单job上限；不要把QoS GPU8的历史记录直接当作当前可申请8张的充分证据。按核验后的最大合法数量组织单个allocation，禁止特权绕过或修改集群限制。

每GPU一个worker、packing1/batch1，68cell在同job内动态分配以减少尾部空闲。明确绑定CUDA设备，保持每配置独立进程/结果及科学隔离。最多3个subagent的协作限制不变，与GPU worker数量无关。

总100GPUh和首批40GPUh仍有效，此次授权增加并发数量而非总预算。按整个allocation的GPU数×walltime计费，包含空闲；提交预留与实际消耗不得越界。若8GPU，则剩余预算允许的walltime小于等于(100减已消耗GPUh)/8小时，再与集群合法时限取小值；必须纳入loading和尾部开销。首批9cell在同job内计时后按原预算条件自动展开。如最高数量方案不能在预算/时限内完成，向planning给出最小资源决策，不擅增预算或减科学集合。

整体runner先debug同路径验证；debug不必申请正式全数GPU（若debug合法上限不足），可在debug合法数量内验证多worker设备隔离与任务调度，并以CPU定向测试覆盖正式worker数量的清单分配；记录debug实际GPU数与正式数差异。禁止把额外工程验证投到正式分区。正式每个worker使用同已验证源码/runtime/producer路径。无需为GPU数增大重做已通过的科学评分等价测试。

## 资源选择最终决定

2026-10-07 planning向用户解释现场限制并推荐“单节点7张A800、最高优先级、一个作业”，用户回复“允许”。据此明确采用该推荐方案：emergency_gpu单节点7张A800，使用现场核验可合法满足7卡的QoS。此决定取代此前“最大合法数量”可能包含多节点的歧义；不授权多节点或8卡低优先级专用分区。

保持单job动态调度68cell、每GPU一个worker、packing1/batch1、总100GPUh、首批40GPUh、预算准入及标签隔离。walltime预留按7×time计算并扣除已有D消耗，不能只算繁忙worker；不得超过现场合法时限和总预算。首批完成后同job内自动估算并继续的条件不变。若预算不能容纳完整面板，提交具体追加需求，不删配置。

执行者已报告旧2GPU正式job12932810仅PENDING时撤下，0秒/0GPUh并留receipt；提交新job前仍核查无重复正式任务。完成新runner合法debug预检后，可直接提交单节点7卡正式作业，无需再确认。最多3个subagent、暂不定时监控的用户要求不变。

## 用户最新决定：恢复多任务提交

2026-10-07用户认为单job七卡排队更严重，明确要求“还是采取以往的方案，多个任务提交”。此决定覆盖本文此前单job要求：恢复独立小资源正式作业（默认每job 1 A800、packing1/batch1，每完整cell独立输出），同一最高合法优先级分区按现场资格提交。保持本阶段已批准最多7GPU总并发，但不再要求一次申请7GPU；用有界批次/依赖限制峰值及预算，不能无界堆积全部任务。

先核实唯一旧正式job12932927的实时状态、已分配资源和输出。若仍PENDING，授权executor以scancel --state=PENDING取消，复核取消终态/分配GPUh再释放其预算预留；保留receipt，然后才提交替代，禁止旧job与替代重复执行。若已RUNNING，先核对完整cell/执行中任务/写盘可恢复性，保存成功结果；仅当能确认不会丢失已完成结果并消除重叠时，允许有序停止该D job后仅补缺失。若停止无法可靠保留结果，向planning报告具体状态，不盲取消或整批重跑。若已COMPLETED，直接验收已有结果，不能因调度变更重跑。

复用已经验证的每cell producer/launcher路径；若恢复以往已通过的原路径且关键源码/runtime未变，可引用既有相应debug，不为纯sbatch资源拆分重复科学检查。若有关键runner/序列化变化，先最小debug再正式。工程任务仍debug，正式最高合法A800。

先完成原首批9cell（7迁移Online+2Native）并按实测预算准入剩余59；若旧job已完成其中部分则仅补缺失。总100GPUh/首批40GPUh不变，累计实际消耗加所有运行/待运行job GPU数×time预留<=100GPUh，取消job仅确认终态后释放预留。每job时限按既有实测及现场合法限制合理设置，不复制七卡14h到每个小job。账户/分区更低并发限制始终优先。所有科学配置、完整面板、标签隔离、最多3subagent及暂不定时监控要求不变。提交一次确认后保存接续点，不人工轮询。

## 最新：取消GPU小时硬上限

2026-10-07用户在D线程直接要求“不需要设置gpu小时数上限”，planning核实原文。覆盖全部旧100GPUh/首批40GPUh/1.2估算资源准入/硬预算停止条件；保留实际用量和预测/时限预留记录，不因预测超旧上限停工。当前多job每1A800/max7并发、最高合法优先级、现场合法QoS/MaxSubmitJobsPU等限制不变；不绕过50待提交数量上限。科学、完整性、标签隔离及无monitor要求不变。
