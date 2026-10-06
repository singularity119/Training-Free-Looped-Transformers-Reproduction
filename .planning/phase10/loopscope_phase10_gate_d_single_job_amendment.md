# Gate D 单次正式allocation补充

2026-10-07用户要求ARC尽量合在一个任务提交，减少各任务重复排队。此补充覆盖D handoff原先首批9job/余59job分批和正式单job最多8h规定，其余科学、权限、预算、无监控要求不变。exact executor=01a111f8-7ba3-7862-b0d6-78c0a5a2f767。

目前executor报告仅debug12932349提交，正式未提交。先查现场，勿重复提交或盲取消已有debug。

优先一次sbatch申请同一allocation的2张A800（若现场时间/资源可行性更适合1张，可用1张并记录依据），每GPU一个worker、packing1/batch1；68cell工作清单由父runner在allocation内调度，worker顺序跑各cell，不为各配置另起排队job/array。各cell独立保存结果/退出状态，配置/题/candidate状态彻底释放，避免跨cell缓存污染。可沿现有每cell独立进程runner，减少改动。

首批7迁移Online+2Native在同job内先执行，计时/预算准入后直接跑余59cell，不需退出allocation等人工批准。首批累计40GPUh的边界和总100GPUh保留；总预算按实际allocation GPU数×walltime计费，包含worker空闲、debug、失败和重试，不能只累计有效forward时长。runner在首批完成时自动核实已消耗+1.2×剩余保守预测<=100GPUh、剩余walltime可容纳；否则保留成功cell并正常停止后报planning资源缺口，不读outcome/不丢配置。首批40GPUh内无法闭合也停止报告。提交前要求既有消耗+该job GPU数×time限制<=100GPUh，并满足现场分区/QoS合法时限；优先选能容纳全部68cell的walltime，不能保留旧8h人为限制。分区仍最高合法A800。

新增整体runner属于关键launcher变化：必须先在debug对同runner/commit/env/两GPU worker调度（若正式2GPU）/producer/每cell验证跑最小<30min端到端预检。已提交旧debug结果可保留为评分证据，但不能声称覆盖新runner。不得为此取消已成功或将工程测试改投A800正式分区。

如果现场合法walltime/GPU资源/硬预算不能容纳单job全量，提供实测/现场依据，只拆成尽量少的必要长job，而非每配置单独提交；不擅自扩大100GPUh/max2GPU。不为了强求一次运行修改科学配置。失败保存成功结果和原attempt，仅后续补缺失，不重跑成功cell。正式job结束后沿D原合同全量闭合再一次分析。用户暂不定时监控继续有效。
