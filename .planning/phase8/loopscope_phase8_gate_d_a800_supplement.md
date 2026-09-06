# Gate D 操作补充：A800最高合法优先级

2026-09-06。用户明确要求Gate D不再长时间等待A40，改用HPC A800并开最高优先级。
exact executor仍为`01a071e9-d133-7742-8826-e0cf5993a1f1`，planning为`01a06fe9-c7bb-7d72-a906-234f301de317`。
按照research-gate-orchestrator skill/protocol执行；本补充只改变调度操作边界，contract_v1全部科学参数和全18闭合后解封规则不变。

planning现场只读确认12651219/12651220均PENDING，原因分别Resources/Priority；A800 partition emergency_gpu为UP、RootOnly=NO、PriorityTier/JobFactor=300。最终account/QoS及节点资格由executor提交前再查普通用户association和当前资源。

授权立即由同一executor完成：

1. 暂停/更新既有唯一monitor，查询两个旧job和有效输出。仍PENDING且无有效输出时，取消这两个本Gate A40排队作业并保留scheduler记录；这是用户要求的资源迁移，单列为USER_REQUESTED_RESOURCE_MIGRATION，不记为工程失败。不取消未知状态作业。若已开始运行，保留已有效分片并据剩余进度决定保留该运行或安全停止后仅迁移未完成工作，禁止双份正式采集。
2. 将缺失的正式test区间迁至A800，使用普通用户实际可用最高优先级partition/QoS（现场候选emergency_gpu，Tier300）；不能申请管理员优先级/修改账户权重或抢占无关作业。不再另排A40重复任务。保留两个模型各18cell合计的第一512题A40 canary，不因换卡重跑有效数据。
3. 复用producer2dd3dfd及已通过的同版功能debug（未变的commit/launcher/env/runtime/serialization）；A800更换卡型的资源形状需实测，不能直接把A40 packing吞吐当A800结论。先小批可保留A800正式canary后扩展余量，可以同一受控作业内先完成小分片并验证再运行剩余。若代码/launcher/runtime改变则按原handoff重做对应debug，不以提速为由绕过真实变化验证。
4. A800初始packing可在1–3独立进程内根据实际显存决定，保持至少15%实测瞬时余量；可复用已知A40安全packing2/3作保守起点，之后测量。batch1/dtype/模型/方向/样本顺序/统计不变，不开新的数值性能设置。记录实际GPU类型与混合硬件来源，不把A40/A800差别解释成干预效应。各配对arm尽量采用同一硬件分片安排；已有A40 canary两arm保持一致。
5. 维持全Gate同时≤2GPU、每job≤6h、累计≤64GPUh；CPU/RAM沿用handoff及集群每GPU限额。新A800 job/root采用新时间/attempt路径，仅补剩余样本。旧第一512及其它有效分片按canonical identity合并，最终仍18×14042，不能重复计数。
6. 重定向原唯一full60min monitor到新job集，主动终态唤醒路由不变。切换提交后向planning回传旧job状态/取消确认、新job IDs、partition/QoS/account/GPU、root和当前排队/运行状态，不等待整个实验终态才反馈本次调度。后续继续既有Gate D直至完成，不重新请求已授权的普通调度权限。

无需新增Gate或executor。队列最高优先级不能保证立即开跑，需如实报告实际状态。planning只下发补充及核对送达，不代executor操作Slurm。
