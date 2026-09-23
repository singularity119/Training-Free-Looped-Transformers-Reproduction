# Gate E：A800 分区优先级补充（2026-09-23）

用户最新指令：所有涉及HPC A800的任务都使用最高优先级分区。本补充覆盖原Gate E handoff 的“debug固定debug分区”及任何较低优先级A800默认选择；科学方法、12-cell面板、160GPUh总预算和同时最多2GPU不变。

Gate E executor在**每一次新增A800提交**前，按实时Slurm `sinfo`/`scontrol show partition`、账户可用性和QoS证据确定普通用户可合法使用且支持A800的最高优先级分区；有多个可用QoS时用其中合法最高优先级。不得根据分区名称猜测，不使用管理员专属分区，不因为排队或申请方便自行选较低优先级。A800的debug/preflight、smoke、canary、probe、正式full均遵守；短预检仍`<30min`、同commit/launcher/env/model/runtime/producer/verifier全路径，后续formal须以前检通过为条件。非A800 debug仍依原skill使用debug分区。

在本补充生效前已提交的job保留其既有分区、状态和证据，不因新规则自动取消/重排；若后续需要补新的A800 job，按此规则执行。若最高优先级合法A800分区不能接纳必要资源或时限，报告确切调度限制给planning，不静默降级或改科学。现有作业若已经满足规则无需重复提交。唯一执行任务、监控路由和终态协议不变。
