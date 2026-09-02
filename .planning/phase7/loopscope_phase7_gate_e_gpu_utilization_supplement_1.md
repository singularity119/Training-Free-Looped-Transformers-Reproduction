# LoopScope Phase 7 Gate E GPU Utilization Supplement 1

```text
GATE_E_OPERATIONAL_SUPPLEMENT
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=E
PLANNING_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR=01a00569-e66d-7eb2-9499-56b01d268366
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-TopAvailable-Outcome-第7阶段-Gate E
SCIENCE_CHANGE=NONE
PANEL_CHANGE=NONE
RESOURCE_OBJECTIVE=MAXIMIZE_SAFE_GPU_MEMORY_UTILIZATION_AND_TIME_TO_RESULT_WITHOUT_OOM
```

## 1. 用户资源原则

Gate E 正式全量实验必须把“不 OOM”作为硬边界，并在该边界内尽量提高单卡有效显存占用、
可用 GPU 并行度与总体吞吐。不得把“一张 GPU 只运行一个 cell”当作默认保守配置；也不得为了
追求显示占用率而牺牲正确性、造成频繁 OOM 或延长整体 time-to-result。

本补充只改变 scheduler/resource packing。冻结的 35-cell panel、模型/revision、MMLU test-14042、
5-shot renderer/evaluator、`batch_size=16`、BF16、loop recipe、输出 schema、information barrier、
combined analysis 与 verifier 均不变。不得通过调大 batch size、跨 cell batch、共享 model/tensor
state 或修改科学配置来填满显存。

## 2. Preflight 与 packing 决策

1. `debug` `<30min` representative preflight 除原 handoff 的决策路径覆盖外，必须记录每个独立
   process 的 GPU peak memory、OOM 状态、elapsed/throughput，并抽样记录整卡 memory/utilization；
   不记录 prompt、token、gold 或 partial outcome。
2. 允许同一 GPU 上运行多个彼此隔离的 cell process。Executor 应依据实测峰值和目标卡物理显存，
   为 allocator/transient peak 保留约 10% 实用余量，先计算一个固定、有界的
   `max_concurrency_per_gpu`，再用同卡并发 preflight 验证。
3. 若预测并发发生 OOM，则逐级降低并发；若 PASS 且仍明显有安全余量，则再测试下一档并发。
   停止于“最高已通过并发”，或下一档按实测峰值已不能保留上述余量。正式 launcher 必须冻结并
   记录该并发上限，不得运行无界的“尽可能多”。
4. 正式 job 内允许使用固定上限的 worker pool：child 完成后立即从尚未运行的 frozen cells 中补入
   下一项，以持续利用显存；每个 child 使用独立模型实例、日志、输出与 write-once path，且不得
   跨 child 共享 tensor/model state。
5. Scheduler 请求必须满足 normal-user CPU/GPU 限制；必要时限制每个 child 的 CPU thread，避免
   CPU 过度订阅成为 GPU 空转来源。GPU/partition 按合法最高优先级、显存适配、availability、queue、
   实测吞吐和总体 time-to-result 选择，不锁死卡型。

## 3. Formal 运行、失败与监控

- Formal submission 使用 preflight 证明的最高安全 packing，并尽可能并行占用当前合法可用 GPU。
  调度不得读取 partial accuracy/correctness，也不得根据 outcome 改变 cell 顺序或资源配置。
- 每个 formal parent 必须记录 GPU 型号/显存、固定 concurrency、child PID/cell ID、各 child exit、
  peak memory、OOM count、elapsed/throughput 和输出路径；这些仅是资源/完整性证据。
- 若 formal 出现不可预见 OOM，保留失败 root/job/log，降低 concurrency 后只在 fresh write-once path
  重跑 invalid/missing cells；这是已授权的 Gate 内普通资源修复，不要求扩大科学权限。
- 获得真实 formal job IDs 后仍只创建一个 60-minute full heartbeat，覆盖全部 parent jobs、worker
  pools 和 child progress；terminal 后删除该 heartbeat，并由 exact executor 向 planning 交付唯一
  `GATE_E_FINAL_AUDIT` 或真正 `BLOCK`。

## 4. 当前 attempt 的过渡规则

- 若尚未提交 formal jobs：直接应用本补充。
- 若 formal jobs 仅 PENDING 且没有有效输出：允许证据保留式 cancel，并在 fresh sibling root 按最高
  安全 packing 重提。
- 若已有 formal job RUNNING 或 cell 已完成：保留全部有效 completed cells，不为重新 packing 而
  重跑；只对尚未运行或 invalid/missing cells 应用本补充。任何旧 root 均不得覆盖或删除。

## 5. 未授权事项

不授权改变 batch size、科学矩阵、窗口、模型、数据、split、renderer、probe、dtype、loop/cache
语义、metrics/schema、bootstrap 或 outcome barrier；不授权生成、额外窗口/模型/task、Gate F、
planning closeout、privileged/admin-only 资源或 cryptographic/digest 治理。

