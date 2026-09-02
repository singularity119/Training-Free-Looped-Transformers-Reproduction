# LoopScope Phase 7 Gate E Gemma Batch Recovery Supplement 4

```text
GATE_E_OPERATIONAL_AND_RUNTIME_SUPPLEMENT
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=E
PLANNING_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR=01a00569-e66d-7eb2-9499-56b01d268366
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-TopAvailable-Outcome-第7阶段-Gate E
SUPERSEDES=.planning/phase7/loopscope_phase7_gate_e_gemma_two_a800_supplement_3.md
USER_AUTHORIZATION=GEMMA_BATCH_SIZE_MAY_BE_LOWERED_TO_AVOID_OOM
PANEL_CHANGE=NONE
RESULT_SEMANTICS_CHANGE=NONE
CURRENT_DECISION=GATE_E_GEMMA_BATCH_RECOVERY_AUTHORIZED_CANARY_REQUIRED
FORMAL_EXPANSION_CONDITION=FIVE_CELL_CANARY_PASS_WITH_FROZEN_GEMMA_BATCH
```

## 1. 决定与冻结边界

用户在 Gemma 单 A800、`batch_size=16` 的 cell 26/34 均发生真实 OOM 后，明确允许降低 batch size。
因此不进入尚未投递执行的 two-A800 device-map/model-sharding 主路径；Supplement 3 只保留为被本补充
取代的规划记录。最小恢复路径改为：Qwen/Llama 保持已验证的 `batch_size=16`，Gemma 使用单张
A800，在 panel 内真实 canary 上寻找最高不 OOM batch size。

Batch size 只作为 Gemma evaluator 的运行时吞吐/显存参数调整；模型/revision、BF16、MMLU
test-14042 identities/57 subjects、5-shot renderer/evaluator、样本顺序与完整 membership、window、`k`、
cache、mode、strategy、alpha/beta、decode、预测与评分语义、输出 schema、35-cell panel、combined
analysis 和 information barrier 全部不变。不得根据 partial outcome 选择 batch size。

当前 array task 15 与现有 heartbeat 必须先自然终态并正确关闭；保留所有 valid cells、失败 roots 和
resource receipts。Cell 26/34 的 OOM attempts 均为 invalid、`record_count=0`，只补跑这两个及之后的
missing Gemma cells；任何 valid cell 不得重跑。

## 2. Gemma batch-size ladder

1. 首选 `batch_size=8`，单 A800、单 process、concurrency=1。
2. 若 representative full cell 在 batch 8 仍 OOM，保留失败证据并依次试 `4 -> 2 -> 1`；每次只允许
   下调一档并使用 fresh write-once path。
3. 第一档在真实 formal canary 上 PASS 后立即冻结为 `GEMMA_FORMAL_BATCH_SIZE`，用于 Gemma 全部
   remaining cells；不得为了排队或追求更低显存继续下调，也不得在不同 Gemma cells 间混用 batch。
4. 禁止 batch >8 的中间猜测、动态/逐样本 batch、microbatch 合并、跨 GPU data parallel、模型分片、
   CPU/disk offload、量化或改变 evaluator/logits 语义。若 batch 1 仍 OOM，停止并向 planning 请求新决定。
5. Resource receipt 必须记录实际 batch、GPU 型号/总显存、peak allocated/reserved、OOM、elapsed 与
   throughput；不得记录 prompt、token、gold、partial accuracy/correctness 或 outcome aggregate。

## 3. 最小实现与 staged canary

1. 只实现 per-model frozen batch binding 与 receipt/verifier closure；做 targeted tests，确认 Qwen/Llama
   仍为 16、Gemma ladder/freeze 正确、35-cell panel 和 membership 不变、invalid-only retry 与 barrier
   生效。使用 purpose-specific commit/push，并将 HPC2 dedicated checkout clean fast-forward。
2. Launcher/runtime 改变后，在 `debug` partition 做一个 `<30min` Gemma functional-path preflight，
   覆盖真实模型、同 evaluator、一个 loop/cache path、安全写入与 fresh pre-outcome verifier；debug
   不承担 formal 显存结论。
3. Debug PASS 后，在 fresh formal root 用单 A800、Gemma batch 8 先只运行最高需求 frozen cell 34。
   Producer/resource receipt 与 fresh pre-outcome verifier PASS、14,042 identities/57 subjects、无 OOM、
   无 output collision/barrier violation 后，该 batch 才冻结。若 OOM，按第 2 节 ladder fresh retry。
4. Cell 34 retainable PASS 后，用同一已冻结 Gemma batch 在另一 fresh path 只补跑 baseline cell 26。
5. Cell 26/34 PASS 后，从 retained roots 联合验证 exact canary indices `0,8,17,26,34`。只有五项全部
   membership/config/resource/barrier 闭合，才允许 formal 扩展。

## 4. Formal 扩展与监控

五-cell canary PASS 后立即只提交 remaining missing cells：Qwen/Llama 单 A800、batch 16、
concurrency=1；Gemma 单 A800、已冻结最高通过 batch、concurrency=1。只补 invalid/missing cells，
不重跑 valid cells。

任一时刻只允许一个 executor-owned 60-minute full heartbeat；替换任务前删除旧 heartbeat，terminal
时 self-wake exact executor。所有旧 roots/jobs/logs 保留且不得覆盖或 destructive cleanup。

## 5. 未授权事项

不授权改变模型/data/split/identity、panel/window/k/cache/mode、renderer/evaluator、dtype、metrics/
schema/bootstrap、outcome barrier；不授权 partial outcome inspection、Gate D 重评分、额外模型/window/
task、two-A800/model sharding（除非 planning 以后重新授权）、privileged 资源、Gate F、planning
closeout、最终绘图/报告或 digest 治理。唯一 Gate 终态仍为 exact executor 可见投递的
`GATE_E_FINAL_AUDIT` 或真正 `BLOCK`。
