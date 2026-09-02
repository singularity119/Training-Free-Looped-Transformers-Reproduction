# LoopScope Phase 7 Gate E Gemma Two-A800 Supplement 3

```text
GATE_E_OPERATIONAL_SUPPLEMENT
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=E
PLANNING_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR=01a00569-e66d-7eb2-9499-56b01d268366
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-TopAvailable-Outcome-第7阶段-Gate E
SUPERSEDES_RESOURCE_INTERPRETATION=.planning/phase7/loopscope_phase7_gate_e_resource_recovery_supplement_2.md
SCIENCE_CHANGE=NONE
PANEL_CHANGE=NONE
RUNTIME_RESOURCE_AMENDMENT=GEMMA_ONLY_EXPLICIT_TWO_A800_MODEL_SHARDING
CURRENT_DECISION=GATE_E_GEMMA_TWO_A800_RECOVERY_AUTHORIZED_CANARY_REQUIRED
FORMAL_EXPANSION_CONDITION=FIVE_CELL_CANARY_PASS_WITH_GEMMA_TWO_A800
```

## 1. 触发证据与决定

Gate E canary array `10227425` 中 Gemma no-loop cell 26 与 Gemma loop cell 34 均在独占、顺序使用的
单张 A800 80GB 上发生真实 `OutOfMemoryError`，`record_count=0`，没有 producer/cell receipt，且没有
读取 partial outcome。两次失败的 peak allocated 约 60.892 GB、reserved 约 66.551 GB；当前 normal-user
可见 GPU 只有 A800 80GB 与 A40 48GB，没有更大显存单卡。因此将 Gemma exact frozen workload 从
单 A800 改为单进程显式分片到两张同节点 A800，属于必要的 Gate-scoped runtime/resource amendment，
不改变科学合同。

当前 array 的 task 15 仍在运行时不得取消、干扰或替换其唯一 heartbeat。必须先等整个当前 array
terminal，删除现有 heartbeat 并保留所有 valid completed cells 和失败证据，之后才能进入本补充的
Gemma 双卡修复。Cell 26/34 的失败输出不得作为有效 outcome；cell 0/8/17 只要各自 verifier PASS
就保留，不得重跑。

## 2. 双 A800 修复边界

仅允许为 `google/gemma-2-2b` 增加一个显式、确定性的 two-A800 device-map/model-sharding 路径：

- 每个 Gemma producer process 精确申请同节点 `gres=gpu:a800:2`，一次只运行一个 cell；不得与另一
  Gemma/Qwen/Llama process 共享这两张卡。
- 保持同一 model/tokenizer revision、BF16、逻辑与物理 `batch_size=16`、输入 membership/order、
  evaluator、window、`k`、cache、mode、strategy、alpha/beta、decode、输出 schema 和 verifier。
- 使用显式记录的 deterministic device map；模型参数必须全部且仅一次分配到两张 A800。禁止
  `device_map=auto`、CPU/disk offload、量化、模型副本/data parallel、microbatch、batch splitting、
  tensor/model state sharing 或因显存改变 logits/evaluator 语义。
- 输入放到 embedding 所在设备；跨设备层边界、loop window、FinalNorm/lm-head 和 evaluator 输出必须
  由实现显式处理。不得用隐式 `.to(cuda:0)` 把已分片模型或张量重新聚集到单卡。
- Qwen/Llama 继续使用已通过的单 A800、concurrency=1 路径。Gemma 双卡 PASS 不能作为提高
  Qwen/Llama packing 的依据。

若 deterministic model sharding 在保持以上冻结项时无法实现，或 representative `batch_size=16`
仍在任一 A800 OOM，不得自行改用 microbatch、减小 batch、CPU offload、tensor parallel、量化或
修改 evaluator；保留证据并向 planning 请求下一项最小决定。

## 3. 实现、测试与 staged canary

1. 等当前 array terminal 并正确关闭旧 heartbeat 后，先完成一个 purpose-specific implementation
   commit。只修改 Gate E 已授权实现/launcher/tests/docs 路径；不修改 protected paths、control、
   scientific amendment 或既有 sealed evidence。
2. Targeted tests 至少覆盖：Gemma-only 双卡开关；显式 device-map 完整且无重复/遗漏；单卡默认路径
   不漂移；embedding/input device、层边界、loop window、FinalNorm/lm-head device closure；禁止
   auto/offload/microbatch；resource receipt 记录两张 GPU 的型号、总显存、peak allocated/reserved、
   OOM 与 elapsed。测试后做 focused commit/push 和 HPC2 dedicated checkout clean fast-forward。
3. 在 `debug` partition 运行 `<30min` 双 GPU functional-path preflight，覆盖真实 Gemma 权重、同一
   launcher/venv/evaluator、至少一个冻结 loop/cache path、安全写入与 fresh pre-outcome verifier。
   Debug 只证明功能，不用于推导 A800 显存；不得读取或汇报 accuracy/correctness。
4. Debug PASS 后，在 fresh write-once root 用 final partition 与两张 A800 先只运行最高需求的 frozen
   Gemma cell 34，保持完整 test-14042、`batch_size=16`。该 cell 必须 producer/resource receipt 与
   fresh pre-outcome verifier PASS，且两卡均被实际使用、无 OOM、无 output collision/barrier violation。
5. Cell 34 retainable PASS 后，才在另一 fresh path 重跑 invalid Gemma baseline cell 26；要求相同闭合。
   任一失败均是 attempt event，先诊断并在本授权边界内 fresh retry；不得直接扩展 Gemma panel。
6. Cell 26/34 PASS 后，从 retained valid roots 联合验证 exact canary indices `0,8,17,26,34`。只有五项
   全部 14,042 identities/57 subjects、配置/资源/信息屏障闭合，才满足 formal expansion condition。

## 4. Formal 扩展与监控

五-cell canary PASS 后立即只提交 remaining missing cells：

- Qwen/Llama：单 A800、concurrency=1。
- Gemma：每个 process 两张同节点 A800、concurrency=1；只补 invalid/missing cells，不重跑 valid cells。

任一时刻只允许一个 executor-owned 60-minute full heartbeat，覆盖真实 active formal job IDs；替换前
先删除旧 heartbeat，terminal 时必须 self-wake exact executor。保留所有旧 roots、jobs、logs 和失败
resource receipts；不得覆盖或 destructive cleanup。

## 5. 仍未授权事项

不授权改变 batch、microbatch、模型/data/split/identity、35-cell panel、window/k/cache/mode、renderer/
evaluator、dtype、metrics/schema/bootstrap、outcome barrier；不授权 partial outcome inspection、Gate D
重评分、额外模型/window/task、privileged/admin-only 资源、Gate F、planning closeout、最终绘图/报告
或 cryptographic/digest 治理。唯一 Gate 终态仍须由 exact executor 可见投递
`GATE_E_FINAL_AUDIT` 或真正 `BLOCK` 给 planning thread。
