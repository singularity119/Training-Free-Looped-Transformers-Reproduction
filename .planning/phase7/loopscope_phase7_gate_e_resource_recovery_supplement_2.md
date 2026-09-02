# LoopScope Phase 7 Gate E Resource Recovery Supplement 2

```text
GATE_E_OPERATIONAL_SUPPLEMENT
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=E
PLANNING_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR=01a00569-e66d-7eb2-9499-56b01d268366
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-TopAvailable-Outcome-第7阶段-Gate E
SUPERSEDES_RESOURCE_INTERPRETATION=.planning/phase7/loopscope_phase7_gate_e_gpu_utilization_supplement_1.md
SCIENCE_CHANGE=NONE
PANEL_CHANGE=NONE
CURRENT_DECISION=GATE_E_RESOURCE_RECOVERY_AUTHORIZED_FORMAL_CANARY_REQUIRED
FORMAL_EXPANSION_CONDITION=VALID_RETAINABLE_CANARY_PASS
```

## 1. 目的与冻结边界

本补充修正 Gate E 资源预检和 formal 扩展顺序，不改变三模型、35-cell panel、MMLU test-14042、
标准 5-shot evaluator、`batch_size=16`、BF16、loop/cache 配置、输出 schema、pre-outcome barrier、
combined analysis 或 verifier。Supplement 1 保留为历史授权，不覆盖；从本补充可见投递起，以本文件
作为 Gate E 当前资源恢复与 formal 扩展的 canonical operational authority。

已观察到的资源事实只用于制定当前恢复路径：tiny debug/underfilled batch 曾低估 formal 峰值，
正式 cell 曾出现约 44 GB 单进程峰值；后续多轮 0--3 秒退出来自计算节点的 Modules/Git 环境差异，
而非模型、数据或 outcome。历史 job/root/log 均保留，不覆盖；用户主动取消必须与异常失败分开记录。

## 2. 两类 preflight 必须分别闭合

1. **Functional-path preflight**：继续使用 `debug` partition、`<30min`，覆盖冻结 commit、launcher、
   venv、模型/数据加载、renderer/evaluator、loop/cache 路径、安全输出和 verifier。
2. **Resource-shape evidence**：只有真正形成 `batch_size=16`，并覆盖代表性 upper-tail prompt/sequence
   或等价最高 activation demand 的测量，才能用于 formal 显存和 packing。`limit=2`、四条样本、
   underfilled batch 或仅覆盖短序列的 debug 只能证明 functional path，禁止据此推导 formal 并发。
3. 按模型及资源行为不同的 cell family 分别记录 peak allocated/reserved memory、throughput、OOM 和
   GPU 型号/总显存；不得从 Gate C、另一模型、另一 cell family、A40 小样本或另一 GPU 类型直接外推。
4. 不得为了制造资源压力而持久化 prompt、token IDs、gold、partial correctness/accuracy 或其他被
   outcome barrier 禁止的字段。

## 3. Launcher 自包含要求

- 登录节点在 submission 前完成 repository/branch/commit/clean 校验。计算节点不得要求 Git
  executable；若仍需运行时 provenance，只能读取已冻结 manifest 和共享 checkout 的必要只读元数据。
- 正式与 debug launcher 使用显式 Bash script/shebang 和已审计 Python/venv 路径；不得依赖
  `sbatch --wrap` 的隐式 `/bin/sh`、计算节点 Modules 安装或登录节点 PATH。
- Modules 缺失不得影响已固定 venv/CUDA 路径的正常启动。任何 launcher 修复通过 targeted tests 后，
  先运行一个 child 或 formal canary；该项闭合前不得重新铺开完整 array。

## 4. 可保留 formal canary 与扩展

1. 使用 final partition、final launcher、final frozen inputs 和 `batch_size=16` 运行最小 formal canary
   set。Canary 必须是 35-cell panel 内的真实 cell；成功记录可直接计入最终 completeness，不是额外
   smoke，也不得读取其 partial outcome。
2. Canary set 至少覆盖每个模型的一个预测最高显存 frozen loop cell；若不同 cache/cell family 的
   内存等价性没有实现或实测依据，则继续以 panel 内真实 cells 补齐这些资源 family 的 canary。
3. 每个 canary 必须完成 producer receipt、resource receipt 和 pre-outcome verifier，并记录真实 formal
   peak memory/throughput。只有 canary 全部 PASS，才能提交其余 cells。
4. 扩展时按模型/cell-family formal 峰值做有界 packing，并保留 practical transient/allocator headroom。
   约 44 GB 峰值的 cell 在 80 GB A800 上不得并发两个独立进程。若没有充分证据证明更高并发安全，
   使用 concurrency=1；最大化显存利用不能越过不 OOM 的硬边界。
5. Formal OOM 或 launcher failure 是 attempt event：保留失败证据，修复后先重新闭合一个 child/canary，
   再只补跑 invalid/missing cells；不得因资源修复更改科学矩阵。

## 5. 已提交 `10226946[0-7]` 的过渡规则

Executor 在收到本补充后先只读核对 scheduler 和 write-once artifacts，不读取 partial outcome：

- 若数组仍全部 `PENDING` 且没有有效新 cell，授权先删除旧 heartbeat、证据保留式取消数组，并在
  fresh sibling root 按第 4 节 staged canary 路径重提；取消完成后才能创建 replacement heartbeat。
- 若已有元素 `RUNNING` 或已完成有效 cell，将这些工作视为当前 formal canary evidence 并保留；不得
  重跑 valid completed cells。能安全单独取消的未启动元素可取消并在 canary 闭合后再扩展；若调度器
  无法在不影响运行元素的情况下安全拆分，则允许已提交的 concurrency=1-per-GPU 元素继续，但它们
  不构成将来 packing>1 的证据。
- 用户此前要求中断 `10226670` 所产生的 `CANCELLED` 终态必须记录为 user-directed cancellation，
  不得计入异常退出或 launcher failure 统计。

## 6. 监控、报告与权限边界

- 任一时刻仍只允许一个 executor-owned heartbeat。真实 formal job ID 返回后使用 `full` 60-minute
  cadence；替换任务前先关闭旧 heartbeat，terminal 时必须 self-wake exact executor。
- 问题、ordinary operational permission 和 terminal packet 只报告 planning thread
  `01a0013e-71c3-7c90-a547-4059b462dc7e`。不得静默终止；唯一终态仍为可见投递的
  `GATE_E_FINAL_AUDIT` 或真正 `BLOCK`。
- 不授权改变 batch、panel、模型、数据、split、renderer/evaluator、loop/cache 语义、dtype、schema、
  bootstrap、information barrier，不授权额外模型/task/Gate F、planning closeout 或 privileged 资源。

