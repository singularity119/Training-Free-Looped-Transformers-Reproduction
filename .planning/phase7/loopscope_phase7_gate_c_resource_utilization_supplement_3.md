# GATE_C_RESOURCE_UTILIZATION_SUPPLEMENT_3

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=C
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a0036f-fc1d-7412-b043-b93b5f088398
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-Full-Acquisition-第7阶段-Gate C
SUPPLEMENT_STATE=AUTHORIZED
CURRENT_ATTEMPT_DECISION=EVIDENCE_PRESERVING_CANCEL_AND_FRESH_RELAUNCH
SCIENCE_CHANGE=NONE
BATCH_SIZE_GT_1_AUTHORITY=NONE
FOUR_CHOICE_REDUCED_PROJECTION_AUTHORITY=NONE
GATE_D_AUTHORITY=NONE
GATE_E_AUTHORITY=NONE
TERMINAL_EVENT=GATE_C_FINAL_AUDIT_OR_BLOCK
```

## 1. 协作与投递协议继续强制生效

本补充只扩展 `.planning/phase7/loopscope_phase7_gate_c_recovery_handoff_2.md` 的 Gate C
恢复权限，不替代原 handoff。Executor 必须完整遵守
`/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md` 及其
`references/protocol.md`：同一 Gate 内完成低风险工程修复、preflight、正式重跑和封存；仅使用一个
executor-owned heartbeat；最终向 exact planning thread 投递并确认唯一一个
`GATE_C_FINAL_AUDIT` 或真正的 `BLOCK`。不能向用户请求普通 operational permission，不能静默终止，
不能进入 Gate D/E。可按原 handoff 使用最多三个窄 subagent，但 Git、远端写入、Slurm/job control、
集成和 terminal delivery 仍由 bound executor 亲自负责。

## 2. Planning 只读诊断与取消决定

Planning 于 2026-08-15 对现有正式尝试做了一次有界只读检查：

```text
run_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase7-gate-c-20260815T033502Z-formal-recovery2
running=10219148 qwen shard-00;10219149 qwen shard-01
pending=10219150;10219151 llama shards;10219152;10219153 gemma shards
partition=emergency_gpu
resource_per_job=1*A800-80GB,8CPU,64GB,time=12:00:00
observed_elapsed=02:36:43;02:11:41 for the two running jobs
observed_batch_cpu=00:03:30;00:03:04
accepted_output_records=0
```

两个正在运行的 Qwen batch script 只设置了 `HF_HOME`、`HF_DATASETS_CACHE`、
`TRANSFORMERS_CACHE` 和 `PYTHONPATH`，没有设置 `HF_HUB_OFFLINE=1`、
`HF_DATASETS_OFFLINE=1` 或 `TRANSFORMERS_OFFLINE=1`。stderr 在模型约 7 秒加载完后持续对
`https://huggingface.co/datasets/cais/mmlu/...` 发出 HEAD 请求并因
`ConnectionResetError(104)` 按 1/2/4/8/8 秒退避；随后才回退到本地 cache。两个日志分别已经出现
181/151 次 cached-dataset fallback，而 producer 仍未写出任何正式 JSONL。短时低 GPU utilization
和约 9 GiB 显存占用因此不是“模型太小”的主要材料性问题，而是 formal launcher 漏传离线环境导致的
正常路径网络等待。

这不是值得保留的有效前向进度；继续运行预计会重复同一退避，并有明显 12 小时 timeout 风险。
因此授权 executor：

1. 先保存现有六个 job 的 `scontrol` batch script、`sacct/squeue` 状态、日志路径/大小/mtime 和
   当前无正式 output 的事实，不复制或展开日志中的敏感内容；
2. 取消 jobs `10219148`–`10219153`，等待并记录六个 terminal scheduler states；
3. 保留整个旧 run root，不移动、不删除、不覆盖、不把其中任何 partial 当作 Gate C evidence；
4. 随六个旧 job terminal，暂停并删除旧 heartbeat
   `loopscope-phase-7-gate-c-full-jobs-10219148-10219153`，记录 lifecycle closure；
5. 不在旧 root 中重跑，后续 preflight 和 formal 均使用 fresh sibling roots。

## 3. 授权的 exact-equivalent 工程修复

### 3.1 必须修复：formal 离线环境

每个新的 debug/formal Slurm command 必须显式设置：

```text
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
PYTHONPATH=src
```

模型与 tokenizer 继续 `local_files_only=True`，数据继续使用既有 audited cache；不得 fallback 到
official Hub、mirror、代理或下载。该变化只恢复已接受 preflight/renderer 的本地数据路径，不改变任何
model、row、prompt、token、forward 或 scalar 计算。

### 3.2 最小代码修复：一次进程只构建一次 task map

当前 `phase7_renderer._state_for_subject()` 在每个新 subject 上重新调用 `_task_map()`；
`_task_map()` 又通过 lm-eval 构建全部 57 个 MMLU tasks。授权在
`src/tflt/loopscope/phase7_renderer.py` 增加进程内 lazy task-map cache，使同一 producer process
只构建一次完整、同顺序的 57-task map，随后仍按 subject 调用原 `_safe_dataset_pair()`，保留相同
safe validation/dev projection、同一 lm-eval renderer、同一 seed 和同一 tokenizer 调用。缓存不得保存
prompt、token IDs、gold/correctness、hidden state 或 logits。

### 3.3 最小可观测性：安全进度计数

授权在 `scripts/loopscope/run_phase7_base_trajectory.py` 每完成固定间隔（建议 25 条，末条必报）向
stderr 输出仅含以下字段的 flush progress event：

```text
event=phase7_progress
completed_records=<integer>
total_records=<integer>
```

不得输出 identity、subject、prompt、token、序列内容、gold/correctness、metric value 或 outcome。
正式 JSONL schema、顺序和数值不得改变。该计数用于 heartbeat 估计 throughput/ETA，不是科学 artifact。

### 3.4 明确不授权的“优化”

- `batch_size>1` 是 frozen science amendment；本补充不授权，仍严格为 `batch_size=1`。
- CPU render/tokenization prefetch 在单生产者/有界队列下可能保持科学语义，但会引入线程安全、顺序和
  ephemeral-input lifetime 的新 runtime path；修复离线环境和 task-map 重建后尚无证据需要它，当前
  不授权。
- 用四行 lm-head weight 直接计算四 choice logits 在实数代数中等价，但当前 BF16 producer 已经有
  “不同 GEMM shape 导致 rounding 不同”的直接 Gate B 证据；它不能视为当前数值实现的 exact-equivalent
  修复，可能改变 entropy/KL 和 V3 排名。本补充不授权 reduced-vocabulary projection、boundary stacking、
  mixed precision 或 tolerance-based substitution。
- 不授权多 process 共用同一 GPU、改 shard membership、量化、offload、dtype、probe、renderer、metrics、
  schema 或 model forward。

## 4. Git、测试与 preflight

这是 decision-critical renderer/runtime 变化，必须使用一个 purpose-specific commit；允许修改的代码/测试
路径仅为：

```text
src/tflt/loopscope/phase7_renderer.py
scripts/loopscope/run_phase7_base_trajectory.py
tests/test_phase7_renderer.py
tests/test_phase7_launcher.py
```

不得触碰七个 protected dirty paths。Executor 运行 targeted tests，至少证明：

1. 两个不同 subject 的 rendering 只构建一次 task map；task order、safe dataset projection、seed 和
   renderer result contract 不变；
2. progress 计数只含安全整数元数据，write-once JSONL bytes/schema/record order 不因计数改变；
3. 原 Phase 7 launcher/renderer focused tests 通过。

只有上述 focused checks 通过后才显式 stage 四个允许路径中的实际改动，commit/push，并对 clean HPC2
dedicated checkout 做 `pull --ff-only`。不需要为这次窄修复重复无关的历史 Gate 或全仓 suite。

因为 launcher environment 与 renderer code 均变化，旧 preflight 不可复用。必须在 `debug` partition
为三个模型各跑一个 fresh、少于 30 分钟的 preflight；每个 preflight 使用从 frozen common manifest
按自然顺序取出的至少四个 safe identities，并覆盖至少两个 subjects，以真正经过 task-map reuse 与
subject transition。preflight 必须使用新 commit、同一正式 launcher、上述离线环境、真实 CUDA、
相同 bfloat16/batch1/producer/schema，并通过 fresh verifier。额外 must-pass：stderr 中没有
`huggingface.co`/mirror 请求、retry/fallback 网络信息；安全 progress counter 单调并闭合到 expected count。

任一模型 preflight 不通过时不得提交 formal。可在上述四个路径和同一科学合同内做低风险修复并重跑
affected preflight；若需要 prefetch、reduced projection、batch/dtype/model/data/metric 改动，则向 planning
报告，不得自行扩权。

## 5. Fresh formal relaunch 与 heartbeat

三个 preflight 全部 PASS 后，才在一个新的 sibling formal root 提交完整六 shards。保持原 common
manifest、每模型两份 contiguous shards、模型 revisions、renderer、probe、bfloat16、batch1、
`output_hidden_states=True`、one native zero-loop `use_cache=false` forward、六个 trajectory arrays 和
write-once output 不变。资源仍按原 handoff选择 normal user 合法最高优先级且显存足够的路径；A800
80GB 可继续使用，但“把 80GB 填满”不是 acceptance condition。

新六个真实 job IDs 核实后，立即创建唯一一个 executor-owned 60-minute `full` heartbeat；它只覆盖新
jobs/root。不得同时保留旧 heartbeat，也不得为 Qwen/Llama/Gemma 分建 monitor。每次 wakeup 只读检查
scheduler、safe progress counter、日志 mtime 和是否出现 offline/network violation；正常非终态保持安静。

## 6. Throughput 证据与 stop rule

修复是否成功由 time-to-result 而非显存占用率判定：

- preflight 三模型均 `<30min` 且没有外网请求；
- formal progress counter 在相邻正常观察点单调增长，能够从
  `delta completed_records / delta elapsed` 给出每 shard 的 ETA；
- projected shard completion 保持在 Slurm time limit 内；GPU 进程存在且没有 launcher/runtime error；
- 最终三模型各 1,531/57、共同 membership、finite sanitized schema、fresh verifier 和 seals 全 PASS。

如果 corrected formal 的 progress 正常，即使瞬时 `nvidia-smi` utilization 为 0 或显存仍约 9 GiB，也
继续运行，不得仅为提高表面 utilization 再改 science。若任一 shard 连续两个 60-minute heartbeat 都无
progress counter 增长，或 ETA 超过其 time limit，heartbeat 必须触发 exact executor；executor 先诊断
该 attempt 并使用现有 Gate 内低风险恢复。只有诊断证明必须采用本补充禁止的 science/numerical path、
需要新外部权限，或已无可推进的低风险路径时，才向 planning 投递真正的 `BLOCK`。

一旦六 shards、三 merge、三 fresh verifier 和三 seals 完整 PASS，删除 heartbeat、投递唯一
`GATE_C_FINAL_AUDIT` 并停止；不得增加预取、投影或其它可选优化。

## 7. 继续冻结的边界

本补充不改变 model/data/split/population/membership/renderer semantics/few-shot/probe/dtype/batch/forward/
choice surfaces/metrics/schema，也不解封 validation gold/correctness、test 或 outcome。不得运行 V3、
bootstrap、ranking、plotting、loop、generation、Gate D 或 Gate E。旧失败 root、protected dirty state、
凭证和所有已封存 evidence 均保持不变。
