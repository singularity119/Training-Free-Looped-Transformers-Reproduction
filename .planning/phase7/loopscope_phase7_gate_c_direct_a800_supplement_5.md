# GATE_C_DIRECT_A800_SUPPLEMENT_5

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=C
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a0036f-fc1d-7412-b043-b93b5f088398
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-Full-Acquisition-第7阶段-Gate C
SUPPLEMENT_STATE=AUTHORIZED
USER_OVERRIDE=SKIP_A40_PREFLIGHT_DIRECT_A800_PACKED_FORMAL
SUPERSEDES_ONLY=SUPPLEMENT_4_A40_PREFLIGHT_AND_PREFLIGHT_GATED_FORMAL
FROZEN_COMMIT=bf37259a71ef4220236f51e5b659141748653f0b
PACKING=3_INDEPENDENT_BATCH1_PROCESSES_PER_A800
SCIENCE_CHANGE=NONE
GATE_D_AUTHORITY=NONE
GATE_E_AUTHORITY=NONE
TERMINAL_EVENT=GATE_C_FINAL_AUDIT_OR_BLOCK
```

## 1. 用户最新指令

用户明确纠正 Supplement 4 的 A40 preflight 安排：“不用，直接上 A800”。按 authority precedence，
本补充取消 A40/debug packed preflight 及其数值对照、显存阈值和 preflight-gated submission；executor
直接在 A800 上提交正式 packed attempt。Supplement 4 其余固定三进程 mapping、child 隔离、资源、
offline 环境、信息屏障、fallback、heartbeat 与 science 禁令继续有效。

若 A40 preflight 尚未提交，不得再提交。若因消息交叉已经创建 A40 job，保存 job ID/状态后精确取消，
保留其 root，不把它作为 admission evidence；随后直接进入 A800 formal。

## 2. Direct A800 formal

在 Supplement 4 所述旧六个 zero-output jobs terminal、旧 heartbeat 删除后，直接创建 fresh sibling formal
root 并提交两个 packed A800 jobs：

```text
packed-job-00=qwen25_3b/shard-00+llama32_3b/shard-00+gemma2_2b/shard-00
packed-job-01=qwen25_3b/shard-01+llama32_3b/shard-01+gemma2_2b/shard-01
per_job=1*A800,24CPU,192G,12:00:00
children_per_gpu=3 independent Python processes
batch_size_per_child=1
```

不需要新 code commit。提交前只做不占 GPU 的最小 launcher sanity：batch script shell syntax 可解析、
三个 child command 都绑定 exact frozen commit/model/revision/manifest、六个 output 与 stdout/stderr path 全部
互异且 fresh、parent 能分别保留三个 PID/exit code、offline variables 完整。检查通过后立即提交，不增加
模拟 preflight、A40、额外 renderer run 或数值对照。

## 3. Formal startup containment

两个真实 job IDs 返回后立即创建唯一一个 60-minute full heartbeat，覆盖两个 parents、六个 child
progress/log streams 和 fresh root。每个 packed job 首次进入 RUNNING 后，executor 可做一次有界 startup
check，确认：三个 child process 均存在、GPU 上没有 OOM、日志没有网络访问/output collision、progress
开始增长。该检查属于正式 attempt 观察，不是 preflight，也不阻止 job 继续运行。

如果 packed formal 出现 OOM、child 非零退出、输出冲突或进度停滞，保留 attempt 与已通过 fresh verifier
的 child shard，只把 missing/invalid shards 回退到 Supplement 3 的 one-process-per-GPU corrected fresh
paths；不得改变 concurrency 数、batch、dtype、projection 或 science。Attempt 失败仍不是自动 Gate-level
`BLOCK`。

## 4. 继续冻结

继续禁止 `batch_size>1`、共享 model/tensor state、CUDA MPS、CPU prefetch、reduced projection、boundary
stacking、model/data/split/membership/renderer/probe/dtype/metric/schema 变化、V3/ranking/plotting、validation
gold/correctness、test/outcome、loop/generation、Gate D 与 Gate E。Gate C 仅在六 shards、三 merge、三
fresh verifier 和三 seals 完整通过后投递唯一 `GATE_C_FINAL_AUDIT`。
