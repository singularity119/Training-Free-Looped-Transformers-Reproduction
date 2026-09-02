# GATE_C_CPU_PER_GPU_SUPPLEMENT_6

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=C
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a0036f-fc1d-7412-b043-b93b5f088398
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-Full-Acquisition-第7阶段-Gate C
SUPPLEMENT_STATE=AUTHORIZED
SUPERSEDES_ONLY=SUPPLEMENT_4_AND_5_PACKED_JOB_CPU_REQUEST
FROZEN_COMMIT=bf37259a71ef4220236f51e5b659141748653f0b
PACKING=3_INDEPENDENT_BATCH1_PROCESSES_PER_A800
PACKED_JOB_CPUS=8_TOTAL
SCIENCE_CHANGE=NONE
GATE_D_AUTHORITY=NONE
GATE_E_AUTHORITY=NONE
TERMINAL_EVENT=GATE_C_FINAL_AUDIT_OR_BLOCK
```

## 1. 已核实的普通调度冲突

Planning 对 executor 报告的 evidence 做了有界只读核实：

```text
rejected_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase7-gate-c-20260815T080000Z-packed-formal-recovery5
requested=1*A800,24CPU,192G,12h
result=REJECTED_BEFORE_JOB_CREATION
scheduler_rule=CPU_PER_GPU_MAX_8
job_ids=NONE
heartbeat=NONE
second_submission=NOT_ATTEMPTED
```

该 attempt 没有获得 GPU、没有运行 producer、没有生成 output，不是 Gate-level blocker。保留 recovery5
root、launcher sanity 与 `submission_rejection.txt`；不得在 recovery5 内重试或覆盖。

## 2. 唯一资源修正

Supplement 5 的 direct A800、固定三 child mapping、192G、12h 和所有 science/runtime contract 不变。
仅把每个 packed Slurm job 的 CPU request 从 24 改为 scheduler 允许的最大值 8：

```text
packed-job-00=qwen25_3b/shard-00+llama32_3b/shard-00+gemma2_2b/shard-00
packed-job-01=qwen25_3b/shard-01+llama32_3b/shard-01+gemma2_2b/shard-01
per_job=1*A800,8CPU-total,192G,12:00:00
children_per_gpu=3 independent Python processes
batch_size_per_child=1
```

每个 child command 必须设置：

```text
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
RAYON_NUM_THREADS=2
```

三个 child 最多使用六个 CPU threads，剩余两个 cores 留给 parent supervision、Python/runtime 和 I/O。
不得把 `8 CPU` 解释成每个 child 8 cores，也不得通过额外 Slurm step、第二块 GPU、超配线程或另一个
parent job 绕过 CPU/GPU 限制。线程上限只控制 CPU oversubscription，不改变 tokenizer 内容、renderer、
CUDA BF16 数值路径或 record。

## 3. Fresh retry

创建新的 sibling root（建议标记 `packed-formal-recovery6`），重新生成两个 parent batch scripts。复用
Supplement 5 的最小 non-GPU sanity，但把 must-pass 更新为：

- 每个 parent 精确请求 `1*A800,8CPU,192G,12h`；
- 三个 child 的上述三个 thread-cap variables 全部为 2；
- exact frozen model/revision/manifest/offline bindings 不变；
- 六个 child output/stdout/stderr paths 全部 fresh、互异；
- parent 分别保存三个 PID、三个 exit status，不能掩盖 child failure。

Sanity PASS 后直接提交两个 packed jobs；不运行 A40/debug preflight，不修改 repo code，不产生新 commit。
若第一个 submission 仍在 job creation 前被不同 resource policy 拒绝，保存 evidence 并向 planning 报告
新的 exact rule，不自行继续改 memory/GPU/concurrency。第一个成功获得 job ID 后继续提交第二个；若第二个
被拒，保留第一个正常排队/运行，不自动取消，向 planning 报告缺失的 exact resource permission。

## 4. Heartbeat 与 fallback

只有两个真实 job IDs 都存在后，立即创建唯一一个 executor-owned 60-minute full heartbeat，覆盖两个
parents、六个 child progress/log streams 和 fresh root；在此之前 heartbeat 保持不存在。Job 首次
RUNNING 后按 Supplement 5 做一次有界 startup check。

运行期 OOM、child failure、output collision 或停滞仍按 Supplement 5：保留 attempt 和已由 fresh verifier
通过的 child shards，只把 missing/invalid shards 用 fresh paths 回退 one-process-per-GPU corrected path。
调度 submission rejection 或单个 child failure 都不是自动 Gate-level `BLOCK`。

## 5. 不变边界

继续冻结 `batch_size=1`、三个独立 process/A800、model/revision、data/split/membership、renderer、probe、
BF16、metrics、schema、offline environment、write-once 和信息屏障。继续禁止共享 model/tensors、CUDA MPS、
prefetch、reduced projection、boundary stacking、V3/ranking/plotting、validation gold/correctness、test/outcome、
loop/generation、Gate D 与 Gate E。最终仍需六 shards、三 merge、三 fresh verifiers、三 seals 完整闭合并
向 planning 确认投递唯一 `GATE_C_FINAL_AUDIT`。
