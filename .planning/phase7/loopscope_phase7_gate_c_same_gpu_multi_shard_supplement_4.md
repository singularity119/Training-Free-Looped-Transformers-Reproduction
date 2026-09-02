# GATE_C_SAME_GPU_MULTI_SHARD_SUPPLEMENT_4

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=C
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a0036f-fc1d-7412-b043-b93b5f088398
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-Full-Acquisition-第7阶段-Gate C
SUPPLEMENT_STATE=AUTHORIZED
SUPERSEDES_ONLY=SUPPLEMENT_3_MULTI_PROCESS_SAME_GPU_PROHIBITION
CURRENT_ATTEMPT_DECISION=EVIDENCE_PRESERVING_CANCEL_AND_FRESH_PACKED_RELAUNCH
FROZEN_COMMIT=bf37259a71ef4220236f51e5b659141748653f0b
PACKING=3_INDEPENDENT_BATCH1_PROCESSES_PER_GPU
SCIENCE_CHANGE=NONE
BATCH_SIZE_GT_1_AUTHORITY=NONE
GATE_D_AUTHORITY=NONE
GATE_E_AUTHORITY=NONE
TERMINAL_EVENT=GATE_C_FINAL_AUDIT_OR_BLOCK
```

## 1. 协作协议与本补充边界

Executor 必须完整遵守
`/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md` 及其
`references/protocol.md`。本补充仅推翻
`.planning/phase7/loopscope_phase7_gate_c_resource_utilization_supplement_3.md` 中“同一 GPU 不得运行
多个 producer process”的一项限制；Supplement 3 的离线环境、task-map cache、安全 progress、信息
屏障、write-once、preflight、heartbeat、terminal delivery 与所有 scientific 禁令继续有效。

同一 Gate executor 仍是 Git、远端写入、Slurm/job control、heartbeat lifecycle、结果集成和唯一终态
投递的责任人。问题必须报告 planning thread `01a0013e-71c3-7c90-a547-4059b462dc7e`，不能静默终止、
向用户索取普通 operational permission 或进入 Gate D/E。

## 2. Live disposition：取消当前六个零进度 jobs

Planning 于 2026-08-15 有界只读确认：

```text
commit=bf37259a71ef4220236f51e5b659141748653f0b
run_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase7-gate-c-20260815T070000Z-formal-recovery3
jobs=10220717;10220718;10220719;10220720;10220721;10220722
states=PENDING/Priority for all six
elapsed=0 for all six
assigned_nodes=none
trajectory_outputs=none
offline_env=present in all six batch scripts
heartbeat=loopscope-phase-7-gate-c-corrected-full-jobs-10220717-10220722
```

不存在已开始或已完成的 valid shard。授权 executor：

1. 保存简洁的 `squeue/sacct`、batch-script paths 和 zero-output 事实；
2. 取消六个 jobs 并确认 scheduler terminal closure；
3. 原 recovery3 root 整体保留，不移动、不覆盖、不复用为新 formal root；
4. 六个 jobs terminal 后删除旧 heartbeat，并保留 lifecycle closure；
5. 在旧 heartbeat 删除前不得创建 preflight/formal 新 heartbeat。

## 3. 冻结 packed runtime

### 3.1 固定 concurrency 与 shard mapping

固定为 `3 processes / GPU`，不授权动态探测或“尽可能多”。正式运行只提交两个 Slurm jobs：

```text
packed-job-00 = qwen25_3b/shard-00 + llama32_3b/shard-00 + gemma2_2b/shard-00
packed-job-01 = qwen25_3b/shard-01 + llama32_3b/shard-01 + gemma2_2b/shard-01
```

每个 child 都是独立 Python process、独立模型实例、独立 CUDA context、独立 manifest、独立 stdout、
stderr、write-once output 和 verifier input。三个 child 只共享同一个 Slurm 分配的物理 GPU；不得共享
model/tensor/cache state，不得做 cross-process batching、IPC tensor、CUDA MPS、boundary stacking 或结果
合并计算。每个 child 内仍是严格 `batch_size=1`、一次 identity 一个原生 forward。

Parent batch script 必须记录三个 PID，分别 `wait` 并保留每个 exit code；一个 child 失败不能掩盖另外
两个 child 的状态，也不能把 parent 报为成功。允许其余 child 完成以保留有效 write-once shard。

### 3.2 资源 envelope

Planning 只读确认 `emergency_gpu` 的 A800 nodes 每台为 64 CPU、约 1 TB RAM、8×A800，partition 对
normal user 开放且没有更小的 per-node CPU/memory cap。正式 packed job 固定请求：

```text
partition/QoS=normal-user legally available highest-priority path chosen under the existing handoff
gres=gpu:a800:1
cpus-per-task=24
mem=192G
time=12:00:00
processes_per_gpu=3
cpu_budget_per_child=8 threads maximum
```

可用 `OMP_NUM_THREADS=8`/`MKL_NUM_THREADS=8` 控制三个 child 的 CPU oversubscription，但不得改变
tokenizer、renderer 或 GPU 数值计算。不得申请第二块 GPU 给同一个 packed job，也不得把六个 child
塞进一张卡。

## 4. Fresh same-GPU concurrency preflight

取消旧 jobs 并关闭旧 heartbeat 后，必须先在 `debug` partition 提交一个 fresh packed preflight：

```text
gpu=1*A40-48GB
cpus=24
mem=192G
time<=00:29:00
concurrent_children=3
children=qwen25_3b,llama32_3b,gemma2_2b
rows_per_child=4 safe identities covering >=2 subjects
commit=bf37259a71ef4220236f51e5b659141748653f0b
```

三个 child 必须同时运行在同一张 A40，使用 formal 将采用的同一 parent supervision、三进程启动方式、
离线环境、producer、renderer、进度日志、per-child output/log layout 和 verifier。无需代码改动或新 commit；
packed launcher 作为 fresh run-root 内的明确 batch script 保存。若实现必须修改 repo code，先停下向
planning 报告，不能自行改变 frozen commit。

并发 preflight 必须与 corrected serial path 做数值对照：优先复用相同 commit、相同四 identities、同为
A40 的现有 serial corrected preflight；若没有 exact matching baseline，则在同一 debug allocation 内先
顺序运行三个 fresh serial controls，再启动 concurrent triple。对照要求 record order、schema、membership
和六个 scalar arrays 逐字段相同；只允许日志路径、wall time、PID/GPU telemetry 这类非 record 元数据
不同。任一数值漂移都视为 packed path 不成立，直接走第 7 节 serial fallback，不用设计 tolerance repair。

Packed preflight must-pass：

- 三个 child exit 0，parent exit 0，三个 fresh verifier 全 PASS；
- 无 OOM、process kill、output/log collision、网络访问/retry 或信息屏障违反；
- progress counter 对每个 child 单调闭合到 4；
- A40 峰值总显存 `<42 GiB`，保留至少约 6 GiB representative headroom；
- 从 parent start 到最慢 child 完成的保守线性外推，对 766-row shard 的 ETA `<10h`，为 12h limit
  保留至少 2h margin；
- debug 总 elapsed `<30min`。

短 job 若在约一分钟内完成无需 heartbeat；若稳定运行超过约一分钟，按协议只创建一个 10-minute
smoke heartbeat，并在 preflight terminal 后删除，formal 前不得遗留。

## 5. Fresh packed formal relaunch

只有 packed preflight 全部 must-pass 后，才创建一个新的 sibling formal root（不得复用 recovery3），
提交本补充第 3 节的两个 packed jobs。每个 child 继续显式设置 Supplement 3 的：

```text
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
PYTHONPATH=src
```

两个真实 formal job IDs 核实后立即创建唯一一个 executor-owned 60-minute `full` heartbeat。Heartbeat
必须同时覆盖两个 parent jobs、六个 child progress/log streams 和 fresh root；不得为每个 child 或模型
另建 monitor。每次 wakeup 只做一次有界只读检查，正常非终态保持安静。

## 6. 科学合同与信息屏障不变

每个 child 必须保留 exact model/revision、原 contiguous shard、标准 MMLU 5-shot renderer、final
non-padding `Answer:` probe、BF16、`output_hidden_states=True`、`use_cache=false`、one native zero-loop
forward per identity、FinalNorm/native logit closure、六个 trajectory arrays、sanitized schema 和 write-once
serialization。Concurrency 只改变 scheduler packing，不改变任一 record 的计算。

继续禁止：`batch_size>1`、CPU prefetch、reduced four-choice projection、boundary stacking、dtype/quantization/
offload、model/data/split/membership/renderer/probe/metric/schema 变化、validation gold/correctness、test/outcome、
V3/bootstrap/ranking/plotting、loop/generation、Gate D 和 Gate E。

## 7. Stop rule 与已授权 fallback

若 packed preflight 出现 OOM、峰值显存达到 42 GiB、任一 child/verification/数值对照失败、网络访问、
output collision、ETA `>=10h` 或 debug 超过 30 分钟，则不提交 packed formal，也不继续调高/调低 concurrency。
直接保留 preflight evidence，回退 Supplement 3 的 corrected `1 process / 1 GPU` 路径：使用同一 frozen
commit、既有 passing serial preflight 和新的 sibling formal root，重新提交六个独立 jobs，并建立唯一
的新 60-minute full heartbeat。该 fallback 已获授权，不需要再次向 planning 申请。

Packed formal 启动后，如果 child 连续两个正常 heartbeat 没有 progress、出现 OOM/非零退出，或 ETA
超过 12h limit，executor 先保留 attempt。已完成且 fresh verifier PASS 的 child shard可以保留；只对
missing/invalid shards 用 fresh paths 回退 one-process-per-GPU corrected path。不得因一个 child 失败覆盖或
重跑其它已验证 shard。若问题要求改变本补充冻结的 science/numerical path 或已无低风险路径，才投递
真正的 `BLOCK`。

六 shards 完成后仍按原 Gate C handoff merge 三模型、检查共同 1,531/57 membership、运行 fresh
verifiers 并 seals。全部闭合后删除 heartbeat，向 planning thread 投递并确认唯一
`GATE_C_FINAL_AUDIT`，然后停止。
