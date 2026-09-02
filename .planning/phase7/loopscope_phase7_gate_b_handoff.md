# GATE_B_HANDOFF

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=B
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a001c1-3ff9-7370-abbb-b32128897a08
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-Debug-Smoke-第7阶段-Gate B
AUTHORIZATION_STATE=AUTHORIZED
TERMINAL_EVENT=GATE_B_FINAL_AUDIT_OR_BLOCK
NEXT_GATE_AUTHORITY=NONE
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
```

本 handoff 的协作与生命周期必须完整遵守 `research-gate-orchestrator` 的 `SKILL.md` 和
`references/protocol.md`：一 Gate 一独立 executor、最多三个有界 subagents、executor 独占 Git/
remote-write/GPU/Slurm/terminal 权限、长任务只使用一个匹配 cadence 的 heartbeat，并且退出前必须向
planning 线程投递且确认唯一 `GATE_B_FINAL_AUDIT` 或真正的 `BLOCK`；无法确认时必须使用
`TERMINAL_DELIVERY_UNCONFIRMED` fallback，不得静默结束或跨 Gate。

## 1. Objective 与 stop condition

Gate B 只回答：三个冻结 Base 模型能否在 HPC2 真实 CUDA 路径上，用同一标准 MMLU 5-shot
renderer、同一 probe 语义和同一 sanitized producer，对每模型四个固定 validation identities
各产生完整、有限、可由 fresh process 验证的逐层 scalar trajectory，并在每个 debug job 中
于 30 分钟内完成。

三模型均通过后发送一个 `GATE_B_FINAL_AUDIT` 并停止；任一模型存在材料 blocker 且 Gate 内正常
低风险修复无法消除时，发送一个窄 `BLOCK` 并停止。不得自行创建或进入 Gate C。

## 2. Authoritative sources 与执行协议

执行前完整重读并遵守：

- `research-gate-orchestrator` 的 `SKILL.md` 与 `references/protocol.md`；
- `hpc2-hkustgz-ssh` skill；
- `loopscope-tflt/AGENTS.md` 第 16 节；
- `.planning/phase7/loopscope_phase7_control.md`；
- `.planning/phase7/loopscope_phase7_scientific_contract.md`；
- `.planning/phase7/loopscope_phase7_plan.md`；
- `loopscope-tflt/docs/loopscope_phase7.md`；
- 本 handoff。

优先级为用户最新指令 > live Git/HPC fact > `AGENTS.md` > control > 本 handoff/plan/runbook >
历史。executor ID、Gate、权限或 base 不一致时，在 mutation 前 `BLOCK`；不得自行调和。

## 3. Starting provenance 与 protected state

```text
local repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
required branch=loopscope
starting HEAD=37241f6fb52aa6ee89a5b4e24ac607f7312e77ac
required ancestry base=4f59bd93eca4da3cbf458a93508f91c5b23912bc
remote host=hpc2-hkustgz
remote checkout=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
remote python=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python
HF cache=/hpc2hdd/home/xhuang225/shared/hf_home/hub
HF token path=/hpc2hdd/home/xhuang225/.cache/huggingface/phase7_read_token
```

Gate A 的 Phase 7 工作集当前为 untracked、未提交、已通过 17 个 focused tests；staged membership
为空。下列 tracked dirty paths 是 planning/前序任务的受保护状态：

```text
AGENTS.md
docs/loopscope_phase2.md
docs/loopscope_phase4.md
docs/loopscope_phase5.md
docs/loopscope_phase6.md
scripts/loopscope/run_qwen4_phase4_p4b.py
src/tflt/loopscope/phase4_outcome.py
```

不得 reset、checkout、stash、clean、覆盖、暂存或提交这些路径。不得 broad `git add`。所有 Git
integration 由 bound executor 本人完成；subagent 不得执行 Git integration。

## 4. Frozen science

```text
models=Qwen/Qwen2.5-3B;meta-llama/Llama-3.2-3B;google/gemma-2-2b
revisions=3aab1f1954e9cc14eb9509a215f9e5ca08227a9b;13afe5124825b4f3751f836b40dafda64c1ed062;c5ebcd40d208330abc697524c919956e692655cf
layer_counts=36;28;26
dataset=cais/mmlu
split=validation
num_fewshot=5
fewshot_split=dev
renderer=lm-eval 0.4.11 compatible standard plain non-chat MMLU
probe=full rendered current-query Answer: prefix final non-padding token before continuation
runtime=bfloat16,batch_size=1,no quantization,no CPU offload
forward=one native zero-loop use_cache=false per identity
choice_surfaces=" A";" B";" C";" D"
```

Raw boundaries、FinalNorm capture、choice distribution 和六类 scalar 指标严格按 scientific
contract。不得运行 loop、generation、test split、selector/V3、bootstrap 或 outcome；不得读取
validation target/gold/label/correctness。标准 renderer 只可在内存中读取四个 smoke questions/
choices 和 dev demonstrations（含 demonstration answers），prompt、input/token IDs 与完整 logits/
probabilities/hidden tensors 均不得持久化。

## 5. Authorized actions

### 5.1 Focused integration 与最小工程修复

允许在以下 Phase 7 专用范围内完成现有 Gate A 工作集的 focused integration，并在真实路径暴露
材料问题时作最小修复：

```text
configs/loopscope/phase7_*
docs/loopscope_phase7.md
scripts/loopscope/run_phase7_*.py
scripts/loopscope/verify_phase7_*.py
src/tflt/loopscope/phase7_*.py
tests/test_phase7_*.py
```

先运行 focused tests、CLI help 与 `git diff --check`；仅显式暂存上述 Phase 7 files，核对 staged
membership 后作一个 purpose-specific commit。只有 origin ancestry 支持普通 fast-forward 时才 push
到 `origin/loopscope`；禁止 merge、rebase、force push、改 main 或提交 planning/protected files。
远端 dedicated checkout 只允许在 clean、同 branch、同 ancestry 条件下 `git pull --ff-only` 到该
commit。若 remote dirty 或不能 fast-forward，停止并报告，不得清理/覆盖。

允许实现/修复标准 renderer entrypoint、安全 manifest builder、Slurm launcher 或 Gate B verifier
辅助脚本，但必须留在上述 Phase 7 范围、保持 science 不变并通过相应 targeted tests。任何
decision-critical launcher/runtime/hook/serialization/producer 修改后，必须产生新的 focused commit、
远端 fast-forward，并只对受影响模型使用 fresh debug root 重跑。

### 5.2 HPC2、network、GPU 与 job control

所有 SSH 使用 `BatchMode=yes`、有界 `ConnectTimeout` 和 `ClearAllForwardings=yes`。允许：

- 有界读取 repo/env/cache/dataset/Slurm 状态；
- 通过 `HF_TOKEN_PATH` 与每命令 `HF_ENDPOINT=https://hf-mirror.com`，将三个 exact revisions 的
  必需模型权重与 index/shards 下载到上述共享 cache；
- 在 remote project 内创建 Gate B launcher/log，以及在 fresh write-once Gate B run roots 中写入
  manifest、sanitized records 与 verifier receipts；
- 在 `debug` partition 提交、观察、取消本 Gate jobs，并进行最小 Slurm/resource 调整；
- 低风险、信息不解封的 fresh retry，保留失败 root/log，不覆盖旧工件。

绝不读取、打印、复制、修改或删除 token 文件/内容；不得 `hf auth login`、持久化 `HF_ENDPOINT`、
安装/升级环境、使用非冻结模型或 alternate repo。模型权重下载可在 login/mgmt 节点作为纯数据传输
完成，但模型构造和 forward 只能在 Slurm `debug` GPU allocation 中进行。

### 5.3 Resource policy 与 heartbeat

先只读检查当前合法 `debug` GPU 类型、显存、queue 和预计启动时间。选择能容纳该模型 bfloat16、
完整 hidden-state trajectory 和最长 smoke prompt 的最小充分 GPU；以显存适配、当前可用性、排队
估计和 time-to-result 为依据，不预锁卡型，不使用 privileged QoS。每个 debug job wall time 必须
`<30min`。

三模型可并行或顺序运行，但 root/job/log/records/receipt 必须按模型分离。若 job 未稳定 RUNNING
约 60 秒便结束，不创建 heartbeat；若仍需等待，只创建一个 executor-owned 10-minute smoke
heartbeat 覆盖整个 Gate B job set。heartbeat 只观察并唤醒 exact executor，不拥有 Gate continuation，
终态后立即停用。

### 5.4 Bounded subagents

可按 skill 在确实缩短 Gate 时同时使用最多三个 subagents，任务只能是当前 Gate 内的 narrow code
inspection、targeted test 或授权路径内 read-only evidence extraction。subagent 不得独立执行 Git
integration、remote write、weight download、GPU/Slurm/job control、跨 Gate 或 terminal delivery。
bound executor 亲自整合、运行 external actions 并发送唯一终包。

## 6. Exact smoke membership

只建立一个跨模型共用的四 identity manifest。按 canonical MMLU validation natural order，基于
`canonical_identity` 与 `subject` 两个安全字段确定：从自然顺序最先出现的两个不同 subjects 中，
各取前两个 identities；若某 subject 不足两个则继续到下一个满足者。选择必须在任何 model forward
前冻结，三模型完全复用，且 manifest 只持久化 `canonical_identity`、`subject` 和 renderer 定位所需
的非敏感索引；不得读取或保存 validation gold/target。若 live lm-eval identity 语义要求不同的
等价定位字段，先用最小代码检查闭合并在终包说明，不得凭 batch 顺序猜测。

## 7. Shortest execution path

1. **Authority/provenance check**：重读权威文件，确认 exact executor、branch/HEAD/dirty/staged、
   remote checkout、env 和三模型 admission。
2. **Integrate once**：补齐必要 renderer/manifest/Slurm glue，运行 focused tests；显式 commit/push
   Phase 7 paths，remote `--ff-only` 同步并冻结 exact smoke commit。
3. **Acquire exact weights**：existing-cache first；缺失时走 token-backed mirror，仅下载三个 exact
   revisions 的运行必需文件。记录可读 snapshot path、来源和文件类型，不输出凭证。
4. **One-identity preflight**：对每模型先在 fresh debug root 用共同 manifest 的第一条 identity
   执行一次真实 CUDA producer，再由 fresh process 以 expected-count=1 验证；必须 `<30min`。
5. **Four-identity smoke**：预检通过后，对每模型在新的 write-once root 跑完整四 identities，fresh
   verifier 使用 expected-count=4、expected-subjects=2。不得把 one-identity root 原地扩写。
6. **Stop and report**：三模型 receipts 均 PASS 后发送终包；不执行 Gate C。

## 8. Must-pass per model

```text
record_count=4
unique_identity_count=4
subject_count=2
forward_count=4
loop_insertions=0
probe=last non-padding token of rendered current-query Answer: prefix
boundary_count=L+1
transition_count=L
final_norm_prehook_calls_per_forward=1
FinalNorm(raw B_L) closes native final hidden/logits path
choice surfaces remain single-token and distinct
four choice logits finite; reduced probabilities sum to one
all six metric arrays finite with exact lengths
forbidden persisted fields absent
fresh_process_verifier=PASS
elapsed<30 minutes per debug job
```

一个模型失败不会授权缩小三模型 membership。保留已经通过且 decision-critical path 未变化的其他
模型证据；只有受影响模型需要 fresh rerun。三模型 4/4 全部通过才可请求 Gate C admission。

## 9. Agility budget

```text
earliest real experiment=one-identity CUDA producer+fresh verifier per model after focused integration
minimum decisive checks=CUDA/device binding, raw B_L closure, exact scalar/schema lengths, fresh verifier
deferred=performance tuning,general framework,extra models,extra prompts,robustness matrix,formal acquisition,V3,plots
deletion test=remove any check that cannot change Gate C admission or interpretation of the three true-CUDA paths
```

正常路径失败时可在本 Gate 内持续做低风险工程修复与 fresh retry；不要为 speculative attacker、
digest/receipt tree 或已通过 Gate 的重复全审计延迟 smoke。若需要改科学合同、扩大样本/模型矩阵、
读取被禁信息、进行 destructive action、改变凭证或跨 Gate，立即 `BLOCK` 请求 planning 决定。

## 10. Terminal delivery

向 planning task `01a0013e-71c3-7c90-a547-4059b462dc7e` 发送一次结构化
`GATE_B_FINAL_AUDIT` 或 `BLOCK`，至少包含：final branch/commit/dirty/staged、Phase 7 commit/push/
remote sync、exact model snapshots、四 identity membership/subjects、Slurm job IDs/resources/runtime、
每模型 run root/record/receipt、exact commands 与 exit codes、repair/retry、信息屏障、protected-state
确认，以及明确请求 `PASS` 或 `BLOCK`。不得包含 token、prompt、token IDs、gold/outcome 或 digest。

发送后立即停止。若无法确认投递，输出 `TERMINAL_DELIVERY_UNCONFIRMED` paste-ready packet；不得轮询
planning 或自行进入 Gate C。
