# Principle

1. **Fail Fast / Errors Never Pass Silently**：不得吞掉异常、伪造默认值或让缺失工件静默降级。
2. **Fix the Cause, Not the Symptom**：先构造可复现失败，再修根因；不得用只适配单个样本或单个 run 的补丁掩盖问题。
3. **Make It Observable**：controller 决策、signal、operator body call、cache/restore、revision 和样本 join 必须可审计。
4. **Design for Traceability**：每个实验都必须能从结果追溯到 commit、manifest、模型 revision、数据 renderer、完整 argv 和 Slurm 工件。
5. **Living Documentation / Single Source of Truth**：科学契约、Gate 或执行边界变化时同步更新本文件与 `LoopPilot_全局实现规划.md`。
6. **Don't Break Mainline**：LoopPilot 只能在专用 clone 和 `looppilot` 分支开发；基础复现 checkout 永远只读。

# LoopPilot 项目执行约定

> 本文件的权威位置是 `looppilot-tflt/AGENTS.md`，适用于 LoopPilot 专用代码仓库及其所有子目录。它定义 LoopPilot 第一阶段实现、HPC2 验证和实验的强制规则。用户最新明确指令优先；若新指令改变科学契约或授权边界，先暂停并更新文档，不得边执行边解释。

## 0. 计划、状态与授权

- `LoopPilot_全局实现规划.md` 是当前全局计划和第一阶段 Gate 定义。
- `AGENTS.md` 只保存长期稳定的工程、科学和执行约束，不记录工具流水账。
- 当前没有任何 Gate 因文档存在而自动获得授权。
- 规划/审计任务负责制定 Gate、签发 handoff、只读审计和给出 `PASS / PASS_WITH_FIXES / BLOCK`。
- Gate A-E 各使用一个独立执行任务；一个执行任务只能承担一个完整 Gate，不得跨 Gate 复用或把当前 Gate 的授权解释为后续 Gate 授权。
- 执行任务命名统一为 `execute-LoopPilot 1.7B 第一阶段-Gate X`。只有前一 Gate 经规划/审计任务独立审计为 `PASS` 后，才允许创建并绑定下一 Gate 的新执行任务。
- Gate handoff 必须明确包含 Gate 名、允许路径、基础 commit、目标 branch、允许修改、禁止动作、测试、远端权限和终态回传对象。
- 每个完整 Gate handoff 必须落盘到 `.planning/looppilot_gate_x_handoff.md` 并发送给对应执行任务；计划/审计任务还必须在自己的对话中同时给出一份简洁版，概括目标、准入条件、允许动作、禁止动作、验收与终态要求，便于用户阅读。对话简洁版不替代完整 handoff，也不扩大授权。
- 规划/审计任务在 handoff 后停止轮询，只等待当前 Gate 执行任务的一次终态 `GATE_X_FINAL_AUDIT` 或 `BLOCK`。审计 `PASS` 后关闭当前 Gate 责任边界，再创建下一 Gate 的新执行任务并签发新 handoff。
- `PASS_WITH_FIXES` 或 `BLOCK` 不得创建下一 Gate 执行任务；有界修复或补充取证仍返回当前 Gate 的同一执行任务。
- 未收到规划/审计任务明确 `PASS`，不得自动进入下一 Gate。
- 历史聊天、旧报告和 memory 只作辅助；当前代码、manifest、工件和本规划才是执行证据。

## 1. 项目与路径边界

只读基础复现：

```text
local=/Users/huangxutao/Desktop/Training-free looped transformer/Training-Free Looped Transformers Reproduction
expected_base_commit=4f59bd93eca4da3cbf458a93508f91c5b23912bc
remote=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction
```

计划中的专用开发位置：

```text
local=/Users/huangxutao/Desktop/Training-free looped transformer/LoopPilot_Input-Adaptive Training-Free Looping for Frozen Transformers/looppilot-tflt
branch=looppilot
remote=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_looppilot
workspace_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot
run_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot/runs
```

强制规则：

- 写入前必须确认 `git rev-parse --show-toplevel` 精确等于专用 clone 路径。
- 只有当前分支精确为 `looppilot` 才允许写代码、提交或生成版本化配置。
- 不得在基础复现 checkout 中 fetch、switch、checkout、建 worktree、安装环境、写文件或运行 LoopPilot。
- 专用 clone 尚不存在时，必须先核对目标目录、远端 branch 和基础 commit；意外已有状态立即 `BLOCK`，不得覆盖或删除重建。
- 不得自动 merge 回 `main`，不得 rebase、force-push 或重写历史。
- 不得提交模型、数据集、完整 run、虚拟环境、cache、GPU 日志或大体积 JSONL。
- 每个 commit 单一目的，并包含对应测试或可复现证据。

每次写入、提交、推送或远程执行前至少记录：

```bash
git status --short --branch
git rev-parse --show-toplevel
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git merge-base --is-ancestor 4f59bd93eca4da3cbf458a93508f91c5b23912bc HEAD
```

## 2. 第一阶段冻结范围

第一阶段只研究：

```text
model=Qwen/Qwen3-1.7B-Base
task=mmlu
num_fewshot=5
dtype=float16
window=[12,15]
k=2
iteration_mode=block
strategy=damped_euler
alpha=1.0
beta=0.0
cache_strategy=last
decode_mode=bypass
actions={BASELINE, LOOP_K2}
```

model/tokenizer revision 必须在 Gate A 通过 authoritative resolved artifact 固定。正式 signal run 前 batch size 必须冻结为显式整数；禁止使用 `auto` 产生不可审计的批组成变化。

第一阶段禁止：

- 动态 window、其他 window 或 LoopScope 联动；
- K=3、动态 K、其他 solver；
- cache/decode/mode 消融；
- 4B、MoE、Llama、蒸馏模型和生成任务；
- 训练型 controller、tuned lens、额外参数或 checkpoint 更新；
- 使用正式 test 标签拟合 signal 权重、阈值或规则；
- 从 limit smoke 的准确率选择指标或控制器。

## 3. Baseline 与 loop 语义不可变

设 `y0 = g(x0)`：

- `BASELINE` 必须返回 `y0`，不得返回 `x0`。
- 返回 `x0` 表示跳过 `[12,15]`，是科学语义错误，必须由测试钉死。
- 第一次 `g(x0)` 是 baseline 必需调用，同时作为 probe；不得再额外调用一次相同窗口。
- `LOOP_K2` 复用 `y0-x0` 作为第一 Euler residual，总 operator body call 必须为 2。
- controller 为 `None` 时，旧 wrapper/strategy 热路径必须保持原样。
- `AlwaysLoop` 必须与 legacy K=2 对齐；`NeverLoop` 必须与未 patch baseline 对齐。
- 同一 batch 只要仍对全部 row 计算第二次 operator，就不得宣称获得真实 FLOPs 或墙钟节省。
- 任何 fallback 都必须显式记录 reason；不得在异常时静默改成 always-loop 或 baseline。

## 4. 决策与数据单位

- MMLU 决策单位是 doc，不是 choice request。
- 同一题四个 request 必须共享一个 action。
- controller 不得读取 choice continuation、gold answer 或 correctness。
- signal 只聚合题目 token；few-shot 模板、padding、special token 和 continuation 必须有显式 mask。
- doc join 使用 task/version、renderer hash、doc hash 和 occurrence id；不得依赖行号或 eval 顺序。
- batch 内数值差异导致同 doc 多个 action 时，整个 run 判为无效并 `BLOCK`。
- 需要离线 decision map 时，必须记录生成它的 commit、配置、input manifest 和 hash；不得人工改表。

## 5. 第一阶段信号边界

在线候选信号：

- `R0`：题目 token 的首步相对窗口更新；
- `C0`：第一次正常窗口前向的层更新路径一致性；
- `N0`：首步 hidden norm 漂移；
- `Q1` 与 residual cosine：第二次调用后的稳定性诊断。

离线输出指标：

- doc-level choice loglikelihood margin；
- entropy、top-1 flip、endpoint JS/KL；
- both-correct、both-wrong、loop-helps、loop-hurts。

强制规则：

- 每个 signal 必须分别落盘和报告；不得先看标签再临时组合权重。
- signal schema、token mask、聚合方法和 epsilon 必须版本化。
- raw hidden state、full logits 和 router tensors 默认不得落盘，只保存必要标量和 hash。
- logit margin/entropy 在第一阶段仅用于离线评价，不得偷偷进入在线 controller。
- `Q1` 已经支付第二次 operator 调用，只能用于稳定性分析或回退研究，不得计入“提前跳过节省计算”的证据。
- `C0` 是待验证假设，不得写成已有论文结论。

## 6. 评价与泄漏约束

每次正式分析至少报告：

- paired accuracy delta；
- n01/n10 与 McNemar；
- hurt detection PR-AUC；
- helped-vs-hurt AUROC；
- 伤害阻断率与帮助保留率；
- policy gain/coverage curve；
- 同覆盖率随机门控 null；
- subject-only、length-only、margin-only baseline；
- oracle headroom capture；
- baseline duplicate/self-flip 噪声地板；
- 分 subject、长度和 margin 的稳定性。

禁止：

- 只报告被 controller 选中子集上的准确率；
- 只报告最优 threshold；
- 从 MMLU test 反复调参后仍声称 test-set untuned；
- 把同题四个 choice 当四个独立样本；
- 把小 benchmark 或 limit smoke 的偶然 delta 当显著结论；
- 在 baseline/loop 配置、revision、renderer 或 batch 不一致时做 paired join。

## 7. 代码修改边界

优先新增：

```text
src/tflt/looppilot/
configs/looppilot/
scripts/looppilot/
tests/test_looppilot_*.py
```

允许在 Gate A handoff 明确授权后最小修改：

- `src/tflt/config.py`
- `src/tflt/strategies.py`
- `src/tflt/wrapper.py`
- `src/tflt/eval_runner.py`
- `src/tflt/cli.py`
- `src/tflt/remote.py`
- `src/tflt/audit.py`

默认禁止第一阶段修改：

- `src/tflt/cache.py`
- 模型 remote code；
- 既有 Phase 2/Phase 3 manifests；
- 原复现项目任何文件；
- LoopScope 专用 clone 和工件。

修改 `wrapper.py` 或 `strategies.py` 前必须先有覆盖 baseline/legacy-loop 语义的 red/characterization tests。一个 commit 不得同时改变信号定义、循环数值算法和 eval 调度。

## 8. 测试与可观测性

本地测试必须保持无 torch/transformers 也能运行：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

核心测试不变量：

- controller `None` 不产生新事件；
- `AlwaysLoop == run_loop(K=2)`；
- `NeverLoop == operator(x0)`；
- no-loop 返回 `y0` 而非 `x0`；
- `operator_body_calls == sum(k_used)`；
- 同 doc 四 request action 一致；
- invalid controller spec 在模型加载前失败；
- JSONL 拒绝覆盖；
- NaN/Inf signal 直接失败；
- join 缺失、重复或 revision 不闭合直接失败；
- restore 后 wrapper 不残留 patch。

不得使用 `except Exception: pass`、空 catch、自动填零或静默裁剪来让实验继续。

## 9. 本地与 HPC2 边界

本地：

- 只在 LoopPilot 目录和专用 clone 写入；
- 不下载模型/数据，不运行 GPU，不保存完整 eval 工件；
- 基础复现和 LoopScope clone 均只读；
- 不覆盖已有文件或 run 摘要。

HPC2：

```text
source_checkout=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_looppilot
read_only_reproduction=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction
workspace_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot
run_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot/runs
HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
UV_CACHE_DIR=/hpc2hdd/home/xhuang225/shared/uv
wheelhouse=/hpc2hdd/home/xhuang225/shared/wheelhouse/tflt-cu121
```

- 环境优先只读复用源复现项目 `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction/.venv`；必须从 LoopPilot checkout 运行并用 `PYTHONPATH` 指向当前代码，设置 `PYTHONDONTWRITEBYTECODE=1`，不得在复用 venv 中安装、升级、卸载或生成新文件。只有复用环境不满足冻结依赖时，才可在新 handoff 明确授权后创建 LoopPilot 专用 versioned venv。
- LoopPilot 新建的 venv、Gate 证据、staging 与正式 run 必须全部位于独立 `workspace_root` 下；不得写入或复用 `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/` 中既有项目的目录。只读复用上述源复现 venv 是唯一例外，不得把任何 LoopPilot 工件写回源复现项目。
- 每次实验使用新的 timestamped write-once run root。
- 不得删除、移动、覆盖、复用 claim 或自动重试已提交 Slurm job。
- SSH 只读探针使用 `ClearAllForwardings=yes`。
- 出现 host-key 变化、权限错误、QOS/partition 拒绝、意外目录或 dirty checkout 时立即 `BLOCK`。
- GPU/Slurm 必须在相应 Gate 明确授权后才可使用。

## 10. Provenance 与工件

每个正式 run 至少包含：

- branch、commit、dirty、base ancestor；
- model/tokenizer repo 与 resolved revision；
- task config、renderer YAML/path/hash 和 lm-eval version；
- Python/torch/transformers/tflt 版本；
- 完整 argv、环境快照和显式 batch size；
- input/doc manifest、token-mask schema 和 hash；
- controller/signal schema、criterion config 和 hash；
- command args、results、signal JSONL、join report 和中文摘要；
- operator body-call、decision reason 和 restore audit；
- Slurm job id、partition/QOS/node、ExitCode、stdout/stderr；
- baseline/loop 配对兼容性检查结果。

不得只回传目录名、截图、最终准确率或口头结论。

## 11. Gate 强制门

### Gate A：本地治理与实现

- 专用 clone/branch/base provenance 正确；
- full unit tests、CLI help、dry-run、compile 和 `git diff --check` 通过；
- baseline/legacy-loop/controller 三条语义测试完整；
- 不连接 HPC2。

### Gate B：HPC2 CPU/import

- local/origin/HPC2 commit 一致；
- 优先只读复用源复现 venv，并验证解释器、依赖锁定、当前 `tflt` 源路径和前后未修改；仅在明确授权时创建独立 venv；
- remote tests/import/compile/help 通过；
- 不运行 GPU/Slurm。

### Gate C：GPU probe/audit

- 四题真实模型 probe；
- signal/schema/revision/restore 完整；
- Always/Never 与 legacy/baseline 对齐；
- body-call 计数正确；
- 不按准确率调规则。

### Gate D：limit 工程语义

- baseline/loop/signal join 100%；
- 同 doc decision 一致；
- limit 不用于科学排名；
- 工件 write-once。

### Gate E：完整信号可行性

- 输入、规则、指标和 null 已冻结；
- baseline self-flip、paired analysis 和分层结果齐全；
- 终态仅为 `PASS_TO_ONLINE / PIVOT_TO_PHASE3 / PIVOT_TO_COARSE / BLOCK`；
- 不自动进入在线 full eval。

## 12. 有界自主修复

每个 Gate 默认最多两轮低风险自主修复：

1. 每轮先保存确定性失败证据或新增 red test；
2. 一个 commit 只修一个根因；
3. 修复后运行完整本地测试和适用远端 CPU 回归；
4. 原失败命令只能在全新 write-once staging 中重跑；
5. 原失败证据、JobID、run root 和日志必须永久保留，不得覆盖、移动或复用。

允许自行修复：不改变科学契约的 import、CLI 透传、schema adapter、路径解析、provenance 绑定和纯工程兼容问题。

小型可容许问题不需要逐项或逐轮上报规划/审计任务。执行任务应在 handoff 的自主修复预算内连续完成失败取证、根因修复、回归和 fresh write-once 重跑，并在 Gate 终态包中一次性汇总根因、commit、测试、旧/新路径和 repair/retry 计数。

GPU/Slurm job 默认不得静默重试；仅当 handoff 明确给出 replacement 预算，且失败发生在模型加载或科学 probe/forward 开始前、根因是单一可复现的纯工程启动错误时，才允许保留旧 Job/run、使用完全相同的资源和全新 run root 自主修复重提。scheduler/QOS 拒绝、OOM、超时、cache/model/data 缺失、实际科学执行失败或需要改变 partition/GPU/资源时不属于小问题。

任何 submit 调用若返回 unknown/undefined、超时、断连、工具中断或缺少即时回执，必须视为“提交状态未知”，不能视为“未提交”。执行任务必须停止新提交，检查原 root 的 `control/job_id.txt` 与 submit stdout/stderr，并按 job name/user、`sacct SubmitLine`、`scontrol Command/WorkDir/StdOut/StdErr` 和精确 run root 查询 `squeue/sacct`。发现任一引用该 root 的 JobID 即永久禁止重提；即使单次快照暂未发现 JobID，也必须先证明原远端 submit 进程已终止，再做有界多次只读确认。状态未闭合前不得创建新 run root 或调用第二次 `sbatch`。

必须立即升级并 `BLOCK`：改变 model/task/split/sample/window/K/solver/signal 定义/token mask/schema/阈值/评价标准；修改 cache；扩大远端权限或计算资源；需要破坏性操作；存在多个科学上合理方案；实际科学作业失败；handoff 未授权 replacement；两轮额度耗尽。

执行任务可以保守使用只读分析 subagent，但主执行任务独占共享工作树写入、Git、SSH/HPC2、Slurm、实验状态和最终证据。不得让多个 agent 同时修改同一 clone 或操作同一远端 run。

## 13. 终态回传与阻塞

每个 Gate 最后必须发送一次：

```text
decision requested: GATE_X_FINAL_AUDIT
result: PASS_CANDIDATE / PASS_WITH_FIXES_CANDIDATE / BLOCK
gate:
planning/audit task id:
execution task id:
branch/commit/dirty:
base ancestor:
files changed:
exact commands and exit codes:
tests:
remote host/job ids/run root:
artifact paths and hashes:
scientific invariants:
observed results:
deviations/errors:
actions explicitly not taken:
requested decision: PASS / PASS_WITH_FIXES / BLOCK
```

无法继续时不得静默结束。工具或 usage limit、审批拒绝、SSH/HPC2/Slurm 不可用、host-key/权限失败、Git/path 不匹配、意外已有文件、scheduler 拒绝、job 失败、工件缺失或科学边界不明确，均必须回传结构化 `BLOCK`。若跨任务消息工具不可用，在执行任务 final response 输出同样的完整包，供用户手动转发。
