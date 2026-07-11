# Principle
1. Fail Fast / Errors Never Pass Silently：不要在代码里藏兜底逻辑来吞掉错误、隐藏问题。出了问题就应该让它爆出来，否则你永远找不到真实问题。
2. Fix the Cause, Not the Symptom / Don't Paper Over Bugs：当一个问题出现时，不要用各种 small fix、针对性补丁来掩盖它。必须定位真实根因，彻底修复。在 bug 上糊纸只会让系统积累你不知道的危险暗病。
3. Make It Observable：即使问题很难定位，也绝不要偷懒做表面修复。应该给项目增加充分的日志和可观测性，保证下次问题再现时你有足够信息去定位。问题无法修复时，只需要诚实告诉我信息不足、需新增日志，不要假装修好了。
4. Design for Debugging / Traceability：始终注意在关键路径上给自己留足排查日志，确保每一个关键节点都是可追溯的。
5. Living Documentation / Single Source of Truth：当项目关键技术栈或产品方向发生变更时，同步更新 AGENTS.md。文档必须随代码一起演进，不能让它变成过时的谎言。
6. Don't Break Mainline：大规模重构或实验性改动前，必须先切新分支。

# LoopScope 项目操作约定

> 本文件只适用于 LoopScope 专用新 clone 的 `loopscope` 分支。它定义第一阶段 Qwen3-1.7B LoopScope 增量实现、HPC2 验证和实验的强制边界。原有本地/HPC2 固定复现 checkout 均为只读参照。发现本文件与用户最新明确指令冲突时，以用户最新指令为准并先暂停报告。

## 0. 计划与信息源

- `AGENTS.md` 只保存长期稳定的项目边界、工程规则和 Gate 验收标准，不记录临时进度、当前执行线程或逐次工具日志。
- `../.planning/loopscope_phase1_control.md` 是第一阶段唯一的可变全局计划与控制文件；当前 Gate、Gate 决策、线程分配、已审计 commit、下一步授权条件均以它为准。
- 旧工作区 `.planning/` 包、历史 handoff、线程聊天和 Codex memory 只作历史证据或辅助回忆；如果与当前代码或控制文件冲突，不得据此覆盖当前事实。
- 单线程内的临时步骤优先使用 Codex 原生 plan/goal，不把每一步复制到项目文件。
- 不得自动调用 `planning-with-files` skill。只有用户在当前请求中明确点名或明确要求启用该工作流时才可使用。
- 仅在 Gate 决策、执行线程变更、审计 commit 变更、实验配方变更或授权边界变更时更新控制文件；不得按每次读取、命令或工具调用追加流水账。

## 1. 项目定位

- 基础复现只读基点：`4f59bd93eca4da3cbf458a93508f91c5b23912bc`。
- 开发分支：`loopscope`。
- 本地专用 clone：`/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt`。
- HPC2 专用 clone：`/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`。
- 第一阶段模型：`Qwen/Qwen3-1.7B-Base`。
- 第一阶段任务：`mmlu`，5-shot。
- 第一阶段目标：验证免训练内部信号能否预测不同 loop window 的真实收益；不提前宣称已经得到通用自动选窗器。
- 基础复现分支 `main` 是固定对照，不接受 LoopScope 功能提交。
- 新 clone 的默认 `main` 也只作只读引用；clone 后首次工作分支必须是从固定基点新建的 `loopscope`。

## 2. Git 与分支保护

每次写文件、提交、推送和远程运行前必须记录：

```bash
git status --short --branch
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git merge-base --is-ancestor 4f59bd93eca4da3cbf458a93508f91c5b23912bc HEAD
```

强制规则：

- 只有当前分支精确为 `loopscope` 时才允许写代码。
- 写入前 `git rev-parse --show-toplevel` 必须等于当前机器约定的 LoopScope 专用 clone；路径不符立即停止。
- 不得在 `main` 上创建、修改或提交 LoopScope 文件。
- 不得在原复现 checkout 中 fetch、切分支、建 worktree、写文件、安装环境或执行 LoopScope 命令。
- 不得 rebase、force-push、重写或删除 `main` 历史。
- 不得把 `loopscope` 自动 merge 回 `main`。
- 如果 `loopscope` 已存在，先核对其祖先、远端跟踪和 worktree；不得删除后重建来规避冲突。
- 初始化本轮新 clone 时，如果远端已意外存在 `origin/loopscope`，必须停止并回传其 provenance；不得静默跟踪、覆盖或换名规避审计。
- 每个提交必须单一目的、测试通过、可独立审计。
- 禁止提交模型、数据集、完整运行日志、run 目录、缓存、虚拟环境或大体积生成工件。

## 3. 第一阶段固定实验配方

除 window 位置外，以下配置冻结：

```text
model=qwen3-1.7b-base
task=mmlu
num_fewshot=5
dtype=float16
k=2
iteration_mode=block
strategy=damped_euler
alpha=1.0
beta=0.0
cache_strategy=last
decode_mode=bypass
window_width=4
```

禁止在第一阶段同时进行：

- K、alpha、beta、cache、decode、block/layer 或 Euler/RK 消融。
- 4B、Llama、MoE 或生成任务扩展。
- tuned lens、训练型 probe 或任何模型参数更新。
- 逐输入在线切换 window。
- 使用最终测试标签选择 window。

## 4. 第一阶段信号

必须分别报告，不得在看到测试结果后临时调权重：

- 候选答案分布熵及窗口熵降。
- 中间层 choice 分布到最终层 choice 分布的 KL-to-final。
- 中间层与最终层 choice top-1 一致率。
- 窗口首步相对更新幅度。
- K=2 两次实际 Euler residual 的收缩比。
- 去除特殊 token、单位归一化并中心化后的 answer-position effective rank；它只作辅助诊断。

任何合成 score、阈值和弃权规则必须写入版本化配置并在 held-out 正式评测前冻结。

## 5. 代码边界

优先新增旁路模块：

```text
src/tflt/loopscope/
  __init__.py
  schema.py
  metrics.py
  grid.py
  probe.py
  window_probe.py
  selection.py
  analysis.py
```

允许的既有文件小改动：

- `src/tflt/cli.py`：注册 LoopScope 子命令。
- `README.md` 或 `docs/loopscope_phase1.md`：记录使用方法。
- `configs/loopscope/`：保存固定配置、schema 和 manifest 模板。
- `scripts/loopscope/`：保存可审计的 pool 生成/Slurm 编排脚本。
- `tests/`：新增纯 Python 单元测试。

默认禁止修改：

- `src/tflt/wrapper.py`
- `src/tflt/strategies.py`
- `src/tflt/cache.py`
- 已完成复现实验 manifest

如果确实必须修改默认禁止文件，先停止并提交最小复现、原因、拟议 diff 和回归测试，等待规划/审计线程批准。

## 6. 本地与 HPC2 边界

本地 Mac：

- 只在 `/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt` 写入。
- 原复现 checkout `/Users/huangxutao/Desktop/Training-free looped transformer/Training-Free Looped Transformers Reproduction` 只读，不得在其中切换分支或创建 worktree。
- 只保存代码、配置、测试和文档。
- 不下载模型/数据，不跑 GPU probe，不保存完整 eval 结果。
- 本地测试必须保持无 torch/transformers 也能运行。

HPC2：

```text
source checkout=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
read-only reproduction checkout=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction
reproduction workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_reproduction/{runs,logs,artifacts}
LoopScope workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/{inputs,runs,staging,artifacts}
HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
UV_CACHE_DIR=/hpc2hdd/home/xhuang225/shared/uv
wheelhouse=/hpc2hdd/home/xhuang225/shared/wheelhouse/tflt-cu121
```

- 不得进入固定复现 checkout 执行 fetch、checkout、switch、pull、install 或 LoopScope 运行；只允许必要的只读 provenance 核对。
- LoopScope checkout 使用同一 lock 文件建立独立 versioned venv；不得原地升级固定复现 `.venv`。
- 运行时 job 只加载 Python/CUDA modules 并激活已审计 venv，不得依赖 `uv` modulefile；`uv` 仅用于环境构建、下载或冻结维护。
- LoopScope 后续新 input、run、staging 与 artifact 必须进入专属 workspace；共享 input/cache 不为目录整齐而复制。
- Phase 1 canonical run 及全部历史 siblings 永久留在旧根 `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers`；不得移动、改名、替换或用 symlink 冒充。
- 每个实验使用新的 timestamped run root；不得覆盖、移动或删除既有 runs/cache。
- SSH 只读探针使用 `ClearAllForwardings=yes`；出现 host-key 变化必须停止并走专用恢复流程。

## 7. 数据与泄漏约束

- 工程 smoke 可以使用极小固定 prompt 集，但不能据此判断指标优劣。
- 正式 probe pool 必须记录来源、split、样本 ID/hash、生成 seed、prompt 模板和数量。
- 正式测试集样本不得进入 window 选择或阈值调节。
- 推荐从非测试 split 构建 500–1000 条多选格式校准池；如果找不到可证明隔离的 split，停止并报告，不得静默改用 MMLU test。
- probe 报告不得依赖 gold label；gold label 只允许在冻结选择规则后的收益评估中使用。

## 8. 输出与 provenance

每个远程工件至少包含：

- Git branch、commit、dirty 状态。
- 模型 repo 与 revision、tokenizer revision。
- Python/torch/transformers/lm_eval/tflt 版本。
- 完整命令与环境变量快照。
- 校准池 manifest/hash 与 candidate-window manifest/hash。
- `command_args.json`、`env.json`、`results.json` 或同等工件。
- `probe_report.json`、扁平 `probe_summary.json` 和中文摘要。
- audit 的 `audit_report.json`、`audit_summary.md`。
- Slurm job id、partition/node、ExitCode、stdout/stderr 路径。

不得只报告目录名、最终准确率或截图。

## 9. 强制门

### Gate A：本地代码门

- `git diff --check` 通过。
- `PYTHONPATH=src python3 -m unittest discover -s tests` 全绿。
- CLI help/dry-run 通过。
- branch/commit/diff/test 证据回传规划线程审计。

### Gate B：HPC2 CPU/import 门

- 远端 branch/commit 与本地已推送 commit 一致。
- 固定依赖版本验证通过。
- 远端 unit tests、compile/help 通过。

### Gate C：GPU probe/audit 门

- 1.7B 四样本 probe 成功且 schema 完整。
- `window=12:15` 的 audit 判定为 `loop_effective_logits_changed`。
- 每个 prefill prompt 的 `operator_body_calls=2`。
- `restore_allclose=true`，无 NaN/OOM/cache traceback。

### Gate D：limit 网格门

- baseline 与全部候选 window 的 `--limit 5` 工程 smoke 结束。
- 样本 ID、结果 schema 和配置一致。
- limit 结果不得用于科学结论。
- 回传完整证据并等待规划线程明确 `PASS` 后，才能提交 full grid。

### Gate E：正式网格门

- candidate-window manifest 已冻结并记录 hash。
- 所有 full runs 使用同一模型 revision、MMLU 5-shot、dtype 和非 window 配置。
- 所有窗口保留逐题 samples，能够进行 paired 分析。
- 失败不得自动重试；先保存错误与命令，报告后由用户决定。

## 10. 审计回传格式

每个 Gate 回传：

```text
Gate:
branch/commit/dirty:
files changed:
exact commands:
tests and exit codes:
remote host/job ids/run root:
artifact paths:
observed result:
deviations/errors:
requested decision: PASS / PASS_WITH_FIXES / BLOCK
```

未收到 `PASS` 时不得自动进入下一门。
