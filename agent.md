# LoopScope 项目操作约定

> 本文件只适用于 LoopScope 专用新 clone 的 `loopscope` 分支。它定义第一阶段 Qwen3-1.7B LoopScope 增量实现、HPC2 验证和实验的强制边界。原有本地/HPC2 固定复现 checkout 均为只读参照。发现本文件与用户最新明确指令冲突时，以用户最新指令为准并先暂停报告。

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
run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs
HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
UV_CACHE_DIR=/hpc2hdd/home/xhuang225/shared/uv
wheelhouse=/hpc2hdd/home/xhuang225/shared/wheelhouse/tflt-cu121
```

- 不得进入固定复现 checkout 执行 fetch、checkout、switch、pull、install 或 LoopScope 运行；只允许必要的只读 provenance 核对。
- LoopScope checkout 使用同一 lock 文件建立独立 versioned venv；不得原地升级固定复现 `.venv`。
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
