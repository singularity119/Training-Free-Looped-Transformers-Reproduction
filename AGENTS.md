# Principle
1. Fail Fast / Errors Never Pass Silently：不要在代码里藏兜底逻辑来吞掉错误、隐藏问题。出了问题就应该让它爆出来，否则你永远找不到真实问题。
2. Fix the Cause, Not the Symptom / Don't Paper Over Bugs：当一个问题出现时，不要用各种 small fix、针对性补丁来掩盖它。必须定位真实根因，彻底修复。在 bug 上糊纸只会让系统积累你不知道的危险暗病。
3. Make It Observable：即使问题很难定位，也绝不要偷懒做表面修复。应该给项目增加充分的日志和可观测性，保证下次问题再现时你有足够信息去定位。问题无法修复时，只需要诚实告诉我信息不足、需新增日志，不要假装修好了。
4. Design for Debugging / Traceability：始终注意在关键路径上给自己留足排查日志，确保每一个关键节点都是可追溯的。
5. Living Documentation / Single Source of Truth：当项目关键技术栈或产品方向发生变更时，同步更新 AGENTS.md。文档必须随代码一起演进，不能让它变成过时的谎言。
6. Don't Break Mainline：大规模重构或实验性改动前，必须先切新分支。

# LoopScope 项目操作约定

> 本文件只适用于 LoopScope 专用 clone 的 `loopscope` 分支。它定义 LoopScope 第一阶段及后续阶段增量实现、HPC2 验证和实验的长期强制边界；阶段进度与动态授权分别记录在对应 control 文件中。原有本地/HPC2 固定复现 checkout 均为只读参照。发现本文件与用户最新明确指令冲突时，以用户最新指令为准并先暂停报告。

## 0. 计划与信息源

- `AGENTS.md` 只保存长期稳定的项目边界、工程规则和 Gate 验收标准，不记录临时进度、当前执行线程或逐次工具日志。
- `../.planning/phase1/loopscope_phase1_control.md` 至 `../.planning/phase6/loopscope_phase6_control.md` 是已关闭第一至第六阶段的历史控制面；`../.planning/phase7/loopscope_phase7_control.md` 是第七阶段当前唯一的可变全局控制文件。当前 Gate、Gate 决策、executor 绑定、已审计 commit/run 与下一步 admission 均以 Phase 7 control 为准。
- `../.planning/selector_rules/` 保存可跨阶段复用的版本化选窗方法；它只定义方法，不授予 Gate、executor、GPU/Slurm、outcome 或文件写入权限。每个 Phase 仍须在自己的 contract/control 中冻结具体方法版本和 cell-specific 参数。
- 旧工作区 `.planning/` 包、历史 handoff、线程聊天和 Codex memory 只作历史证据或辅助回忆；如果与当前代码或控制文件冲突，不得据此覆盖当前事实。
- 单线程内的临时步骤优先使用 Codex 原生 plan/goal，不把每一步复制到项目文件。
- 不得自动调用 `planning-with-files` skill。只有用户在当前请求中明确点名或明确要求启用该工作流时才可使用。
- 仅在 Gate 决策、执行线程变更、审计 commit 变更、实验配方变更或授权边界变更时更新控制文件；不得按每次读取、命令或工具调用追加流水账。
- 面向用户、导师或论文阅读者的人类交付物统一放在专用 clone 外层的 `../资产/`：
  中文报告和结果表放在对应的 `../资产/报告/phase<N>/`；图片、绘图脚本、source-data、QA 和
  PNG/PDF/SVG 等展示输出放在 `../资产/figures/`。不得在 Git 项目内新建或镜像
  `报告/`、`figures/` 或其他人类交付目录；这些输出默认不进入代码仓库提交。
- `research-gate-orchestrator` 的协作上下文继续按信息分层保存：稳定规则在本
  `AGENTS.md`，当前授权与 Gate 决策在 `../.planning/*_control.md`，Gate contract、
  handoff 与 repair supplement 在 `../.planning/`，可执行 runbook 在 `docs/`，
  card/schema/launcher/tests 在各自 repo 路径，机器运行证据在 write-once HPC run root。
  不得为了“集中资产”把这些 agent/执行上下文移动到 `../资产/`。

## 1. 项目定位

- 基础复现只读基点：`4f59bd93eca4da3cbf458a93508f91c5b23912bc`。
- 开发分支：`loopscope`。
- 本地专用 clone：`/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt`。
- HPC2 专用 clone：`/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`。
- 第一阶段模型：`Qwen/Qwen3-1.7B-Base`。
- 第一阶段任务：`mmlu`，5-shot。
- 第一阶段目标：验证免训练内部信号能否预测不同 loop window 的真实收益；不提前宣称已经得到通用自动选窗器。
- 第二阶段目标：围绕第一阶段局部结果验证“可迭代精炼区”机制——区分持续有益修正、短暂最优、无益扰动和错误过度自信；先用小范围 `k` 轨迹与逐样本分析理解机制，只有机制得到支持后才讨论无标签选窗或跨模型扩展。
- 第三阶段目标：固定 Qwen3-1.7B × MMLU × K=2 配方，只利用 validation 全量 1,531 条样本的一次 no-loop 层级 choice trajectory，检验局部 entropy–KL 双重反转能否在同一 model-task cell 内恢复、富集或排除 loop windows；测试/full outcome 不得进入 selector。
- 第四阶段目标：直接在 Qwen3-4B-Instruct-2507 × MMLU-Pro 5-shot CoT 上检验，能否仅凭生成首 token 前的一次普通 no-loop prefix forward 全词表 entropy–KL trajectory，前瞻性预测完整 CoT decoding 的 width-4 loop window；同一 12,032 test identity 总体可同时承担 outcome-blind selector 与后续 outcome，但必须按时间和字段严格隔离。
- 第五阶段目标：在 `Qwen3-4B-Base × MMLU 5-shot` 上填补与第三阶段同任务的尺度迁移 cell；仅用 validation-1531 一次 native no-loop choice/hidden trajectory 冻结 width-4 selector，再在与其样本级隔离的 test-14042 sealed panel 上区分 ranking enrichment、absolute gain、known-comparator competitiveness 和 `ABSTAIN`。
- 第六阶段已终态：在完成 `Qwen/Qwen3-4B-Instruct-2507 × MMLU 0-shot`、MMLU 5-shot prefix trajectory 与 V3 selector 后，固定四个 loop 窗口加 no-loop baseline 的 MMLU 5-shot test-14042 panel 已完成。`15:18` 是唯一 paired bootstrap 95% CI 全部高于零的 cell（`+0.5697 pp`，CI `[+0.1496,+0.9970]`）；V3 selected `14:16` 为 `+0.0926 pp` 且 CI 跨零。该 outcome 不回写 selector，也不授权继续加窗、调参或启动下一 Gate；旧 V2 `ABSTAIN` 与此前终止的 MMLU-Pro D-2 分支继续作为独立历史证据。
- 第七阶段目标：在 `Qwen/Qwen2.5-3B`、`meta-llama/Llama-3.2-3B`、`google/gemma-2-2b` 三个 Base 模型上，对同一 MMLU validation-1531、plain 5-shot `Answer:` 末端 probe 采集一次 native zero-loop 逐层轨迹，按 V3.1 分别产生模型内 `SELECTED_WINDOW` 或合法 `ABSTAIN`，由 planning 先完成 outcome-blind 绘图后，再在 MMLU test-14042 运行按有限 width-4 V3 score 形成的 3/3/2 不对称 fixed exploratory outcome panel：Qwen/Llama 各取 top-3，Gemma 取全部 top-2，共 35 cells。`Qwen3-1.7B-Base` 与 `Qwen3-4B-Base` 仅作历史背景，不在本阶段重采集、重评分、重绘或加入 Gate E。
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
- `src/tflt/eval_runner.py`：第二阶段只允许新增显式 opt-in 的 final-output sidecar adapter；默认关闭时 parser、`LoopConfig.audit_collector=None`、HFLM/TaskManager/`simple_evaluate` 参数、revision closure、`results.json` 和科学行为必须保持不变。adapter 只从 evaluator 已完成的逐样本记录提取四个 raw choice scores，并用稳定 sample identity 作顺序无关的精确 join；不能按 callback/batch 顺序猜测。当前 H1 不允许通过该 adapter 接入 wrapper residual collector，也不允许为 full residual 新增 HFLM request router。
- `README.md` 或 `docs/loopscope_phase*.md`：记录使用方法与工件契约。
- `configs/loopscope/`：保存固定配置、schema 和 manifest 模板。
- `scripts/loopscope/`：保存可审计的 pool 生成/Slurm 编排脚本。
- `tests/`：新增纯 Python 单元测试。

默认禁止修改：

- `src/tflt/wrapper.py`
- `src/tflt/strategies.py`
- `src/tflt/cache.py`
- `src/tflt/config.py`
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
- 上一条是 Phase 1–3 的默认边界。Phase 4 的唯一例外由本文件第 13 节和 `../.planning/phase4/loopscope_phase4_control.md` 冻结：允许同一 MMLU-Pro test-12032 identity 总体用于无标签 prefix selector 与后续 outcome，但 selector freeze 前不得读取 target `cot_content`、gold/answer、生成 token/答案、correctness 或任何 baseline/loop outcome。
- Phase 6 MMLU 5-shot selector 阶段只读 validation-1531 的 prompt/choice identity 与 sanitized choice/hidden trajectory；标准 renderer 可读取 dev split 的五个 demonstrations 及其示例答案。Gate N PASS 后，用户另行授权 Gate O 在与 validation selector 隔离的 canonical MMLU test-14042 上运行固定 outcome panel；只有五个 cells 全部完成并通过 identity/completeness closure 后才可一次性读取 gold/correctness/accuracy/gain。已完成的 0-shot 与已终止的 MMLU-Pro 分支都只作历史对照，不得把其 trajectory 或 outcome 混入 Gate O。
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

## 11. 第二阶段稳定科学与治理边界

### 11.1 阶段定位

- 第一阶段 `Qwen3-1.7B-Base × MMLU 5-shot` 已经看过完整 accuracy，是第二阶段提出机制假设的开发锚点，不是 held-out 证据。
- 第二阶段核心是验证“可迭代精炼区”：某个 block 在单次前向已有明显作用，但重复调用时在观测到的 K≤4 范围仍保留方向有益的修正能力。是否以及何时饱和必须单独报告；未在 K≤4 观察到饱和不等于 H1 失败，也不自动授权 K>4。
- `Native-Continuation Alignment (NCA)` 是 H1 V2 拟在 Gate A 冻结的无标签、相关性方向代理：它比较真实 loop residual 与同一样本原生下游延续方向。只有 Gate A 审计 `PASS` 后才可称为预注册。NCA 正值只表示与原模型原生路径一致，不证明朝正确答案移动，也不能替代 gold-label 纠错裁决。
- 当前不得直接建设新的加权 selector 或 NCA certificate selector。先判断 `12:15` 的收益究竟来自持续有益修正、一次性关键层作用、普通扰动、错误过度自信，还是仅仅保持原生路径。
- 正结果、负结果和“仅 K=2 短暂最优”都必须如实报告。Gate 的 `PASS` 只表示契约和证据完整；H1 结果与 NCA 诊断必须分开记录。
- 旧版多模型、多任务、全窗口信号竞赛不是第二阶段核心门；它只在核心机制得到支持且用户另行授权后，作为可选扩展。
- `configs/eval/phase2_qwen17_mmlu.json` 属于基础复现项目的历史 Phase 2 命名，不得覆盖或改作 LoopScope 配置。LoopScope 第二阶段新配置只进入 `configs/loopscope/`。

### 11.2 核心实验不变量与测量分层

Gate A 拟冻结的核心机制卡草案：

```text
card=H1_ITERATIVE_REFINEMENT_ZONE_V2
model=Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1
task=mmlu
num_fewshot=5
dtype=float16
windows=11:14,12:15,13:16  # inclusive, width=4
iteration_mode=block
strategy=damped_euler
beta=0.0
cache_strategy=last
decode_mode=bypass
primary continuation=(k,alpha)=(2,1.0),(3,1.5),(4,2.0)  # 固定 h=alpha/k=0.5
fixed-horizon control=(k,alpha)=(1,1.0),(2,1.0),(3,1.0),(4,1.0)
nca_position=final_pre_answer_prompt_token
nca_calibration=Phase 1 frozen validation 512; labels sealed during probe
nca_bootstrap=10000; seed=0
nca_role=prospective_secondary_direction_proxy; never selector in H1
```

- 当前 `damped_euler` 的单步大小为 `h=alpha/k`。主轨迹固定 Phase 1 的 `h=0.5` 并随 K 延长总迭代时长，用于直接检验“继续计算”；固定 `alpha=1` 的控制轨迹只检验同一总时长下更细的子步，二者不得混写。
- `(k=1,alpha=1)` 必须先在小样本上证明与普通单次前向等价；若不等价，立即 `BLOCK`，不得把它静默当 baseline。
- 主 continuation 的 step 都是 `0.5`，因此 K=3/4 的前两次调用状态/residual tensors 必须与 K=2 在同样输入上于内存中 allclose；工件只保存 max-abs/scalar 证明。Gate B 若不能证明 prefix consistency，必须 `BLOCK`，不能解释为“继续同一迭代”。
- Phase 1 的 baseline 与三个窗口 `k=2` 结果优先只读复用；不得为目录整齐或补字段自动重跑。若缺少关键逐样本字段，先报告并由规划线程决定是否授权 write-once trace 补跑。
- 每个新工件的 identity/path 必须同时编码 `protocol、window、k、alpha`；不得仅按 K 命名而让 `(3,1)` 与 `(3,1.5)` 冲突。
- 指标必须标明来源，禁止混写：
  1. **baseline layer probe / provenance A**：boundary entropy、KL、effective rank，以及 baseline window update 与原生下游延续的 NCA；
  2. **actual loop trajectory / provenance B**：每轮 residual norm、`q_t`、相邻方向和每轮 NCA，来自实际重复调用；
  3. **final model output by k / provenance C**：完整前向结束后的 choice probability、entropy、top-1/top-2 margin、JS divergence 与答案；
  4. **gold-label adjudication / provenance D**：card 与无标签字段冻结后才计算的 `wrong→right`、`right→wrong`、正确答案 margin 和错误过度自信。
- NCA 的原生延续为 `c_(w,i)=B_N[i,p,:]-B_(b+1)[i,p,:]`，其中 `p` 是冻结的 final pre-answer prompt token。body-call index 冻结为 zero-based `t=0..K-1`：`t=0` 是 initial call，baseline NCA 使用普通单次前向 `B_(b+1)-B_a`，actual-loop NCA 只汇总 repeated calls `t=1..K-1`；`K=1` 不产生 repeated-step NCA。近零向量、非有限值、位置或边界 identity 不一致必须显式 invalid/fail-fast，不得静默当作普通零分。
- NCA 默认只在冻结 512 校准池上采集。向量可在设备内以 float32 临时计算，但工件只持久化 per-sample cosine、`||c||`、`||delta||`、validity mask 和汇总值；不得持久化完整 hidden/residual tensors。
- H1 V2 采用尺度分离：冻结 512 calibration 是 actual-loop residual、`q_t`、相邻方向、NCA 和无标签 final-choice 轨迹的唯一 mandatory trajectory scale；14,042-sample full 只保存 evaluator 产生的每样本四个 raw choice scores、概率、entropy/margin/JS、答案与 gold paired outcome，不要求也不允许新增 full residual/NCA trace。
- full final-output sidecar 必须用 canonical `task/doc_id/doc_hash` manifest 作唯一、完整、原序 closure；evaluator 输出可任意排序，但 join 后必须恢复 canonical natural order。若 lm-eval 真实 logged sample 缺少足以证明该 join 或 raw four-choice scoring 的字段，Gate B 必须 `BLOCK`，不得自动增加 HFLM router、改 batch size、改 renderer 或改用 direct full producer。
- full-population residual–answer-flip 个体相关性不属于当前 H1 的 confirmatory claim。Gate C 对 residual/NCA 的机制解释引用经审计的 512 calibration；14,042 full 只负责最终决策轨迹和 gold-label outcome。若以后确需 full residual correlation，必须另立 hypothesis card 和独立预算。
- baseline layer probe 的 entropy drop 不能表述为“loop 后 entropy 下降”。中间 block hidden state 经 final norm/lm_head 得到的 lens 量只作辅助；未经有效性门不得当作 primary evidence。
- NCA、`q`、entropy 和 eRank 都不能单独证明方向有益。H1 的最终科学裁决仍由 final-output 与 gold-label paired outcome 给出；NCA 只回答它是否提供额外的无标签区分力。
- 三窗口的六个 prospective incremental contrasts 构成唯一 primary confirmatory family，使用 two-sided exact McNemar 与 Holm step-down、family-wise `alpha=0.05`；fixed-horizon 是独立、内部 Holm 校正的 secondary control family，cumulative-vs-baseline 只作 secondary context，二者均不得单独触发 `REFINEMENT_SUPPORTED`。NCA diagnosis 必须独立于 H1 outcome，使用预先声明的 pointwise、未校正 secondary intervals，不作 FWER-controlled confirmatory claim。Gate A 必须把 H1/NCA 标签的判定顺序、符号条件、区间比较、valid-fraction 下限和 multiplicity（包括 NCA 明确不校正）写入 versioned config/hash，规划线程审计后才可冻结。
- H1 outcome 只回答 `12:15` 在当前 model-task cell 的 continuation 结果；“目标窗口显著、邻窗不显著”不能证明窗口间差异，不得据此声称 `12:15` 唯一或显著优于邻窗。窗口选择性只能由未来独立 direct contrast/selector card 检验。
- 第一阶段 effective rank 继续作为解释性 covariate；10k-token pooled eRank、angular/BI、tuned lens 和新复合分数均不属于核心必做项，需独立 hypothesis card。

### 11.3 敏捷 hypothesis card

- 第二阶段按小型 hypothesis card 推进，而不是一次冻结整个研究项目。每张卡只能回答一个主要问题，并明确 model/task/window/k、指标来源、需要的新计算、成功/反证模式和最大新 full-run 数。
- NCA 在 H1 V2 冻结前由用户明确加入，因此可作为 prospective secondary observable；此后出现的新指标不能回写为 H1 的预注册证据，所有 post-hoc 发现必须标记 exploratory。
- 只读分析现有工件的卡不需要重复 GPU Gate；新增采集代码才进入本地门；新增 GPU 字段先过小样本门；只有小样本闭环后才能 full。
- 优先复用既有 repo、venv、缓存、renderer、Phase 1 samples 与 paired-analysis 基础设施。不得为了通用化而先做大规模重构。
- Gate A 初始授权不得修改 `wrapper.py`、`strategies.py`、`cache.py` 或 `config.py`。现有 `audit_collector`/`record_tensor_diff` 事件应先由新的 Phase 2 collector 消费；只有执行线程给出最小失败用例、规划线程另发 superseding authorization 后，才可讨论在现有 payload 中增加纯索引 metadata，且不得改变 tensor 数值、loop/cache/restore 语义或旧 collector。
- 用户在 Gate A `BLOCK` 后选择尺度分离路线；当前 superseding contract 仍不授权修改 `wrapper.py`、`strategies.py`、`cache.py`、`config.py`，也不授权 HFLM request fingerprint/router、独立 full direct residual producer 或任何 14,042-sample residual/NCA 补跑。
- B2 的 deterministic no-loop boundary 与 15 个 loop logical cells 必须在同一受控进程/session 中顺序完成或使用等价的内存内复用设计，使 native-continuation vectors 只短暂驻留内存；不得为跨进程复用持久化完整 hidden/residual vectors，也不得把一次 baseline 静默扩成 15 次未声明重复运行。
- 本地无 torch/transformers 仍须通过 lazy import 与 pure-Python/fake-tensor contract tests；真实 CUDA dtype/device/数值闭环留给 Gate B1，不得在 Gate A 伪造 GPU 证明。

### 11.4 第二阶段核心 Gate

1. **Gate A：轻量轨迹/NCA 实现与 H1 V2 冻结**——只实现 512 calibration 的每轮 residual/NCA producer、独立 full final-output sidecar、从 canonical per-sample 工件重建统计量的分析 producer，以及所需最小 closed-world schema、配置和测试；不得远程运行。
2. **Gate B：HPC2 CPU + smoke/limit + bounded NCA calibration 门**——先做 CPU/provenance，再做四样本与 limit 轨迹；验证 `k=1` 等价、fixed-step prefix consistency、调用次数、NCA 位置/边界/字段、样本 identity 和资源上限。工程闭环后可在同一 Gate 内执行冻结 512 池、三核心窗口的无标签 NCA/final-choice probe；B2 最多一个 no-loop boundary cell 加十五个 loop cells、单 GPU 总计不超过 18 GPU-hours。不得运行 full MMLU，也不得按 NCA 改 Gate C 窗口或 K。
3. **Gate C：三窗口 focused full 门**——复用 Phase 1 baseline 与 `(k=2,alpha=1)`；主轨迹只新增三窗口 `(3,1.5)/(4,2)` 六个配置，并为 `12:15` 新增 `(3,1)/(4,1)` 两个 fixed-horizon controls，默认最多八个 full 配置；full 只产出 canonical final-choice/gold paired evidence，不采集 residual/NCA。执行一次 write-once paired mechanism analysis，引用 Gate B 已审计的 512 residual/NCA evidence，并分别报告 H1 outcome 与 NCA diagnosis。
4. **Gate D：Gate C 后的可选条件分支**——使用互斥矩阵：H1=`TRANSIENT_ONLY/PERTURBATION`（任意 NCA）停止 selector；H1=`SUGGESTIVE/INCONCLUSIVE`（任意 NCA）只可停止或另立最小 H1 消歧卡；只有 H1=`REFINEMENT_SUPPORTED` 时才按 NCA 分流，其中 `NCA_INCONCLUSIVE` 只可停止或最小 NCA 消歧，`NCA_DIRECTION_SUPPORTED` 可由用户另立 `H2_NCA_CERTIFICATE`，`NCA_NONDISCRIMINATIVE/NCA_NATIVE_FIDELITY_ONLY` 可选择一个新任务或一个新模型做最小机制确认。任何分支都需独立授权。

完整的全窗口 NCA 排名、top-3 shortlist、唯一胜者/no-loop verifier、window-identity permutation 和固定部署协议不属于 H1，也不构成 Gate C admission；它们只可在 Gate C 审计后作为独立 H2 card 冻结。旧版 16/17-window grid、三模型×两任务、depth-controlled signal competition、10k-token eRank 和生成任务继续保留为 optional backlog。

每一 Gate 使用独立执行线程；规划/审计线程不亲自执行、不轮询、不预建未来 Gate 执行线程。执行线程只在终态主动发送一次结构化 `GATE_X_FINAL_AUDIT` 或 `BLOCK`，规划/审计线程收到后重新读取 live Git、Slurm 和不可变工件再决定。

Gate 终态经规划/审计线程判定后，只撤销该 executor 对 Git、数据、网络、GPU、Slurm、repair 和后续 Gate 的全部执行权限，不归档、不关闭其线程；完成、作废或 `BLOCK` 的 executor 均保留在项目下作为只读 provenance，且不得复用为下一 Gate executor。线程保留不等于继续授权；除非规划/审计线程明确要求补充终态说明，否则该 executor 不得继续执行、轮询、修复或写入。

凡当前 Gate executor 已提交 HPC2/GPU/Slurm 作业，且作业在一次提交确认后仍需较长时间排队或运行，executor 必须停止人工轮询、shell sleep、循环 `squeue/sacct` 或为了维持 thread active 而空转，改为在自己的执行线程中创建真实的 Codex automation/heartbeat。默认按预计时长采用约 10 分钟、30 分钟或 60 分钟 cadence；等待中的 executor 显示 idle 是正常状态。每次 automation 唤醒只允许执行一次有界、只读状态检查（例如带 `BatchMode=yes`、`ConnectTimeout` 与 `ClearAllForwardings=yes` 的 SSH，加一次 `squeue/sacct`、必要的短日志尾部和预期工件存在性检查），然后立即结束本次唤醒；状态未发生材料变化时不打扰用户。automation 不获得提交、重试、requeue、取消、改参、改 throttle、修复、写工件或科学裁决权限；这些动作仍须来自该 Gate 原有 handoff 或新的明确授权。monitor prompt 必须冻结 exact Gate、executor thread、planning/audit thread、job IDs、run root、terminal criteria、terminal 后下一项已授权动作及 fallback recipient。遇到 terminal success、terminal failure、SSH trust 异常或其他材料性 blocker 时，automation 在完成一次有界只读检查后必须先暂停/删除自身并保留 lifecycle evidence，再向精确 executor 投递 `AUTOMATION_TERMINAL_RESUME` 以触发新的 executor turn，并保存可见投递确认；heartbeat 的 final answer 本身不等于 executor 已恢复，禁止只写“executor 可恢复”后结束。若无法触发或确认 executor continuation，必须向 planning/audit thread 发送并在自身输出完整 `AUTOMATION_RELAY_REQUIRED`，由 planning/audit thread 只做一次有界 relay recovery；不得重跑作业、创建新 executor 或转为轮询。executor 的 `GATE_X_FINAL_AUDIT` / `BLOCK` 也只有在精确 planning/audit thread 的投递得到可见确认后才算送达；无法确认时必须输出完整 `TERMINAL_DELIVERY_UNCONFIRMED`。planning/audit 线程不替 executor 做 heartbeat 轮询。

## 12. 第三阶段稳定科学与治理边界

- 第三阶段 primary selector source 是 MMLU validation 全部 1,531 条唯一样本的一次普通 no-loop forward；Phase 1 validation-512 只作 secondary robustness，不参与 variant 选择。
- selector 只读取逐层四选项 choice distribution、entropy、KL-to-final、CE 及其局部 SHIFT/FLANK/CONSENSUS 结构；target gold、correctness、loop residual、full accuracy 与答案翻转均禁止进入 selector producer。
- 正式 loop outcome 固定使用与 validation-1531 样本级不相交的 MMLU test-14042 paired evidence；历史 13 窗只作 development，尚未查看 outcome 的 12 窗才承担 width-4 blind completion。
- 核心 loop 配方继续冻结为 `K=2、block、damped_euler、alpha=1、beta=0、cache_strategy=last、decode_mode=bypass`；任何 K、alpha、strategy、model 或 task 变化必须另立科学卡。
- Phase 3 Gate 名称和当前状态以 `../.planning/phase3/loopscope_phase3_control.md` 为准；本文件第 9 节 A–E 是 Phase 1 历史门，不得用来推断 Phase 3 当前授权。
- 第三阶段不设置 GPU 数量、GPU-hours 或墙钟硬上限。GPU Gate 只有在明确授权、科学矩阵冻结并满足 admission 后，才能按 HPC 空闲资源合理并行并尽快提交；资源变化不得改变样本、窗口、配方、统计或 retry 语义。

## 13. 第四阶段稳定科学与治理边界

- Phase 4 使用 current-checkpoint reproduction：`Qwen/Qwen3-4B-Instruct-2507 × mmlu_pro 5-shot CoT`，不是原论文 checkpoint 的 bit-exact 复现。
- selector 检测段只允许生成首 token 前的一次普通 no-loop prefix forward；读取 `B_0..B_36` 最后一个有效 prefix token，经 frozen final norm 与 LM head 得到完整词表的逐层 entropy 和 `KL(p_l || p_36)`。完整词表 entropy 不得称为 Choice Entropy。
- 正式 outcome 配方冻结为 `window_width=4、K=3、block、Euler、step_size=1/3、total_horizon=1、cache_strategy=first、decode_mode=full`；只有 width-4 window 可变。
- width-4 主组覆盖 33 个候选、其中 25 个具备严格 CONSENSUS 支持，并独占 selected、blind high3/low3 与 Phase 4 outcome。width `2..8` 的 strict-154 与 edge-aware-224 仅作 outcome-blind 次要排名；不得自动进入 Phase 4 full-decode outcome。
- selector 与 outcome 共用一个 canonical MMLU-Pro test-12032 identity manifest。隔离对象是信息与时间而非样本身份；任何正结论都必须限定为 `same-population outcome-blind transductive`，不得声称 unseen-sample generalization。
- Phase 4 Gate 顺序固定为 P4-A 本地最小实现、P4-B 真实 prefix acquisition/selector freeze、P4-C width-4 frozen panel outcome acquisition、P4-D 一次性 unseal/analysis/phase-end audit。每 Gate 一个独立 executor；只有 planning/audit thread 可以裁决并授权下一 Gate。
- Phase 4 的动态状态、exact executor、commit/run identity、权限和 admission 只以 `../.planning/phase4/loopscope_phase4_control.md` 为准；详细命令与工件合同见 `docs/loopscope_phase4.md`。

## 14. 第五阶段稳定科学与治理边界

- Phase 5 冻结 model cell 为 `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539 × cais/mmlu × 5-shot`，checkpoint config 为 36 decoder layers、hidden size 2560、bfloat16。
- selector 只读 MMLU validation 全量 1,531 / 57 subjects 的一次 native no-loop forward；记录 `B_0..B_36` final pre-answer position 的 A/B/C/D choice entropy、`KL(p_l||p_36)` 与 final-norm hidden RMS-L2/cosine。target gold、correctness、test outcome 和 loop outcome 严禁进入 selector。
- primary selector 是 width-4 entropy–KL `CONSENSUS`；hidden geometry 只是 diagnostic，不得加入 selector 或与 cosine distance 重复计票。selection frequency 阈值为 0.80；无 eligible、top tie 或 frequency 不足时必须 `ABSTAIN`。
- loop outcome 冻结 `width=4、K=3、block、Euler alias=damped_euler、alpha=1、beta=0、step=1/3、horizon=1、cache=first、decode CLI=bypass / scientific N/A`，只允许 window 变化。
- outcome panel 最多八个 unique cells：no-loop、known fixed `15:18`、registry-excluded selected-inclusive High3 和 disjoint Low3；同时保留 raw High3 metadata。baseline/`15:18` 默认 fresh acquisition，只有 bit-exact per-sample closure 才可例外复用。
- formal outcome 使用 MMLU test 全量 14,042 canonical identities，panel 必须先 hash-close 再运行。Gate C 只见 completeness/identity/scheduler/hash，Gate D 才一次性 unseal 全 panel。
- success 与 ranking 必须分开：selected-baseline 要求 point gain>0；selected 对 `15:18` 的 non-inferiority margin 为 0.30 pp 且 panel regret<=0.30 pp；High3−Low3 为正不得单独宣称 selection success。
- bootstrap 冻结 2,000 replicates；trajectory seed=`20260722`，outcome seed=`20260723`。禁止 K sweep、variable-width outcome、dynamic gating、selector 调权、Orthogonal Residual、template projection、CAST、SEAL、PASTA、SADI 或其他 representation intervention。
- Phase 5 Gate 顺序为 A 合同/provenance、B validation trajectory/selector freeze、C sealed test outcomes、D one-shot unseal/terminal analysis。每 Gate 一个独立 executor，只有 planning/audit 任务可裁决并授权下一 Gate。
- Phase 5 动态状态、exact executor、commit/run identity、权限和 admission 只以 `../.planning/phase5/loopscope_phase5_control.md` 为准；详细命令与工件合同见 `docs/loopscope_phase5.md`。

## 15. 第六阶段稳定科学与治理边界

- 用户已终止原 `Qwen3-4B-Instruct-2507 × MMLU-Pro 5-shot CoT` pre-answer D-2 分支。已完成 6/8 shard、失败/取消 shard、D-2P `EXPLORATORY_PARTIAL_6_OF_8` 与全部 write-once roots 仅作历史证据；不得补齐、merge、正式选窗、unseal、转成 selected-window 结论或与当前分支混合。
- 当前 Phase 6 follow-up 冻结 cell 为 `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 × cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe × 5-shot`。Evaluator 使用 lm-eval 0.4.11 标准 MMLU choice-loglikelihood、plain prompt、`num_fewshot=5`、`fewshot_split=dev`、`apply_chat_template=false`、`fewshot_as_multiturn=false`，不生成 CoT 或答案文本；continuation surfaces 仍为 exact ` A/ B/ C/ D`。0-shot 的 model/tokenizer/choice-token closure 可复用，但 5-shot rendered prompt projection 与 hash 必须 fresh acquisition。
- Probe position 是 rendered prefix 的最后一个非 padding token。每个 validation identity 只做一次 `use_cache=false`、zero-loop native forward，并从同一 forward 采集 raw `B0...B36`；B0...B35 只应用一次 final norm，B36 不得 double norm。逐层 choice distribution 使用 exact A/B/C/D logits 的 stable float64 softmax。Selector 输入仅为 choice entropy `H` 与 `KL(p_l||p_36)`；hidden RMS-L2-to-final、hidden cosine-to-final、cosine-distance-to-final 与 raw adjacent angular distance 只作诊断，不得进入 eligibility、ranking、frequency 或 panel membership。
- Selector source 固定为 MMLU validation 全量 1,531 identities / 57 subjects。Validation gold、test split、correctness、accuracy、gain、flip 与任何 baseline/loop outcome 全程禁止读取；本分支不创建 outcome panel、不运行 test-14042。标准 evaluator 只允许 frozen dev 5-shot demonstrations，不授权生成式 direct-letter fallback、chat template、multiturn 或 CoT。
- Gate O 的 TFLT 配方冻结为 bfloat16、K=3、block Euler、step size=1/3、total horizon=1、cache=first、decode=full；固定 cells 为 no-loop、inclusive `14:16`、`13:16`、`15:18`、`12:16`。标准 MMLU 5-shot 采用 full-sequence choice-loglikelihood；当前 wrapper 的 `bypass` 只跳过 cached incremental one-token decode，因此在该 evaluator 路径上 `full` 与 `bypass` 预期执行等价，但 Gate O 只运行用户指定的 `full`，不新增 decode 消融。
- Phase 6 当前 V3 重打分采用 `.planning/selector_rules/aggregate_common_turn_v3_absolute_rate.md` 中冻结的 `AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE@3.1.0`：中央 blocks `11...24` 内 width 3/4/5/6 共 42 candidates；category-macro point/bootstrap estimand、`NetPositive`、`RateStable`、`AggregateCommonTurn`、`S_RATE_TURN=sqrt(G_H*G_K*Q_H*Q_K)`、2,000 次 bootstrap、seed `20260801`、combined-rank frequency `>=0.80` 和 V3 终态必须按 exact shared-rule hash 复算。旧 V1/V2 的 `BiphasicStable`、direction margins、`tau_pair_frequency` 与 `SoftRelativeStable` 不得进入 V3 admission 或 ranking。合法 V3 `ABSTAIN` 是终态，不得为获得窗口而调规则。
- 0-shot 的 Gate H/I/J、5-shot 的 Gate K/L/M、V3 rescore Gate N 与 fixed-panel outcome Gate O 均已闭合。Gate M 的 V2 `ABSTAIN_NO_RATE_STABLE_ELIGIBLE`、Gate N 的 V3 selected inclusive `14:16` 与 Gate O 的五-cell test outcome 必须分别解释；Gate O 后没有授权其他 loop 实验。每 Gate 一个独立 executor，只有 planning/audit task 可裁决。
- Phase 6 executor-owned 低风险工程修复次数不限，但必须保持在同一 Gate、冻结科学合同、授权路径/权限、protected state、信息屏障与已授权 fresh-retry 语义内，并持续取得材料性进展；该规则不授权科学修改、outcome 读取、正式节点重跑、破坏性操作或跨 Gate。
- 所有工程 debug 与 smoke 实验必须使用 `debug` partition；所有正式长任务必须先有同 commit/launcher/env/model/data/producer/verifier 的 `<30min` debug preflight PASS。5-shot Gate M 正式任务按用户指令冻结 A800，并使用普通用户合法的最高优先级 A800 partition/QoS；正式 job ID 核实后即启用唯一 full 60 分钟 heartbeat（含 PENDING），debug heartbeat 仅在稳定 RUNNING 约 60 秒后启用。smoke/probe/full cadence分别为 10/30/60 分钟。
- Phase 6 动态状态、exact executor、commit/run identity、权限、seed、panel 和 admission 只以 `../.planning/phase6/loopscope_phase6_control.md` 为准；runbook 见 `docs/loopscope_phase6.md`。

## 16. 第七阶段稳定科学与治理边界

- Phase 7 正式新模型 membership 仅为 `Qwen/Qwen2.5-3B`、`meta-llama/Llama-3.2-3B`、`google/gemma-2-2b`。Gate A–D 的共同 selector 任务为 `cais/mmlu` validation 全量 1,531 identities / 57 subjects、标准 plain non-chat 5-shot renderer、dev demonstrations、batch size 1、bfloat16、无量化；这些 Gate 与 planning plotting checkpoint 不得运行 test、loop、generation 或 outcome panel。用户追加的 Gate E 仅在 plotting checkpoint 完成后运行独立 `cais/mmlu` test 全量 14,042 identities / 57 subjects outcome panel。
- Probe 固定在当前 validation query 完整 5-shot rendered prefix 的最后一个非 padding token，即答案 continuation 尚未加入时 `Answer:` 结尾的 tokenizer-final token。每条 identity 只做一次 native zero-loop、`use_cache=false` forward；不得改成 demonstration answer、目标答案字母、生成 token、EOS 或 padding。
- 对每模型真实 `L`，raw residual boundaries 为 embedding output `B0` 和每个 decoder block 后、FinalNorm 前的 `B1...BL`。若原生 `hidden_states[-1]` 已 post-norm，使用临时 FinalNorm forward-prehook 捕获 raw `B_L`，每次 forward 恰好调用一次并在 `finally` 移除；`FinalNorm(raw B_L)` 必须闭合原生 final hidden，禁止 double norm。
- 四个 continuation surfaces 精确为 `" A"/" B"/" C"/" D"`，每个 tokenizer 上都必须各为一个且彼此不同的 token；任一模型失败即 `BLOCK_CHOICE_SURFACE_NOT_SINGLE_TOKEN`，不得自行采用多 token、首 token、裸字母、chat template 或模型特定 prompt fallback。
- V3 selector 唯一输入为四选项 stable-float64 softmax 的 Choice Entropy `H` 与 `KL(p_l||p_L)`。hidden RMS-L2-to-final、hidden cosine-to-final、cosine distance `1-cosine` 和 FinalNorm 前相邻 raw residual angular distance 只作诊断与绘图，不得进入 eligibility、ranking、frequency、tie-break 或终态。
- 跨深度候选域按每模型真实 `L` 生成：`t(L)=ceil(0.30L)`，允许 blocks 为闭区间 `[t(L), L-t(L)-1]`，widths 为 `{3,4,5,6}`，只发布完整包含于该区间的半开窗口，并按 width、start 升序枚举。V3 方法固定为 `AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE@3.1.0`，2,000 次 within-category equal-category-macro joint bootstrap、seed `20260801`、三模型共用 canonical draw map但独立评分，combined-rank frequency 门槛为 `0.80`；不得跨模型汇总 score 或强制共同窗口。
- 每模型唯一合法科学终态为 `SELECTED_WINDOW`、`ABSTAIN_NO_V3_ELIGIBLE` 或 `ABSTAIN_COMBINED_RANK_UNSTABLE`。合法 `ABSTAIN` 不触发调参；输入、membership、公式或 independent verifier 不闭合是工程 `BLOCK`，不能伪装成科学弃权。
- Gate 顺序固定为 A provenance/最小实现、B 每模型四 identity 的 `debug` GPU preflight、C 三模型 validation-1531 正式 acquisition、D CPU-only V3.1 分析与 source-data verifier、planning-owned outcome-blind plotting/QA checkpoint、E 三模型 test-14042 full outcome。每 Gate 一个新的独立用户可见 executor；只有 planning/audit task 可判定 `PASS/PASS_WITH_FIXES/BLOCK` 并授权下一 Gate。
- Phase 7 executor 标题保持相同 Project 与 Phase，并严格使用 `execute-<Project>-<Gate-local-topic>-第<Phase>阶段-Gate <Gate>`：Gate B=`Debug-Smoke`、Gate C=`Full-Acquisition`、Gate D=`V3-Scoring`、Gate E=`TopAvailable-Outcome`；已完成的 Gate A 历史标题保留。替代 executor 仍使用该科学 Gate 的同一 canonical title，恢复/重试编号只写入 exact thread ID、control 或 recovery 文件名，不得擅自改成 `Gate C-2`。每个 executor 须在执行前完整重读 `research-gate-orchestrator`，以最短可信实验主线、材料性和 deletion test 为准；不得把 Gate 扩展成通用框架、重复全量审计或非正常路径防御平台。
- Phase 7 以后每次启动新 Gate executor，handoff 的首个机器可读区块必须显式声明 `COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR`，并紧接着说明：exact executor ID 与 canonical title、问题/ordinary operational permission 只报告 planning thread、不得绕过 planning 直接向用户停摆、subagent 不拥有 Git/remote-write/GPU/Slurm/terminal 权限、长任务唯一 heartbeat 及 terminal self-wake 路由、唯一 `GATE_X_FINAL_AUDIT`/真正 `BLOCK`、可见投递确认与 `TERMINAL_DELIVERY_UNCONFIRMED` fallback。Planning 在 dispatch 前必须从 phase plan 的 canonical mapping 生成标题，创建后立即用 title 工具返回值核对 exact string，再绑定 control、投递完整 handoff 并确认一次；任一项未闭合不得把 Gate 描述为已 dispatch。
- Phase 7 executor 可在确实缩短当前 Gate 时使用 subagent，任一时刻最多三个。Subagent 只能承接 handoff 明确委派的当前 Gate 内窄任务，不得跨 Gate、改科学合同、独立执行 Git integration/GPU/Slurm、发送 terminal event 或替代绑定 executor 的整合和终态责任。
- Gate A 可用 `ClearAllForwardings=yes` 只读访问 HPC2 项目代码、环境、缓存、配置与 provenance 以闭合背景信息，但不授权远端写入、GPU、Slurm、模型 forward 或数据 acquisition。所有后续 debug/smoke 只能使用 `debug` partition 且 `<30min`；正式长任务必须先有同 commit/launcher/env/model/runtime/producer/verifier 的有效 preflight，之后使用普通用户合法可用的最高优先级 partition/QoS，并依据显存适配、实测运行时间、当前可用性、排队估计与 time-to-result 选择合适 GPU，不预先锁死卡型。
- Gate D `PASS` 后先由 Phase 7 planning thread 亲自完成 trajectory/V3 绘图、图表 QA、outcome-blind 中文报告，并从每模型 Gate D candidate table 过滤 `width=4`、按 finite `S_RATE_TURN` 降序及 frozen tie-break 取 top available；该 checkpoint 不读取 test/outcome。用户已在 blocker 出现后明确批准 3/3/2 科学修订：Qwen/Llama 各取 top-3，Gemma 取仅有的 top-2。Gate E 分别固定 13/13/9 cells：每模型一个 no-loop baseline，加每个冻结窗口的 `k={2,3} × cache={first,last} × mode=block`；三模型合计 35 cells。Gemma 的结果只能称为 ABSTAIN 后的 post-ranking exploratory outcome。
- Gate E 使用标准 MMLU 5-shot test 全量 14,042 identities / 57 subjects。35/35 cells、共同 identity/completeness closure 和 pre-outcome verifier 全部闭合前不得读取部分 accuracy/gain 或据 outcome 改 panel；完整后只做一次 combined analysis 与 fresh verifier。正式任务必须先有同 immutable decision-critical path 的 `debug` `<30min` preflight，并把“代码路径正确”与“资源形态具有代表性”分别闭合：未真正形成冻结 `batch_size=16` 与代表性长序列/激活峰值的小样本或 underfilled batch 只能证明代码路径，不能用于估算 formal 显存。Packing 必须按模型及资源行为不同的 cell family 分别取证，不得从 Gate C、其他模型、小样本 A40 或其他 GPU 类型直接外推。若 debug 无法复现 formal 资源形态或计算节点镜像，正式运行先提交最小、可保留并计入 35 cells 的 formal canary，canary PASS 后才扩展剩余任务；launcher 修复后同样先验证一个 child/canary，不得立即重铺完整数组。计算节点 launcher 必须自包含，固定解释器与 shell，不依赖 Git executable、Modules 安装或登录节点 PATH；完整 Git/clean 校验在提交前闭合。正式资源遵循 normal-user 合法最高优先级，并以不 OOM 为硬边界最大化有效显存占用和总体吞吐，允许经上述证据证明的有界 worker-pool/backfill，但不得以调大 batch size、跨 cell batch 或共享 model/tensor state 改变科学路径。用户主动取消与异常退出必须分别记录。GPU 仍按显存/runtime/availability/queue/time-to-result 选择，并由 Gate E executor 使用唯一 60-minute full heartbeat。Gate E planning `PASS` 后由 planning 更新 outcome 报告、完成 consolidated audit 并终结 Phase 7；不自动创建 Gate F。
- 用户在 Gate E terminal 后明确追加 Gate F，覆盖上一条默认停止分支但不改变 Gate A–E 结论。Gate F canonical title 为 `execute-LoopScope-V3-Top2-Outcome-第7阶段-Gate F`；每模型取 Gate D V3.1 全候选 point-rank 前两名：Qwen `[14,17)`/`[12,15)`、Llama `[11,14)`/`[10,13)`、Gemma `[13,16)`/`[10,13)`，逐窗遍历 `k={2,3} × cache={first,last} × mode=block`，共 24 个新 loop cells。复用 Gate E 三个已验证 no-loop baselines，不重跑；Qwen/Llama batch 16、Gemma batch 8。24/24 新 cells 与三个 baseline 的共同 completeness/pre-outcome barrier 闭合前不得读取局部 outcome，闭合后仅做一次 combined analysis 与 fresh verifier。结果必须标记 `POST_TERMINAL_V3_POINT_RANK_TOP2_OUTCOME_EXPLORATORY`；Gate F planning PASS 后重新封存 Phase 7，不自动创建 Gate G。
- Phase 7 新证据使用可读路径、配置、model/data identifiers、命令、job IDs、manifest membership 和独立复算；不得新增、要求或比较 cryptographic/content digest。Phase 7 动态状态、exact executor、权限、audited commit/run 与 admission 只以 `../.planning/phase7/loopscope_phase7_control.md` 为准；科学合同与总体计划分别见同目录的 `loopscope_phase7_scientific_contract.md` 和 `loopscope_phase7_plan.md`。
