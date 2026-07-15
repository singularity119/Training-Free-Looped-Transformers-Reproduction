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
- `../.planning/loopscope_phase1_control.md` 与 `../.planning/loopscope_phase2_control.md` 分别是已关闭第一、第二阶段的历史控制面；`../.planning/loopscope_phase3_control.md` 是第三阶段当前唯一的可变全局控制文件。当前 Gate、Gate 决策、线程分配、已审计 commit、下一步授权条件均以 Phase 3 control 为准。
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
- 第二阶段目标：围绕第一阶段局部结果验证“可迭代精炼区”机制——区分持续有益修正、短暂最优、无益扰动和错误过度自信；先用小范围 `k` 轨迹与逐样本分析理解机制，只有机制得到支持后才讨论无标签选窗或跨模型扩展。
- 第三阶段目标：固定 Qwen3-1.7B × MMLU × K=2 配方，只利用 validation 全量 1,531 条样本的一次 no-loop 层级 choice trajectory，检验局部 entropy–KL 双重反转能否在同一 model-task cell 内恢复、富集或排除 loop windows；测试/full outcome 不得进入 selector。
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

凡当前 Gate executor 已提交 HPC2/GPU/Slurm 作业，且作业在一次提交确认后仍需较长时间排队或运行，executor 必须停止人工轮询、shell sleep、循环 `squeue/sacct` 或为了维持 thread active 而空转，改为在自己的执行线程中创建真实的 Codex automation/heartbeat。默认按预计时长采用约 10 分钟、30 分钟或 60 分钟 cadence；等待中的 executor 显示 idle 是正常状态。每次 automation 唤醒只允许执行一次有界、只读状态检查（例如带 `BatchMode=yes`、`ConnectTimeout` 与 `ClearAllForwardings=yes` 的 SSH，加一次 `squeue/sacct`、必要的短日志尾部和预期工件存在性检查），然后立即结束本次唤醒；状态未发生材料变化时不打扰用户。automation 不获得提交、重试、requeue、取消、改参、改 throttle、修复、写工件或科学裁决权限；这些动作仍须来自该 Gate 原有 handoff 或新的明确授权。遇到 terminal success、terminal failure、SSH trust 异常或其他材料性 blocker 时，automation 只把证据送回原 executor；executor 恢复后先暂停/删除该 monitor，再按 Gate 协议继续验证或发送唯一终态事件。planning/audit 线程不替 executor 做 heartbeat 轮询。

## 12. 第三阶段稳定科学与治理边界

- 第三阶段 primary selector source 是 MMLU validation 全部 1,531 条唯一样本的一次普通 no-loop forward；Phase 1 validation-512 只作 secondary robustness，不参与 variant 选择。
- selector 只读取逐层四选项 choice distribution、entropy、KL-to-final、CE 及其局部 SHIFT/FLANK/CONSENSUS 结构；target gold、correctness、loop residual、full accuracy 与答案翻转均禁止进入 selector producer。
- 正式 loop outcome 固定使用与 validation-1531 样本级不相交的 MMLU test-14042 paired evidence；历史 13 窗只作 development，尚未查看 outcome 的 12 窗才承担 width-4 blind completion。
- 核心 loop 配方继续冻结为 `K=2、block、damped_euler、alpha=1、beta=0、cache_strategy=last、decode_mode=bypass`；任何 K、alpha、strategy、model 或 task 变化必须另立科学卡。
- Phase 3 Gate 名称和当前状态以 `../.planning/loopscope_phase3_control.md` 为准；本文件第 9 节 A–E 是 Phase 1 历史门，不得用来推断 Phase 3 当前授权。
- 第三阶段不设置 GPU 数量、GPU-hours 或墙钟硬上限。GPU Gate 只有在明确授权、科学矩阵冻结并满足 admission 后，才能按 HPC 空闲资源合理并行并尽快提交；资源变化不得改变样本、窗口、配方、统计或 retry 语义。
