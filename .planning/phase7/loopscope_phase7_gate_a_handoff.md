# GATE_A_HANDOFF

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=A
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a00159-6c55-72b3-bbf8-5ac798868258
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-三模型Provenance与轨迹探测实现-第7阶段-Gate A
AUTHORIZATION_STATE=AUTHORIZED
TERMINAL_EVENT=GATE_A_FINAL_AUDIT_OR_BLOCK
NEXT_GATE_AUTHORITY=NONE
```

## 1. Objective 与 non-objectives

Gate A 只回答：三个冻结 Base 模型是否能在同一 Phase 7 科学合同下被准确定位、采集和复算，
以及支持 Gate B 四 identity 真实 smoke 所需的最小 producer/verifier/V3 depth-general 代码是否
就绪。

本 Gate 不运行模型 forward、GPU/Slurm、MMLU trajectory、bootstrap selector、loop、test、
outcome 或绘图；不创建 Gate B executor，也不进入 Gate B。

## 2. Authoritative sources

- stable policy: `loopscope-tflt/AGENTS.md`，尤其第 16 节；其中现有未提交修改为 planning-owned
  protected state，executor 不得修改或提交。
- mutable control: `.planning/phase7/loopscope_phase7_control.md`
- scientific contract: `.planning/phase7/loopscope_phase7_scientific_contract.md`
- phase plan: `.planning/phase7/loopscope_phase7_plan.md`
- shared selector method:
  `.planning/selector_rules/aggregate_common_turn_v3_absolute_rate.md`，版本
  `AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE@3.1.0`。若其中历史 digest 约定与 Phase 7
  的 no-digest 规则冲突，以 Phase 7 contract 和 skill 的 no-digest 规则为准。
- orchestration skill: `research-gate-orchestrator`; 使用前完整重读其 `SKILL.md` 与
  `references/protocol.md`，并以敏捷实验主线、材料性验收和 deletion test 为准，避免通用
  框架化、重复全量审计或非正常路径防御。访问 HPC2 时同时遵循 `hpc2-hkustgz-ssh` skill。

优先级为：用户最新指令 > live Git/HPC fact > `AGENTS.md` > control > plan/runbook > 历史。
handoff 与 control 若在 executor/Gate/base/权限上不一致，停止并 `BLOCK`，不得自行调和。

## 3. Starting provenance 与 protected dirty state

```text
local repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
required branch=loopscope
base commit=37241f6fb52aa6ee89a5b4e24ac607f7312e77ac
required ancestry base=4f59bd93eca4da3cbf458a93508f91c5b23912bc
remote source checkout=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
remote read-only reproduction checkout=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction
```

激活时以下 tracked files 已有前序未提交修改，全部受保护：

```text
AGENTS.md
docs/loopscope_phase2.md
docs/loopscope_phase4.md
docs/loopscope_phase5.md
docs/loopscope_phase6.md
scripts/loopscope/run_qwen4_phase4_p4b.py
src/tflt/loopscope/phase4_outcome.py
```

不得 reset、checkout、stash、clean、覆盖、整理、暂存或提交这些文件。不得用 broad `git add`；
每次提交前只显式暂存本 handoff 授权的 Phase 7 paths，并检查 staged membership。若发现新的
未知 dirty overlap 会影响 Phase 7 provenance，停止并报告；无关 dirty state 保留并在终态说明。

## 4. Frozen science

### 4.1 Model-task cells

```text
Qwen/Qwen2.5-3B
meta-llama/Llama-3.2-3B
google/gemma-2-2b
dataset=cais/mmlu
split=validation
population=1531 identities / 57 subjects
num_fewshot=5
fewshot_split=dev
renderer=lm-eval 0.4.11 compatible standard plain non-chat MMLU
probe=full rendered Answer: prefix final non-padding token before answer continuation
runtime_dtype=bfloat16
batch_size=1
forward=one native zero-loop use_cache=false per identity
```

`Qwen/Qwen3-1.7B-Base` 和 `Qwen/Qwen3-4B-Base` 不属于本阶段 membership，禁止 fresh
acquisition、重评分、重绘或作为 candidate/window 输入。

### 4.2 Choice、boundary 与 selector

- Exact continuation surfaces 仅为 `" A"`, `" B"`, `" C"`, `" D"`；每个 tokenizer
  必须各为一个、彼此不同的 token。检查在内存中完成，不持久化 token/input IDs。任一失败：
  `BLOCK_CHOICE_SURFACE_NOT_SINGLE_TOKEN`，不得 fallback。
- 对真实 `L`，boundaries 为 raw `B0...BL`；`B0` 是 embedding output，`Bj` 是 block j
  后且 FinalNorm 前的 residual。必要时用 temporary final-norm forward-prehook 捕获 raw
  `B_L`，每个 forward hook call 恰为 1，`finally` 移除；`FinalNorm(raw B_L)` 闭合原生
  final hidden，禁止 double norm。
- V3 唯一输入为 choice Entropy `H` 和 `KL(p_l||p_L)`；choice softmax 使用 float64 stable
  implementation。hidden RMS-L2/cosine/cosine-distance 与 raw adjacent angular distance
  只作 diagnostic。
- `t(L)=ceil(0.30L)`，central blocks 为闭区间 `[t(L),L-t(L)-1]`，widths `{3,4,5,6}`，
  只保留完整包含的半开窗口，顺序 `width ascending, start ascending`。
- V3.1 的 equal-category macro、joint bootstrap、eligibility、`S_RATE_TURN`、ranking frequency
  与三类终态严格按 frozen contract；Gate A 只实现和用 synthetic fixtures 验证，不读取正式数据
  或执行 2,000-replicate 正式 scoring。

## 5. Allowed actions

### 5.1 Local implementation

允许在 `loopscope-tflt` 内新建或修改以下范围：

```text
src/tflt/loopscope/phase7_*.py
configs/loopscope/phase7_*
scripts/loopscope/phase7_*.py
scripts/loopscope/run_phase7_*.py
scripts/loopscope/verify_phase7_*.py
tests/test_phase7_*.py
docs/loopscope_phase7.md
src/tflt/loopscope/__init__.py        # 仅必要的最小导出，若当前不脏
src/tflt/cli.py                       # 仅必要的显式 opt-in 最小注册，若当前不脏
```

优先保持 Phase 7 sidecar 独立，不为通用化重构旧 Phase 1–6 代码。可以读取全仓库背景；可以
运行 targeted local tests、compile、CLI help/dry-run、`git diff --check`。本地不得下载模型或
数据，不得运行 torch/model forward。

Git 允许：read-only status/diff/log/fetch；对授权路径作 focused commit；仅当 `origin/loopscope`
与本地 ancestry 允许普通 fast-forward、staged membership 纯净且测试通过时，push 自己的
Phase 7 commit 到 `origin/loopscope`。禁止 force push、rebase、merge、改 main、提交 protected
dirty paths 或把 planning files 纳入 Git commit。若安全 push 条件不成立，保留本地 commit
并在终态明确报告，不得改写历史。

### 5.2 HPC2 read-only background closure

允许使用 `hpc2-hkustgz` / account `xhuang225`，所有 SSH 明确设置
`ClearAllForwardings=yes`，对以下 project-relevant roots 作有界只读检查：

```text
/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction
/hpc2hdd/home/xhuang225/shared/hf_home
/hpc2hdd/home/xhuang225/shared/datasets
/hpc2hdd/home/xhuang225/shared/uv
/hpc2hdd/home/xhuang225/shared/wheelhouse/tflt-cu121
```

允许读取项目代码、lock/env/version 信息、已有 model/tokenizer snapshot 的 config/tokenizer
metadata、MMLU renderer/task code和 cache membership；允许在已审计环境中运行不加载模型权重、
不读取 validation rows/gold 的 CPU config/tokenizer admission，以闭合 exact resolved revision、
tokenizer revision、真实 `L`、decoder/embedding/final-norm/lm-head 路径及 choice-surface 单 token
性质。不得输出或落盘 token IDs。

Gate A 禁止远端写入、`git pull/switch/checkout/install`、环境升级、cache 修改/下载、license
接受、credential 操作、模型权重加载/forward、dataset sampling、CUDA、GPU、Slurm、job 提交、
run root 创建或 artifact mutation。缓存缺失、模型 gated/不可读或必须新增下载/credential 时，
返回窄 `BLOCK_MODEL_ACCESS`，不得替换模型。

### 5.3 Bounded delegation

Executor 被明确允许按 `research-gate-orchestrator` 使用 subagent，但仍为唯一负责人。只有
当确实缩短 Gate A 时，最多同时使用三个 bounded subagents，
范围只能是：一个特定架构代码定位、一个 isolated synthetic test、或上述 exact roots 内的
read-only evidence extraction。Subagent 不得写远端、操作 Git integration、运行模型/GPU/Slurm、
跨 Gate、修改科学合同或发送 terminal event；executor 必须亲自复核并整合其结果。

## 6. Required work packages

1. **Admission closure**：从 live local/HPC2 evidence 冻结三模型 resolved model/tokenizer
   revision、`L`、decoder container、embedding、final norm、lm head 和 cache/access 状态；在内存
   验证 choice surfaces。发布 readable 三模型 admission table 和由公式生成的 candidate-domain
   table，不记录 token IDs 或 digest。
2. **Narrow producer**：实现三架构 adapter、一次-forward raw-boundary capture、exactly-once
   FinalNorm、choice distribution 与六类 scalar trajectory reduction、closed sanitized schema 和
   write-once producer/launcher contract。不得修改 TFLT loop wrapper/strategy/cache/config。
3. **Verifier**：实现 formal record membership/schema/shape/finite/zero-loop/probe/final-norm
   closure verifier；禁止字段至少覆盖 gold/label/correctness/test/outcome/raw/full logits/probs、
   hidden tensors、token/input IDs、prompt text、loop residual。
4. **V3 depth-general core**：实现任意 `L` 候选域、V3.1 point/bootstrap/rank/terminal 逻辑与
   独立复算入口；正式数据和正式 scoring 仍锁定到 Gate D。
5. **Focused verification and docs**：运行下节命令；记录 Gate B 的 exact launcher/verifier
   contract、三个模型四 identity smoke 所需资源字段和 `<30min` debug preflight 条件。

## 7. Required validation 与 must-pass

实现后执行并回传 exact command/exit code；若最终文件名与下列 glob 匹配，可在终态展开为
具体文件列表：

```bash
git status --short --branch
git rev-parse --show-toplevel
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git merge-base --is-ancestor 4f59bd93eca4da3cbf458a93508f91c5b23912bc HEAD
python3 -m compileall src/tflt/loopscope/phase7_*.py scripts/loopscope/phase7_*.py scripts/loopscope/run_phase7_*.py scripts/loopscope/verify_phase7_*.py tests/test_phase7_*.py
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase7_*.py'
PYTHONPATH=src python3 scripts/loopscope/run_phase7_base_trajectory.py --help
PYTHONPATH=src python3 scripts/loopscope/verify_phase7_base_trajectory.py --help
PYTHONPATH=src python3 scripts/loopscope/run_phase7_v3.py --help
PYTHONPATH=src python3 scripts/loopscope/verify_phase7_v3.py --help
git diff --check
git diff --cached --name-only
```

若最小实现合理合并了入口，可省去不存在的冗余脚本，但必须保留一个 trajectory producer、
一个 record verifier、一个 V3 analyzer 和一个 fresh-process V3 verifier 的清晰可调用入口，并
在终态给出实际等价命令。不得为了满足文件名模板制造无用 wrapper。

Must-pass：

- 三个 exact Base repo 可读，未替换为 Instruct、量化、镜像或其他规模；
- 三个 tokenizer 的四个 exact choice surfaces 均 single-token 且互异；
- 三架构 raw `B0...BL`/prehook/final-norm 路径由实现与 focused fake-module tests 覆盖；真实
  CUDA 数值闭合明确留给 Gate B；
- 任意 `L` candidate enumeration、V3 formulas/terminal 与 forbidden-field rejection 有
  targeted tests；
- local compile/tests/help/dry-run/diff-check 通过；
- staged/commit membership 不包含 protected dirty paths；
- Gate B 所需 launcher/verifier/schema 已就绪，不需要新科学选择。

## 8. Agility budget 与 repair

```text
smallest increment=one narrow three-architecture sidecar producer + closed verifier + depth-general V3 core + focused tests/CLI
earliest real experiment=Gate B four-identity debug smokes after Gate A PASS
minimum decisive checks=choice-token closure, raw-final-boundary design, arbitrary-L candidates, forbidden-field rejection
stop rule=three-model CPU admission and focused verification pass -> send terminal packet; no framework refactor/full suite/GPU
```

同一 Gate、同一科学合同和授权路径内的低风险 local engineering repair loops 可继续，保留材料性
进展即可。不得把修复扩大到科学参数、protected files、remote mutation、下载、GPU/Slurm 或
下一 Gate。真实 model access/choice surface 科学合同失败、必须改 prompt/metric/model、或安全
push/dirty provenance 无法闭合时发送真正的 `BLOCK`。

Planning acceptance 只抽查 1–3 个决定性事实：三模型 admission/choice closure、Phase 7 diff 与
protected membership、focused tests/CLI。更广的跨 Gate audit 留到 Phase 7 末尾。

## 9. Terminal delivery

完成后向 planning thread `01a0013e-71c3-7c90-a547-4059b462dc7e` **精确发送一次**：

```text
GATE_A_FINAL_AUDIT
Project/phase:
Gate:
Execution thread:
Authorized starting branch/commit/dirty state:
Final branch/commit/dirty state:
Files changed:
Exact commands and exit codes:
HPC2 read-only checks and exact roots:
Three-model admission/candidate-domain result:
Tests and observed result:
Deviations/errors/repair loops:
Actions deliberately not taken:
Protected-state confirmation:
Decision requested: PASS / PASS_WITH_FIXES / BLOCK
```

使用 app 的 cross-thread message 工具并保留可见成功响应。若无法确认投递，executor 必须在
自身最终输出给出完整 `TERMINAL_DELIVERY_UNCONFIRMED` paste-ready packet，不得声称 planning
已收到。发送 terminal 后立即停止；不得轮询 planning、修改控制文件、创建 Gate B 或继续执行。

Gate A 无长任务、无 job、无 heartbeat。Gate B、C、D 与 planning closeout 仍锁定。
