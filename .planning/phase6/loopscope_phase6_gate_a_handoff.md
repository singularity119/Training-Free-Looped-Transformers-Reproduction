# GATE_A_HANDOFF

```text
PROJECT=LoopScope
PHASE=6
CARD=PreAnswer-RBRV2
GATE=A
PLANNING_THREAD=019fb3de-2298-75f2-a083-0dca453ea79c
PLANNING_THREAD_TITLE=plan-audit-LoopScope 第六阶段
AUTHORIZED_EXECUTOR=019fb3fd-8cb5-7980-a0fb-6baaa48bd0d0
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-PreAnswer-RBRV2-第6阶段-Gate A
HANDOFF_STATE=AUTHORIZED_OPERATIONAL_SUPPLEMENT_1
HANDOFF_REVISION=3
SUPERSEDES=GATE_A_HANDOFF_REVISION_2_AFTER_BLOCK_PROVENANCE_MISMATCH
PLANNING_DECISION=BOUNDED_HPC2_READ_ONLY_PERMISSION_GRANTED
REPAIR_SCOPE=EXACT_EXTRACTOR_SOURCE_PROVENANCE_ONLY
TERMINAL_EVENT=GATE_A_FINAL_AUDIT_OR_BLOCK
```

Revision 3 保留 revision 2 的 authority-state reconciliation，并增加用户明确授权的
Gate A bounded HPC2 read-only source retrieval。它不改变 Gate A scientific/engineering
scope、Git base 或 Gate B...G 锁；executor 必须从头重跑只读 admission，一致后继续原
Gate A 工作。

## 1. Objective

以最短可审计路径实现 Phase 6 的本地纯 CPU 合同层，为 Gate B/C 的远端 provenance 与
真实四样本 smoke 提供唯一入口：

1. machine-readable Phase 6 card 与 closed schemas；
2. span-preserving final-answer extractor 与 generated-ID/text aligner；
3. 两遍式 generation/replay producer 的纯函数、sanitizer 与 CLI skeleton；
4. B0...B36 全词表 entropy/KL-to-final、hidden diagnostics 与相邻角距离计算；
5. 完整 V2 central-40% width 3/4/5/6 selector、panel freezer；
6. 独立 verifier 与 targeted synthetic tests。

本 Gate 不运行真实模型、数据或实验，不回答选出了哪个窗口。

## 2. Authoritative sources

按以下顺序重新读取，冲突时停止并向 planning 发送 `BLOCK`：

1. 用户最新授权；
2. live Git state；
3. `loopscope-tflt/AGENTS.md`；
4. `.planning/loopscope_phase6_control.md`；
5. 本 handoff；
6. `loopscope-tflt/docs/loopscope_phase6.md`。

研究协议使用
`/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md`。执行线程可以在本
Gate 范围内使用最多三个并发 subagent；不得 nested delegation。executor 对集成、
Git、测试、终态包和跨线程投递负唯一责任。

## 3. Admission 与 starting provenance

```text
repository=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
required_branch=loopscope
authorized_base_commit=e5563629aec404030c4d10f973a1532a771ede68
required_ancestor=4f59bd93eca4da3cbf458a93508f91c5b23912bc
remote=origin
```

授权时的预期既有 dirty state：

```text
 M AGENTS.md
?? docs/loopscope_phase6.md
```

- Phase 6 之前的资产路径与 Phase 4/5 文档 diff 已由 planning 以
  `e5563629aec404030c4d10f973a1532a771ede68` 提交并推送到 `origin/loopscope`。
- `AGENTS.md` 包含规划线程的 Phase 6 稳定政策修改：只读、不得覆盖、stash、reset、
  stage 或 commit。
- `docs/loopscope_phase6.md` 是规划线程创建的 Gate A 允许路径，可以在合同不变的前提下
  修订并由 Gate A commit。
- 若出现上述列表之外的既有修改，先判断是否落在本 Gate 路径且可明确归因；不能明确时
  `BLOCK_DIRTY_STATE_DRIFT`，不得自行清理。

任何写入、测试、提交、推送前运行：

```bash
git status --short --branch
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git merge-base --is-ancestor 4f59bd93eca4da3cbf458a93508f91c5b23912bc HEAD
git rev-parse --show-toplevel
```

必须精确满足路径、分支、base 与祖先合同。Gate A 可在 base 之上产生一个或少量单一目的
commit，但不得吸收受保护的既有 dirty 文件。

## 4. Frozen scientific contract

### 4.1 Cell 与 probe

```text
model=Qwen/Qwen3-4B-Instruct-2507
revision=cdbee75f17c01a7cc42f958dc650907174af0554
dataset=TIGER-Lab/MMLU-Pro
revision=b189ec765aa7ed75c8acfea42df31fdae71f97be
split=test
population=12032
fewshot=5 validation demos
decoder_layers=36
dtype=bfloat16
generation=greedy,do_sample=false,temperature=0,max_gen_toks=2048,until=Question:
loop=K3,block,Euler,step=1/3,horizon=1,cache=first,decode=full
```

Pass 1 是 deterministic native no-loop full CoT generation。Exact lm-eval 0.4.11
`custom-extract` 来自 `_default_template_yaml@356e937a...`：

```text
regex_pattern='answer is \(?([ABCDEFGHIJ])\)?'
filter tail=take_first
```

span-preserving extractor 必须逐字使用 case-sensitive regex/capture group 1、收集所有
match spans，并只接受唯一 span；多个 spans 不得沿用 outcome filter 的 `take_first`，
而必须 fail closed。`utils.py@74ab409c...` 只定义 renderer，不是 regex source。
Versioned aligner 将 span start 映射到承载第一个答案内容字符的 generated token，半开
区间边界归右侧 token。probe 是该 token 之前的最后一个 token。

Pass 2 重放：

```text
rendered_prompt_ids + generated_ids strictly before answer_first_token
use_cache=false
output_hidden_states=true
loop_insertions=0
```

任一 missing、ambiguous、unalignable、no preceding token、generation/replay ID mismatch
必须 fail closed；不得 fallback、丢样本或 complete-case。

### 4.2 Trajectories

读取 raw residual boundaries `B0...B36`：

- selector-only: full-vocabulary entropy `H[37]`；
- selector-only: `D_l=KL(p_l||p_36)[37]`；
- diagnostic-only: `RMS-L2(FinalNorm(B_l),FinalNorm(B_36))[37]`；
- diagnostic-only: cosine-to-final `[37]`；
- diagnostic-only: cosine-distance-to-final=`1-cosine` `[37]`；
- diagnostic-only: raw adjacent angular
  `acos(clamp(cos(B_l,B_l+1),-1,1))/pi` `[36]`。

`D_36=0`、final hidden RMS-L2/cosdist `=0`、cosine `=1` 必须按冻结 tolerance 验证。
hidden diagnostics 不得影响 eligibility、score、selection frequency 或 panel membership。

### 4.3 V2 selector

```text
central blocks=11...24
w3 starts=11...22 count=12
w4 starts=11...21 count=11
w5 starts=11...20 count=10
w6 starts=11...19 count=9
candidate_count=42
NewEligible=Scorable & NetPositive & SoftRelativeStable & BiphasicStable
S_RATE=sqrt(G_H*G_K)
RateStable=within-window H/K one-sided fixed-SE max-t lower bounds >0
bootstrap=category-stratified joint,R=2000,seed=20260801,ddof=1
tie=math.isclose(rel_tol=1e-12,abs_tol=1e-12)
selection_frequency_threshold=0.80
```

科学终态只能是：

```text
SELECTED_WINDOW
ABSTAIN_NO_RATE_STABLE_ELIGIBLE
ABSTAIN_NO_UNIQUE_TOP1
ABSTAIN_RATE_RANK_UNSTABLE
```

### 4.4 Panel 与 information barrier

Known same-cell registry：

```text
no-loop,15:18,6:9,10:13,25:28,4:7,5:8,22:25
```

known loop windows 不从 42 candidates 删除，也不改变 selected/ABSTAIN，但不能占
registry-excluded diagnostic High3/Low3 slots。

Gate E/F 的冻结 panel 是 unique union：

```text
fresh baseline
fixed 15:18
registry-excluded positive-finite diagnostic S_RATE High3
registry-excluded positive-finite diagnostic S_RATE Low3
selected window if distinct
```

若 registry-excluded positive-finite diagnostic universe 少于 6，必须
`BLOCK_DIAGNOSTIC_PANEL_UNDERPOPULATED`。

sanitized selector record 允许 span offsets、token indices、extractor/aligner hashes 与
scalar arrays；禁止 prompt/generated text、token IDs、answer content、answer-span hash、
prediction、gold、label、correctness、accuracy、gain、flip 或 outcome。

## 5. Allowed implementation

### 5.1 Writable repository paths

```text
configs/loopscope/phase6_*.json
src/tflt/loopscope/phase6_*.py
scripts/loopscope/run_qwen4_phase6_*.py
scripts/loopscope/verify_phase6_*.py
tests/test_loopscope_phase6_*.py
docs/loopscope_phase6.md
```

可读取并最小复用既有 Phase 4/5 模块与 fixtures，但不得修改它们。不得修改
`src/tflt/loopscope/__init__.py`、`src/tflt/cli.py`、wrapper/strategies/cache/config 或
其他既有 source。Gate A 的 CLI 入口直接使用 `scripts/loopscope/*.py`。

### 5.2 Local actions

- 纯 Python/NumPy CPU implementation、unit tests、compile、CLI help/dry-run；
- 只读检查本地 Git 历史与既有代码；
- 只对上述 allowlist 路径执行 explicit `git add`；
- 产生单一目的 commit，并 `git push origin loopscope`；
- 通过 `ssh -o BatchMode=yes -o ConnectTimeout=10 -o ClearAllForwardings=yes
  hpc2-hkustgz` 只读以下 remote source scope：
  - `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/lib/python3.11/site-packages/lm_eval/tasks/mmlu_pro/`
  - 同一 venv 的 `lm_eval-0.4.11.dist-info/METADATA`；
  允许 `stat/find/sed/grep/wc/sha256sum`，只用于 exact source/version/hash closure；
- 在同一 Gate 内最多两个 executor-owned 低风险修复 loop；
- 最多三个活跃 subagent，任务必须具体、有限且完全属于 Gate A；subagent 不得 Git
  commit/push、不得发送 terminal packet。

## 6. Forbidden actions

- Gate B 或之后任何工作；
- 上述 exact read-only scope 外的 HPC2 读取；
- 任何 remote write、remote Git sync、package mutation 或 environment mutation；
- remote model/data/cache/outcome/artifact 读取；
- 模型或数据集 load/download、network acquisition；
- torch/transformers/CUDA/GPU/Slurm、真实 generation/replay；
- 读取 Phase 4 outcome 数值/结果路径；
- 读取 gold、label、correctness、accuracy、gain；
- selector 正式运行、真实 panel 冻结或 outcome acquisition/unseal；
- 修改模型、任务、split、population、candidate、formula、seed、threshold、anchor、
  panel 或信息屏障；
- `git add -A`、`git add .`、stash、reset、checkout 覆盖、rebase、force push、
  merge main、删除或移动既有工件；
- 修改/提交受保护的既有 dirty 文件。

## 7. Required implementation and validation

### 7.1 Smallest sufficient increment

优先复用既有 `phase4_*` producer/schema 与 `phase5_relative_biphasic_v2_absolute_rate.py`
语义，只增加 Phase 6 所需的 anchor、two-pass、full-vocab/hidden diagnostics、closed
schema 与 multiwidth V2 glue。不得建设通用 hook framework、跨模型 adapter、在线 gating
或无关 refactor。

建议最小模块：

```text
phase6_anchor.py
phase6_schema.py
phase6_acquisition.py
phase6_selector.py
phase6_verifier.py
```

具体拆分可在 allowlist 内调整，只要职责和测试覆盖不变。

### 7.2 Must-pass targeted checks

至少覆盖：

1. exact source-bundle three hashes、literal case-sensitive regex 与 single-match equivalence；
2. extractor unique/missing/ambiguous/重复 cue，多个 spans 必须 fail closed；
3. span start 在 token 内、恰在 token 边界、Unicode/byte offset、special token；
4. answer-first-token 与 preceding probe off-by-one；
5. generation-prefix/replay IDs exact，不一致 fail closed；
6. raw `B0...B36`、H/D/hidden 37、angular 36 schema；
7. KL-to-final、hidden endpoints、raw-vs-FinalNorm angular；
8. full-vocabulary entropy/KL 与 selector 输入字段闭合；
9. 42-candidate exact enumeration；
10. V1 compatibility fixtures、V2 rate/RateStable/tie/frequency/三类 ABSTAIN；
11. hidden diagnostics 改值不能改变 eligibility/selected/High3/Low3；
12. known registry exclusion、deduplicated panel union、少于 6 blind diagnostic window
    fail closed；
13. forbidden-field rejection 与 sanitizer 不泄漏 answer/outcome；
14. independent verifier 对正常与关键失败 fixtures 判定正确。

执行并回报 exact command/exit code：

```bash
git diff --check
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_loopscope_phase6_*.py'
python3 -m compileall -q src/tflt/loopscope/phase6_*.py scripts/loopscope/*phase6*.py
python3 scripts/loopscope/run_qwen4_phase6_prepare.py --help
python3 scripts/loopscope/run_qwen4_phase6_prepare.py --dry-run
python3 scripts/loopscope/verify_phase6_gate_a.py --help
python3 scripts/loopscope/verify_phase6_gate_a.py --card configs/loopscope/phase6_pre_answer_v2_card.json --self-test
```

若 executor 采用不同但等价的 phase6 script 名，必须在 terminal packet 给出实际命令，
且仍满足 help、dry-run 与 independent verifier。无需跑整个历史测试套件，除非修改虽被
禁止但意外发生；发生时应先停止而不是扩大测试。

### 7.3 Required artifacts

至少产生：

```text
configs/loopscope/phase6_pre_answer_v2_card.json
configs/loopscope/phase6_trajectory_schema.json
configs/loopscope/phase6_selector_freeze_schema.json
Phase 6 source modules
Phase 6 prepare/verify scripts
Phase 6 targeted tests
docs/loopscope_phase6.md
```

card 必须包含 frozen cell、anchor、candidate/selector、seeds、known registry、information
barrier、allowed scientific states 与 engineering BLOCK states。schemas 必须
`additionalProperties=false` 或实现等价 closed-world validation。

## 8. Executor verification vs planning acceptance

Executor 做上述 targeted engineering verification，记录 exact diff、commands、exit codes
并提交/push。不要再做独立“自我审计”、全历史套件或假想 tamper matrix。

Planning 收到 terminal event 后只检查：

1. exact executor/branch/commit 与受保护 dirty state；
2. card/schema/selector 的 42-candidate、anchor、hidden-only 与 information barrier；
3. targeted tests/verifier 的真实通过证据。

跨 Gate provenance、远端 runtime 与最终科学结论留到 Gate B/C 和 phase-end audit。

## 9. Agility budget

1. smallest increment：仅实现当前 Qwen3-4B-Instruct/MMLU-Pro Phase 6 路径；
2. tests：只跑 Phase 6 targeted suite、compile、help/dry-run 与 verifier；
3. earliest real experiment：本 Gate 没有真实模型；完成后立刻交给 Gate B 远端 CPU/
   provenance，再由 Gate C 运行四样本；
4. stop rule：must-pass checks 全部通过并产生可审计 commit 后立即回传，不增加框架化、
   portability、全面文档或 defense-in-depth hardening。

## 10. Repair and escalation

- executor-owned repair：最多两个低风险循环，只能修复同一 Gate 正常路径工程缺陷；
- planning audit-returned repair：普通情况下最多一个 bundled cycle；
- 任何科学公式/candidate/anchor/schema semantics/seed/threshold/panel/model/data/split
  变化，立即 `BLOCK`；
- 若只是普通运维权限不足，发送 `BLOCK`，planning 可按用户委托直接授予最小充分补充
  权限；executor 不得自授权；
- 不可逆破坏、凭据/秘密、提前 outcome unseal 或 Phase 6 外扩必须停止。

## 11. Terminal delivery

完成时只发送一次结构化：

```text
GATE_A_FINAL_AUDIT

Project/phase: LoopScope Phase 6
Gate: A
Execution thread: 019fb3fd-8cb5-7980-a0fb-6baaa48bd0d0
Result: AUDIT_REQUESTED
Authorized starting state:
Final branch/commit/dirty:
Files changed:
Exact commands:
Tests and exit codes:
Remote host/job IDs/run root: NONE
Decision-critical artifact paths and hashes:
Observed result:
Deviations/errors:
Repair loops used:
Subagents used and integrated:
Completed state:
Actions not taken:
Protected-state confirmation:
Decision requested: PASS / PASS_WITH_FIXES / BLOCK
```

用 Codex thread message 工具精确投递到 planning thread
`019fb3de-2298-75f2-a083-0dca453ea79c` 并保留 delivery success。executor 本地 final 只是
副本。若工具不可用或确认失败，输出同一 packet，并标记
`TERMINAL_DELIVERY_UNCONFIRMED`，不得声称已通知 planning。

在 terminal event 前不得进入 Gate B。
