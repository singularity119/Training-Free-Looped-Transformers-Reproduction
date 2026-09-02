# GATE_A_HANDOFF

```text
Project/phase=LoopScope Phase 5
Hypothesis=H5_QWEN4B_BASE_MMLU_TRAJECTORY_SELECTION_V1
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019f8631-3b66-7d40-8cae-d21c37ed27c1
Executor title=execute-LoopScope-Trajectory-Selection-第5阶段-Gate A
Gate=A
State=AUTHORIZED
Terminal event=GATE_A_FINAL_AUDIT or BLOCK
```

## 1. Objective and non-objectives

Gate A 只完成 Phase 5 合同/provenance 最小闭环：把已批准的 model/data/renderer/
trajectory/selector/panel/statistics 配方写入 hash-closed card 与窄 schema/verifier，用本地
synthetic/contract tests 证明关键数学与拒绝边界，并将 historical baseline/`15:18`
决定为 bit-exact reuse 或 fresh acquisition。

本 Gate 不做 formal validation trajectory，不运行 MMLU test outcome，不选窗，不产生
High3/Low3，不提交 GPU/Slurm，不制定 Gate B 具体执行计划。

## 2. Authoritative sources

1. policy: `loopscope-tflt/AGENTS.md`
2. mutable control: `.planning/loopscope_phase5_control.md`
3. scientific plan: `报告/LoopScope_第五阶段总体目标与Gate计划.md`
4. runbook skeleton: `loopscope-tflt/docs/loopscope_phase5.md`
5. exact handoff: this file

冲突优先级为用户最新指令 > live Git/HPC immutable evidence > `AGENTS.md` > control >
runbook/history。若 handoff 与 control 的 executor/Gate/权限不一致，在任何 mutation 前
`BLOCK`。

## 3. Starting provenance

```text
repository=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope
base_commit=ee676a8d13d1bed4aa783da0606f5dd60ec8a11f
origin_commit=ee676a8d13d1bed4aa783da0606f5dd60ec8a11f
expected_preexisting_dirty=modified AGENTS.md; untracked docs/loopscope_phase5.md
planning_only_files_outside_git=../.planning/loopscope_phase5_control.md; ../.planning/loopscope_phase5_gate_a_handoff.md; ../报告/LoopScope_第五阶段总体目标与Gate计划.md
protected_reference_base=4f59bd93eca4da3cbf458a93508f91c5b23912bc
```

你不是代码库中唯一工作的 agent。保留并适配 planning 线程已有改动；不得 reset、
stash、checkout 或回退他人文件。执行前必须记录 repo root、branch、HEAD、status 和
protected-base ancestry。

## 4. Frozen scientific contract

### Model/tokenizer

```text
repo=Qwen/Qwen3-4B-Base
commit=906bfd4b4dc7f14ee4320094d8b41684abff8539
architecture=Qwen3ForCausalLM
layers=36
hidden_size=2560
vocab_size=151936
dtype=bfloat16
tie_word_embeddings=true
final_norm_weight=model.norm.weight
tied_output_weight=model.embed_tokens.weight
spaced_choice_ids=A:362,B:425,C:356,D:422
plain_choice_ids=A:32,B:33,C:34,D:35
```

### Data/evaluator

```text
dataset=cais/mmlu
dataset_revision=c30699e8356da336a370243923dbaf21066bb9fe
task_group=mmlu
num_fewshot=5
fewshot_split=dev
trajectory_split=validation
trajectory_population=1531 identities / 57 subjects
outcome_split=test
outcome_population=14042 identities
lm_eval=0.4.11
chat_template=false
multiturn=false
```

validation/test canonical identities 必须严格分离。Gate A 产物不得包含 target answer/gold/
correctness/outcome values。

### Trajectory/selector

- boundaries=`B_0..B_36`; width-4 starts=`0..32`; SHIFT=`1..31`; CONSENSUS=`4..28`;
  `15:18`=`B_15 -> B_19`.
- choice space 为 exact spaced A/B/C/D continuation logits 的 stable float64 softmax。
- 五轨迹：choice entropy、`KL(p_l||p_36)`、final-norm RMS-L2、cosine、`1-cosine`。
- generic deltas 为 exit-entry；selector 派生 `E=H_entry-H_exit`、`K=D_exit-D_entry`。
- primary selector 只有 entropy–KL CONSENSUS；geometry 不入 selector，cosine distance 不重复投票。
- eligibility、tie tolerance 1e-12、selection frequency >=0.80 和 ABSTAIN 逻辑按 control。
- trajectory bootstrap: subject-stratified joint bootstrap, R=2000, seed=20260722.

### Outcome panel/statistics

- baseline + known `15:18` + registry-excluded selected-inclusive High3 + disjoint Low3，<=8 unique
  cells；保留 raw High3 metadata。
- `15:18` 和任何已知 outcome 不占 blind group 名额。
- selected-baseline success uses point gain>0 and always reports paired CI.
- comparator margin=0.30 pp; outcome bootstrap R=2000, seed=20260723.
- default historical baseline/15:18 verdict=`FRESH_ACQUISITION_REQUIRED`；只有 exact model/
  tokenizer/data/renderer/evaluator/recipe 和逐样本 artifact 全部闭合才可例外复用。

## 5. Allowed actions

### Local files

优先只新增窄边界：

```text
configs/loopscope/phase5_card.json
configs/loopscope/phase5_card_schema.json
configs/loopscope/phase5_known_outcome_registry.json
configs/loopscope/phase5_gate_a_receipt.json
src/tflt/loopscope/phase5_schema.py
scripts/loopscope/verify_phase5_gate_a.py
tests/test_loopscope_phase5_gate_a.py
```

可按必要最小调整 `docs/loopscope_phase5.md`。planning 已修改的 `AGENTS.md` 与
`docs/loopscope_phase5.md` 已经授权由 Gate A executor 审阅后与 Gate A 单目的提交
一并 commit/push，但不得改写 planning authority 或删减冻结边界。

若一个更少文件的实现足以完成 must-pass，可不创建上述某个 optional file；不得
为了 schema 优雅建设通用 framework。

### Git

- 可在 `loopscope` 上使用 one-purpose commit，并普通 push 到 `origin/loopscope`。
- 不得 rebase、force-push、merge main、改写历史或提交模型/数据/大工件。
- 仅 stage Gate A 授权文件和 planning 明确交付的 `AGENTS.md`/runbook。

### HPC2/network/data

- host=`hpc2-hkustgz`; project clone=`/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`;
  shared HF root=`/hpc2hdd/home/xhuang225/shared/hf_home`.
- 可用 `ClearAllForwardings=yes` 连接 login node，只读检查 source/env/cache/config/provenance。
- 可用已有 venv 做 CPU import/config/tokenizer 检查和最小 fake/synthetic Euler 3-call
  reproducer；不得实例化正式 model weights 或做 forward。
- 可对冻结 `cais/mmlu` revision 做 renderer/fingerprint/identity provenance 采集；输出必须
  strip target gold/answer/correctness，不得打印或人工查看任何 test answer。
- 如冻结 model/dataset 还缺必要文件，可用 `HF_ENDPOINT=https://hf-mirror.com`
  下载精确 pinned revision，不得改 repo/revision。
- historical reuse 审计只允许检查已知 project roots 中的 argv/revision/manifests/
  per-sample artifact presence/hash/schema；禁止解析 accuracy、gain、predictions 或未登记
  window outcome。若不能快速闭合，直接冻结 fresh，不扩大搜索。

## 6. Forbidden actions and protected state

- Gate B/C/D 任何工作；formal 4-identity smoke、validation-1531 trajectory、test-14042
  outcome、selector ranking/panel construction。
- GPU、Slurm/scheduler submission、job retry/cancel/requeue、正式模型 forward。
- 读取、解析或暴露任何尚未登记的 Phase 5 test/window outcome。
- 修改 `src/tflt/wrapper.py`、`strategies.py`、`cache.py`、`config.py`、`eval_runner.py`
  或现有 Phase 1–4 cards/selectors/outcomes/reports/run artifacts。
- K/width/model/task/seed/threshold/panel/margin 变更；geometry selector；任何 representation
  intervention。
- destructive operations、overwrite old artifacts、删除/stash/reset 现有改动。

若 protected implementation path 存在对 normal-path must-pass 的材料性阻塞，先提供最小
reproducer 和拟议 diff，发 `BLOCK`；不得自行改权。

## 7. Required implementation and artifacts

### Card/schema

card 必须无 placeholder 地冻结：

- exact model/tokenizer/data/evaluator/renderer 身份；
- 36-layer boundary/support universe 与 inclusive window mapping；
- five-trajectory scalar/numeric/persistence contract；
- CONSENSUS formulas、eligibility、tie/frequency/ABSTAIN；
- known registry、selected-inclusive High3/Low3 panel constructor；
- outcome metrics、bootstrap streams、terminal labels/margin；
- explicit forbidden selector/outcome fields and fail-closed policies。

schema/verifier 只需保护 Gate B 会实际消费的 normal path。不做穷尽式 tamper matrix。

### Known-outcome registry

必须登记用户提供公开背景 baseline=73.27%、`15:18`=73.61%/+0.34 pp，
明确标记 `KNOWN_BACKGROUND_NOT_FORMAL_RESULT`，将 `15:18` 排除于 blind group。
若 exact per-sample closure 不成立，冻结 baseline 与 `15:18` 都是 fresh。

### Gate-level receipt

一个 receipt 足够，记录 card/schema/registry hashes、local tests、remote provenance 摘要、reuse
verdict、Git base/final commit 和未执行的禁止动作。不建 micro-receipt tree。

## 8. Targeted executor verification

先实现最小增量，然后运行决定性测试。至少覆盖：

1. card/schema/registry exact validation 与 self-hash/receipt 重算。
2. L=36 得到 B0..B36、33 starts、SHIFT 1..31、CONSENSUS 4..28，`15:18`
   严格是 B15->B19。
3. spaced/plain A/B/C/D 的唯一单-token IDs 重现；不同则 fail closed。
4. synthetic 五轨迹的 entropy/KL/RMS-L2/cosine/derived-distance 重算和 final-boundary
   closure；非 finite、zero-norm cosine、禁止字段必须失败。
5. CONSENSUS 中心/参照符号、lower95 eligibility、score、tie 和三种 ABSTAIN；
   geometry 不影响 score。
6. known-window exclusion、selected-inclusive High3、raw High3 preservation、Low3 disjoint、
   dedup 和 <=8 cells。
7. current strategy alias 的 K=3/alpha=1 最小 CPU/synthetic reproducer 得到 3 body calls、
   step=1/3、horizon=1，不改 protected strategy code。
8. 现有 Phase 3 choice-space 关键测试中与复用路径直接相关的最小子集。

推荐命令由 executor 按实际文件冻结；不要为了形式运行多次 full suite。
只在更改了广泛共享路径且存在材料回归风险时运行一次 full suite；默认不需要。

## 9. Must-pass acceptance

Gate A 可请求 planning 审计的最小条件：

1. formal card 无 placeholder，其决策相关字段与 control 完全一致。
2. schema/verifier 可从 card/registry/receipt 和 synthetic records deterministic 重算关键
   contracts，目标测试全部 exit 0。
3. exact snapshot/tokenizer、dataset/renderer/evaluator provenance 可解释，不依赖
   test outcome；baseline/15:18 有明确 reuse/fresh verdict。
4. protected paths 和 Phase 1–4 artifacts 未改，没有 GPU/Slurm/formal forward/outcome access。
5. 产物已 one-purpose commit 并普通 push，working tree 只允许 planning-only outer files
   保持在 repo 之外。

style、文字、额外 schema 美化、手工 tamper 防御和通用化不是 must-pass。

## 10. Agility budget and repair

1. 最小实现：一张 card + 一个窄 schema/verifier + 一个 known registry + 一个
   targeted test file + 一个 Gate receipt。
2. 最早证据：先让 boundary/token/trajectory/selector/panel synthetic tests 通过，再做一次
   bounded HPC provenance 检查。
3. 删除测试：不能改变科学配方、数值结论、泄漏风险、protected state 或
   Gate B admission 的检查，移出 critical path。
4. 停止规则：must-pass 一旦满足就 commit/push 并发送终态包，不继续建框架或
   寻找假想缺陷。

executor 可在不改冻结科学/权限/路径的前提下做最多两次低风险自主 repair
loops。planning 审计返回的普通 repair cycle 上限一次。任何可能改变
model/data/split/sample/seed/metric/threshold/candidate/panel/schema 科学语义、protected path
或外部权限的选择必须 `BLOCK`。

## 11. Executor delegation

如确实可并行缩短 Gate A，executor 可同时使用最多三个 active subagents，仅限
具体有界的代码检查、单个 schema/test 组件或只读 HPC evidence extraction。
subagents 不得改科学、提交/操作 scheduler、写远程、发终态包、做 Gate 决策或
再生 subagents。executor 独自负责集成、Git/push、证据和唯一终态事件。

## 12. Terminal delivery

Gate A 完成或阻塞时，executor 只能向 planning thread
`019f8604-4717-7be2-8bf8-9d4a26a3d7f7` 发送一次：

```text
GATE_A_FINAL_AUDIT
Project/phase: LoopScope Phase 5
Gate: A
Execution thread: 019f8631-3b66-7d40-8cae-d21c37ed27c1
Result: AUDIT_REQUESTED
Authorized starting state:
Final branch/commit/dirty:
Files changed:
Exact commands:
Tests and exit codes:
Remote host/job IDs/run root:
Decision-critical artifact paths and hashes:
Observed result:
Deviations/errors:
Repair loops used:
Completed state:
Actions not taken:
Protected-state confirmation:
Decision requested: PASS / PASS_WITH_FIXES / BLOCK
```

如无法继续，改发 `BLOCK`，并附 narrow blocker、evidence、缺失权限/决策、必须
保持不动的状态。必须通过 Codex `send_message_to_thread` 取得可见投递成功；
如无法确认，在 executor final answer 输出完整的
`TERMINAL_DELIVERY_UNCONFIRMED` 备份，不得声称 planning 已被唤醒。

Gate A 没有长作业，不创建 automation/heartbeat。终态包发出后立即停止，不进入
Gate B，不自行声称 `PASS`。
