# GATE_C_HANDOFF

~~~text
Project/phase=LoopScope Phase 5
Hypothesis=H5_QWEN4B_BASE_MMLU_TRAJECTORY_SELECTION_V1
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019f884d-e5c0-7750-bc8e-7735c7984bb9
Executor title=execute-LoopScope-Trajectory-Selection-第5阶段-Gate C
Gate=C
State=AUTHORIZED_AFTER_VISIBLE_DELIVERY
Terminal event=GATE_C_FINAL_AUDIT or BLOCK
~~~

## 1. Objective and claim boundary

Gate C 的唯一目标是在冻结的 Qwen3-4B-Base × MMLU 5-shot test-14042 上，fresh acquisition
完成 Gate B 冻结的八个 unique cells，并将逐样本结果封存为 hash-closed artifacts。Gate C
只证明 scheduler terminal、recipe/identity/completeness 和 seal；不读取或报告 accuracy、gain、
flip、prediction distribution、窗口优劣或任何 Phase 5 科学结论。

Gate B 合法输出 `ABSTAIN_NO_POINT_ELIGIBLE`。这不取消 ranking panel：Gate C 仍完整运行
High3/Low3，以便 Gate D 区分 ranking enrichment、absolute benefit 和 ABSTAIN。由于 baseline
与 `15:18` 均为 `FRESH_ACQUISITION_REQUIRED`，八个 cells 全部 fresh。

## 2. Admission and authoritative frozen inputs

~~~text
admitted_commit=0d7e33d7b095fcae4b5dd7fbc14fdb21f2410205
branch=loopscope
gate_b_run_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-b-repair-b36-native-20260722T035452Z
gate_b_trajectory_file_sha256=aa4894e632f557ddc4614e88d06ec1fc4c525463be7531aef69fa3c28671edf9
selector_freeze_file_sha256=7c42f0e22ce2caa5784edb33dcfc271e31291229e107b478928fd77181cf1a0b
selector_freeze_manifest_sha256=4c66ef75ee8efc492245b341ee79c0147a3ba3b973a0de569ef6ca5dcb858d97
outcome_panel_file_sha256=3ea83d12e5185edffbf01d5b916612e616d3a7cf41881601fa7b8931918be0e9
outcome_panel_manifest_sha256=f28ce84d54e69f2179209f276ecd69f39c3e94359eac4f2aed84b09167be4961
gate_b_receipt_file_sha256=bd720065beb2cd7387322346aafdb06e55c6c0131896d6df10164da0e427d941
phase5_card_file_sha256=8d25996f994d131ea0fd9ca78f83c90680a67c41c7c460f7e4e8b4ae5ea42a73
phase5_card_manifest_sha256=75eed3e30c623f5eedd63e84e93bfd3e4de50cfb490b23f3a2d31683f9947143
test_identity_manifest_file_sha256=59500340aa64d91810cd5a5d40f3d96ded5618ab5508543b31eea93050e7f3e4
test_identity_manifest_internal_sha256=ec494ff6dd55b4398cd8151eed4609916e8d4c272df4583a4a1574cab3c25386
test_ordered_identity_sha256=2c4557079fca1a7c071460f779ed714805fc96c3391560825cd34386d4051532
canonical_count=14042
subjects=57
~~~

Gate B 的旧 invalid root `phase5-gate-b-20260721T201330Z` 与 repair preflight-only root 都不是
Gate C 输入。开始 mutation 前必须核对 executor ID、repo/branch/HEAD/origin/dirty、protected
ancestor、上述 upstream hashes，并确认 Gate B executor 已无权限。任何漂移立即 `BLOCK`。

## 3. Frozen panel and recipe

按以下顺序冻结，禁止替换、重排、删减或新增：

| index | role | cell |
|---:|---|---|
| 0 | baseline | `no-loop` |
| 1 | fixed comparator | `15:18` |
| 2 | blind High1 | `4:7` |
| 3 | blind High2 | `5:8` |
| 4 | blind High3 | `13:16` |
| 5 | blind Low1 | `8:11` |
| 6 | blind Low2 | `9:12` |
| 7 | blind Low3 | `6:9` |

除 window 外全部相同：

~~~text
model=Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539
tokenizer_revision=906bfd4b4dc7f14ee4320094d8b41684abff8539
dtype=bfloat16
dataset=cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe
task_group=mmlu
split=test
num_fewshot=5
lm_eval=0.4.11
metric=acc,none
renderer=exact Gate A MMLU renderer; apply_chat_template=false; fewshot_as_multiturn=false
population=14042 identities / 57 subjects
window_width=4
k=3
strategy_cli=euler
strategy_kernel=damped_euler_alias
alpha=1.0
beta=0.0
step_size=1/3
total_horizon=1
iteration_mode=block
cache_strategy=first
decode_mode=bypass
log_samples=true
~~~

Baseline 不加载 wrapper。每个 loop cell 的 inclusive `s:e` 必须绑定 `B_s -> B_(e+1)`。
不得复制 Phase 4 MMLU-Pro 的 free decode/CoT evaluator；必须复用 Phase 3 已审计的 MMLU
multiple-choice/loglikelihood 路径和 Phase 4 的 sealed eight-cell control pattern。

## 4. Minimal implementation and protected paths

推荐最小新增：

~~~text
src/tflt/loopscope/phase5_outcome.py
scripts/loopscope/run_qwen4base_phase5_gate_c.py
tests/test_loopscope_phase5_gate_c.py
~~~

仅在实际 artifact contract 需要时新增一个窄 Gate C schema JSON，或更新
`docs/loopscope_phase5.md` 的 Gate C 命令骨架。不要建设通用 outcome framework，不重构 Phase
3/4 runner。优先复用：Phase 3 exact MMLU evaluator/identity primitives、Phase 4 write-once
launch/seal/sacct pattern、当前 wrapper/LoopConfig 正常路径。

Gate A/B files、`phase5_schema.py`、`phase5_acquisition.py`、`phase5_selector.py`、Gate B
artifacts、Phase 1-4 artifacts 全部只读。以下稳定实现路径仍禁止修改：

~~~text
src/tflt/wrapper.py
src/tflt/strategies.py
src/tflt/cache.py
src/tflt/config.py
src/tflt/eval_runner.py
~~~

若 normal path 证明必须修改任一保护路径，先 `BLOCK`，附最小 reproducer 和拟议 diff；不得
自行修改。

## 5. Sealed information barrier

Gate C 新 run root：

~~~text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-c-20260722T053058Z
~~~

正式 `results.json`、logged-sample JSON/JSONL 和任何 evaluator outcome payload 全部 sealed。
执行线程不得用 `cat/head/tail/jq/python`、日志摘要或其他方式查看其中的 accuracy、prediction、
gold、correctness、raw choice scores、gain 或 flip。launcher stdout/stderr 不得打印 metrics 或
sample outcome。

允许 producer 以严格 allowlist 机械投影并持久化 outcome-free identity sidecar：task、doc_id、
doc_hash、subject、split、canonical order/hash、count。允许 seal verifier 读取该 sidecar、
command/revision receipts、文件 stat/SHA 与 scheduler rows；不得解析 sealed outcome fields。
完整 evaluator payload 只供 Gate D 一次性 unseal。

Gate C completion receipt 必须写明：

~~~text
outcome_values_consumed=false
accuracy_or_gain_computed=false
results_payload_parsed_by_sealer=false
gate_d_unseal_performed=false
status=SEALED_COMPLETE_READY_FOR_PLANNING_AUDIT
~~~

## 6. Ordered execution

### C0. Admission, narrow implementation, targeted tests

1. 从 clean admitted commit 实现 Gate C producer/launcher/sealer。
2. fail closed 绑定 panel、card、model/data/renderer、test manifest、all-cell recipe 与 write-once
   paths；单测覆盖 panel drift、baseline/loop config、identity allowlist、partial/failed scheduler
   refusal、sealed no-parse 和重复写拒绝。
3. 运行 Gate A verifier、Gate C targeted tests，以及直接复用的最小 Phase 3 MMLU/Phase 4
   outcome regression；不重复 full suite。
4. `py_compile`、`git diff --check`、protected-path diff 通过，创建一个 implementation commit
   并普通 push。远端只允许 clean fast-forward。

formal launch 前最多两次 executor-owned low-risk repair loops；每次必须有可复现工程根因、
一个窄 commit 和 fresh smoke。formal 任一 outcome cell 开始后禁止修改 producer/sealer code。

### C1. CPU/provenance preflight and eight-cell smoke

先在 HPC2 login node 做 config/tokenizer/cache/task/manifest/import/hash preflight，不实例化权重。
模型缺失文件时可按已授权的 pinned revision 使用：

~~~text
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
HF_HUB_DISABLE_XET=1
~~~

随后对八个 frozen cells 各做一次相同 fixed smoke identity set 的 GPU limit smoke。建议使用两
个固定 MMLU subjects 各两条，共四条 identities；必须在 launch 前按 canonical identity/hash
冻结，不得根据 smoke outcome 换样本。

每个 smoke cell 必须验证：

- exact model/tokenizer revision、MMLU renderer 与 evaluator logged-sample identity；
- baseline 零 loop activity；七个 loop windows 均为 K3、三次 body calls、step=1/3、block、
  cache-first、decode bypass；
- loop-effective logits change、finite output、restore 后 native allclose；
- smoke artifacts 明确 `NON_SCIENTIFIC_LIMIT_SMOKE`，不进入 formal 或 Gate D analysis；
- 不向 planning 汇报 smoke accuracy 或窗口相对表现。

八个 smoke 全部 PASS 后，用实测更新 batch size、partition、array throttle、time/memory。预计
Gate C 总预算 16–64 A800-equivalent GPU-hours、存储 <4 GB；若外推 >64 GPU-hours 或 >4 GB，
先 `BLOCK` 请求 planning 运维裁决，不通过减样本、删 Low3、改 dtype/recipe 解决。

### C2. Freeze once and run full sealed panel

在 fresh root 冻结 full launch manifest、八 cell argv/environment、implementation hashes、test
identity/panel/card hashes与 scheduler plan，然后以一个 exact eight-task Slurm array 提交；每
task 一个 cell，`--requeue` 关闭，禁止 automatic retry。

每个 cell 必须产生：command/env/model revision receipt、gold-free identity sidecar、sealed
results/logged samples、producer completion receipt。八 cell 每个都必须 exact 14,042 unique
identities / 57 subjects，与冻结 manifest 无 missing/duplicate/extra，且同一 ordered identity
hash。baseline 与所有 loop cells 使用完全相同 evaluator seeds、batch semantics 和 renderer。

任一 formal task FAILED/OOM/CANCELLED、出现 partial artifact、identity/recipe drift 或 outcome
意外泄露，立即 `BLOCK`；不要重提、续跑、覆盖或先查看其他 cell 的科学结果。只有明确
`forward/evaluator requests=0` 且没有 cell output/partial record 的纯 scheduler failure，才可在
原 handoff 的最多一次 fresh attempt 内按完全相同 manifest/argv重提；否则由 planning 裁决。

### C3. Seal without scientific unseal

仅当 exact 8/8 scheduler tasks 都 `COMPLETED/0:0` 后，sealer 才能核对 scheduler rows、文件
existence/nonempty/stat/SHA、producer receipts、identity sidecars 和八 cell exact equality。
不得计算或读取 accuracy/gain/flip。写一次 Gate C completion receipt 和 resource receipt，暂停
Gate C monitor，发送终态包。

## 7. Agility budget and stop rule

- 最小增量：Phase 5 MMLU MC eight-cell sealed adapter，不做通用化。
- 决定性工程检查：panel/recipe freeze、identity allowlist、eight-cell smoke、8/8 terminal seal。
- 最早真实实验：八 cell 四身份 smoke；通过后立即 full，不追加穷尽 tamper matrix。
- stop rule：full seal 与 identity/hash closure 通过即请求 audit；不在 Gate C 做分析、解释或额外
  复跑。

## 8. Authorized and forbidden actions

授权：在上述窄文件写代码/测试/commit/普通 push；HPC2 clean ff-only sync；write-once Gate C
root；pinned cache 下载；必要 GPU/Slurm smoke 与 exact eight-cell full；executor-owned 只读
monitor；机械 identity projection 与 stat/SHA seal。

禁止：修改 selector/panel/card；读取或报告 formal outcome values；按 partial result 调度、改参
或停掉不喜欢的 window；运行未冻结 window；K/width/model/task sweep；Gate D unseal/analysis；
重跑已完整 cell；修改保护路径/历史 artifacts；force-push/rebase/destructive filesystem；创建
Gate D executor 或跨 Gate 继续。

## 9. Terminal delivery

完成或阻塞时，只向 planning thread `019f8604-4717-7be2-8bf8-9d4a26a3d7f7` 发送一次：

~~~text
GATE_C_FINAL_AUDIT
Project/phase: LoopScope Phase 5
Gate: C
Execution thread: 019f884d-e5c0-7750-bc8e-7735c7984bb9
Result: AUDIT_REQUESTED
Admission and final Git state:
Files changed and commits:
Exact commands/tests/exit codes:
HPC2 environment and implementation hashes:
Smoke job/cells/identities/structural result:
Formal array job and exact 8/8 scheduler states:
Run root and sealed artifact paths/SHA256:
Per-cell 14042/57 identity and ordered-hash closure:
Panel/card/test-manifest closure:
Information-barrier confirmation:
Resource accounting and monitor lifecycle:
Repair loops/deviations:
Actions not taken:
Decision requested: PASS / PASS_WITH_FIXES / BLOCK
~~~

如不能继续，改发 `BLOCK`，保留全部 evidence，不做未经授权的 retry/unseal。必须取得可见
cross-thread delivery confirmation，发送后立即停止，不进入 Gate D。
