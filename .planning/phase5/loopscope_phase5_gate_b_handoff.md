# GATE_B_HANDOFF

~~~text
Project/phase=LoopScope Phase 5
Hypothesis=H5_QWEN4B_BASE_MMLU_TRAJECTORY_SELECTION_V1
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019f864e-9b57-73e1-b56c-e7e40e7ad28d
Executor title=execute-LoopScope-Trajectory-Selection-第5阶段-Gate B
Gate=B
State=AUTHORIZED_AFTER_VISIBLE_DELIVERY
Terminal event=GATE_B_FINAL_AUDIT or BLOCK
~~~

## 1. Objective and non-objectives

Gate B 回答一个 outcome-blind 工程与科学冻结问题：能否在 exact
Qwen3-4B-Base × MMLU validation-1531 上，以每条 identity 一次 native no-loop
forward 同时产生 choice/output 与 final-norm hidden geometry 五轨迹，并严格按 Gate A
冻结的 entropy–KL CONSENSUS 生成 ranking、eligibility、selected/ABSTAIN 和完整
High3/Low3 outcome panel。

本 Gate 的终点是 hash-closed selector/panel freeze，不是验证 accuracy。它不读取或运行
MMLU test，不产生 baseline/window outcome，不解释 gain，不进入 Gate C。

## 2. Admission and authoritative state

Gate A planning acceptance 已裁决 PASS：

~~~text
admitted_commit=60e39f6f77c4ff09f4723274e4d6cfaa7ef7281d
origin/loopscope=60e39f6f77c4ff09f4723274e4d6cfaa7ef7281d
gate_a_card_sha256=8d25996f994d131ea0fd9ca78f83c90680a67c41c7c460f7e4e8b4ae5ea42a73
gate_a_card_manifest_sha256=75eed3e30c623f5eedd63e84e93bfd3e4de50cfb490b23f3a2d31683f9947143
known_registry_sha256=714d8d253f69883d902b922e4927e3d41f20e4b5dec93df0a826c3f5e5797f7f
baseline_reuse=FRESH_ACQUISITION_REQUIRED
15:18_reuse=FRESH_ACQUISITION_REQUIRED
~~~

权威顺序：

1. 用户最新明确指令；
2. live Git/HPC2/Slurm 与 write-once artifacts；
3. loopscope-tflt/AGENTS.md；
4. .planning/loopscope_phase5_control.md；
5. 本 handoff；
6. docs/loopscope_phase5.md 和历史材料。

开始任何 mutation 前，必须重读上述 policy/control/handoff，并核对 executor thread ID、
repo root、branch、HEAD、origin、dirty 与 protected-base ancestry。若绑定或控制状态不一致，
立即 BLOCK。

## 3. Frozen scientific contract

不得修改 configs/loopscope/phase5_card.json、card schema、known registry 或其中任何
model/data/trajectory/selector/panel/statistics 语义。关键冻结值：

~~~text
model=Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539
dtype=bfloat16
layers=36
hidden_size=2560
dataset=cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe
renderer=lm-eval 0.4.11 MMLU 5-shot exact existing anchor
population=validation 1531 identities / 57 subjects
forward=one native no-loop forward per formal identity
position=final non-padding prompt token
boundaries=B_0..B_36
choice_tokens=spaced A:362 B:425 C:356 D:422
selector=ENTROPY_KL_CONSENSUS_WIDTH4_V1
bootstrap=subject-stratified joint, R=2000, seed=20260722
selection_frequency_threshold=0.80
panel=known-excluded selected-inclusive High3 + disjoint Low3 + baseline + fixed 15:18
max_unique_cells=8
~~~

对每个 boundary，先得到 z_l：B_0..B_35 只应用一次 frozen final norm；B_36 使用 native
model 已 final-normalized 的 hidden，不得 double norm。choice logits 只投影 exact 四个
tied LM-head rows；float64 stable softmax 后保存四概率。hidden reduction 用 float32 临时
计算 RMS-L2、cosine，cosine distance 必须由 1-cosine 派生。所有 full logits、全词表概率、
hidden/residual tensors 在写盘前必须消失。

## 4. Minimal implementation path

优先复用：

- Phase 3 的 exact MMLU renderer/validation safe-pool、decoder boundary、choice projection、
  native no-loop guard、write-once shard/merge 与 subject bootstrap primitives；
- Phase 4 的 36-layer bfloat16 runtime、final B36 projection closure与同-forward scalar
  reduction 思路；
- Gate A 的 phase5_schema trajectory/CONSENSUS/panel deterministic helpers。

只建设 Phase 5 窄适配，不建设通用 hook framework。默认允许新增：

~~~text
src/tflt/loopscope/phase5_acquisition.py
src/tflt/loopscope/phase5_selector.py
scripts/loopscope/run_qwen4base_phase5_gate_b.py
tests/test_loopscope_phase5_gate_b.py
configs/loopscope/phase5_trajectory_schema.json
configs/loopscope/phase5_selector_freeze_schema.json
~~~

可按必要最小更新 docs/loopscope_phase5.md 的 current Gate banner/commands。若更少文件足够，
不要为凑清单创建空文件。

以下 Gate A 冻结产物默认只读：

~~~text
configs/loopscope/phase5_card.json
configs/loopscope/phase5_card_schema.json
configs/loopscope/phase5_known_outcome_registry.json
configs/loopscope/phase5_gate_a_receipt.json
src/tflt/loopscope/phase5_schema.py
scripts/loopscope/verify_phase5_gate_a.py
tests/test_loopscope_phase5_gate_a.py
~~~

如果实际 normal path 证明 phase5_schema.py 必须做纯 additive producer/manifest validation
扩展，先向 planning 发 BLOCK，附最小 reproducer、拟议 diff 与不改变 Gate A 数学的证明；
不得自行改写。

永久保护且不得修改：

~~~text
src/tflt/wrapper.py
src/tflt/strategies.py
src/tflt/cache.py
src/tflt/config.py
src/tflt/eval_runner.py
Phase 1-4 cards/selectors/outcomes/reports/run artifacts
~~~

## 5. Ordered execution stages

### B0. Admission, local implementation and targeted tests

1. 从 clean admitted commit 开始；保留其他线程的任何新改动，不 reset/stash/checkout。
2. 实现最小 producer/analyzer/launcher。selector consumer 必须 fail closed 拒绝 gold、
   correctness、test/outcome 字段及非有限/identity 漂移。
3. 运行 Gate A verifier 加 Gate B targeted tests；只在真实共享路径发生材料变化时运行一次
   broader suite。
4. git diff --check、protected-path diff 和 forbidden placeholder scan 必须通过。
5. 创建 one-purpose implementation commit，普通 push origin/loopscope。不得 rebase、
   force-push 或 merge main。

Gate B artifact 必须绑定该最终 implementation commit；smoke 后若出现材料工程 bug，可在
冻结科学不变时做最多两次低风险 repair loop，每次一个窄 commit/push，并用 fresh smoke
attempt。formal acquisition 一旦开始，不得再改 producer/analyzer code。

### B1. HPC2 sync and CPU/provenance preflight

HPC2:

~~~text
host=hpc2-hkustgz
repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope
run_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-b-20260721T201330Z
HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
model_snapshot=/hpc2hdd/home/xhuang225/shared/hf_home/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
~~~

对 remote clone 只允许 clean fast-forward sync 到 pushed implementation commit。复用已审计
兼容 venv；若缺依赖，可在独立 versioned venv 中做最小维护，不得原地升级历史复现环境。
模型缺文件时可在 login node 使用：

~~~text
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
HF_HUB_DISABLE_XET=1
~~~

只下载 pinned revision 到共享 cache。login node 只做 config/tokenizer/hash/import/CPU
preflight，不实例化正式权重做 forward，不使用 CUDA。

CPU preflight 必须闭合 exact commit/venv/package versions、card/hash、model snapshot shards、
renderer/pool identities、validation count=1531、subjects=57、无 test import，并生成
write-once admission receipt。

### B2. Exact four-identity GPU smoke

在 frozen validation membership 中按 canonical order 预注册四条 identity，至少覆盖两个
subjects；不得看 trajectory 数值后换样本。只提交最小 GPU Slurm smoke，并验证：

1. 每 identity 恰好一次 native forward，use_cache=false，loop modules 未 import、
   active wrapper=none、loop_insertions=0；
2. prompt/hash/answer position 与 exact renderer anchor 一致；
3. B_0..B_36 顺序/shape 正确，B36 native logits 与 tied LM-head choice projection 闭合，
   且没有 B36 double norm；
4. spaced choice token IDs 精确，四概率、choice entropy、KL-to-final、hidden RMS-L2、
   cosine、derived distance 全部 finite；
5. final boundary：KL=0、L2=0、cosine=1、distance=0；
6. 持久化 JSON 不含 full logits、full-vocab probability、hidden/residual tensor、gold、
   correctness、test/outcome；
7. exact forward count=4，membership 完整无重复；估算正式 1531 throughput、峰值显存、
   每样本存储与 GPU-hours。

smoke 只是工程门，不用于 selector 或科学报告。smoke 未全部通过时禁止 formal acquisition。
可对明确工程根因做最多两次低风险修复；不得改 model/data/renderer/metrics/selector。

### B3. Freeze formal membership and launch once

smoke PASS 后，在同一 run root 冻结 formal manifest/shard membership/argv/environment/
implementation hashes。每个 validation identity 必须且只属于一个 shard；同一 identity 的
choice 和 geometry 来自同一 forward。

根据 smoke throughput 选择普通合法 GPU partition、batch/shard/concurrency；这些仅是调度
参数，不得改变 identity、prompt、dtype、数值合同或 selector。预算：

~~~text
expected_A800_80G_equivalent=2..6 GPU-hours
storage_limit=<250 MB
planning_reapproval_trigger=smoke_extrapolation >12 GPU-hours or storage >500 MB
~~~

若超过 reapproval trigger，先 BLOCK，不通过减样本、改 precision、删 geometry、减少
bootstrap 或改变 selector 解决。

formal producer 必须在离线 cache 模式运行，不得访问 test split。每条 formal identity
只允许一个成功 forward。不得自动重跑已完整 identity；失败 shard 只有在 receipt 明确证明
forward_calls=0 且未写任何 identity record 时，可用完全相同 commit/argv 在 fresh attempt
目录重提一次。若是否已经 forward 不可证明，或已有任何 partial identity，立即 BLOCK，由
planning 决定，避免把一次采集静默变成重复采集。

### B4. Merge, deterministic selector and panel freeze

只有所有 1531 records terminal-complete 后才 merge；必须 exact manifest equality，无 missing、
duplicate、extra、partial。独立 verifier 从 persisted scalar records重算：

- 每条 trajectory 与 five-metric/final-boundary closure；
- 33 个 width-4 window delta 和 25 个 CONSENSUS support starts；
- 57-subject joint bootstrap 的 exact PRNG/index digest、SE、lower95；
- ranking score、eligibility 与 failure reasons；
- selected_window 或三种 ABSTAIN 原因、selection frequency；
- registry-excluded raw_high3、selected-inclusive panel_high3、disjoint blind_low3；
- baseline、fixed 15:18、complete unique panel identities 与 max-8；
- card/source/trajectory/analyzer/panel/implementation hashes。

selector freeze 是 outcome-blind 的正式结果，可以读取 validation trajectory scalars，但
不得读取 test identity 内容、test logged samples、target label/gold/correctness、baseline
或任何 window outcome。不得根据 trajectory 重新调权、改 eligibility/threshold、把 geometry
加入 score，或手工替换排名。

freeze 完成后写一次 Gate-level receipt 和简洁中文 selector 摘要；摘要可报告 ranking、
eligibility、selected/ABSTAIN 与 panel identities，但不得出现 accuracy/gain/outcome。

## 6. Authorized actions

- 本地在专用 clone 的 loopscope 分支修改第 4 节允许的 Gate B 文件，targeted tests，
  one-purpose commits 与普通 push。
- 使用 ClearAllForwardings=yes SSH 连接 HPC2，远端 clean ff-only sync、只读环境/cache/
  provenance 检查、独立 run root 写入。
- 在 B1 完成后提交必要的 GPU/Slurm smoke 与 formal validation jobs；可按 smoke throughput
  合理选择合法 partition、GPU count、memory/time、array concurrency。
- 在冻结科学不变时做最多两次 executor-owned 低风险 engineering repair loops。
- 按第 5 节精确 retry 语义，仅重提可证明 forward_calls=0 的 failed attempt。
- 创建 executor-owned read-only automation/heartbeat 监控已提交长作业。

## 7. Forbidden actions

- 任何 MMLU test dataset/logged sample/outcome/prediction/accuracy/gain/flip 读取或运行。
- baseline、15:18 或冻结 panel 的 loop outcome；Gate C/D 工作或计划扩展。
- K/width/model/task/dtype/renderer/choice space/bootstrap seed/threshold/panel rule 改动。
- geometry-augmented selector、动态 gating、K sweep、variable-width outcome、任何
  representation intervention。
- formal identity 重复 forward、choice/geometry 分离重跑、根据 smoke/trajectory 值改样本。
- 修改保护实现路径、Phase 1-4 artifacts、Gate A card/schema/registry 或历史 run。
- destructive Git/filesystem、覆盖旧 artifacts、force-push/rebase、自动 merge main。
- executor 自行宣称 PASS、创建 Gate C executor、跨 Gate 继续。

## 8. Evidence and write-once artifact contract

run root 中至少包含：

~~~text
admission/
  gate_b_admission_receipt.json
smoke/
  smoke_membership_manifest.json
  smoke_trajectory_records.jsonl
  smoke_receipt.json
formal/
  formal_membership_manifest.json
  formal_shard_manifest.json
  shards/<shard_id>/<fresh_attempt>/...
  validation1531_phase5_trajectories.jsonl
  validation1531_phase5_trajectory_manifest.json
selector/
  selector_report.json
  selector_ranking.csv
  selector_freeze.json
  outcome_panel.json
  selector_summary_zh.md
verification/
  trajectory_verifier_receipt.json
  selector_verifier_receipt.json
  gate_b_receipt.json
slurm/
  launch receipts, stdout/stderr and job terminal evidence
~~~

实际文件名可由窄 producer 稳定冻结，但不得减少决定性内容。所有 JSON/JSONL 必须 strict
JSON、finite、canonical hash；run root 与正式 artifacts write-once。不要建设 micro-receipt
tree：一个 smoke receipt、一个 formal/selector Gate receipt 加 scheduler evidence足够。

## 9. Targeted executor verification

本地最小检查：

1. Gate A verifier；
2. Gate B pure-Python/fake-tensor tests：boundary/B36 no-double-norm、same-forward
   choice+geometry、forbidden persistence、identity/shard exactness、bootstrap/ABSTAIN/panel；
3. directly reused Phase 3 renderer/acquisition 与 Phase 4 runtime/scalar 的最小 regression
   subset；
4. py_compile、git diff --check、protected-path diff。

远端最小检查：

1. CPU admission/import/provenance receipt；
2. exact four-identity GPU smoke；
3. 1531 completeness/identity/trajectory merge verifier；
4. independent selector/panel recomputation。

一旦这些决定性检查通过，停止额外 hardening；不要运行无材料回归风险的 full suite 或构造
穷尽 tamper matrix。

## 10. Must-pass acceptance

Gate B 只有同时满足以下条件才可请求 planning audit：

1. implementation commit(s) 已普通 push；local/origin/remote scientific commit 和 producer
   hash闭合，repo clean。
2. smoke exact 4 identities，全部 native no-loop/boundary/token/final-logit/five-trajectory/
   no-persistence checks pass。
3. formal validation 正好 1531 identities/57 subjects；每 identity 一次成功 forward；
   choice 与 geometry 来自同一 forward；无 test/outcome access。
4. trajectory merge/hash/identity exact，所有 scalars finite，final boundary闭合。
5. outcome-blind selector deterministic 重算一致；selected 或 ABSTAIN 合法；High3/Low3/
   baseline/15:18 panel hash-closed且 <=8 unique cells。
6. protected paths/history 未改；jobs terminal；resource/accounting/automation lifecycle
   evidence完整。

非材料 style、命名美化、额外 schema 防御或通用化不阻塞 PASS。

## 11. Monitoring, repair and escalation

长作业由本 executor 自己创建 automation/heartbeat；planning 不轮询。默认 cadence：
预计 <2h 用 10 分钟，2–8h 用 30 分钟，更长队列/运行用 60 分钟。每次只做一次有界、
只读 squeue/sacct、短日志尾与 expected artifact existence 检查。

monitor prompt 必须冻结 Gate B、executor/planning thread IDs、host、job IDs、run root、
terminal criteria 和下一授权动作。terminal 时 automation 先暂停/删除自身并保留 lifecycle
evidence，再向精确 executor 发送 AUTOMATION_TERMINAL_RESUME；若不能确认，向 planning
发送 AUTOMATION_RELAY_REQUIRED。automation 不得提交、重试、取消、改参或做科学裁决。

普通 repair budget：最多两次 executor-owned 低风险工程修复。以下必须 BLOCK：

- 需要改变冻结科学/card/schema语义或保护路径；
- 需要接触 test/outcome；
- formal attempt 已发生部分/不明 forward，需要决定是否重复；
- 资源外推超过 planning reapproval trigger；
- 两次修复后仍不能满足 normal path；
- 凭据、安全、破坏性或阶段扩展选择。

## 12. Terminal delivery

完成或阻塞时，只向 planning thread
019f8604-4717-7be2-8bf8-9d4a26a3d7f7 发送一次：

~~~text
GATE_B_FINAL_AUDIT
Project/phase: LoopScope Phase 5
Gate: B
Execution thread: 019f864e-9b57-73e1-b56c-e7e40e7ad28d
Result: AUDIT_REQUESTED
Admission and final Git state:
Files changed:
Exact commands/tests/exit codes:
HPC2 environment and implementation commit:
Smoke job IDs/run paths/result:
Formal job IDs/shard attempts/terminal states:
Forward-count and identity closure:
Trajectory/selector/panel artifact paths and hashes:
Selected window or exact ABSTAIN reason:
Frozen raw High3/panel High3/Low3/fixed comparator identities:
Outcome-blind and forbidden-action confirmation:
Resource accounting and automation lifecycle:
Deviations/repair loops:
Actions not taken:
Decision requested: PASS / PASS_WITH_FIXES / BLOCK
~~~

如无法继续，改发 BLOCK，说明 narrow blocker、证据、已保持不动的 artifacts/jobs 和需要的
planning/user decision。必须通过 Codex send_message_to_thread 得到可见投递确认；无法确认时，
executor final answer 输出完整 TERMINAL_DELIVERY_UNCONFIRMED。发送终态包后立即停止，不进入
Gate C，不自行声称 PASS。
