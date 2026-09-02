# GATE_O_AUTHORIZATION — LoopScope Phase 6 MMLU 5-shot fixed-panel outcome

```text
HANDOFF_ID=LOOPSCOPE_PHASE6_GATE_O_HANDOFF_V1
HANDOFF_STATE=AUTHORIZED
PLANNING_THREAD=019fb3de-2298-75f2-a083-0dca453ea79c
EXECUTOR_THREAD=019fc345-d707-78a1-844d-d6755f03bfa1
EXECUTOR_TITLE=execute-LoopScope-MMLU5-V3-Outcome-第6阶段-Gate O
GATE=O
BASE_COMMIT=f95aeb4f3583d30f0bca2b351b741b039a1084eb
BRANCH=loopscope
TERMINAL=GATE_O_FINAL_AUDIT_OR_MATERIAL_BLOCK
```

## 1. Exact admission

执行前完整读取并核对：

- `loopscope-tflt/AGENTS.md` SHA-256
  `d0995e1f3e62e31c39a9fe756e0d4e0a697b88dd1369e81ec40d9768cb8683cd`
- `.planning/loopscope_phase6_control.md` SHA-256
  `c91871696384eede8a973b3b031c9bdb4eca9de8686e449c5b8dae9bf5549204`
- `.planning/phase6/loopscope_phase6_mmlu5_v3_outcome_plan.md` SHA-256
  `3693643cd5188cbdc78152e2e8f041d911418daf2f001385a3f41658808b4030`
- 本 handoff 的 fresh SHA-256（执行线程自行核对并回传）
- `/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md` SHA-256
  `186d38ed283aa393ffd94158f5ef5854f959c4ac8e40a4f3451d03a7a16115cb`
- `/Users/huangxutao/.codex/skills/hpc2-hkustgz-ssh/SKILL.md` SHA-256
  `b97016cb9a7693d879506537b0447e6a5d8e5120866eef104c7fee59f3ad23ae`

本地 repo 必须是专用 clone、branch `loopscope`、HEAD 与 origin 均为
`f95aeb4f3583d30f0bca2b351b741b039a1084eb`，且历史基点
`4f59bd93eca4da3cbf458a93508f91c5b23912bc` 是祖先。

预期 dirty state 只有以下七个 planning/user-owned 未暂存文件，均不得 restore、覆盖、
stage 或 commit：

```text
AGENTS.md d0995e1f3e62e31c39a9fe756e0d4e0a697b88dd1369e81ec40d9768cb8683cd
docs/loopscope_phase2.md 021f4fe274dc4130e4287d4dfb26aece0ee49fd78e335fa93d6c0aa4b3488c0d
docs/loopscope_phase4.md 29609d2a9a2f8d539944c148e5e69b671556536522b1748508b6f682a8084879
docs/loopscope_phase5.md a4faf9f95657db47e782b84907390de5681a7e6b958d20489e69e62c6b72ef1a
docs/loopscope_phase6.md 2f5c797b47c4e692a9e269a978bd859ceea6b25e68079e285156615e5e39bda8
scripts/loopscope/run_qwen4_phase4_p4b.py 7c5614a3845fe34f7a232b1038be36019f17d36f9c5006eb3df74aeb695f9e68
src/tflt/loopscope/phase4_outcome.py 5b315c64af1ac0ed5e9cf4d284c462cffb50a771f82ebc24f5a9f745af072df0
```

不一致时先做 bounded read-only characterization；只有材料性 provenance、科学、信息、
安全、破坏性外部权限或真正 no-progress 边界才发送 `BLOCK`。普通工程问题不是 Gate 终态。

## 2. Scientific contract

固定五 cells（TFLT CLI inclusive）：

```text
0 no_loop
1 14:16
2 13:16
3 15:18
4 12:16
```

不得增删、替换、按 outcome 重排或增加 `decode=bypass` cell。

```text
model=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554
dataset=cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe
evaluator=lm-eval 0.4.11 standard MMLU choice-loglikelihood
num_fewshot=5
fewshot_split=dev
prompt=plain
apply_chat_template=false
generation=false
population=test 14042 / 57 subjects
dtype=bfloat16
k=3
iteration_mode=block
strategy=euler
alpha=1.0
beta=0.0
cache_strategy=first
decode_mode=full
batch_size=16 common to all cells
```

Canonical identity source：

```text
root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3c-20260716T064601Z
manifest=test14042_identity_content_manifest.json
manifest_file_sha256=59500340aa64d91810cd5a5d40f3d96ded5618ab5508543b31eea93050e7f3e4
metadata=test14042_identity_content_metadata.jsonl
metadata_file_sha256=3c281447cc7618e58c292a20c13b73d8f42f081692a6927372cf1ee1eebf84df
ordered_identity_sha256=2c4557079fca1a7c071460f779ed714805fc96c3391560825cd34386d4051532
```

`decode=full` 与 `bypass` 的等价只限当前 full-sequence MMLU choice-loglikelihood
代码路径：wrapper 的 bypass 条件仅针对 `seq_len=1` 且带历史 cache/cache_position 的
incremental decode。Gate O 必须仍实际传入并验证 `decode_mode=full`。

## 3. Authorized implementation

只允许新增或修改 Gate O 自有路径：

```text
configs/loopscope/phase6_mmlu5_v3_outcome_card.json
scripts/loopscope/run_qwen4_phase6_mmlu5_gate_o.py
src/tflt/loopscope/phase6_mmlu5_gate_o.py
tests/test_loopscope_phase6_mmlu5_gate_o.py
```

可只读复用 Phase 3/5/6 canonical identity、renderer、outcome、audit、schema 与 verifier
helper。不得修改 `wrapper.py`、`strategies.py`、`cache.py`、`config.py`、历史 Phase 3/5/6
artifacts 或上述七个 protected dirty paths。若复用 helper 不足，优先在 Gate O 新模块内做
最小实现。

执行线程负责 focused tests、compile/help/dry-run、明确 staged-path audit、单目的 commit、
push 到 exact `origin/loopscope`，并将 HPC2 专用 clean clone fast-forward only 到同一 commit。
允许对 exact remote `git@github.com:singularity119/Training-Free-Looped-Transformers-Reproduction.git`
的 `refs/heads/loopscope` 执行必要 push；禁止 force push、rebase、merge、stash 或 cleanup。

## 4. Debug and formal execution

1. CPU admission 闭合 card、模型/数据/evaluator/cache/package、test manifest、五 cell manifest、
   output allowlist 和 fresh-root 语义。
2. 在 `debug` partition/A40 上，以最终提交和相同 launcher/env/model/data/producer/verifier
   对全部五 cells 跑 limited smoke。loop cells 每个 audited prompt 必须
   `operator_body_calls=3`、`loop_effective_logits_changed`、`restore_allclose=true`、无 bypass/
   NaN/OOM；baseline 为 zero loop activity。结果 accuracy 不得作为改合同依据。
3. debug PASS 后，在全新 write-once root 创建 formal manifest，用最高合法优先级
   `emergency_gpu/emergency_gpu` 提交 `0-4%5` A800 array：每 task 1×A800、8 CPU、64G、
   `06:00:00`。
4. verified formal Job ID 产生后立即建立唯一 60-minute heartbeat，PENDING 期也监测。
   每次只做一次 bounded read-only SSH + scheduler/artifact check；不得由 heartbeat retry、
   cancel、requeue、repair、读取 outcome 或裁决科学结果。terminal 时先暂停/删除 heartbeat，
   再显式恢复 exact executor。
5. 禁止读取任何 partial cell 的 accuracy/correctness/gain。五 cells scheduler terminal success、
   artifact stat/hash 与 canonical identity closure 全部 PASS 后，才可在一次 analysis 中读取
   outcomes；随后只运行一次 fresh-process independent verifier。

若 batch size 16 在真实 debug 中 OOM，允许低风险工程修复为一个更低的共同 batch size，
但必须全部五 cells 同步采用、重新 fresh debug、更新 frozen receipt，并禁止混用旧 batch-size
artifacts。其他 model/data/evaluator/loop/window 改动不属于低风险修复。

## 5. Frozen analysis

对每个 cell 报告：

- `correct_count / 14042` 与 accuracy；
- 相对 no-loop 的 accuracy delta（percentage points）；
- paired baseline→loop 的 `n01/n10` flip counts；
- subject-stratified paired percentile bootstrap 95% CI，2,000 replicates，seed `20260803`。

这四个 loop-vs-baseline contrast 均报告，不作新 selector，也不按 outcome 重新定义主窗口。
可以按 accuracy 展示排序，但不得将排序回写为 V3 selector 结论。Gate O 不运行其他 full loop。

## 6. Repair, reporting, and terminal

执行线程低风险工程修复次数不限，只要保持本 Gate、冻结科学合同、fresh write-once、
protected state 与材料性进展。失败 job 是 attempt-level event；诊断根因、修复测试并 fresh
retry，不得因普通 launcher/path/schema/OOM/shape/dtype/loader/SSH 暂时错误直接终止 Gate。

终态回传必须包含：admission hashes、branch/commit/dirty、changed files、tests、全部 debug/
formal job IDs 与 scheduler terminal evidence、run roots、五 cell manifest、每 cell identity/result/
receipt hashes、analysis/verifier hashes、五个 accuracy 与 delta/CI、heartbeat lifecycle、repairs、
protected-state confirmation，以及中文报告路径
`资产/报告/phase6/LoopScope_第六阶段GateO_MMLU5_V3固定Panel全量Acc.md`。

只有一次 `GATE_O_FINAL_AUDIT` 或材料性 `BLOCK`，并必须可见投递到 planning thread
`019fb3de-2298-75f2-a083-0dca453ea79c`。Gate O PASS 后 authority 自动撤销，不自动开始后续 Gate。
