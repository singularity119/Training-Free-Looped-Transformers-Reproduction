# GATE_C_REPAIR_1_HANDOFF

~~~text
Project/phase=LoopScope Phase 5
Hypothesis=H5_QWEN4B_BASE_MMLU_TRAJECTORY_SELECTION_V1
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019f884d-e5c0-7750-bc8e-7735c7984bb9
Executor title=execute-LoopScope-Trajectory-Selection-第5阶段-Gate C
Gate=C
Planning decision=PASS_WITH_FIXES
Repair cycle=1 of 1 planning-audit-returned cycle
State=AUTHORIZED_AFTER_VISIBLE_DELIVERY
Terminal event=GATE_C_FINAL_AUDIT or BLOCK
~~~

## 1. Audited root cause

Smoke array `10032360` 的八个 exact cells 全部完成了模型结构审计与四身份 lm-eval 运行，随后
一致失败于：

~~~text
Phase5OutcomeError: lm-eval identity projection is incomplete; missing=4 extra=4
~~~

根因是 `project_identity_only` 直接信任 lm-eval sample row 的 `doc_hash`。lm-eval 的内部
`doc_hash` 与 Phase 3/5 冻结的 gold-free canonical hash 不是同一语义；后者必须由
`question + choices + subject` 的 canonical projection 计算。planning 只读诊断在 baseline
smoke 上用 Phase 3 已冻结的 `safe_test_dataset_fields` / `safe_test_doc_sha256` 重算，得到
`safe_identity_count=4, safe_identity_matches=4`，且没有输出任何 outcome 字段。

这是 identity adapter 的窄工程缺陷，不改变 model、dataset/split、14042 membership、renderer、
metric、panel、window、K、dtype、scheduler science 或 seal。原 Gate C handoff 未改变；本补充
只授权修复 identity projection 并在 fresh write-once root 重跑。

## 2. Preserve the failed smoke attempt

以下 root/job 永久保留为 non-formal failed smoke evidence：

~~~text
old_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-c-20260722T053058Z
smoke_job=10032360
scheduler=exact tasks 0..7 all FAILED/1:0
formal_submitted=false
formal_artifacts=false
canonical_gate_c_input=false
~~~

旧 root 不得覆盖、删除、追加、seal、salvage 或作为 repaired Gate C canonical input。其中 smoke
`results.json` 只属于 `NON_SCIENTIFIC_LIMIT_SMOKE` 工程证据，不得查看/报告 accuracy 或其他
outcome。新 receipt 只记录旧 root、job、failure string 和 invalidation reason。

## 3. Authorized narrow repair

从 clean pushed commit `f5aea974fd4eb1b2335cb165075768778e2922ed` 开始：

1. `project_identity_only` 以 `(task, decimal doc_id)` 定位冻结 test metadata record。
2. 只暂时访问 sample row `doc.question`、`doc.choices`、`doc.subject`，复用 Phase 3 已审计的
   `safe_test_dataset_fields`、`safe_test_doc_sha256`，并可同时核对 frozen
   `sanitized_content_sha256`。
3. canonical identity 中的 `doc_hash` 必须来自上述 gold-free safe projection；不得信任或持久化
   lm-eval 内部 `doc_hash`。
4. 禁止访问 `target`、answer/gold、prediction、correctness、resps/raw scores、metric 或 accuracy。
   question/choices/subject 只用于临时 hash，identity sidecar 仍只持久化原冻结 identity、subject、
   split、ordinal/hash/count。
5. 新增一个真实形状 regression：lm-eval internal `doc_hash` 与 canonical safe hash 不同，但
   safe doc 精确闭合时 projection PASS；safe doc 或 membership 漂移时 fail closed；outcome
   sentinel 不被访问。

默认 edit scope：

~~~text
src/tflt/loopscope/phase5_outcome.py
tests/test_loopscope_phase5_gate_c.py
scripts/loopscope/run_qwen4base_phase5_gate_c.py   # only if fresh-root binding requires it
~~~

不修改 Gate A/B artifacts、Phase 5 card/selector/panel、Phase 1-4 artifacts 或保护路径。允许
一个 identity-fix commit 和至多一个纯 fresh-root provenance binding commit；二者合计仍是一次
planning-audit repair cycle。普通 push，远端 clean fast-forward only。

最小验证：Gate C targeted tests、新 regression、Gate A verifier、直接相关 Phase 3 safe identity
与 Phase 4 seal regression、py_compile、`git diff --check`、protected-path diff。不要跑 full suite
或扩充 tamper matrix。

## 4. Fresh smoke and formal continuation

新的唯一 canonical Gate C root：

~~~text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-c-repair-identity-20260722T072256Z
~~~

在最终 repair commit/hash/argv/root 全部冻结后，使用与旧 attempt 完全相同的八 cells、四个
smoke identities、renderer、batch semantics、recipe 和 information barrier，提交一次 fresh
eight-cell smoke array。重跑四身份是显式授权的工程 smoke，不得复用旧 smoke results。

新 smoke 必须同时证明：8/8 `COMPLETED/0:0`、原结构审计全部通过、每 cell 的 safe canonical
identity sidecar 精确为四 identities、lm-eval internal hash 未被当作 canonical hash、没有 outcome
字段被访问或报告。若失败，再次 `BLOCK`；不得开启第二次 audit-returned repair。

新 smoke PASS 后，立即按原 Gate C handoff freeze/submit exact eight-cell formal panel。formal
一旦开始，禁止代码 mutation；任一 partial/failed formal cell 继续适用原 BLOCK/retry 规则。
只有 fresh root 的 8 × 14,042 sealed outputs/identity sidecars 才可成为 Gate C completion input。

## 5. Information barrier and terminal packet

Gate D 继续锁定。executor 不得查看旧或新 results/sample 中的 accuracy、gain、prediction、gold、
correctness、raw choice score、flip 或窗口相对表现；sealer 仍只能读取 identity sidecar、
command/revision receipts、stat/SHA 和 scheduler rows。

终态仍按原 handoff 发送唯一 `GATE_C_FINAL_AUDIT` 或 `BLOCK`，并额外列出：

- repair commit(s) 与 targeted regression；
- old invalid root/job 不变且未被消费；
- fresh root、新 smoke/formal job IDs；
- safe doc hash projection contract 与 exact 4/4 smoke、14042/57 formal identity closure；
- `lm_eval_internal_doc_hash_consumed_as_canonical=false`；
- repair accounting=`1 planning-audit-returned repair cycle`；
- outcome barrier 与 Gate D lock 未违反。

取得可见 cross-thread delivery confirmation 后停止。
