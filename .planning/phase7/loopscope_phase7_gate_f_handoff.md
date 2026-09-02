GATE_F_HANDOFF

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=F
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a00965-fa7f-7411-b060-0c8315a94711
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-V3-Top2-Outcome-第7阶段-Gate F
AUTHORIZATION_STATE=AUTHORIZED
TERMINAL_EVENT=GATE_F_FINAL_AUDIT_OR_TRUE_BLOCK
NEXT_GATE_AUTHORITY=NONE
PLANNING_RECLOSE_AUTHORITY=NONE
```

## Mandatory collaboration behavior

执行前必须完整重读并遵守
`/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md`。本 task 是唯一 Gate F
executor，但不是 planning/audit authority。普通工程问题或 ordinary operational permission 缺口
只报 planning thread `01a0013e-71c3-7c90-a547-4059b462dc7e`，不得绕过 planning 直接向
用户停摆。

Executor 可在确实缩短本 Gate 时使用 subagent，任一时刻最多三个；subagent 不拥有
Git integration、remote write、GPU/Slurm、job control、scientific amendment 或 terminal-event 权限。
绑定 executor 独自负责整合、调度、修复和终态交付。

本 Gate 任意时刻只允许一个 long-job heartbeat。Formal job ID 确认后立即创建唯一
60-minute `full` monitor（含 PENDING）。Terminal heartbeat 必须先删除自身，再向 exact
executor 投递 `AUTOMATION_TERMINAL_RESUME`；不可达时路由
`AUTOMATION_RELAY_REQUIRED` 给 planning。

Gate 结束时只发送一个 `GATE_F_FINAL_AUDIT` 或真正的 `BLOCK`，并保留向 planning
投递成功的可见确认。不可确认时输出完整 `TERMINAL_DELIVERY_UNCONFIRMED` packet。
Executor 不自宣 `PASS/PASS_WITH_FIXES`。

## Objective

在不重算 Gate D、不改动 Gate E 封存结果的前提下，对三模型 Gate D V3.1 模型内
point-rank 前二窗口运行 MMLU test-14042 全量 loop outcome，共 24 个新 loop cells。
复用 Gate E 已验证的三个 no-loop baselines，闭合 27-cell 分析 panel，再只做一次
combined analysis 和 fresh verifier。

Non-objectives：不重跑 baseline、Gate C/D/E，不重算 V3，不改候选/排名，不增加第三窗口、
其他模型/task/mode，不据 partial outcome 调参，不绘图，不更新 planning 报告，不创建 Gate G。

## Authoritative sources

- Stable policy: `loopscope-tflt/AGENTS.md`
- Live control: `.planning/phase7/loopscope_phase7_control.md`
- Scientific amendment: `.planning/phase7/loopscope_phase7_gate_f_v3_top2_amendment_20260816.md`
- Canonical panel: `资产/报告/phase7/phase7_gate_f_v3_top2_panel_manifest.json`
- Gate D scoring source: `资产/figures/phase7_base_trajectory/source_data/phase7_v3_candidates.csv`
- Gate E canonical outcome root: `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase7-gate-e-20260816T041724Z-formal-remaining24-gitless`
- Runbook: `loopscope-tflt/docs/loopscope_phase7.md`

用户最新指令 > live artifacts/code/scheduler > `AGENTS.md` > control > runbook > 历史 handoff/chat。

## Starting provenance and protected state

```text
local_repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
remote_repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
branch=loopscope
starting_commit=65b39762caed6ab2443d03314d29c58d73d20f3d
required_ancestor=4f59bd93eca4da3cbf458a93508f91c5b23912bc
remote_host=hpc2-hkustgz
```

七个既有 tracked dirty paths 全部是 planning/user protected state，不得 reset、checkout、stash、clean、
覆盖、暂存或提交：`AGENTS.md`、`docs/loopscope_phase2.md`、`docs/loopscope_phase4.md`、
`docs/loopscope_phase5.md`、`docs/loopscope_phase6.md`、`scripts/loopscope/run_qwen4_phase4_p4b.py`、
`src/tflt/loopscope/phase4_outcome.py`。Planning-owned `.planning/phase7/`、`资产/报告/phase7/` 和
`资产/figures/phase7_base_trajectory/` 只读，不由 executor 修改或提交。

## Frozen scientific/runtime contract

```text
models/revisions=identical to Gate E
dataset=cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe
split=test; population=14042 exact identities / 57 subjects
num_fewshot=5; fewshot_split=dev; prompt=plain non-chat
evaluator=lm-eval 0.4.11 choice-loglikelihood; generation=false; metric=acc,none
dtype=bfloat16; quantization=none
qwen25_3b batch_size=16 windows=[14,17),[12,15)
llama32_3b batch_size=16 windows=[11,14),[10,13)
gemma2_2b batch_size=8 windows=[13,16),[10,13)
k={2,3}; cache_strategy={first,last}; iteration_mode=block
strategy=euler; alpha=1.0; beta=0.0; decode_mode=full
new_loop_cells=24; retained_gate_e_baselines=3; analysis_panel=27
bootstrap=subject-stratified paired percentile 95%; R=2000; seed=20260803
interpretation=POST_TERMINAL_V3_POINT_RANK_TOP2_OUTCOME_EXPLORATORY
```

Half-open width-3 windows 映射到 inclusive TFLT layer ranges：Qwen `14:16/12:14`；Llama
`11:13/10:12`；Gemma `13:15/10:12`。Gemma 保留 `ABSTAIN_COMBINED_RANK_UNSTABLE`标记。

## Allowed implementation/Git actions

优先复用 Gate E outcome implementation，只做运行 24-cell Gate F panel、baseline reference closure 和
fresh analysis/verifier 所必需的最小增量。允许路径：

```text
configs/loopscope/phase7_gate_f_*.json
scripts/loopscope/run_phase7_gate_f_*.py
scripts/loopscope/verify_phase7_gate_f_*.py
src/tflt/loopscope/phase7_outcome.py
tests/test_phase7_outcome.py
docs/loopscope_phase7.md
```

允许 focused commit/push `origin/loopscope` 与 HPC2 dedicated checkout `pull --ff-only`。只能显式暂存授权
路径；不得 broad add、提交 protected paths、rebase、force push、merge main。正常路径确需
其他文件时，先向 planning 提交最小原因/diff/targeted regression 请求。

## Allowed HPC2/data/GPU/scheduler actions

- 依 `research-gate-orchestrator` 与 `hpc2-hkustgz-ssh` 访问 exact dedicated checkout、venv、cache 和 workspace；SSH 使用 `BatchMode=yes`、bounded `ConnectTimeout`、`ClearAllForwardings=yes`。
- 只在 fresh timestamped/write-once Gate F roots 写入 card/manifest、debug/formal outputs、logs、receipts 与 final analysis；历史 roots 全部只读。
- 允许加载三模型、读取同一 MMLU test membership/gold，并运行已冻结 24 loop cells。不允许重跑 baseline。运行保持 offline。
- Gate E 的同 commit-family、launcher/env/model/data/evaluator/loop/cache/write/verifier 路径与真实 width-4 formal cells 已通过，width-3 不增加模型或 batch 显存上界。若仅是 panel/config 增量且 targeted tests/static verifier 闭合，允许复用 Gate E 的 resource-shape evidence；仍需一个 `<30min` `debug` functional preflight 覆盖 fresh Gate F launcher/manifest/verifier。修改 launcher/runtime/serialization 则重做受影响 representative debug；debug 无法闭合时才使用一个可保留 formal canary。
- Formal 使用 normal-user 合法最高优先级。现有 A800 真实峰值不支持安全双进程 packing，默认每 GPU 一进程、concurrency=1；仅在 direct model/cell-family 证据证明无 OOM、有头间且 time-to-result 更好时使用有界 packing，不共享 model/tensor state，不改 batch/science。
- 低风险工程/launcher/environment/scheduler/resource/serialization 失败允许诊断和 fresh retry，只补 invalid/missing cells，不覆盖旧 root。

## Information barrier and forbidden actions

1. Panel、cell IDs、baseline references 与 output roots 必须在首个 Gate F forward 前冻结。
2. 24/24 新 loop cells 和 3 baseline-reference membership 全部闭合、pre-outcome verifier `PASS` 前，不得人工读取、打印、排名或汇报 partial Gate F accuracy/gain/correctness。
3. 完整后只运行一次 Gate F combined analysis 与 fresh verifier；不据 outcome 增删窗口/配置。
4. 禁止 raw prompt、token/input IDs、full logits/probabilities、hidden tensors 和无关生成内容。
5. 禁止 digest/hash 证据、destructive cleanup、credential/license 行为、Gate G 或 planning recclose。

## Ordered execution and agility budget

1. 重读权威文件，验证 exact executor/title、Git/protected state、Gate D window 映射和 Gate E baseline references。
2. 最小实现 Gate F 24-cell card/launcher/pre-outcome verifier/combined analyzer/fresh verifier；只跑 targeted tests。
3. Focused commit/push/remote fast-forward 后，冻结 fresh run root、exact 24 cells 和 3 baseline references。
4. 运行一个 `<30min` debug functional preflight。该路径通过且没有新 resource-shape 变化后，立即提交 formal 24-cell panel，不继续添加硬化。
5. Formal job IDs 确认后创建唯一 60-minute full heartbeat 并停止人工轮询。
6. Terminal wake 后只补 invalid/missing。24/24 及 baseline references 闭合后做唯一 analysis/verifier，向 planning 投递唯一终包。

```text
smallest_increment=Gate F panel binding plus baseline-reference reuse on the proven Gate E outcome path
earliest_real_experiment=one fresh Gate F functional debug, then formal 24 cells
minimum_decisive_checks=24-cell expansion; 3 baseline refs; exact config/membership; pre-outcome barrier; one fresh paired recomputation
stop_rule=debug PASS -> formal; 24/24 complete -> analyze once -> terminal packet
```

## Must-pass and terminal requirement

```text
new_loop_cells=24
new_loop_cells_per_model=8,8,8
retained_baselines=3
analysis_panel_cells=27
test_identities_per_cell=14042
subjects=57
no missing/duplicate/extra/partial cells
exact windows/k/cache/mode/strategy/alpha/beta/decode/batch/model/revision
pre-outcome verifier=PASS
one combined analysis + fresh independent verifier=PASS
interpretation=POST_TERMINAL_V3_POINT_RANK_TOP2_OUTCOME_EXPLORATORY
```

终包必须包含 exact executor/title、Git/protected state、files/commands/exit codes/tests、HPC2 jobs/resources/run roots、24-cell completeness、baseline-reference closure、analysis/verifier、repair/deviation、heartbeat lifecycle 和 deliberately-not-taken actions；不包含 digest/hash。

发送：

```text
GATE_F_FINAL_AUDIT
Project/phase: LoopScope Phase 7
Gate: F
Execution thread: 01a00965-fa7f-7411-b060-0c8315a94711
Decision requested: PASS / PASS_WITH_FIXES / BLOCK
```

收件人只能是 planning thread `01a0013e-71c3-7c90-a547-4059b462dc7e`。
