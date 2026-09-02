GATE_E_HANDOFF

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PROJECT_PHASE=LoopScope Phase 7
GATE=E
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a00569-e66d-7eb2-9499-56b01d268366
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-TopAvailable-Outcome-第7阶段-Gate E
AUTHORIZATION_STATE=AUTHORIZED
TERMINAL_EVENT=GATE_E_FINAL_AUDIT_OR_TRUE_BLOCK
NEXT_GATE_AUTHORITY=NONE
PLANNING_CLOSEOUT_AUTHORITY=NONE
```

## Mandatory collaboration behavior

执行前必须完整重读并遵守
`/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md`。本 task 是唯一 Gate E
executor，但不是 planning/audit authority。任何普通工程问题或 ordinary operational permission
缺口都必须报告 exact planning thread `01a0013e-71c3-7c90-a547-4059b462dc7e`；不得绕过
planning 直接向用户停摆或请求普通权限。

Executor 可在确实缩短本 Gate 时使用 subagent，任一时刻最多三个；subagent 只能处理
明确委派的 Gate E 内窄任务，不拥有 Git integration、remote write、GPU/Slurm、job control、
scientific amendment 或 terminal-event 权限。绑定 executor 独自负责整合、调度、修复和终态交付。

本 Gate 只允许一个 long-job heartbeat：formal job ID 真实存在后立即创建 60-minute
`full` monitor（包括 PENDING）。Terminal heartbeat 必须先暂停/删除自身，再向本 exact
executor 投递 `AUTOMATION_TERMINAL_RESUME`；失败时转 planning 的 `AUTOMATION_RELAY_REQUIRED`。
禁止只说“executor 可恢复”后结束。

Gate 结束时只发送一个 `GATE_E_FINAL_AUDIT` 或真正的 `BLOCK`，并保留向 planning
投递成功的可见确认。若无法确认，必须输出完整
`TERMINAL_DELIVERY_UNCONFIRMED` paste-ready packet，不得静默结束。Executor 永远不自己宣布
`PASS/PASS_WITH_FIXES`。

## Objective

在不改动 Gate D 评分或 planning 绘图证据的前提下，对已冻结的 3/3/2 width-4
top-available panel 执行三模型 MMLU test-14042 全量 outcome，闭合 35/35 cells 的共同
identity/completeness，然后只做一次 combined outcome analysis 与 fresh verifier。

Non-objectives：不重跑 Gate C/D，不重算/改 V3，不增加窗口/模型/任务，不据部分 outcome
删减或调参，不绘图，不更新最终中文报告，不执行 Phase 7 consolidated audit，不进入 Gate F。

## Authoritative sources

- Stable policy: `loopscope-tflt/AGENTS.md`
- Live control: `.planning/phase7/loopscope_phase7_control.md`
- Scientific contract: `.planning/phase7/loopscope_phase7_scientific_contract.md`
- 3/3/2 amendment: `.planning/phase7/loopscope_phase7_gate_e_332_panel_amendment_20260815.md`
- Canonical panel: `资产/报告/phase7/phase7_gate_e_panel_manifest.json`
- Runbook/background: `loopscope-tflt/docs/loopscope_phase7.md`

用户最新指令 > live artifacts/code/scheduler > `AGENTS.md` > control > runbook > 历史 handoff/chat。

## Starting provenance and protected state

```text
local_repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
remote_repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
branch=loopscope
starting_commit=6547f68715421a0725a3816b397b814717c10653
required_ancestor=4f59bd93eca4da3cbf458a93508f91c5b23912bc
remote_host=hpc2-hkustgz
```

写入前必须核对 exact repo、branch、HEAD、ancestry、origin 和 dirty/staged state。下列七个既有
tracked dirty paths 全部是 planning/user protected state，不得 reset、checkout、stash、clean、覆盖、
暂存或提交：

```text
AGENTS.md
docs/loopscope_phase2.md
docs/loopscope_phase4.md
docs/loopscope_phase5.md
docs/loopscope_phase6.md
scripts/loopscope/run_qwen4_phase4_p4b.py
src/tflt/loopscope/phase4_outcome.py
```

Planning-owned `.planning/phase7/`、`资产/报告/phase7/` 和 `资产/figures/phase7_base_trajectory/`
不得由 executor 修改、暂存或提交。Gate C/D sealed roots 与 Gate D final analysis 只读。

## Frozen scientific contract

### Models

```text
qwen25_3b=Qwen/Qwen2.5-3B@3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
llama32_3b=meta-llama/Llama-3.2-3B@13afe5124825b4f3751f836b40dafda64c1ed062
gemma2_2b=google/gemma-2-2b@c5ebcd40d208330abc697524c919956e692655cf
dtype=bfloat16
quantization=none
```

### Dataset/evaluator

```text
dataset=cais/mmlu
dataset_revision=c30699e8356da336a370243923dbaf21066bb9fe
split=test
population=14042 exact identities / 57 subjects
sampling=none
num_fewshot=5
fewshot_split=dev
prompt=plain non-chat
apply_chat_template=false
fewshot_as_multiturn=false
generation=false
evaluator=lm-eval 0.4.11 standard MMLU choice-loglikelihood
metric=acc,none
log_samples=true
batch_size=16
```

### Exact windows and cells

```text
qwen25_3b windows=[14,18),[19,23),[16,20); cells=1+3*2*2=13
llama32_3b windows=[10,14),[11,15),[9,13); cells=1+3*2*2=13
gemma2_2b windows=[10,14),[12,16); cells=1+2*2*2=9
total_cells=35
k={2,3}
cache_strategy={first,last}
iteration_mode=block
strategy=euler (TFLT damped_euler equivalent alias)
alpha=1.0
beta=0.0
decode_mode=full
window_width=4
no_loop_baseline=one per model
```

Half-open windows above map to inclusive TFLT layer ranges by subtracting one from stop:
Qwen `14:17/19:22/16:19`; Llama `10:13/11:14/9:12`; Gemma `10:13/12:15`.

### Outcome analysis

```text
primary=standard accuracy
contrasts=each loop cell versus the same-model no-loop baseline
report=correct_count,accuracy,delta_percentage_points,n01,n10
paired_ci=subject-stratified paired percentile bootstrap 95%
bootstrap_replicates=2000
bootstrap_seed=20260803
multiple_comparison_status=not frozen; panel is exploratory and nominal single-cell results are non-confirmatory
```

## Allowed implementation and Git actions

最小新增/修改范围：

```text
configs/loopscope/phase7_gate_e_*.json
scripts/loopscope/run_phase7_gate_e_*.py
scripts/loopscope/verify_phase7_gate_e_*.py
src/tflt/loopscope/phase7_outcome*.py
tests/test_phase7_outcome*.py
docs/loopscope_phase7.md
```

允许读取 Phase 6 Gate O 实现作为标准 MMLU 5-shot outcome 路径参考，但不得复制其
cryptographic/digest 治理；Phase 7 只使用可读路径、配置、membership、commands、job IDs 和
fresh recomputation。若正常路径确实必须修改上述范围外的已有实现文件，先向 planning 发送最小
复现、原因、拟议 diff 和 targeted regression；不自授权。

允许一目的一 commit 的 focused integration，使用显式文件列表暂存、提交并 push
`origin/loopscope`；随后将 HPC2 dedicated checkout `pull --ff-only` 到同一 commit。不得 broad add、
commit protected paths、rebase、force push、merge main、修改或推送 main。

## Allowed HPC2/data/GPU/scheduler actions

- 通过 `hpc2-hkustgz` 读取/操作 dedicated LoopScope checkout、已审计 venv、三模型既有
  snapshots、MMLU cache 和 LoopScope workspace。SSH 必须使用 `BatchMode=yes`、bounded
  `ConnectTimeout` 和 `ClearAllForwardings=yes`。
- 在新 timestamped/write-once Gate E roots 中构建 canonical test identity manifest、pre-outcome freeze、
  debug/formal outputs、logs、receipts 与 combined analysis。不得覆盖、移动或删除旧 roots。
- 允许加载三模型权重、读取 MMLU test identities/gold，执行已冻结 no-loop/loop outcome，
  并在 35/35 completeness 闭合后做一次 combined outcome 分析。运行时必须保持 offline，
  不依赖新的 Hub 网络请求。
- 先在 `debug` partition 运行一个 `<30min` representative end-to-end preflight，覆盖同一
  immutable commit/launcher/venv/model/data/renderer/loop-cache/write/verifier path，至少包含三模型、
  baseline 与一个 loop cell，并覆盖 `k=2/3` 和 `cache=first/last` 的组合路径。
- Preflight PASS 后，正式任务使用 normal user 合法可用的最高优先级 partition/QoS；
  GPU 按显存适配、preflight 实测 runtime、availability、queue 与 time-to-result 选择，不使用
  privileged/admin-only 资源。
- 允许在同一 GPU 上 packing 多个彼此独立的 cell process，但只能在 `debug` 同路径
  preflight 证明无 OOM、无 output collision 且确实缩短 time-to-result 后使用；不允许共享
  model/tensor state 或改变 batch/science。
- 允许对工程、launcher、environment、scheduler/resource、serialization 或 OOM 失败做合同内诊断与
  fresh write-once retry；只补 invalid/missing cells。失败 job 是 attempt event，不是自动 Gate BLOCK。

## Information barrier and forbidden actions

1. Panel/config/cell IDs 和 canonical test identity manifest 必须在首次 outcome run 前冻结；窗口不得
   根据 test 或 outcome 变更。
2. 35/35 cells 全部完成且 pre-outcome completeness verifier PASS 前，不得人工读取、打印、
   排名或汇报 partial accuracy/gain/correctness。完整后只运行一次预注册 combined analysis。
3. 禁止按部分 outcome 取消、重排、调参、补窗或重新评分 Gate D；禁止运行第四个
   Gemma window、non-finite V3 窗口、其他 width、其他模型或其他 task。
4. 禁止持久化 raw prompt、token/input IDs、full logits/probabilities、hidden tensors 或无关生成内容。
   允许 canonical identity、subject、cell config、prediction/correctness 和 paired analysis 必需的最小字段。
5. 不得创建/要求/比较 cryptographic/content digests，不得把 Phase 6 的 digest 实现引入 Phase 7。
6. 禁止修改 protected paths、Gate C/D sealed evidence、planning figures/reports/control/amendment。
7. 禁止 destructive cleanup、无授权 credential/license/network action、Gate F、planning closeout、最终绘图或阶段终审。

## Ordered execution

1. 重读权威文件和 live Git/HPC provenance；确认 exact executor/title/control 一致。
2. 用最小 Phase 7 sidecar 实现 35-cell card、launcher、sanitized per-sample outcome、pre-outcome
   completeness verifier、combined analyzer 和 fresh verifier；只运行 targeted normal-path tests。
3. 显式列表暂存授权文件，做单一目的 commit/push，将 HPC2 dedicated checkout fast-forward
   到同一 clean commit。
4. 在首次 model outcome forward 前，冻结 exact 35 cells、canonical test-14042 identity manifest、配置和
   fresh/write-once roots；pre-outcome verifier 必须确认 13/13/9 静态 membership。
5. 在 HPC2 `debug` partition 运行同路径 `<30min` end-to-end preflight。失败时在本 Gate 内诊断并
   fresh retry；修改 decision-critical launcher/runtime/serialization 后必须重跑受影响 preflight。
6. Preflight PASS 后提交 35-cell formal outcome jobs。获得真实 job IDs 后创建唯一 60-minute
   full heartbeat，然后停止人工轮询。
7. Terminal wake 后由 exact executor 核对所有 jobs/cells/identities/artifacts；只在 35/35 completeness
   闭合后一次性运行 combined analysis 与 fresh verifier。
8. 向 planning 发送唯一 terminal packet 并停止。不绘图、不更新 final report、不自宣 PASS。

## Required validation and must-pass criteria

- Targeted tests 必须覆盖 3/3/2 exact windows、35-cell expansion、baseline 去重、`k/cache` 笛卡尔积、
  half-open/inclusive 映射、canonical membership、partial-outcome barrier、paired analysis 和 fresh recomputation。
- Local 与 remote focused tests/CLI/help/`git diff --check` PASS；remote checkout clean 且与 pushed commit 一致。
- Debug preflight 在 `debug` partition、`<30min`、真实 CUDA/model/data/loop/cache/write/verifier path PASS。
- Formal 每个 cell 精确 14,042 unique identities / 57 subjects；35/35 cells 无 missing/duplicate/extra/partial；
  每模型 baseline 唯一且与该模型所有 loop cells 同 membership。
- 每个 loop cell 的 model/revision/window/k/cache/mode/strategy/alpha/beta/decode/dtype 与冻结 card 一致。
- 35/35 闭合后的唯一 combined analysis 报告 accuracy、paired delta、n01/n10 与 subject-stratified
  95% paired bootstrap CI；fresh verifier 从 sealed minimal outcome rows 独立复算一致。
- Gemma 所有 outcome 保留 `ABSTAIN_COMBINED_RANK_UNSTABLE` / post-ranking exploratory 标记；不得发布
  prospective selected-window success。
- 信息屏障、protected paths、write-once roots、single-heartbeat lifecycle 和 terminal delivery 闭合。

## Agility budget and repair boundary

```text
smallest_increment=one narrow Phase 7 outcome card/runner/pre-outcome verifier/combined analyzer/fresh verifier
earliest_real_experiment=one representative three-model debug preflight immediately after targeted tests
minimum_decisive_checks=35-cell expansion; static membership; one real baseline/loop path; exact config; completeness barrier; fresh paired recomputation
stop_rule=preflight PASS -> submit formal panel; 35/35 complete -> analyze once -> terminal packet; no general framework or extra hardening
```

Executor-owned low-risk engineering repair loops 在本 Gate 合同、路径、信息屏障和 fresh-retry 语义内次数不限，
但必须持续取得材料性进展。凡需改 model/data/split/prompt/identity/panel/window/k/cache/mode/
strategy/alpha/beta/decode/batch/metric/statistics、提前解封部分 outcome、扩大信息或文件权限、
不可逆破坏、credential/license 或外部权限，必须向 planning 发送真正 `BLOCK` 或最小权限
请求；不得自行修订。

## Terminal requirement

终包必须包含：exact executor/title、authorized/final Git 状态、文件、commands/exit codes、tests、
HPC2 jobs/resources/run roots、preflight 证据、35-cell completeness、combined analysis/fresh verifier 路径、
repair/deviation、heartbeat lifecycle、protected-state 与 deliberately-not-taken actions。不包含 digest/hash 证据。

发送：

```text
GATE_E_FINAL_AUDIT
Project/phase: LoopScope Phase 7
Gate: E
Execution thread: 01a00569-e66d-7eb2-9499-56b01d268366
Decision requested: PASS / PASS_WITH_FIXES / BLOCK
```

收件人只能是 planning thread `01a0013e-71c3-7c90-a547-4059b462dc7e`。可见投递确认失败时，
输出完整 `TERMINAL_DELIVERY_UNCONFIRMED` 供手工路由并立即停止。
