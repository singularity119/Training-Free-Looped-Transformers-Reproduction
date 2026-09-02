# LoopScope 第四阶段全局计划与控制

最后更新：2026-07-17

## 1. 文件职责与权威优先级

本文件是 LoopScope 第四阶段的 prospective mutable control plane。它记录 Phase 4 的规划状态、未来 Gate、executor binding、授权边界、冻结科学合同、审计对象、blocker 与 next admission。

`loopscope-tflt/AGENTS.md` 已由 governance-only commit `47e94fcb8c59153b25a83535ea4e564b1ffb5bc4` 将 active-control pointer 切换到本文件。Phase 1–3 controls 仅保存已完成历史；任何 Phase 3/X4 executor 不得承担 Phase 4。

信息源分工：

| 层 | 权威文件 | 职责 |
|---|---|---|
| 稳定政策 | `../loopscope-tflt/AGENTS.md` | 长期 Git、数据、HPC2、保护路径、审计和 terminal 规则 |
| Phase 4 当前控制 | 本文件 | Phase 4 状态、executor、授权、冻结卡、决定与 admission |
| 科学计划 | `../报告/LoopScope_第4阶段总体目标与Gate计划.md` | 假设、公式、对照、实验矩阵、统计与结论上限 |
| Runbook | `../loopscope-tflt/docs/loopscope_phase4.md` | schema、命令形状、工件布局与逐 Gate 操作顺序 |
| Evidence | Git、tests、Slurm、write-once artifacts | 实际发生了什么 |
| History | Phase 1–3 controls、旧报告、线程、memory | 背景与 provenance，不授予当前权限 |

权威优先级：

```text
用户最新明确指令
> live Git / scheduler / immutable artifacts
> loopscope-tflt/AGENTS.md
> 本 Phase 4 control（activation 后）
> runbook
> 报告与历史材料
```

本文件只在 Gate/decision、executor、audited object、科学合同、权限、blocker 或 admission 变化时更新，不记录逐命令日志或普通运行状态。

## 2. 当前身份与状态

```text
phase=LoopScope Phase 4
hypothesis=H4_PV_EK_TRS_QWEN4B_MMLUPRO_V1
planning/audit thread=019f6bf0-d1ec-7d53-ba06-391e9db2bd88
phase3 planning/audit predecessor=019f670d-24ca-7ad0-b92e-64438aa04ef1; historical authority only

control status=PHASE_TERMINAL
current state=PHASE4_TERMINAL_PASS
current Gate=none; P4-A/P4-B/B-2/B-family reporting/P4-C/P4-D all closed PASS
audited P4-C executor=019f6d14-bf10-73c0-8220-2b5a38fd22b9; historical title=execute-LoopScope-PV-EK-TRS-第4阶段-Gate P4-C; all authority revoked
active P4-C authority=none; exact eight-cell completion receipt accepted PASS without outcome unseal
audited P4-D executor=019f78aa-a161-7af0-a080-b7a03f556495; historical title=execute-LoopScope-PV-EK-TRS-第4阶段-Gate D; all authority revoked
active P4-D authority=none; one-time outcome analysis accepted PASS
P4-B/B1 center reporting executor=019f6ca9-ef16-7a53-8949-e47d89fd2dc2; title=execute-LoopScope-PV-EK-TRS-第4阶段-Gate P4-B
active P4-B/B1 reporting authority=none; presentation-only report planning audit PASS and temporary authority revoked
B-2 audited executor=019f6ed2-1bbb-74d2-80bf-f20846137d49; title=execute-LoopScope-Hidden-Geometry-Probe-第4阶段-Gate B-2; all execution authority revoked, historical read-only only
standing Phase 4 HPC resource authority=for authorized inference/acquisition jobs, choose queue/QoS and GPU class adaptively to minimize total time-to-result while using resources efficiently; A40 is suitable for small debug/smoke, while higher-priority queues and A800 are allowed when they improve formal-run throughput
standing delegated operational-permission authority=planning/audit thread may directly grant reasonable permissions needed to complete the current admitted Gate after auditing a permission-only BLOCK; no per-instance user confirmation required
standing phase execution mandate=true; P4-A through P4-D mainline remains serial; P4-B is B1 and its center thread may receive explicit read-only reporting handoffs for completed additive B sub-Gates
next planned Gate=none; Phase 4 terminal reached; any follow-up requires a new hypothesis card/Gate and user authorization
terminal events currently expected=none

phase3 terminal status=PASS
post-P3-X4 terminal status=PASS
phase3/post-P3 active authority=none

dedicated repo=../loopscope-tflt
branch=loopscope
planning base commit=fb5ead49c6ad7b79fe8fd92e44e8ec6aa65d5c2d
planning base origin/loopscope=fb5ead49c6ad7b79fe8fd92e44e8ec6aa65d5c2d
activation-time AGENTS.md SHA256=9784d91d48813932feb547436f4d79677aa847e70242e917780bbe4e6064fc73
activation commit=47e94fcb8c59153b25a83535ea4e564b1ffb5bc4
machine-readable Phase 4 card=configs/loopscope/phase4_pv_ek_trs_card.json; SHA256=980955386907a1865699808219da1379031c1395d9cdb77029585cf266f160f8; status=P4A_PROVENANCE_CLOSED; closure_mode=NOT_REUSABLE_FRESH_REQUIRED
audited Phase 4 commit=ffaa20eaf1c68727b0107ccb1c50ac58001dede2; P4-B PASS; P4-A PASS base remains 670a62f
audited P4-B run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-p4b-20260716T204154Z
authorized P4-C run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-p4c-20260716T223844Z
P4-C audited implementation commit=a210da31d734d539b198d939118136e3e035a36f; clean/pushed; completion receipt SHA256=acc00860913970f622729c9ebc29e72ae3a3457651765367aff1695696c6b7b8; internal manifest=5999ed08879474ea42b1b3f9703c90e93575f826a36a963c1a5b4cebd3bef59e; planning decision=PASS
authorized P4-D write-once analysis root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-p4d-20260719T044036Z; confirmed absent before authorization
P4-D audited implementation commit=ee676a8d13d1bed4aa783da0606f5dd60ec8a11f; clean/pushed; analysis SHA256=cd54251d65da12307927d2cb648c5ffd460cfa4fbb53dfb686c87f1154fb3d9d; table SHA256=12ad365c7bbb018941aa917a1e9c9a7e6a5813a514a72110307263bcd1aca5ef; verifier SHA256=5a5ba98c41427851f52fe9cc5b89b464fc2c473e225b1c486a8c2ac178fae37e; Gate receipt SHA256=9487d62847ba2df6fdf5eb9d4ed53878ddf19864294495530236926b8413e84d; planning decision=PASS
P4-D terminal scientific result=selector ABSTAIN; G_enrich=+4.7207446809 pp, paired 95% CI [+4.2553191489,+5.2111729832] pp, ENRICHMENT_SUPPORTED; blind-six Spearman rho=-0.0857142857, exact p=0.9194444444; fixed 15:18 panel-local best gain=+1.0472074468 pp; scientific label=PV_EK_TRS_PROSPECTIVE_RANKING_SIGNAL_ONLY
B-2 isolated base commit=a210da31d734d539b198d939118136e3e035a36f
B-2 local clone=../loopscope-tflt-phase4-b2; branch=loopscope-phase4-b2
B-2 remote clone=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope_phase4_b2; branch=loopscope-phase4-b2
authorized B-2 run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-b2-hidden-geometry-20260717T064424Z
B-2 audited commit=1b530667eb71c0f1341890582c741e3997bf4e12; receipt SHA256=89961524aabd7a9050cc3d90e2b2fec7f4df32bd3806d2cbc1d3506bd30b08a9; analysis SHA256=9cb7138faa4e15d5948d5053ce0483fdd8b73d360d6d6ed408f118f897114732
authorized B-family report root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-b-family-report-20260717T081130Z
B-family report audited artifacts=CSV SHA256 4f569c61549ce7985561d3296b8eb646fefa027a7925d4a5ea7c2089f2b3b9d0; Markdown SHA256 3470d9fbdf5a3976d4bc190dbb501514b8ee63bff1121649c05cc7fb31daade2; receipt SHA256 48e509c9c37ef41f8d18a7a4213fa616b4c7a8a1bd076828da878c6783ff4395
```

用户已于 2026-07-17 明确授权 Phase 4 activation，并要求本规划线程按 Gate 次序持续分发、验收并推进直至 P4-D phase terminal。P4-A/P4-B/B-2、B-family presentation report、P4-C 与 P4-D 均已关闭 `PASS`，Phase 4 正式终止。冻结 selector 保持 `ABSTAIN`；预注册 High3-vs-Low3 相对富集成立，但六个 blind 窗口绝对 gain 全为负，且 blind-six score/gain 相关不显著。所有 executor 权限均已撤销，既有工件保持只读；任何后续 sweep、variable-width outcome 或新 selector 必须另立 hypothesis card/Gate 并重新获得用户授权。

## 3. 冻结与待绑定科学合同

### 3.1 用户已冻结的内容

```text
reproduction_type=current-checkpoint reproduction; not paper-checkpoint bit-exact
model=Qwen/Qwen3-4B-Instruct-2507
task=mmlu_pro
num_fewshot=5
target_behavior=full CoT generation

selector=one native no-loop prefix forward before the first generated target token
selector_position=last effective prefix token from exact renderer
boundaries=B_0...B_36
distribution=full-vocabulary next-token distribution
primary layer scalars=H_l; KL(p_l || p_36)
formal outcome window_width=4
primary selector width=4; SHIFT/FLANK/CONSENSUS local reversal; CONSENSUS min-z
secondary selector widths=2..8; strict154 and edge-aware224 outcome-blind rankings only
eligibility separated from ranking=true
abstain allowed=true

selector population=MMLU-Pro test all 12032 identities
outcome population=the same MMLU-Pro test all 12032 identities
isolation=information and chronology, not sample identity
claim type=outcome-blind transductive prospective selection within one cell

loop K=3
iteration_mode=block
strategy=euler
step_size=1/3
total_horizon=1
cache_strategy=first
decode_mode=full

forbidden selector inputs=target cot_content; target gold/answer; correctness; generated target tokens/answers; baseline or loop outcome; teacher forcing
diagnostic only=normalized entropy; logit RMS; top1/top10/top100 mass; template-only 14; deterministic option-permutation 512
```

### 3.2 P4-A 必须绑定但不得擅自选择的内容

以下字段必须从对应 baseline 的 immutable artifacts、cached snapshots 和 exact task renderer 闭合，并写入 machine-readable card；若证据不足，P4-A `BLOCK`：

```text
model snapshot revision and tokenizer revision
MMLU-Pro dataset revision/fingerprint
lm-eval version and exact mmlu_pro task/YAML/renderer hashes
chat template and rendered-prefix contract
generation kwargs and primary metric name
historical baseline and 15:18 artifact identities
known_outcome_registry before any Phase 4 selector scoring
remote venv/cache/source clone identities
```

不能为闭合方便切换到最新 dataset/model revision，也不能混用与 historical comparator 不同的 renderer。

### 3.3 用户已确认的 population 规则

用户于 2026-07-17 选择 shared-population design：

```text
selector targets=all 12032 MMLU-Pro test identities
formal outcome targets=the same 12032 identities
validation 70=fixed 5-shot demonstration source only
sample-held-out claim=false
outcome-blind claim=true only if selector freeze predates all Phase 4 outcome access
```

shared population 必须使用一个 canonical identity/renderer manifest。selector producer 与 outcome analyzer exact-close 到同一 bytes/SHA256；任何 target `cot_content`、gold、answer 或 outcome-derived field 在 selector freeze 前可见都是材料性 blocker。

### 3.4 用户已确认的分析与 outcome defaults

```text
blind prospective panel=top3 and bottom3 CONSENSUS scores excluding known_outcome_registry
fixed comparator=15:18
competitive/noninferiority margin=0.30 percentage points
template-only control=14 prompts; one per category; weighted by shared population category counts
option-permutation control=category-stratified deterministic 512-subset by identity hash; compare to original on the same subset
selector bootstrap=2000 category-stratified joint replicates; seed 20260717
outcome bootstrap=2000 category-stratified paired replicates; seed 20260718
selection frequency threshold=0.80
variable-width normalization=E_rate=E/w; K_rate=K/w
variable-width role=secondary outcome-blind scoring only; no automatic outcome
```

用户已于 2026-07-17 接受以上值与双组输出方案。P4-A handoff 必须按此冻结；除非用户在 activation 前发出新的明确指令，否则不得修改。P4-A 开始后不得根据 signal 或 outcome 改动。

## 4. 候选、公式与结论合同

### 4.1 候选与 score

```text
decoder layers=0..35
boundaries=B_0..B_36
width4 start universe=0..32; 33 windows
SHIFT support=1..31
FLANK support=4..28
CONSENSUS support=4..28; 25 scored windows

E(s)=mean_i[H_i(s)-H_i(s+4)]
K(s)=mean_i[D_i(s+4)-D_i(s)]
O_H=E(s)-0.5*(E(s-1)+E(s+1))
O_K=K(s)-0.5*(K(s-1)+K(s+1))
F_H=E(s)-0.5*(E(s-4)+E(s+4))
F_K=K(s)-0.5*(K(s-4)+K(s+4))
CONSENSUS score=min(z_OH,z_OK,z_FH,z_FK)
rank order=score descending; start ascending
```

Full-vocabulary `H` 不得命名为 Choice Entropy。本阶段不计算 A–J answer entropy，不构造 choice-space CEdrop。

### 4.2 Variable-width 次要排名

```text
widths=2..8
full coverage starts=0..36-w; count=37-w; total=224
strict CONSENSUS starts=w..36-2w; count=37-3w; total=154
counts by width full=35,34,33,32,31,30,29
counts by width strict=31,28,25,22,19,16,13

E_rate(s,w)=mean_i[H_i(s)-H_i(s+w)]/w
K_rate(s,w)=mean_i[D_i(s+w)-D_i(s)]/w
O_H=E_rate(s,w)-0.5*(E_rate(s-1,w)+E_rate(s+1,w))
O_K=K_rate(s,w)-0.5*(K_rate(s-1,w)+K_rate(s+1,w))
F_H=E_rate(s,w)-0.5*(E_rate(s-w,w)+E_rate(s+w,w))
F_K=K_rate(s,w)-0.5*(K_rate(s-w,w)+K_rate(s+w,w))
strict score=min(z_OH,z_OK,z_FH,z_FK)

R(s,w)={r in {s-1,s+1,s-w,s+w} | 0<=r<=36-w}
A_H=E_rate(s,w)-mean_r_in_R E_rate(r,w)
A_K=K_rate(s,w)-mean_r_in_R K_rate(r,w)
edge score=min(z_AH,z_AK)
cross-width rank=score descending; width ascending; start ascending
```

strict 154 与 edge-aware 224 分开标准化、分开排名。strict width-4 子集必须复现主 selector 的 25 个 score/rank/eligibility。variable-width 只输出 outcome-blind 诊断，不产生 Phase 4 selected/high3/low3，不进入 P4-C；任何 variable-width outcome 需要 P4-D 后的新 card、Gate 与用户授权。

### 4.3 Eligibility 与 abstention

CONSENSUS point eligibility：

- center `E>0` 且 `K>0`；
- `s±1` 与 `s±4` 四个 comparison windows 各自 `E<0` 且 `K<0`；
- `O_H/O_K/F_H/F_K` 的 95% bootstrap lower bound 全部严格 `>0`；
- equality at zero fails。

25 个 supported windows 全部 score/rank。选择只在 eligible set 中进行；无 eligible、top tie 或 selection frequency `<0.80` 均 `ABSTAIN`。ABSTAIN 后若 score 非退化，仍可运行预注册 top3/bottom3 enrichment，不得把 ineligible top1 改称 selected。

### 4.4 Outcome panel 与 endpoints

```text
baseline=no-loop
fixed comparator=15:18
blind high group=highest three width-4 scores excluding known_outcome_registry
blind low group=lowest three width-4 scores excluding known_outcome_registry
maximum outcome matrix=baseline + 7 loop windows
```

Primary：

```text
gain(w)=accuracy(w)-accuracy(baseline)
G_enrich=mean gain(blind high3)-mean gain(blind low3)
SUPPORTED if lower95CI(G_enrich)>0
REFUTED if upper95CI(G_enrich)<0
INCONCLUSIVE otherwise
```

Secondary：blind-six Spearman/exact permutation、selected vs baseline、selected vs `15:18`、exact McNemar、panel regret。panel regret 不得称 global regret。

Phase labels：

```text
PV_EK_TRS_PROSPECTIVE_SELECTION_SUPPORTED
PV_EK_TRS_PROSPECTIVE_RANKING_SIGNAL_ONLY
PV_EK_TRS_ABSTAIN_WITH_RANKING_RESULT
PV_EK_TRS_NOT_SUPPORTED
PV_EK_TRS_INCONCLUSIVE
```

任何正结论必须带 `same-population outcome-blind transductive` 限定；不能声称 unseen-sample generalization。

## 5. Gate 状态表

| Gate | Objective | Executor | State | Audited object | Next admission condition |
|---|---|---|---|---|---|
| P4-A | 卡、schema、shared manifest contract、prefix scalar 与双组 selector/analyzer 最小本地实现 | `execute-LoopScope-PV-EK-TRS-第4阶段-Gate P4-A` (`019f6c53-1d0b-72a0-8081-226a0fc7e2f6`) | `PASS` | commit `670a62f`; card `9809553…`; verifier manifest `a78b58c…`; historical artifacts `NOT_REUSABLE` | satisfied; P4-B admitted |
| P4-B | 4-identity GPU smoke 后采集 shared-12032 trajectories/controls，冻结 width-4 主排名/panel 与 variable-width 次排名 | `execute-LoopScope-PV-EK-TRS-第4阶段-Gate P4-B` (`019f6ca9-ef16-7a53-8949-e47d89fd2dc2`) | `PASS` | clean/pushed `ffaa20e`; source/smoke/trajectory/selector/panel byte-closed；`ABSTAIN`；panel=`no-loop` + `15:18` + blind high3/low3 | scientific authority remains revoked；separate B1 reporting authority listed below |
| B-2 | 同一 no-loop prefix 的 final-normalized hidden L2/cosine-to-final 轨迹、窗口 delta 与 output-geometry agreement 诊断 | `execute-LoopScope-Hidden-Geometry-Probe-第4阶段-Gate B-2` (`019f6ed2-1bbb-74d2-80bf-f20846137d49`) | `PASS` | isolated commit `1b53066`; smoke `a06e12e…`; original-12032 manifest `93d423a…`; analysis `9cb7138…`; receipt `8996152…` | satisfied; authority revoked; P4-D B-2 prerequisite closed |
| P4-B/B1 reporting | 只读汇总 P4-B 原始 output-distribution scalars 与 B-2 hidden-geometry scalars，生成新的 37-row presentation table/receipt | 原 P4-B/B1 中心线程 `019f6ca9-ef16-7a53-8949-e47d89fd2dc2` | `PASS` | CSV `4f569c6…`；Markdown `3470d9f…`；receipt `48e509c…`；12,032 identity/order exact；replicated scalar drift=0 | satisfied；reporting authority revoked；P4-D report prerequisite closed |
| P4-C | width-4 exact frozen outcome cells 的 limit smoke 与 full-decode acquisition；结果封存 | `execute-LoopScope-PV-EK-TRS-第4阶段-Gate P4-C` (`019f6d14-bf10-73c0-8220-2b5a38fd22b9`) | `PASS` | clean/pushed `a210da3`; 8/8 Slurm tasks `COMPLETED 0:0`; eight result stat/SHA exact；completion receipt `acc0086…` / manifest `5999ed0…`; no outcome parsed | satisfied；authority revoked；P4-D admitted |
| P4-D | 一次性解封、统计裁决、中文结果与 phase-end audit | `execute-LoopScope-PV-EK-TRS-第4阶段-Gate D` (`019f78aa-a161-7af0-a080-b7a03f556495`) | `PASS` | clean/pushed `ee676a8`; exact 12,032×8 closure；analysis `cd54251…`; table `12ad365…`; verifier `5a5ba98…`; Gate receipt `9487d62…`; `G_enrich=+4.7207 pp [4.2553,5.2112]` | Phase 4 terminal；all authority revoked；new work requires new hypothesis authorization |

状态名不授予权限。只有本 control、精确 handoff、executor identity 与 live provenance 一致时，当前 Gate 才算 `AUTHORIZED`。

## 6. Phase 4 activation 条件

P4-A handoff 前必须依次完成：

1. 用户明确授权 Phase 4 activation 和 P4-A；仅要求写规划文件不算授权。
2. 规划线程只读核对 repo branch/commit/dirty 与 Phase 3/X4 terminal state。
3. `AGENTS.md` 以 governance-only change 将 current-control pointer 从 Phase 3 切到本 Phase 4 control，并写入 Phase 4 稳定边界；不得夹带功能代码。
4. governance commit 普通 push，local/origin clean；记录 commit 与 AGENTS SHA256。
5. 新建一个独立 P4-A executor，绑定 exact base、路径与 terminal recipient。
6. 可见确认 `GATE_P4A_HANDOFF` 已投递，并在本规划线程用简短中文解释执行计划。

在以上条件满足前不得 pre-create P4-A executor。

## 7. Gate authorization envelopes

### 7.1 P4-A：冻结与本地最小实现

**Objective**：materialize byte-frozen Phase 4 card、shared population contract、closed schemas、stable scalar computation、width-4 主 selector 与 variable-width 次排名 analyzer、CONSENSUS/ABSTAIN 和 synthetic verifier。

**Admission**：第 6 节全部满足。

**Proposed writable paths**：

```text
configs/loopscope/phase4_*.json
src/tflt/loopscope/phase4_*.py
scripts/loopscope/*phase4*.py
tests/test_loopscope_phase4*.py
docs/loopscope_phase4.md only if implementation command shapes need closure
```

**External actions**：none。禁止 SSH/HPC2、model/dataset load/download、GPU、Slurm、historical/new outcome read。

**Must-pass**：

- exact frozen card and verifier receipt；
- shared 12,032 identity contract rejects missing/extra/reordered/duplicate rows；
- analyzer rejects target cot/gold/generation/outcome and persisted logits/tensors；
- synthetic float32 log-softmax scalars and `D_36≈0` recompute；
- width-4 33/25 support、min-z、eligibility、ranking、abstention and panel exclusion exact；
- variable-width strict 154 / edge-aware 224、E_rate/K_rate、cross-width tie-break 与 width-4 compatibility exact；
- Phase 4 targeted tests、CLI help/dry-run、`git diff --check` pass。

**Agility budget**：复用 Phase 3 sidecar patterns；最小功能路径；targeted tests 优先；没有材料性 broad risk 不跑 full suite。must-pass 一旦闭合就回传，不扩建通用框架。

**Terminal**：`GATE_P4A_FINAL_AUDIT` 或 `BLOCK`，精确投递本规划线程并确认。

### 7.2 P4-B：真实 no-loop prefix selector

**Objective**：4-identity smoke bundle 证明 producer 正确后，尽快采集 shared-12032 original trajectories 与两个预注册 controls，冻结 width-4 主 selector/panel 与 variable-width 次排名。

**Admission**：P4-A `PASS`；新 executor；exact commit/env/cache/model/data permissions and write-once run root in handoff。

**Terminal audited state**：executor `019f6ca9-ef16-7a53-8949-e47d89fd2dc2` 在 clean/pushed `ffaa20eaf1c68727b0107ccb1c50ac58001dede2` 完成 variable-option repair、真实 GPU smoke、shared-12032/option512/template14 acquisition、单次 analyzer/verifier 与 byte freeze。planning 独立核对 local/origin commit、七个远端冻结工件哈希及 `outcome_access_before_freeze=false` 后判定 `PASS`；该 executor 的 P4-B 科学写入、GPU、Slurm、automation 与 Gate 权限全部撤销。其后用户指定该线程为 B1 中心，planning 仅按 §7.2R 恢复独立 reporting authority，不改变本 Gate 的关闭状态。

**Stages**：

1. remote ff-only sync/import/targeted check；
2. 4-identity GPU smoke bundle：4 original + 同 4 option-permutation + 4 matched-category template-only；确认 exact renderer/last-token boundary、零生成 token、B0..B36、finite scalars、no persisted logits；
3. shared-12032 no-loop acquisition；
4. template-only 14 与 deterministic option-permutation 512 diagnostics；
5. one analyzer/verifier pass；freeze width-4 selected/ABSTAIN、blind high3/low3、variable-width strict154/edge224、known-outcome exclusions and comparator reuse closure。

**Forbidden**：任何 target cot/gold/answer/correctness/generated response/loop outcome read；修改 score/split/population；任何 full-decode outcome；下一 Gate。

**Agility budget**：smoke 通过立即做 acquisition；不建设通用 hook framework、receipt tree、full suite 或旧 Gate re-audit。

**Terminal**：`GATE_P4B_FINAL_AUDIT` 或 `BLOCK`，精确投递本规划线程并确认。

### 7.2R P4-B/B1：B-family presentation reporting

**角色**：用户已明确 `P4-B 就是 B1`，并指定它为 B 系列中心执行线程。P4-B 的 selector/acquisition 科学 Gate 仍保持 `PASS` 且关闭；这里只向同一中心线程恢复一次独立的 reporting-only authority，用于汇总已经完成的附加 B Gate。该 authority 不等于重开 P4-B，不允许重新采集、重新分析 selector 或改变任何冻结结论。

**Objective**：在不修改 P4-B 与 B-2 原始 trajectory/manifest/analysis 的前提下，按 `B_0..B_36` 生成一个新的 presentation-only CSV、Markdown 与 JSON receipt。每行同时展示 P4-B authoritative 的 full-vocabulary entropy、KL-to-final、top1/top10/top100 mass，以及 B-2 authoritative 的 final-normalized hidden RMS-L2、raw cosine similarity、derived cosine distance；每项附 exit-minus-entry delta。

**Exact inputs**：

```text
P4-B root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-p4b-20260716T204154Z
P4-B merged trajectory SHA256=879d63c271bc828cec77f3dafff1661dd36218ffb50decb945ab33dd5ac4332d
P4-B trajectory-manifest file SHA256=522f014ba31a5f69882a9853fe535d0758cc8e5454817c7ea22cd4c1decf8d34
B-2 root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-b2-hidden-geometry-20260717T064424Z
B-2 merged trajectory SHA256=d2f0030ce1c7df07d3341915aba680001efd5a3874a09a1ff07c6c211babbf06
B-2 trajectory-manifest file SHA256=93d423a02c57c55790df3d8c2d23b47218a691a793b1671a45c83af7f29e357a
B-2 analysis file SHA256=9cb7138faa4e15d5948d5053ce0483fdd8b73d360d6d6ed408f118f897114732
output root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-b-family-report-20260717T081130Z
```

**Compatibility gate**：两份 trajectory 必须 exact join 到同一 12,032 canonical identities、顺序与每个 `B_0..B_36`。B-2 中复刻的 P4-B scalar 只用于兼容性检查；以 P4-B 为 output-distribution 权威源，并要求对应 scalar 在 `atol=1e-6, rtol=1e-6` 内一致。任何 identity/order/boundary/finite/drift failure 都 `BLOCK`，不得平均、静默选择或覆盖原表。

**Table contract**：37 行严格按 `B_0..B_36`；每列为 12,032 样本的 arithmetic mean。delta 统一为 current boundary minus previous boundary；`B_0` delta 写为 `NA`，不得伪造为 0。cosine distance 必须只由 `1-mean(raw cosine)` 派生，其 delta 与 raw cosine delta 互为相反数。Markdown 必须解释各列方向和单位，不作 outcome/gain 预测。

**Authority**：允许同一 P4-B/B1 executor 以 SSH 只读进入上述 P4-B/B-2 roots，流式读取 scalars，并只写新 report root。CPU-only aggregation 可在安全 login CPU 上完成，或按需使用 `emergency_cpu`；不授权 GPU、model/dataset load、new forward、trajectory/analyzer rerun、P4-C access、outcome access、selector/panel change、Git/source commit。若需要小型 renderer，只能写在 report root 内并记录 bytes/SHA256。

**Must-pass receipt**：绑定 exact input paths/hashes、12,032 ordered identity hash、37-boundary closure、metric/delta definitions、compatibility max absolute/relative drift、exact command or report-root script hash、CSV/Markdown hashes、zero outcome/P4-C access、原 P4-B/B-2 hashes unchanged。

**Terminal**：同一 P4-B/B1 中心线程只发送一次 `GATE_B_FAMILY_REPORT_FINAL_AUDIT` 或 `BLOCK` 给本 planning/audit thread。报告 audit `PASS` 后立即撤销本次 reporting authority。P4-D 需要该报告与 P4-C 均 planning-audit `PASS`。

**Terminal audited state**：planning 对 exact sender、三份 output byte hashes、37×17 CSV shape、12,032 ordered identities、37 boundaries、七个 replicated P4-B scalar 的 zero drift 与 safety flags 做了最小充分只读核对，判定 `PASS`。v1 renderer 因整数 boundary ID 与展示标签不一致而在任何输出生成前 fail-fast；v2 只修正映射并保留 v1 证据，属于不影响 metric/source/identity 的非材料性工程偏差。临时 reporting authority 已撤销。

### 7.2B Gate B-2：Prefix hidden-geometry diagnostic

**Objective**：在与 P4-B 完全相同的 ordinary no-loop、zero-generation、last-valid-prefix-token forward 上，对 B_0..B_36 新采集 final-normalized hidden geometry：RMS-L2 distance to B_36 与 cosine similarity to B_36；以 cosine distance=`1-cosine` 统一 distance delta 方向，并与既有 KL-to-final 作纯诊断比较。

**Chronology/isolation**：B-2 是用户新增的独立 outcome-blind 支线，可与 P4-C 并行，但不得进入 P4-C clone/branch/run root、读取其 job logs/results 或改变其 commit/资源/job。为避免两个 executor 争用 `loopscope`，B-2 唯一 Gate-scoped branch-policy exception 为：从 clean/pushed `a210da31d734d539b198d939118136e3e035a36f` 创建 sibling local clone `../loopscope-tflt-phase4-b2` 与 branch `loopscope-phase4-b2`，在独立 remote clone `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope_phase4_b2` 使用同名 branch。不得 push/merge/rebase `loopscope`，不得让 P4-C checkout fast-forward 到 B-2 commit；B-2 terminal 后是否整合代码另立 planning decision。

**Terminal audited state**：executor `019f6ed2-1bbb-74d2-80bf-f20846137d49` 在 isolated clean/pushed commit `1b530667eb71c0f1341890582c741e3997bf4e12` 完成 Gate。write-once root `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-b2-hidden-geometry-20260717T064424Z`；P4-B source/trajectory hashes保持不变，P4-C access=false。planning 独立核对隔离 Git、smoke invariants、12,032 manifest、analysis checks 与 receipt 后判定 `PASS`，executor 全部权限撤销。

**Frozen representation**：对最后一个有效 prefix token 的每个 boundary hidden `h_l`，令 `z_l=final_norm(h_l)`，包括 B_36 在内每个 boundary 恰好应用一次 exact frozen model final norm；`z_l,z_36` 在 scalar reduction 前转 float32。记录 `hidden_l2_rms_to_final(l)=||z_l-z_36||_2/sqrt(hidden_dim)` 与 `hidden_cosine_to_final(l)=dot(z_l,z_36)/(||z_l||_2||z_36||_2)`。cosine distance=`1-hidden_cosine_to_final` 由 analyzer 派生，不作为第三个独立 producer metric。zero norm/non-finite 必须 fail closed，不允许 epsilon fallback、clamp 或静默修正。

**Frozen comparability**：model/tokenizer/revision、MMLU-Pro revision、canonical test-12032 identity/order/source manifest、5-shot renderer/tokenization、bfloat16、`use_cache=false`、hidden states、last-valid-token、B_0..B_36、`target_generation_count=0` 与既有 H/K/top-mass scalar 定义保持 P4-B exact。只采集 original-12032；不重跑 template14/option512。不得持久化 hidden/residual/logit/probability/generated/outcome fields。

**Unified deltas**：全部用 exit-minus-entry。per-layer `delta_x(j)=x(B_j)-x(B_{j-1})`；window `delta_x(s,w)=x(B_{s+w})-x(B_s)`，必须等于内部 per-layer deltas 之和。`delta L2<0`、`delta cosine-distance<0`、raw `delta cosine>0`、`delta KL<0` 都表示向 B_36 靠近；`delta H<0` 表示 entropy 下降。表格同时报告 raw cosine 与 cosine distance，或显式记录两者 delta 互为相反数。

**Stages**：

1. 隔离 clone/branch admission，最小 scalar/schema/analyzer/launcher 实现与 targeted tests；
2. 复用 P4-B smoke 中同四个 original identities 做两次 deterministic GPU smoke，验证 37 rows、finite、B_36 L2≈0/cos≈1/cos-distance≈0、L2 nonnegative、cosine range、float32 与 zero generation/no tensor persistence；
3. smoke pass 后复用 audited source/order/shard membership，只对 original-12032 做一次正式 acquisition，工件与 P4-B 分离；
4. 生成 boundary mean + current-minus-previous layer delta 表、width-4 主 window delta 表、预注册 widths 2..8 次要 aggregate delta 表；生成三种 geometry 的 37-point mean-curve pairwise Pearson/Spearman，以及每个 width-4 window 的 sample-level delta sign-agreement；
5. concise diagnostic interpretation 后停止。不得把新 metric 加入 selector、eligibility、ranking/panel，亦不得读取 outcome 或声称预测 gain。

**Must-pass**：exact source/order/prompt/token identities；producer 只新增授权 scalars/metadata；B_36 endpoint tolerance、range、finite、additivity、deterministic rerun pass；original count=12,032 without missing/extra/duplicate/reorder；one Gate receipt binds branch/commit/card/upstream/source/jobs/commands/counts/artifact hashes；P4-B/P4-C protected artifacts unchanged。

**Agility/resource**：优先旁路复用 P4-B producer，不建设通用 hook framework；targeted tests 后立即做 4-identity real smoke，pass 后立即正式采集。A40 适合 smoke/短 prefix shards；formal 可按排队与吞吐选择 A40/A800/高优先级资源，但不得取消、改优先级或操纵 P4-C jobs。最多 3 个 active subagents，GPU/Slurm 只由 executor 操作。

**Retry**：最多 2 个 executor-owned low-risk repair loops 与 1 个 bundled audit-returned cycle。scheduler/transient failure 只允许在 fresh attempt path 上保留证据并最多 2 个 retry cycles；已 valid 的 shard 不覆盖、不重跑，只补 failed/missing shards。deterministic four-identity second pass 是预注册测量，不计 retry。

**Terminal**：`GATE_B2_FINAL_AUDIT` 或 `BLOCK`，只由 exact B-2 executor 精确投递本规划线程并确认。B-2 `PASS` 不解封 P4-C outcome；P4-D 需等待 P4-C 与 B-2 均 `PASS`。

### 7.3 P4-C：full-decode outcome acquisition

**Objective**：只对 frozen width-4 panel 取得 exact full-decode outcomes，同时保持 partial outcomes 封存。

**Admission**：P4-B `PASS`；selector/panel hashes recorded；新 executor；exact scheduler/compute actions explicitly authorized。

**Current exact authority**：executor `019f6d14-bf10-73c0-8220-2b5a38fd22b9`，base commit `ffaa20eaf1c68727b0107ccb1c50ac58001dede2`，write-once root `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-p4c-20260716T223844Z`。只允许 fresh acquisition 八个 cells：baseline `no-loop`；fixed comparator `15:18`；blind high3 `6:9,10:13,25:28`；blind low3 `4:7,5:8,22:25`。panel manifest file SHA256=`02148127ef7b634106b1ee1044f591012d3543563e94344583480a7c3061eb79`，embedded manifest=`abc65342e5444638d3073ce780ec37d8d72eb7c34d87ac809bdb29a24056d094`，selector freeze file SHA256=`a4cc48a9b82b3d89b962cdefb55b6e4133c08a0aff3431f0de72c3750fc062c8`。`ABSTAIN`/`selected_window=null` 不删减 enrichment panel；variable-width outcome 仍禁止。

**Standing resource authorization**：P4-C 被正式 admit 并绑定新 executor 后，其 debug/smoke 与 frozen full-decode outcome jobs 应根据任务规模、排队等待、预计吞吐和总完成时间选择资源。小型 debug/smoke 可优先使用 A40；正式 full runs 可在确有加速价值时使用账户可用的高优先级 partition/QoS 与 A800，也可在更快开跑或整体更高效时继续使用 A40/普通队列。无需为选择 A800 或高优先级资源再次询问用户，但二者不是强制默认。该授权不提前解锁 P4-C，不改变 outcome cell、recipe、retry、unseal 或 information-barrier 规则，实际 partition/QoS/GPU/job IDs 仍须写入 exact handoff 与 immutable run evidence。

**Stages**：

1. historical baseline/`15:18` 均为 `NOT_REUSABLE`；本 Gate 对上述八个明确 cells 全部 fresh acquisition；
2. each new cell one isolated `limit=5` engineering smoke；
3. verify actual loop and `operator_body_calls=3`；
4. smoke pass 后立即提交 exact full cells；
5. executor-owned 10/30/60-minute low-frequency read-only automation；
6. all required cells terminal 后生成 sealed completion/identity receipt，不做 outcome analysis。

**Forbidden**：partial accuracy inspection、panel change、automatic retry after valid outcome、variable-width outcome、new window/K/recipe、planning-thread polling。

**Agility budget**：最小 launcher/manifest adapter、targeted tests、one smoke then full；不扩建 framework 或重复 suite。

**Terminal**：`GATE_P4C_FINAL_AUDIT` 或 `BLOCK`，精确投递本规划线程并确认。

### 7.4 P4-D：unseal、analysis 与 phase audit

**Objective**：零新增 GPU，一次性解封 shared-12032 outcomes，运行 frozen analyzer，生成科学标签和 concise phase-end audit。

**Admission**：P4-C `PASS`；all required cells complete and identity-close；新 executor；exact unseal allowlist in handoff。

**Must-pass**：

- analyzer/card/identity/panel hashes exact；
- full cell membership complete；
- one joint paired analysis produces per-window gain、`G_enrich` CI、blind-six ranking、selected/comparator、panel regret and label；
- claims include transductive/same-population/current-checkpoint limits；
- Chinese summary and compact verifier complete。

**Forbidden**：new GPU/Slurm、rerun、retuning、post-outcome threshold/candidate change、additional sweep。

**Terminal**：`GATE_P4D_FINAL_AUDIT` 或 `BLOCK`，精确投递本规划线程并确认。P4-D planning audit is Phase 4 terminal。

## 8. Protected state 与远程根

```text
local dedicated repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
HPC2 dedicated repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
LoopScope workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope
proposed Phase 4 run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-<gate>-<UTC>/
```

Protected：

- reproduction checkout、Phase 1–3/X4 immutable runs、controls/reports；
- `wrapper.py、strategies.py、cache.py、config.py`；
- lm-eval framework/source/task/YAML/renderer；
- baseline/`15:18` artifacts；
- `.planning/*control.md` 与 `AGENTS.md` 只由 planning/audit thread 更新；
- 禁止 amend/rebase/reset/force-push、覆盖/移动/删除历史工件或自动 merge `main`。

任何修改默认禁止文件的需求必须以最小 reproducer、原因、拟议 diff 和回归测试 `BLOCK` 回规划线程，不得由 executor 自行扩大权限。

## 9. Repair、retry 与审计深度

```text
executor-owned low-risk repair loops per Gate<=2
bundled audit-returned material repair cycles<=1
second audit-returned cycle=only same unresolved material root cause or repair-introduced material regression
```

只允许同一可复现 engineering root cause、保持 frozen science/permissions/data/external effects 的修复。以下变化必须 `BLOCK`：model/data/revision/population/identity、target field visibility、score/eligibility/ranking/panel/threshold/statistics、loop recipe、protected code、resource/scheduler authority 或 scientifically meaningful alternative。

有效 outcome 产生后不得自动 retry。P4-C handoff 若未明确 retry semantics，则 scheduler rejection、partial failure 或 hardware change 均回规划线程。

用户已将 Phase 4 后续 operational-permission 裁决权常设委派给本 planning/audit thread。executor 若因 handoff 权限不足而 `BLOCK`，planning 在核对其必要性、材料性与最小范围后，可直接向同一 Gate executor 授予完成当前 Gate 所需的 host/path、固定版本下载/环境维护、remote debug、项目根写入、带最小 reproducer 的特定 protected implementation path 修改、GPU/Slurm/job control 或 write-once evidence-preserving retry 权限，并同步 control 与 supplemental handoff；无需逐次征求用户。executor 不得自行扩权，下一 Gate 仍保持锁定。

该委派不覆盖：修改冻结 model/data/population/renderer/score/panel/statistics/threshold 或其他科学合同；提前解封 outcome/answer/generated content；扩展 Phase 4 Gate、实验矩阵或外部系统范围；不可逆删除/覆盖/历史重写；凭证、秘密、安全或法律/外部主体授权。混合 blocker 只把这些真正超出委派的部分交回用户，普通工程权限部分由 planning 直接裁决。

Per-Gate planning acceptance 只检查 1–3 个会改变 transition 的事实：authorized sender、exact commit/artifact/job、frozen contract/identity closure。不得重复 executor full verification。P4-D 后只做一次 concise phase-end audit。

## 10. Event-driven terminal 与 long-job continuation

每 Gate 一个独立 executor。规划线程 handoff 后不轮询、不提前创建下一 Gate executor。executor 只能发送一次：

```text
GATE_P4X_FINAL_AUDIT
```

或：

```text
BLOCK
```

terminal packet 至少包含 Gate/executor、authorized/final branch/commit/dirty、files、exact commands/exit codes、host/jobs/run root/artifact hashes、observed result、deviations、repair loops、actions not taken、protected-state confirmation 和请求 `PASS/PASS_WITH_FIXES/BLOCK`。

用户授权的 P4-B/B1 中心汇总是该规则的窄例外：附加 B Gate 仍由独立 executor 完成和终态审计，但 P4-B/B1 可在 planning 发出新的 reporting-only handoff 后汇总其只读结果，并用不同事件名 `GATE_B_FAMILY_REPORT_FINAL_AUDIT` 终止。该例外不允许一个 executor 承担多个科学 Gate，也不复活已关闭 Gate 的科学权限。

未来新建执行线程的稳定命名格式为 `execute-LoopScope-<descriptor>-第<阶段>阶段-Gate <gate>`；`<gate>` 不再重复阶段前缀，例如使用 `Gate A`、`Gate B-2`、`Gate D`，不得使用 `Gate P4-A` 或 `Gate P4-D`。既有线程 title 作为历史 identity 保留，不因本规则追溯改名。

只有 planning/audit thread 作 Gate decision。投递只有在精确 destination 的可见工具确认后才成立；否则 executor 输出 `TERMINAL_DELIVERY_UNCONFIRMED` 与完整 paste-ready packet。

P4-C long-job monitor 是只读 observer。terminal 时必须暂停/删除自身，并向 exact P4-C executor 可见确认投递：

```text
AUTOMATION_TERMINAL_RESUME
```

若无法触发/确认，则向本规划线程发送并输出 `AUTOMATION_RELAY_REQUIRED`。planning 只允许一次 bounded relay recovery，不提交/重试/轮询作业。

## 11. 决策记录

| 日期 | 范围 | 决定 | 依据 | 下一条件 |
|---|---|---|---|---|
| 2026-07-17 | Phase 3/X4 admission | Phase 3 P3-A–P3-G 与 Post-P3-X4 已 `PASS`；无 active authority | Phase 3 control 与 clean `fb5ead4` | 可规划新 hypothesis；不能继承旧 executor |
| 2026-07-17 | Phase 4 insight | 采用 `H4_PV_EK_TRS_QWEN4B_MMLUPRO_V1`，直接进入 4B Instruct × MMLU-Pro CoT，不做 1.7B bridge | 用户最新明确指令 | materialize three planning files only |
| 2026-07-17 | population | selector 与 outcome 共同使用全部 12,032 test identities；隔离 outcome 值与时间，不隔离样本 | 用户对规划问题选择第一种 | claim 限定为 same-population transductive prospective |
| 2026-07-17 | outcome defaults | width-4 blind high3 + low3 panel、baseline、`15:18` 与 `0.30 pp` 竞争阈值获认可 | 用户明确回复“目前默认值认可” | P4-A 按 frozen defaults materialize |
| 2026-07-17 | dual scoring groups | width-4 33/25 为 outcome 主组；width 2..8 strict154/edge224 为 outcome-blind 次组，按 E_rate/K_rate 排名 | 用户明确认可该方案 | variable-width outcome 仅可由 P4-D 后新 Gate 授权 |
| 2026-07-17 | current authorization | 三份规划文件可写；P4-A、executor、commit/push、SSH/GPU/Slurm/outcome 均未授权 | 用户请求范围 + Gate protocol | 用户明确授权 activation/P4-A 后更新 AGENTS 并 dispatch |
| 2026-07-17 | Phase 4 activation | 用户授权本规划线程按 Gate 次序持续分发、验收并推进至 P4-D；治理 commit `47e94fc` 已推送且 local/origin clean；P4-A 绑定 executor `019f6c53-1d0b-72a0-8081-226a0fc7e2f6` | 用户最新明确指令 + live Git + AGENTS SHA256 | 等待唯一 `GATE_P4A_FINAL_AUDIT` 或 `BLOCK`；不轮询 |
| 2026-07-17 | executor limited delegation | 当前及后续 Gate executor 在任一时刻可有最多 3 个 active subagent 处理 Gate 内有界工程子任务；这是并发上限，不是历史累计配额，整个 Gate 总创建数不设硬上限，只要每次委派合理且不以 churn 规避并发限制。executor 仍是唯一责任主体，subagent 无跨 Gate、科学合同、GPU/Slurm、裁决或终态权限 | 用户最新明确指令 + validated skill rule | 当前 P4-B 收到 superseding supplement；未来 Gate exact handoff 继承 |
| 2026-07-17 | P4-A terminal audit | `BLOCK`：本地最小 selector contract 已在 clean/pushed `32e9425` 闭合，但 exact model/tokenizer/data/task/renderer/generation/comparator/environment provenance 17 字段缺失；normal verifier 正确 fail-closed | authorized executor terminal packet + live Git + card/receipt + allowed-path diff | P4-B 锁定；等待用户授权同一 executor 做 bounded outcome-free read-only provenance repair |
| 2026-07-17 | P4-A provenance repair authorization | 用户授权同一 P4-A executor 通过 outcome-free、read-only SSH 检查 HPC2 immutable identity/config/environment metadata；不得读取 outcome/answer/correctness/generated content，不得加载模型/数据或使用 GPU/Slurm | 用户明确回复“授权” | 补齐 17 项 provenance、normal verifier fail-closed 转为 pass、commit/push 后回传新终态；P4-B 仍锁定 |
| 2026-07-17 | P4-A repair cycle 1 audit | `BLOCK`：authorized LoopScope/cache roots 安全闭合字段 1–8、15–17 的候选证据，但未含 current-checkpoint 4B MMLU-Pro historical comparator；9–14 仍不可唯一绑定；local repo 无变化且 outcome barrier 完整 | same-executor terminal packet + live clean `32e9425` + unchanged card/receipt | 同一 root cause；允许 skill 规定的第二且最后 repair cycle，不创建 P4-B |
| 2026-07-17 | comparator non-reuse fallback | Phase 4 预注册 runbook 允许 baseline/`15:18` 无法完整闭合时在 P4-C fresh acquisition；第二 repair 可把 exact historical artifact 判为 `NOT_REUSABLE`，但不得把 current defaults 伪装为 historical | Phase 4 report §1.2/§7 P4-C + runbook §7.1–7.3 | exact reuse closure 或 exact non-reuse closure 二选一；normal verifier 必须识别该预注册分支 |
| 2026-07-17 | exact historical comparator root | 用户提供 exact run root `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs.legacy-project-symlink-20260701-031953/phase3-qwen4b-instruct2507-mmlupro-bs64-bs64-20260701-164838`；第二 repair 不再广搜三个历史根，只对该 root 作 outcome-free metadata/hash closure | 用户最新明确路径 | 可复用则 bind exact baseline/15:18；不可复用则登记 artifact identity/known registry 并冻结 P4-C fresh acquisition |
| 2026-07-17 | default HPC connectivity/debugging | 用户纠正“本地/增量 Gate 不等于离线 Gate”：所有 Gate executor 默认可连接项目已批准 HPC、读取相关 source/cache/env/provenance，并获得与当前 Gate 相称的远程调试权限；GPU/Slurm/outcome/昂贵执行仍逐 Gate 精确授权。受委派 subagent 可在父 executor 窄范围内做只读 SSH | 用户最新明确指令 + updated research-gate-orchestrator | 当前 P4-A 增补 dedicated clone ff-only sync 与 CPU/import/config/verifier debug；不跨入 P4-B GPU smoke |
| 2026-07-17 | P4-A final audit | `PASS`：clean/pushed commit `670a62f` 以 `NOT_REUSABLE_FRESH_REQUIRED` 闭合 17 项 provenance；历史 baseline/`15:18` 仅登记 identity，不读取 outcome，P4-C 冻结为 fresh acquisition；normal verifier manifest `a78b58c…` 通过 | exact executor terminal + live Git/card/receipt + planning normal verifier | 撤销 P4-A 全部执行权限；P4-B admitted |
| 2026-07-17 | P4-B authorization | 新 executor `019f6ca9-ef16-7a53-8949-e47d89fd2dc2` 绑定 Gate P4-B；允许 HPC2 ff-only sync、targeted debug、GPU/Slurm 4-identity smoke、shared-12032 no-loop acquisition、14 template controls、512 option permutations 与 outcome-blind selector freeze | standing phase mandate + P4-A PASS + exact handoff | 等待唯一 `GATE_P4B_FINAL_AUDIT` 或 `BLOCK`；P4-C 保持锁定 |
| 2026-07-17 | Phase 4 adaptive HPC resource authorization | 当前 P4-B 及后续正式 admit 的 inference jobs 按任务规模、排队时间和吞吐平衡资源：小型 debug/smoke 可用 A40，正式任务可在有利时使用高优先级 queue/QoS 与 A800；高级资源是许可上限而非强制默认 | 用户对上一资源规则的明确纠正 | 当前 P4-B 收到 superseding supplement；未来 P4-C exact handoff 继承，Gate、outcome、retry 与信息权限不提前扩大 |
| 2026-07-17 | delegated operational-permission authority | 后续 executor 若因当前 handoff 权限不足而 `BLOCK`，本 planning/audit thread 审查后直接授予完成当前 Gate 所需的合理、最小权限，无需逐次征求用户；executor 仍不得自扩权 | 用户最新明确授权 + updated research-gate-orchestrator | 当前 P4-B 收到 supplement；未来 Gate handoff 继承。科学合同、提前 unseal、phase expansion、不可逆破坏及安全/外部权限仍回用户 |
| 2026-07-17 | P4-B option-cardinality BLOCK audit | `PASS_WITH_FIXES`：clean/pushed `4870fc3` 的正常 source path 错把所有 MMLU-Pro 题目限定为 10 options；planning 在 HPC2 只读复核冻结 12,032 rows 为 3..10 options，且 formal/smoke permutation 均遗漏实际长度。批准 audit-returned repair 1/1，保持 dataset/population/renderer/score 不变 | exact executor BLOCK + live Git/code + outcome-free HPC2 option-length count | 同一 executor 做最小 repair；通过后继续原 P4-B，不创建 P4-C |
| 2026-07-17 | P4-B final audit | `PASS`：clean/pushed `ffaa20e` 完成 12,032 original、512 option、14 template 的真实 no-loop trajectory；真实 smoke、29 shard receipts、单次 analyzer/verifier 与冻结工件闭合。selector 合法 `ABSTAIN`，blind high3=`6:9,10:13,25:28`，low3=`4:7,5:8,22:25`，fixed comparator=`15:18` | exact executor terminal packet + local/origin commit + planning read-only remote SHA/manifest check | 撤销 P4-B executor 全部执行权限；P4-C admitted |
| 2026-07-17 | P4-C authorization | 新 executor `019f6d14-bf10-73c0-8220-2b5a38fd22b9` 绑定 exact frozen eight-cell panel；baseline 与 `15:18` 均 fresh acquisition；允许 adaptive A40/A800/priority Slurm、cell limit smoke、full decode、sealed completion 与只读低频 monitor | P4-B `PASS` + frozen selector/panel hashes + standing phase/resource/permission mandate | 等待唯一 `GATE_P4C_FINAL_AUDIT` 或 `BLOCK`；P4-D 保持锁定且 outcome 不得提前解封 |
| 2026-07-17 | Gate B-2 scope addition | 用户认可 hidden-geometry probe 并明确要求新建独立 B-2 executor；final-normalized RMS-L2/cosine-to-B36、exit-minus-entry deltas、original-12032 acquisition 与 diagnostic-only analysis 获授权 | 用户在 closed P4-B thread 的最新明确指令，经 exact terminal routing 到 planning | B-2 与 P4-C 信息隔离并行；不得回用 closed P4-B executor |
| 2026-07-17 | Gate B-2 isolation/authorization | 为避免 active P4-C 与 B-2 争用 `loopscope` branch/checkout，B-2 从 clean/pushed P4-C implementation commit `a210da3` 建立独立 local/remote clone 与 branch `loopscope-phase4-b2`；新 executor `019f6ed2-1bbb-74d2-80bf-f20846137d49` 获 outcome-blind HPC/GPU/Slurm 权限 | live Git clean `a210da3`; P4-B source/trajectory hashes；new clone/run roots absent；用户新增 Gate 权限 | 等待 `GATE_B2_FINAL_AUDIT` 或 `BLOCK`；P4-C authority/commit/artifacts不受 control drift 影响；P4-D 等待两 Gate PASS |
| 2026-07-17 | Gate B-2 final audit | `PASS`：isolated clean/pushed `1b53066`；双遍四样本 smoke endpoint/range/determinism/zero-generation/no-persistence 全过；24/24 shards 精确 12,032；33 width-4/224 width2..8、additivity 与 cosine-sign checks 闭合。KL/L2/cosine-distance 轨迹高度但非完全一致 | exact executor terminal + planning local/remote isolated Git + smoke/manifest/analysis/receipt read-only checks | 撤销 B-2 executor 全部权限；保留 branch/run artifacts；P4-D 只再等待 P4-C PASS |
| 2026-07-17 | B-family central reporting | 用户明确 `P4-B 就是 B1`，并指定 P4-B/B1 为 B 系列中心执行线程，可汇总完成的附加 B Gate；同一 P4-B thread 恢复一次 reporting-only authority，原 P4-B/B-2 artifacts 只读，只向新的 write-once root 生成 37-row presentation CSV/Markdown/receipt | 用户最新明确纠正 + P4-B/B-2 immutable PASS artifacts | 等待 `GATE_B_FAMILY_REPORT_FINAL_AUDIT` 或 `BLOCK`；P4-D 同时等待该报告与 P4-C audit PASS |
| 2026-07-17 | future executor naming | 后续新执行线程的 Gate 字段不再重复 Phase 前缀：使用 `Gate A/B-2/D`，不使用 `Gate P4-A/P4-D`；既有线程名保留为历史 identity | 用户最新明确指令 + validated `research-gate-orchestrator` rule | 下一新线程从 `execute-LoopScope-…-第4阶段-Gate D` 起执行 |
| 2026-07-17 | B-family presentation report final audit | `PASS`：同一 P4-B/B1 中心线程在新 write-once root 生成完整 37×17 CSV/Markdown/receipt；12,032 ordered identities、37 boundaries exact，七个 replicated P4-B scalars 最大绝对/相对漂移均为 0，P4-C/outcome/GPU/Git access 均为 false。接受 v1 在输出前 fail-fast、v2 仅修 boundary label mapping 的非材料性偏差 | exact terminal sender + planning read-only remote hashes/CSV/receipt sample | 撤销 B1 reporting authority；P4-D 的 B-family report prerequisite 满足，只等待 P4-C PASS |
| 2026-07-19 | P4-C final audit | `PASS`：clean/pushed `a210da3`；job `9992990` 的 8/8 A800 array tasks 全部 `COMPLETED 0:0`；八个 fresh result 文件 stat/SHA 与 terminal packet 完全一致；completion receipt `acc0086…` 为 `SEALED_COMPLETE`，并确认 outcome 尚未解析、P4-D 尚未 unseal | exact executor terminal + planning live Git + read-only Slurm/stat/SHA/receipt checks | 撤销 P4-C 全部权限；P4-D admitted and bound to new executor `019f78aa-a161-7af0-a080-b7a03f556495` |
| 2026-07-19 | P4-D authorization | 新 executor `019f78aa-a161-7af0-a080-b7a03f556495` 绑定一次性 outcome unseal、冻结 paired analysis、中文科学摘要、compact verifier 与 phase-terminal evidence；零新增 GPU/Slurm、禁止 rerun/retune/sweep | P4-C PASS + B-family report PASS + standing phase mandate | 等待唯一 `GATE_P4D_FINAL_AUDIT` 或 `BLOCK`；P4-D planning audit is Phase 4 terminal |
| 2026-07-19 | P4-D final audit / Phase 4 terminal | `PASS`：clean/pushed `ee676a8`；八个 raw group `exact_match,custom-extract` 与 12,032×8 paired analysis 逐项一致；`G_enrich=+4.720745 pp`，95% CI `[+4.255319,+5.211173]`，但 blind-six rho `-0.085714`、exact `p=0.919444`，六个 blind gain 全负；`15:18` 为 panel-local best `+1.047207 pp`；selector 继续 `ABSTAIN` | exact executor terminal + planning live Git/protected diff + remote aggregate exact-match cross-check + analysis/verifier/hash-chain/claim audit | 撤销 P4-D 全部权限；Phase 4 `PHASE4_TERMINAL_PASS`；不得自动扩展实验 |

## 12. 当前 blocker 与下一动作

- Current blocker：none。Phase 4 已在 P4-D planning audit 后终止，全部 Gate 均为 `PASS`，但科学结论是有限的 ranking-signal-only，而不是选窗成功或可部署收益。
- Frozen selector state：`ABSTAIN`，`selected_window=null`，但预注册 enrichment panel 保持完整：fresh `no-loop`、fresh `15:18`、high3 `6:9,10:13,25:28`、low3 `4:7,5:8,22:25`。known registry 继续只排除 `15:18` 的 blind slot，不提供 outcome 值。
- Current executor：none。P4-A/P4-B/B1/B-2/P4-C/P4-D 的所有执行、reporting、outcome、GPU、Slurm、SSH-write 与 monitoring 权限均已撤销；既有 run artifacts immutable。
- Current authorized external actions：none under the Phase 4 Gate control。规划线程可基于已审计 aggregate artifacts 完成用户已授权的导师汇报/Notion 更新，但不得把 reporting 解释为新实验权限。
- Retry envelope：closed。不得重跑、替换、重新筛选或修改任何 Phase 4 outcome/selector/analysis 工件。
- Actions not authorized：任何新 inference/GPU/Slurm、cell rerun、panel/window/K/recipe/threshold/statistic 改动、variable-width outcome、additional sweep、selector retuning、重新分析 frozen outcomes、覆盖/删除既有 run、protected core 修改或历史重写。后续研究必须新建 hypothesis card/Gate 并获得用户授权。
- B-2 closed result：record_count=12,032；curve Pearson/Spearman 为 KL-vs-L2 `0.855278/0.986012`、KL-vs-cosine-distance `0.937548/0.982456`、L2-vs-cosine-distance `0.936369/0.997155`；width-4 mean sign-agreement 为 `0.845513/0.853325/0.954606`，说明三种距离轨迹强相关但并非逐窗等价。该结果只支持 hidden/output geometry agreement 诊断，不支持 gain prediction。
- Terminal interpretation：High3-vs-Low3 的相对富集显著，但它主要来自 Low3 更严重的退化；六个 blind 窗口绝对 gain 全为负，score/gain 六窗单调相关不显著。`PV_EK_TRS_PROSPECTIVE_RANKING_SIGNAL_ONLY` 不能表述为 selected-window success、absolute benefit 或 unseen-prompt generalization。
- Accepted non-material audit note：analysis JSON 的 legacy 字段 `population.raw_and_canonical_order_exact=true` 名称容易误读；权威 `identity_closure` 与实现表明八组 raw evaluator 顺序彼此一致，但不同于 P4-B canonical 顺序，分析通过 unique canonical identity join 恢复后再计算。该命名不改变 identity membership、correctness matrix、统计或结论，禁止为此重跑已冻结分析。
- Authority owner：用户保留任何新 hypothesis、phase scope、实验扩展、不可逆破坏及安全/外部授权。
- Next admission：none。Phase 4 terminal；任何后续动作需独立新授权，旧 executor 不再发送事件或执行操作。
