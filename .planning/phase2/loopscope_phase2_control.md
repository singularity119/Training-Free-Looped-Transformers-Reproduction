# LoopScope 第二阶段全局计划与控制

最后更新：2026-07-16

## 1. 文件职责与权威优先级

本文件是 LoopScope 第二阶段唯一的可变控制面，记录当前 hypothesis card、Gate、动态线程身份、已审计对象、科学判断和下一步授权。长期规则见 `../loopscope-tflt/AGENTS.md`，执行说明见 `../loopscope-tflt/docs/loopscope_phase2.md`。

权威优先级：用户最新明确指令 > live Git/Slurm/不可变工件 > `AGENTS.md` > 本 control > runbook > 报告、历史线程与 memory。

本文件只在 hypothesis card、Gate 决策、执行线程、审计 commit/run、科学配方或授权边界变化时更新，不记录逐命令流水账。动态线程 ID 只写在这里。

## 2. 当前身份与状态

```text
phase=LoopScope Phase 2
planning/audit thread=019f6634-0369-7832-a617-d9f7e225345f
planning/audit predecessor=019f5149-35f6-7383-b3ec-3376ba413d2b; archived; historical authority/provenance only
phase1 historical control=./loopscope_phase1_control.md; closed
phase1 audited code baseline=loopscope@2c3acb34c07362bf82c0f7a0af59e0d435ab44cd
phase1 GitHub state=origin/loopscope@2c3acb34c07362bf82c0f7a0af59e0d435ab44cd
phase1 GitHub rollback marker=annotated tag loopscope-phase1-complete-20260711; peeled commit 2c3acb34c07362bf82c0f7a0af59e0d435ab44cd
phase2 plan baseline=loopscope@a81fba18e8e164f38951ce7ff5db12aa38d39002
phase2 authorized governance baseline=loopscope@9e06dd3953ef64e99b6328907fdbd59f479a7ee6; origin/loopscope identical
current state=GATE_D_PASS; PHASE2_SUPPLEMENTAL_COMPLETE; WAITING_NEXT_HYPOTHESIS_CARD
current hypothesis card=H2_FIXED_HORIZON_STEPWISE_PROFILE_V1; user-authorized exploratory/descriptive mechanism card frozen in section 10.12; it supersedes the prior Gate-D stop only for the exact 12:15 fixed-horizon K1-K4 profile and does not reopen selector expansion
current Gate=D; PASS
current executor=none; Gate D executor 019f668f-e98d-7033-a28a-36d55d0cf5ac closed and deauthorized after PASS
phase2 progression authority=user latest instruction on 2026-07-14 authorizes continuing the ordered post-Gate-A sequence through completion of the Phase 2 implementation and core experiment without requesting routine Gate-transition approval; one Gate and one executor at a time still applies, and any material change to frozen science, protected code, total GPU budget, destructive behavior, or external authority still requires escalation
phase2 planning repo changes=committed and pushed at a81fba18e8e164f38951ce7ff5db12aa38d39002 plus scope clarification 9e06dd3953ef64e99b6328907fdbd59f479a7ee6
Gate A prior terminal event=BLOCK received from the authorized executor and independently audited on 2026-07-14
Gate A latest terminal event=SUCCESS_PENDING_PLANNING_AUDIT received from the same authorized executor at loopscope@ee1638c3bf14adc75b903a08aebae354fe9e3ae4 and independently audited on 2026-07-14
Gate A audited Git state=local loopscope@ee1638c3bf14adc75b903a08aebae354fe9e3ae4 clean; origin/loopscope remains 9e06dd3953ef64e99b6328907fdbd59f479a7ee6; exactly four unpushed commits
Gate A latest audit decision=PASS under the user-authorized experiment-materiality standard; card canonical SHA256 7d682e8e5ddc9e66b8e07770eaa3a8e83f95ba0161ae3a8d34e629b744fc82be is frozen/preregistered by this audit receipt
Gate A repair route=scale-separated evidence: 512 calibration owns residual/NCA; 14042 full owns final-choice/gold paired outcome only
Gate A repair policy anchor=planning-authored AGENTS.md SHA256 dca4a8e216717aef15b3d97c7a4a1a7ceb9c31bdf8c2b558de2c247d019368e7
Gate A accepted non-blocking technical debt=raw full-evidence JSON can exploit an adversarial manifest_sha256 fallback to bypass exact byte-hash checking without changing normal producer outputs, samples, science, statistics, or execution; record for later hardening only
Gate A recovery authorization=revoked by the user-authorized materiality recalibration; executor restored clean ee1638c3 with no new commit
Gate B authorization=push the four audited Phase 2 commits normally, fast-forward the dedicated HPC2 clone, then execute B0→B1→B2 under the frozen H1 V2 card; no full MMLU and no Gate C action
Gate B lm-eval freeze=the installed lm-eval framework, package source, task/YAML registry, HFLM, TaskManager, simple_evaluate, renderer, batching, metric and result semantics are read-only; all compatibility must be implemented in LoopScope-owned opt-in adapters/parsers/manifests/sidecars/tests, and inability to adapt without framework mutation is a mandatory BLOCK
Gate B terminal event=SUCCESS_PENDING_PLANNING_AUDIT received from authorized executor 019f5ebd-87d0-79a1-9633-3e67ad93d5b1 for run gate-b-h1v2-20260714T040409Z, jobs 9977332/9977363/9977369 and loopscope@02cd49619f3d42f2dc062be7d7a17855e6d8f756
Gate B audit decision=PASS on 2026-07-14; B0 live reuse, B1 K1/prefix proof, B2 sealed 512 matrix, Slurm/resource/protected-state and lm-eval framework freeze all independently verified; no material repair is required
Gate C authorization=produce exactly eight new full MMLU cells under the frozen card, reuse the four immutable Phase 1 baseline/K2 cells, then execute one write-once mechanism analysis and return one H1 outcome plus one independent NCA diagnosis; no Gate D action
Gate C terminal event=SUCCESS_PENDING_PLANNING_AUDIT received from authorized executor 019f5f51-bc44-7e62-93e6-d7043e5bea1b for run gate-c-h1v2-20260714T064922Z, jobs 9977778/9977874 and loopscope@96159840525b89240a9b0f70e63764e2110216e2
Gate C audit decision=PASS on 2026-07-14; exact eight-new plus four-reuse full closure, label-unseal chronology, one analysis/verifier, scientific statistics, resources, protected state and lm-eval freeze were independently verified; no material repair remains
Gate C scientific outcome=H1 PERTURBATION; k4 saturation_not_established; independent NCA diagnosis NCA_NONDISCRIMINATIVE; native-fidelity EVALUATED_NOT_SUPPORTED
Gate C accepted provenance deviation=the frozen card retains phase2_analysis.py SHA256 817b24ec... while the actual audited runtime is commit 9615984 with file SHA256 cf60e24a...; the diff only removes routing-only cell_id before the unchanged path/hash loader and does not modify any source row, statistic, threshold or classifier. Future work must bind the actual runtime commit or a new card and must not rewrite the post-unseal H1 card
Gate D terminal event=confirmed ROUTE_REPAIR_RESEND of the existing GATE_D_FINAL_AUDIT from authorized executor 019f668f-e98d-7033-a28a-36d55d0cf5ac; original archived-recipient failure was routing-only and created no second Gate outcome
Gate D audit decision=PASS on 2026-07-16; live Git/Slurm, 512 identity/5120-call scalar closure, key artifact hashes, full K1-K4 statistics, verifier equivalence, protected state and targeted card/tests were independently checked; no material defect remains
Gate D scientific outcome=12:15 fixed-horizon K2 is the observed accuracy peak; K3 and K4 progressively raise entropy and lower margins/accuracy, while late within-K entropy effects attenuate/reverse and eRank continues to rise. This supports a transient refinement followed by over-refinement/perturbation diagnosis, not a selector or causal claim
Gate D accepted non-blocking deviations=one extra deterministic no-write D4 recomputation after the single successful exclusive-create atlas write; stock verifier JSON integer/string key false negative closed by a source-aware canonical roundtrip verifier with all five output hashes equal; one cosmetic EOF blank line; the previously accepted Gate C H1 implementation-hash drift remains historical and unchanged
next admission condition=Phase 2 Gate D supplementary experiment is closed. No executor or post-Gate-D action is currently authorized. Any selector/window-ranking study, other window, K>4, new full MMLU, model/task extension or new metric must begin as a separately frozen hypothesis card and new Gate authorization
```

本线程是第二阶段长期计划/授权/审计线程，不亲自执行 Gate、不轮询、不创建 planning heartbeat。每一 Gate 只绑定一个新的独立执行线程；不得提前创建未来 Gate executor。

## 3. 2026-07-12 重新定靶

新的核心设计输入：

- 用户在分析线程 `019f5243-94fa-7513-950e-a76e20c71c61` 后段形成的结果解释和假设；
- [周报 7.12](https://app.notion.com/p/39a7d1c8d46f80708711c1fd879f9f55) 中的“两类 probe”“可迭代精炼区”和下一阶段实验；
- IdeaSpark 线程 `019f57b6-b0f0-7a40-8afb-5920d3cba444` 形成的 `Native-Continuation Alignment (NCA)` 想法卡、四源碰撞审计与可实现性检查；该结果是方法候选，不是实验成功证据；
- Phase 1 不可变 selection report。

2026-07-11 的旧草案尚未提交、未授权、未执行，因此现被本版安全取代：

- 三模型×两任务、16/17 窗 full correlation 不再是核心；
- 10k-token pooled eRank、angular/BI、tuned lens、生成任务不再是 admission 条件；
- `SELECTOR_GO` 不再是第二阶段首要终点；
- 原 A–G 长链缩为 A–C 核心门与一个可选 D 门。

2026-07-13 在 H1 尚未冻结、未执行前加入 NCA：它拟在 Gate A 被版本化并冻结为 H1-C 的 prospective secondary direction proxy；Gate A 审计 `PASS` 前不能称为已预注册。它不改变三窗口、两条 K/alpha 轨迹或八个 full 上限；完整 NCA selector 延后为 Gate C 后独立 H2 card。

保留的旧原则：逐题 paired、split 隔离、配置冻结、证据 write-once、Phase 1 结果只读复用、每 Gate 独立 executor、规划线程独立审计。

## 4. Phase 1 强制起点

```text
run=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-20260711-053022
selection_report_sha256=2b62df1f48051c0847ee9d2c439f935a7255d68a6c891f7dc6064b738def47e0
model=Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1
task=mmlu; 5-shot; 14042 paired samples
baseline accuracy=62.71186440677966%
11:14 delta=+0.12106537530266315 pp; q=0.9893; entropy_drop=-0.0183; delta_erank=+22.5450
12:15 delta=+0.413046574562026 pp; q=1.0024; entropy_drop=+0.3823; delta_erank=+33.3222
13:16 delta=-0.6694197649907421 pp; q=0.9834; entropy_drop=-0.2260; delta_erank=+32.9835
primary signal=r_times_one_minus_q; rho=-0.22424242424242424; eligible=false
```

关键事实：`12:15` 的提升仍是趋势性证据，paired bootstrap 95% CI 约 `[-0.036,+0.826] pp`，McNemar `p≈0.0748`。第二阶段要解释机制，不能把这一点观察当成已确认规律。

## 5. H1：可迭代精炼区（核心假设）

H1 V2 由三个可分离部分组成：

1. **H1-A 任务相关区域**：block 在 baseline 单次前向中产生适度选择熵下降且跨样本 effective rank 不坍缩，说明它可能参与任务决策而非简单压缩所有表示；给予额外 test-time compute 可能继续明确答案。
2. **H1-B 计算未快速饱和**：实际重复调用时 `r` 足够、`q≈1`，说明第一次调用后 residual 没有迅速消失，block 仍保留相近强度的更新能力；这不等同于“单次贡献不足”，而是“已有明显作用但仍可能继续精炼”。
3. **H1-C 更新方向有益**：NCA 比较每次真实 loop 更新与同一样本原生下游延续方向，作为无标签方向代理；最终是否有益仍必须由 final-output 与 gold-label 纠错、margin 和置信度轨迹独立裁决。

对窗口 `w=[a,b]`、样本 `i`、冻结的 final pre-answer prompt token `p`：

```text
c_(w,i) = B_N[i,p,:] - B_(b+1)[i,p,:]
NCA_(w,i,t) = cos(delta_(w,i,t)[p,:], c_(w,i))
```

body-call index 固定为 zero-based `t=0..K-1`：`t=0` 是 initial call。provenance A 的 baseline NCA 使用普通单次前向 `B_(b+1)-B_a`；provenance B 只把真实 loop 的 `t=1..K-1` 作为 repeated-step NCA。`K=1` 不产生 repeated-step NCA，只作 baseline equivalence。

NCA 正值只表示更新和原模型自身下游路径一致。它可能维持正确 baseline，也可能强化错误 baseline；若 `12:15` 与 harmful `13:16` 的 NCA 同样为正且稳定，则 NCA 不具方向区分力。H1 不用 NCA 排名、筛窗、设阈值或决定部署。

本卡不为 `r`、`q`、entropy、eRank 或 NCA 从 Phase 1 数据 post-hoc 拟合数值阈值。它们作为连续解释变量；只有机制和 NCA 区分力在新 K 轨迹中得到支持，才可另立 `H2_NCA_CERTIFICATE` selector card。

## 6. H1 实验卡 v2（草案）

### 6.1 最小科学矩阵

```text
card=H1_ITERATIVE_REFINEMENT_ZONE_V2
model=Qwen/Qwen3-1.7B-Base
revision=ea980cb0a6c2ae4b936e82123acc929f1cec04c1
task=mmlu
num_fewshot=5
dtype=float16
windows=11:14,12:15,13:16  # inclusive width=4
iteration_mode=block
strategy=damped_euler
beta=0.0
cache_strategy=last
decode_mode=bypass
primary_continuation=(k,alpha)=(2,1.0),(3,1.5),(4,2.0)  # fixed step h=alpha/k=0.5
fixed_horizon_control=(k,alpha)=(1,1.0),(2,1.0),(3,1.0),(4,1.0)
calibration_pool=Phase 1 frozen validation 512; no gold labels for probe metrics
nca_position=final_pre_answer_prompt_token
nca_continuation=B_N-B_(b+1)
nca_bootstrap_replicates=10000
nca_bootstrap_seed=0
nca_role=prospective_secondary_direction_proxy; never selector in H1
full_samples=Phase 1 ordered 14042 MMLU samples
paired_bootstrap_replicates=2000
paired_bootstrap_seed=20260710
```

当前实现的 damped Euler 单步大小是 `h=alpha/k`。因此：

- **主 continuation 轨迹**固定 Phase 1 的 `h=0.5`，从 `(2,1)` 延长到 `(3,1.5)`、`(4,2)`，直接检验 residual 仍存在时继续迭代是否带来净纠错；
- **fixed-horizon control**固定 `alpha=1`，增加 K 只会把相同总更新时长离散得更细，用于区分“更多迭代时长”与“更细数值子步”；full 默认只在 `12:15` 补 `(3,1)/(4,1)`；
- `(k=1,alpha=1)` 预期等价于普通单次 block 通过；Gate B 必须在小样本上用 final outputs 和调用次数证明。若不等价则 `BLOCK`。
- 主 continuation 的三组 step 都是 `0.5`；同一样本、同一窗口下，K=3/4 的前两个 residual/state tensors 必须在内存中与 K=2 allclose，工件只保存 max-abs/scalar proof，作为“延长同一轨迹”的 prefix-consistency admission。

Phase 1 `(k=2,alpha=1)` 是 hypothesis-generating discovery point；新 `(3,1.5)/(4,2)` 和 fixed-horizon controls 才是本卡的 prospective evidence。Gate C 只能形成当前 model-task cell 内的机制判断，不能声称跨任务/模型确认。

所有 manifest/output identity 必须同时包含 `protocol、window、k、alpha`，避免同 K 不同 alpha 冲突。NCA 默认仅在冻结 512 校准池采集：向量在设备内以 float32 临时计算，只持久化 per-sample NCA、`||c||`、`||delta||`、validity mask 和汇总值。14,042 full 只保存 canonical evaluator final-choice/gold paired evidence，不允许 residual、`q_t`、相邻方向或 NCA 字段；不得新增 14,042-sample baseline trace、保存完整 hidden tensor 或重跑 Phase 1。

### 6.2 四层测量口径

| 层级 | 来源 | 指标 | 用途 |
|---|---|---|---|
| A | baseline layer probe | boundary choice entropy/drop、KL、answer-position eRank、baseline window update 与 native continuation 的 NCA | 描述单次前向中的 block 性质与原生方向；无标签解释变量 |
| B | actual loop trajectory | 每轮 `delta_t` norm、`r_t`、`q_t=norm(delta_{t+1})/norm(delta_t)`、相邻更新 cosine、每轮 NCA | 判断 residual 是否消失、稳定、旋转或偏离原生方向 |
| C | 每个 K 完整前向的 final output | choice probability、final entropy、top1-top2 margin、JS-to-K1、top1 retention、answer | 主要决策轨迹；不依赖 raw intermediate lens |
| D | gold-label adjudication | 四类答案转移、correct margin、错误过度自信，以及冻结后 NCA 在 baseline-correct/wrong 组的条件表现 | 独立裁决额外计算是否真正有益 |

任何 per-iteration block-boundary lens 输出必须命名为 `intermediate_lens_*`，仅作辅助并单独通过有效性门；不得与 final-output entropy/margin 混写。

NCA 的位置、inclusive boundary、sample identity、向量 shape/dtype/device 和 finite/near-zero validity 都必须进入 schema。任一不一致不得被零填充或聚合掩盖。NCA 使用 10,000 次、seed 0 的 sample-index bootstrap；不得与 full paired outcome 的 2,000 次、seed 20260710 混成同一统计 family。

### 6.3 逐样本 paired 分析

在 card 与无标签指标冻结后解封 gold label，逐窗口、逐 K 报告：

- accuracy 与 paired bootstrap/McNemar；
- `wrong→right`、`right→wrong`、`right→right`、`wrong→wrong`；
- 相对 baseline 的累计净纠错率，以及主 continuation 相邻点 `(2,1)→(3,1.5)`、`(3,1.5)→(4,2)` 的增量净纠错率；
- final entropy、top1 margin 和正确答案 margin 的变化；
- 上述变化在四类答案转移中的条件分布；
- “错误但更自信”比例；
- 主 continuation 轨迹从 `(K=2,alpha=1)` 到 `(3,1.5)/(4,2)` 的改善、饱和或反转位置；
- `12:15` fixed-horizon `(K=2,3,4,alpha=1)` 与 continuation 轨迹的差异。

六个三窗口 prospective incremental contrasts 组成唯一 primary confirmatory family：每个窗口各有 `K2→K3` 和 `K3→K4` 两项，使用 two-sided exact McNemar 与 Holm step-down，family-wise `alpha=0.05`。`12:15` 两个 fixed-horizon contrasts 构成独立、内部 Holm 校正的 secondary control family；所有 cumulative-vs-baseline / `K4-vs-K2` 结果只作 secondary sign guard/context，均报告 point estimate 与 paired CI，不能单独触发 `REFINEMENT_SUPPORTED`。

冻结 512 校准池的 NCA 先在无标签状态下计算。card、公式、位置、validity 和汇总规则冻结后，才可解封该池 label，作 baseline-correct/wrong 与四类转移的 development-only 条件分析；不得据此调 NCA 阈值、Gate C 窗口或 K。若 full per-sample NCA 需要新增完整 baseline pass，则本卡不要求它，必须报告缺口并保持八个 full 上限。

`13:16` 是核心 falsifier：它同样 `q≈1`、eRank 增长，却在 K=2 明显掉点。`4:7`（大 entropy drop 但 harmful）、`15:18`（高 eRank 增长但 harmful）、`22:25`（原主分数最高但负收益）只使用现有 K=2 工件作为零新增计算的反例，不默认进入 K sweep。

### 6.4 科学判读标签

Gate A 必须把以下规则、优先级和实现 hash 写入 versioned config；它们在 Gate A `PASS` 后才成为预注册判据。这里的 K2/K3/K4 只指 primary fixed-step `(2,1)/(3,1.5)/(4,2)`。令 `d23(w)=acc(w,K3)-acc(w,K2)`、`d34(w)=acc(w,K4)-acc(w,K3)`，`d24(w)=acc(w,K4)-acc(w,K2)`；`p_Holm` 是六项 primary family 的 Holm-adjusted two-sided exact McNemar p-value，`U24` 是 `12:15` 的 d24 使用 full paired bootstrap（2,000、seed 20260710）所得 95% percentile interval 上界。工件/identity/统计字段不完整时直接 Gate `BLOCK`，不得硬分配科学标签。完整数据按下列优先级只取第一个匹配项：

1. `PERTURBATION`：`12:15` 的 `d23` 或 `d34` 为负、对应 `p_Holm<0.05`，且该 contrast 的 `right→wrong > wrong→right`。
2. `REFINEMENT_SUPPORTED`：未命中 `PERTURBATION`；`d23(12:15)>0` 且 `p_Holm<0.05`；`d24(12:15)>=0`。
3. `TRANSIENT_ONLY`：未命中前两项，且 Phase 1 已知 `K2-vs-baseline>0`、但 `U24<0`；仅 d24 点估计为负而 interval 跨 0 时不得使用本标签。
4. `SUGGESTIVE`：未命中前三项；`d23(12:15)>0`、`d24(12:15)>=0`，但 `12:15` 的 `d23` 未达到 `p_Holm<0.05`。
5. `INCONCLUSIVE`：其余完整但互相矛盾或低于当前分辨率的轨迹。

`q`、JS、entropy、margin 与错误过度自信继续作为机制解释和反证定位，不额外创造未冻结的主判据。fixed-horizon family 与 cumulative contrasts 也不能把 `SUGGESTIVE/INCONCLUSIVE` 升格为 `REFINEMENT_SUPPORTED`。`11:14/13:16` 继续作预注册邻窗与 falsifier，但“12:15 显著、邻窗不显著”不得解释为窗口间显著差异；H1 outcome 只声明 `12:15` 在当前 cell 的 continuation 结果，不声明它是唯一或显著优于邻窗的窗口。

K4 是否已经饱和独立报告：若 `d34>0` 且对应 `p_Holm<0.05`，结论是“截至 K4 仍在改善、饱和未观察到”，不判 H1 失败，也不自动授权 K>4；未显著也不能在没有预设 equivalence margin 时宣称已饱和。

NCA 另给一个不依赖 H1 outcome 的独立诊断标签。对 fixed-step primary cells `s∈{(2,1),(3,1.5),(4,2)}`，每个 sample-cell 只有在声明的全部 repeated steps 都 valid 时才 valid；先对这些 steps 的 NCA 取中位数，再跨样本取中位数。用同一组配对 sample-index bootstrap（10,000、seed 0）计算 `12:15` 中位数及 paired-valid intersection 上 `NCA(12:15)-NCA(13:16)` 的 95% percentile interval，记 lower bound 为 `L`。native-fidelity 条件分析还用同一预算的 stratified bootstrap 计算 `right→right` 组的 NCA interval，以及 `median_NCA(wrong→right)-median_NCA(wrong→wrong)` 的 interval；只有 `right→right、wrong→right、wrong→wrong` 三个子组都至少有 10 个 valid samples 时该 cell 才可用于 native-fidelity 判定。完整数据按以下优先级只取第一个匹配项：

1. `NCA_INCONCLUSIVE`：任一三窗口 × 三 primary cell 的 valid fraction `<0.95`，任一 primary paired-valid intersection `<0.95`，或任何 primary overall/difference interval 缺失/非有限。
2. `NCA_NATIVE_FIDELITY_ONLY`：三个 `12:15` primary cell 均有 `L_NCA>0`；且至少两个合格 cell 同时满足 `L_NCA(right→right)>0` 与 `U_[NCA(wrong→right)-NCA(wrong→wrong)]<=0`。
3. `NCA_DIRECTION_SUPPORTED`：未命中第 2 项；三个 `12:15` primary cell 均有 `L_NCA>0`；三个 paired `12:15-13:16` interval 中至少两个 `L_diff>0`。
4. `NCA_NONDISCRIMINATIVE`：其余有效但不能同时证明正向、相对 harmful 窗口区分力与净纠错一致性的情况。

若 native-fidelity 合格 cell 少于两个，必须另报 `native_fidelity_check=NOT_EVALUABLE`，但这本身不把 primary NCA 诊断改成 `NCA_INCONCLUSIVE`；继续按第 3–4 项判定。这些 NCA intervals 均为 pointwise、未校正的 secondary diagnostic intervals，不构成独立 FWER-controlled confirmatory rejection，也不能代替 H1 的 gold-label primary family。

科学标签不替代 Gate 审计结论；完整的负结果也可以 Gate `PASS`。

## 7. 轻量 Gate 计划

| Gate | 目标 | 最大范围 | 退出条件 |
|---|---|---|---|
| A | residual/NCA/final-output 采集、逐样本分析与 H1 V2 冻结 | 本地最小代码/config/schema/tests；无 SSH/GPU | commit 与配置/hash 审计 `PASS` |
| B | HPC2 CPU + smoke/limit + bounded NCA calibration | 先验证三窗口两条轨迹与 NCA schema；工程闭环后才可执行冻结 512、三核心窗口、五个唯一设置的无标签 NCA/final-choice probe；最多 1 个 no-loop boundary + 15 个 loop cells，单 GPU ≤18 GPU-hours；无 full | `(k=1,alpha=1)` 等价、prefix consistent、NCA position/boundary/validity、unlabeled final-choice、trace/schema/revision 与 512 identity 完整；不得据 NCA 改 Gate C；资源预测超界则 B2 前 `BLOCK` |
| C | focused full mechanism test | 复用 baseline/(2,1)；三窗口新增 `(3,1.5)/(4,2)` 六个配置，`12:15` 新增 `(3,1)/(4,1)` 两个控制；默认最多 8 个 full；一次 paired analysis | 工件审计 `PASS/BLOCK` + 一个 H1 outcome + 一个 NCA diagnosis |
| D | `12:15` fixed-horizon K1–K4 逐步机制图谱 | 新增一次冻结 validation-512 的 latent/choice-lens scalar profile；复用 Gate B/C 与 Phase 1 的既有 512/14,042 evidence；一个 A40、GPU-hours 不设硬上限、0 个新 full | K1–K4 raw/applied/cumulative 轨迹、既有 full accuracy/flip/confidence 轨迹与一次 write-once atlas 完整；终态由规划线程审计 |

若 Phase 1 `results.json` 缺少重建 K=2 final choice distribution 所需字段，Gate A/B 必须报告缺口；不得自动把 Gate C 扩成 K=2 全量重跑。任何补跑需 planning 新授权、write-once 路径并保留旧结果。

## 8. 敏捷后续规则

Gate C 后可以快速新增 hypothesis card，例如：

- H2：`H2_NCA_CERTIFICATE`——仅当 H1/NCA 支持时，验证全窗口 baseline ranking、top-3、唯一胜者/no-loop verifier 与 window-identity permutation；
- H3：`q≈1` 的有效区间是否能在一个新任务复现；
- H4：baseline layer entropy/eRank 与 final K 轨迹的关系；
- H5：可选 10k-token pooled eRank 或 angular/BI 对照。

每张新卡先尝试只读复用 H1/Phase 1 工件。只有现有证据无法回答时才新增采集；先 smoke，后 full。不得把探索结果追认成 H1 的预注册证据。

## 9. Optional backlog（不构成当前授权）

- Qwen3 0.6B/1.7B/4B × MMLU/MMLU-Pro；
- 16 点 normalized window grid 与固定 0.525 对照；
- depth-controlled signal competition、random-in-band、angular/BI；
- 10k-token pooled corrected eRank；
- Llama/MoE held-out、生成任务、自动 selector。

这些项目只有在 H1/H2 给出值得扩展的机制或监测信号后，才由用户逐项选择；不得打包成默认 Phase 2 工作量。

## 10. 当前授权边界

用户已授权推送第一阶段回滚边界、提交第二阶段治理基线，并由独立执行任务开始 Gate A。当前唯一执行授权如下：

```text
gate=A
executor=019f5c1d-7355-7130-81e2-125b4f1e21c0
repository=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope
base_commit=9e06dd3953ef64e99b6328907fdbd59f479a7ee6
maximum_new_run_count=0
GPU_hours=0
repair_budget=at most 2 low-risk engineering loops
terminal_recipient=019f5149-35f6-7383-b3ec-3376ba413d2b
```

允许：

- 本地最小增量实现、配置、schema、分析代码、CLI/opt-in eval trace adapter、测试与必要 runbook 同步；
- 新建 `src/tflt/loopscope/phase2_schema.py`、`phase2_trajectory.py`、`phase2_analysis.py`，必要时新建 `phase2_reuse.py`；仅为导出新接口可最小修改 `src/tflt/loopscope/__init__.py`；
- 最小修改 `src/tflt/cli.py` 与 `src/tflt/eval_runner.py`；新建 `configs/loopscope/qwen17_mmlu_phase2_h1_v2.json`、`scripts/loopscope/prepare_qwen17_phase2.py`、`tests/test_loopscope_phase2_*.py`，必要时最小修改 `tests/test_cli.py`；仅同步 `docs/loopscope_phase2.md`；
- `src/tflt/eval_runner.py` 只可新增显式 opt-in 的 Phase 2 trace adapter；默认关闭路径的命令、lm-eval task 语义、结果与 provenance 必须保持不变并有回归测试；
- 在本地创建一个或多个单一目的 Gate A commit；不得 amend 已推送的 `a81fba1`，不得 push，等待规划线程审计。

禁止：

- 修改 `src/tflt/wrapper.py`、`src/tflt/strategies.py`、`src/tflt/cache.py`、`src/tflt/config.py`、`AGENTS.md`、本 control、Phase 1 run/config/manifest/freeze/results/analysis；
- SSH、HPC2 写入、Slurm、GPU、模型/数据/package 下载、venv/cache 修改；
- 任何 Phase 1 baseline/K=2 补跑、实验 run、512 label 解封、full MMLU；
- selector/no-loop verifier、全窗口/多模型/多任务框架、Gate B/C/D 或 optional backlog；
- 把 Gate A 产物称为已预注册，或执行线程自行宣布 Gate `PASS`。

Gate A reuse matrix 只能把证据分为 `local_schema_confirmed / historical_only / requires_gate_b_live_check`；本门不得把未通过 SSH 读取的 HPC2 工件宣称为 live verified。任何会改变模型、任务、split、样本、窗口、K/alpha、seed、primary family 或 NCA/H1 判据的选择必须 `BLOCK` 回传，不得自行扩权。

Gate A 必须把以下数值与输出语义写入 versioned config/schema，并在审计 `PASS` 前保持 freeze candidate：

```text
task=mmlu
primary_metric=acc,none
choice_labels=A,B,C,D in frozen renderer/task order
choice_score_source=the four raw per-choice log-likelihood scores used by lm-eval acc,none
top1_tie_break=first maximum in A,B,C,D order
choice_probability=float32 softmax over raw choice scores
entropy_log_base=e
js_log_base=e
top_margin=raw top1 score - raw top2 score
correct_margin=raw gold score - max raw non-gold score
nca_dtype=float32
nca_min_norm=1e-8; norm < threshold is invalid
k1_equivalence_scores_atol=1e-4
k1_equivalence_scores_rtol=1e-5
prefix_state_residual_atol=1e-5
prefix_state_residual_rtol=1e-3
nca_ci=95% percentile [2.5,97.5], 10000 resamples, seed 0
full_outcome_ci=95% percentile [2.5,97.5], 2000 paired resamples, seed 20260710
```

`top1`/correctness 必须与 evaluator 的 `acc,none` 逐样本一致；不一致 fail-fast。六项 primary contrast ID 与固定顺序为 `fs_11_14_k2_k3、fs_11_14_k3_k4、fs_12_15_k2_k3、fs_12_15_k3_k4、fs_13_16_k2_k3、fs_13_16_k3_k4`，同 p-value 时按该顺序作 deterministic Holm tie-break。Exact McNemar 使用两侧 exact binomial，零 discordant 时 `p=1`。NCA 每个 sample-cell 必须全部声明的 repeated steps valid；overall 和 paired-valid fraction 都以冻结 512 为分母，paired difference 的每个 bootstrap replicate 复用同一 sample index，native-fidelity 在各冻结 subgroup 内 stratified resample。

B2 的一次 no-loop boundary 与 15 个 loop logical cells 必须由同一受控 process/session 的内存内复用设计支持；Gate A 只实现并测试这一契约，不得落盘 native-continuation/full hidden vectors，也不得运行 B2。

所有未来新工件必须进入 `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/`；Phase 1 canonical run 永久只读留在 legacy path。兼容时优先复用既有 venv 与 shared caches，不原地升级。

### 10.1 Gate A 终态审计与当前 blocker（2026-07-14）

执行线程 `019f5c1d-7355-7130-81e2-125b4f1e21c0` 以 control
SHA256 `8c4214f72b23e9a808c8fca049e7fde273c70b59a2ddb52db7eb434afc4307d8`
发送结构化 `BLOCK`。规划/审计线程重新读取 live Git、代码、配置和测试后确认：

- local/origin 仍为 `loopscope@9e06dd3953ef64e99b6328907fdbd59f479a7ee6`；草案只存在于 dirty working tree，没有 commit/push，也没有 SSH、HPC2、Slurm、GPU 或 Phase 1 工件变更；
- 本地 targeted `26/26`、full `169/169`、compile、CLI help、`git diff --check` 与实现 hash 均通过，但这些绿测没有覆盖真实 producer/identity/provenance 闭包；
- 受控 direct probe 通过逐样本 `begin_forward/end_forward` 可以绑定 calibration identity；lm-eval full 路径会批处理/重排四个 choice request，而现有 wrapper event 不含 sample identity，当前 collector 又只接受 batch size `1`，因此不能把 full residual trajectory 诚实绑定到稳定样本；
- `(K=1,alpha=1)` equivalence 与跨 K prefix consistency 目前只有 helper/test，没有 Gate B1 可执行 producer；calibration renderer identity 与 lm-eval full identity 也没有经审计的确定性 bridge；
- 当前 analysis CLI直接信任预聚合 JSON，未从逐样本 sidecar 构建 primary/NCA families，也没有绑定 card/input/source hashes、ordered identity、revision 或唯一执行 provenance；
- schema/analysis 不是 closed-world：独立反例可让改为 `calibration_gold_sealed=false` 的重哈希 card、畸形 choice 字段、重复 identity 和越界 NCA 值通过；large-discordance exact McNemar 可溢出。故 freeze candidate 不满足预注册或 Gate B admission。

当前没有 repair 授权。需要先在以下会改变 Gate B/C 契约的路线中作出明确选择：

1. 为 lm-eval/HFLM request path 增加可验证 fingerprint/router，使 batch row、四个 choice request、sample identity 与 wrapper scalar trace 闭包；
2. 将 full residual 采集改成独立受控 direct producer，并新增 renderer/evaluator 等价与 namespace bridge；
3. 缩减 H1 的尺度契约：512 calibration 负责 residual/NCA，14,042 full 只负责稳定 final-choice 与 gold paired outcome，不要求 full residual trace。

无论选择哪条路线，都必须先修复 closed-world schema、逐样本 analysis producer、stable McNemar、sealed-label 与 write-once provenance，再由同一 Gate A executor完成 bounded repair。不得创建 Gate B executor。

### 10.2 用户选择与 superseding bounded repair 授权（2026-07-14）

用户明确选择第 3 条“尺度分离”路线。它取代 10.1 中的待选状态，但保留原
`BLOCK` 及其证据作为历史。当前 H1 V2 的长期边界已写入 planning-authored
`AGENTS.md`（SHA256
`dca4a8e216717aef15b3d97c7a4a1a7ceb9c31bdf8c2b558de2c247d019368e7`）：

- 冻结 512 calibration 是 residual、`q_t`、相邻方向、NCA 与无标签 final-choice trajectory 的唯一 mandatory scale；
- 14,042 full 只产生 canonical evaluator final-choice、entropy/margin/JS、答案和 gold paired outcome，不采集 residual/NCA；
- 512 calibration identity 与 14,042 full identity 是两个显式 namespace，只允许各自在其 canonical manifest 内闭包，不伪造跨 namespace bridge；
- 当前 H1 不研究 full residual–answer-flip 个体相关性；不授权 HFLM request router、full direct residual producer 或 14,042-sample residual/NCA 补跑。

同一 Gate A executor `019f5c1d-7355-7130-81e2-125b4f1e21c0` 获得一次
superseding local-only repair 授权：

```text
gate=A repair
repository=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope
starting HEAD=9e06dd3953ef64e99b6328907fdbd59f479a7ee6
origin/loopscope=9e06dd3953ef64e99b6328907fdbd59f479a7ee6
expected dirty state=the preserved Gate A draft plus the planning-authored AGENTS.md diff
maximum_new_run_count=0
GPU_hours=0
network/SSH/HPC2/Slurm=forbidden
repair_budget=at most 2 bounded local engineering loops
terminal_recipient=019f5149-35f6-7383-b3ec-3376ba413d2b
```

允许修改的实现范围仍限于现有 Gate A draft：

- `docs/loopscope_phase2.md`、`src/tflt/cli.py`、`src/tflt/eval_runner.py`、`src/tflt/loopscope/__init__.py`；
- `configs/loopscope/qwen17_mmlu_phase2_h1_v2.json`、`scripts/loopscope/prepare_qwen17_phase2.py`；
- `src/tflt/loopscope/phase2_schema.py`、`phase2_trajectory.py`、`phase2_analysis.py`、`phase2_reuse.py`；
- `tests/test_loopscope_phase2_*.py`；
- `AGENTS.md` 只能核对上述 planning-authored SHA 并原样作为独立 policy commit 提交，不得再改写。

repair 必须完成：

1. card、trajectory/final-output sidecar、aggregate envelope、analysis input/report 使用 exact-key closed-world schema；冻结所有 science、scale、seal、statistics、write-once 和 decision-rule 字段，拒绝未知字段、错误类型、非有限值、降序区间、越界 cosine 与 `K<1`；
2. 每个 cell 绑定 `identity_namespace/card hash/protocol/window/k/alpha/revision`；canonical manifest 校验唯一、完整、样本数、natural order 与 ordered identity hash，禁止把列表与自身比较来冒充 closure；
3. 512 direct calibration producer 明确使用自己的 direct-probe score source，不得把 next-token logits 虚称为 lm-eval raw loglikelihood；B1 real producer path 必须实际调用并输出 K1 equivalence 与跨 K state/residual prefix proof，不能只保留 helper；
4. full adapter 改为名实相符的 opt-in final-output adapter；从 evaluator logged samples 作顺序无关的 exact identity join 后恢复 canonical order。不得因 `batch_size=auto` 本身失败，也不得连接 wrapper residual collector；缺失/重复 identity 或 raw four-choice score 时 fail-fast，留给 Gate B live check；
5. analysis 必须从经 hash/identity/revision/seal 校验的 canonical per-sample sidecars重算 transitions、delta、bootstrap、exact McNemar、primary/fixed-horizon Holm families、NCA summaries 和最终标签；不得接受能直接决定结论的裸预聚合 p-value/delta/NCA cells；
6. exact McNemar 必须在 14,042 样本尺度数值稳定，并从 transition counts 重算；bootstrap replicates/seed、family IDs/order 与 NCA denominator 必须等于 card；
7. sealed 512 工件禁止 gold/correctness 字段；解封条件分析必须绑定已冻结 card、sealed artifact hashes 与显式 unseal authorization/receipt schema；当前 Gate A 只实现和测试，不读取 label；
8. Phase 2 analysis 输出使用原子 exclusive-create/write-once 语义，并绑定 card/input/source hashes、canonical identity、revision、attempt/receipt 和完整统计证据；不得修改 Phase 1 公共 schema helper；
9. 把审计反例全部变成 regression tests：unsealed/unknown-field card、malformed choice、duplicate/reordered identity、out-of-range NCA、rogue/missing cell、reversed interval、large-discordance McNemar、伪造 aggregate p-value、错误 seed/replicate、full residual 字段和跨 namespace join 均必须被拒绝。

Git 只允许两个单一目的本地 commit，均不得 push：

1. 先只提交 planning-authored `AGENTS.md`，不得夹带现有 draft；
2. repair 全部测试通过后提交其余 Gate A implementation/config/tests/runbook。

禁止修改 `wrapper.py`、`strategies.py`、`cache.py`、`config.py`、本 control、Phase 1 工件或其他非列出文件；禁止 stash/reset/覆盖草案、amend、push、下载、远程验证或进入 Gate B。若真实修复需要扩大文件边界、改变冻结窗口/K/alpha/seed/family、读取 label、修改 scorer/renderer 或选择另一架构，必须发送 `BLOCK`。

本 repair authorization 的终态必须由执行线程亲自、且确认跨线程工具成功后，向 planning thread 发送一次新的 `GATE_A_FINAL_AUDIT` 或 `BLOCK`；执行线程不得自宣 `PASS`。

### 10.3 Gate A repair 终态独立审计与第二次 bounded repair（2026-07-14）

同一执行线程以 control SHA256
`2b7c03d99b823aa8572db31cf3f9d265394070c0b2287d558b5dbac0ba40dcfc`
回传 `SUCCESS_PENDING_PLANNING_AUDIT`。规划线程独立复核 live Git、card、实现和测试后确认以下正向事实：

- local `loopscope@3c44d2f0343f8d46ff10c01c81ed5a6868e3816e` clean，恰好比 `origin/loopscope@9e06dd3953ef64e99b6328907fdbd59f479a7ee6` 领先两个未推送 commit；Phase 1 tag 与禁止文件未变；
- planning-authored `AGENTS.md` 独立 commit、card canonical hash、八个 implementation hash 和 512/full 尺度分离均闭合；
- 独立重跑 Phase 2 targeted `40/40` 与 full `183/183` tests、compile、CLI help、hash closure 和 diff checks 全部通过；
- B1 producer 确实执行 no-loop、15 logical cells、三个 K1 cells，并在内存比较 K1 choice equivalence 与 K2/K3/K4 prefix；full adapter 确实作 unordered exact identity join、允许 `batch_size=auto`，且未接 residual collector/router。

但绿测之外的 adversarial audit 复现了以下 blocker，因此 Gate A 决定仍为 `BLOCK`，card 不预注册，Gate B 保持锁定：

1. **统计 family 漂移**：control 冻结的是 `12:15` 的两个 fixed-horizon incremental contrasts `K2→K3、K3→K4`；当前 card/analysis 却把历史 `K1→K2` 加入同一 Holm family。`K1→K2` 只能保留为 equivalence/cumulative context。
2. **判读规则未完整版本化**：card 只保存标签顺序和少数阈值；H1 的目标 contrast/符号/transition/`U24`/saturation 条件与 NCA 的 3-cell/2-cell/interval 条件仍只硬编码在 Python，违反“完整 decision rules 写入 versioned config/hash”。
3. **full 实际执行身份未闭合**：adapter 接受 request 自述的 cell 和五个任意 SHA，却没有把它们与实际 `model/revision/task/fewshot/dtype/loop/window/k/alpha/...`、`command_args.json`、`env.json`、`model_revision.json`、attempt/receipt 对照。独立构造不存在任何 provenance 文件的虚假 request 仍被接受。
4. **sealed root 可夹带 label**：`sealed_calibration_512_scalar_aggregate.probe_pool` 只检查为 mapping；独立加入 `gold_index/correctness`、重算 self-hash 后仍通过。
5. **B1→B2 admission 未闭合**：B1 proof 未绑定 source pool/producer，B2 不核对其四个 identity 是当前冻结 512 manifest 的确定性 smoke subset；任意同 card 的四样本 proof 可以解锁 B2。
6. **analysis audit verifier 可接受伪造结论**：正常 producer 会从逐样本 source 重算，但 standalone report validator 只比较 report 内部 aggregate。重哈希后可接受伪造 cumulative CI/H1、越界 JS、负 subgroup count、boolean p-value 及 valid-count/interval 矛盾，而 source hashes 不变。
7. **canonical source provenance 未进入最终 closure**：identity manifest 只约束 count/unique/order/self-hash；analysis input 不引用 B2 sealed aggregate，因而不能从最终报告追溯到 Phase 1 frozen pool/dataset/renderer source root。

这些问题的正确修复由现有 control/AGENTS 唯一确定，不需要改变模型、任务、split、窗口、K/alpha、seed、主 family 或尺度分离路线。同一 Gate A executor
`019f5c1d-7355-7130-81e2-125b4f1e21c0` 获得第二次 local-only bounded repair：

```text
gate=A second repair
repository=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope
starting HEAD=3c44d2f0343f8d46ff10c01c81ed5a6868e3816e
origin/loopscope=9e06dd3953ef64e99b6328907fdbd59f479a7ee6
expected worktree=clean; ahead of origin by exactly 2 commits
maximum_new_run_count=0
GPU_hours=0
network/SSH/HPC2/Slurm=forbidden
repair_budget=at most 3 bounded local engineering loops
Git=preserve both existing commits; at most one new single-purpose repair commit; no amend/rebase/reset/push
terminal_recipient=019f5149-35f6-7383-b3ec-3376ba413d2b
```

允许修改范围与 10.2 相同，但 `AGENTS.md`、本 control、Phase 1 工件及
`wrapper.py/strategies.py/cache.py/config.py` 继续只读。repair 必须同时完成：

1. 将 fixed-horizon Holm family 恢复为且仅为 `fh_12_15_k2_k3、fh_12_15_k3_k4`；`K1→K2` 只留在 equivalence/cumulative context。六项 primary family 不变。
2. 在 card 的 exact-key `decision_rules` 中结构化冻结 control 6.4 的全部 H1/NCA 条件、目标 contrasts、符号与 interval endpoint 比较、valid/subgroup 数量、优先级、saturation 和 multiplicity role；classifier 必须读取这些字段或逐字段验证后执行，任一字段漂移 fail-fast。
3. full adapter 在运行前校验 request cell 与真实 argv；在运行后从实际 write-once `command_args/env/model_revision/attempt/receipt` 构造并核验 producer provenance，确认 loaded model/tokenizer revision closure 与 card 一致。baseline 的 inactive loop defaults 不冒充 active science；loop cells 必须逐字段匹配。默认未启用 adapter 的既有 eval 路径保持不变。
4. Phase 2 adapter 模式的 command/env/revision/results/sidecar 采用 new-root/exclusive-create 语义；request 中的 provenance ref 必须是 path+hash 并实际读取验证，不能只接受格式正确的虚构 SHA。
5. 对 B1/B2 `probe_pool` 使用 exact schema并递归执行 sealed-label guard；B1 proof 绑定 pool source/selected-subset hash、ordered four identities 和真实 producer。B2 必须把 proof 与同一 frozen-512 source manifest、确定性前四 identity 及 card/revision 对照后才 admission。
6. calibration identity/root 绑定 Phase 1 frozen pool manifest、renderer subset和 dataset provenance；full identity 绑定 `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`、5-shot natural-order producer及 Phase 1 run/input provenance。analysis input 必须引用并验证 B2 sealed aggregate，使其 identity、baseline 和 15 cell refs与加载的 16 个 sealed artifacts逐项一致。
7. 增加 source-aware analysis verifier：从 analysis input 重新加载全部逐样本 source、确定性重算 canonical report，并与待审报告逐字段/manifest hash 相等；structural validator不得被当作 source-derived verifier。同步收紧 bool-vs-number、JS数学范围、subgroup count/fraction、NCA valid-count/interval availability、真实 UTC timestamp 等内部约束。
8. B1/B2 aggregate 增加可审计的 per-cell wall-clock、peak GPU memory 与 restore proof schema；Gate A 只用 fake/pure-Python 测试，真实值留给 Gate B。
9. 新 regression tests 必须至少覆盖本节 1–7 的每个反例，并证明 default eval path、Phase 1 tests、scale separation 和 protected files仍不变；targeted/full tests、compile、CLI help、card/hash closure 全绿后才可提交。

若上述 source binding 需要一个尚未 live 核验的 digest，Gate A 只能冻结字段/schema并标记
`requires_gate_b_live_check`，不得编造 digest；若必须更改任何科学集合、protected file 或远程读取，立即回传 `BLOCK`。repair 后执行线程只发送一次新的终态包，仍不得自宣 PASS或进入 Gate B。

### 10.4 Gate A second-repair 终态独立审计 BLOCK（2026-07-14）

同一执行线程以 control SHA256
`85486c59815b65fffa4e8248eb427d613fe6d976a3589b9af2e5524018281e38`
回传 `SUCCESS_PENDING_PLANNING_AUDIT`。规划线程重新读取 live Git、card、实现与测试，并由三条独立只读审计线攻击 Git/card、producer/source provenance 与 source-aware analysis。确认以下正向事实：

- local `loopscope@46e5365efd70561159c36b5447aef49f6111450a` clean，恰好比 `origin/loopscope@9e06dd3953ef64e99b6328907fdbd59f479a7ee6` 领先三个未推送 commit；最新 repair 恰为一个 commit、13 个授权文件，Phase 1 tag 与禁止文件未变；
- card file SHA256 `78aee1ab1484f9ec54c354fc12a470e88be256c62a38b69b01b80f879b015af5`、canonical SHA256 `315d28c54661a1fe277cf0a98e0142560206bd67259314ec80b33de1665b91ff` 与八个 implementation hash 闭合；
- fixed-horizon Holm family 已恢复为且仅为 `K2→K3、K3→K4`，完整 H1/NCA rules 已版本化；sealed-label、B1→B2 first-four、resource/restore、report recomputation 与统计类型/range 反例均已收紧；
- 规划线程独立重跑 Phase 2 targeted `53/53`、full `196/196`、compile、CLI help、card/hash closure 和 diff checks，均通过。

但绿测之外的 adversarial audit 复现了以下 provenance blocker，因此 Gate A 决定仍为 `BLOCK`，card 不预注册，Gate B 保持锁定：

1. **producer kind 未与 cell 集合绑定**：full envelope 对全部十二个 cell 都允许 `phase1_immutable_reuse_adapter`。独立构造本应新运行的 `fixed_step 11:14, K=3, alpha=1.5`、14,042-row envelope，只填五个虚构 SHA，`make_full_final_output_envelope` 与 validator 均接受；八个新 full 因此可被伪装成 Phase 1 reuse。
2. **四个合法 Phase 1 reuse cell 没有真实 producer closure**：仓库没有该 reuse adapter 的 loader/CLI/cell-to-source mapping；producer 不含 results/source artifact ref，full identity 只绑定 baseline 样本 identity，不绑定 baseline/三个 K2 的逐样本 choice scores 与具体不可变结果。
3. **source-aware analysis 未重验 full producer**：analysis 只读取 sidecar path+hash 并作 envelope structural validation，不重新读取 command/env/revision/results/attempt/receipt，也不约束 full sidecar 的 resolved path 与独立 LoopScope write-once root。伪造 sidecar 会被当作 canonical source，并被 verifier 忠实重算成伪 H1。
4. **probe evidence sibling/path closure 仍可绕过**：producer-evidence validator 只要求绝对路径；把伪 command/env/revision/control files 放在系统临时目录，同时自述 governed output root，`load_files=True` 仍可通过。

本次审计不修改 repo、不 push、不访问 HPC2/GPU/label，也不进入 Gate B。任何后续 repair 必须重新显式授权；建议的最小边界是：严格映射四个 Phase 1 reuse cell 与八个 new-run cell、实现真实 Phase 1 reuse source loader、为所有 full/probe evidence 加入同根 symlink-safe artifact-root/path+hash closure，并让 analysis verifier 重新加载 producer evidence及对应逐样本 source。不得改变科学集合、窗口、K/alpha、seed、尺度分离或 protected files。

### 10.5 用户持续推进授权与 Gate A provenance-only bounded repair（2026-07-14）

用户明确授权本规划/审计线程按照 `research-gate-orchestrator` 和既定 Gate 顺序持续推进第二阶段，无需在每个正常 Gate transition 或不改变冻结科学契约的同 Gate 工程修复前再次请求用户确认。该 standing authority 不取消以下边界：一次只授权当前 Gate；`PASS` 后为下一 Gate 创建新的 executor；`BLOCK/PASS_WITH_FIXES` 只回到同一 Gate executor；任何会改变模型、数据/split、样本、seed、窗口、K/alpha、metric/threshold/family、尺度分离、GPU 总预算、protected code 或 destructive/external authority 的选择仍须停止并回传。

基于 10.4 已复现的单一工程根因——full/probe producer evidence 没有形成 cell-aware、source-aware、same-root provenance closure——同一 Gate A executor
`019f5c1d-7355-7130-81e2-125b4f1e21c0` 获得第三次、也是本授权下唯一的 provenance-only local repair：

```text
gate=A provenance repair
repository=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope
starting HEAD=46e5365efd70561159c36b5447aef49f6111450a
origin/loopscope=9e06dd3953ef64e99b6328907fdbd59f479a7ee6
expected worktree=clean; ahead of origin by exactly 3 commits
authorized executor=019f5c1d-7355-7130-81e2-125b4f1e21c0
runtime=gpt-5.6-sol/max
maximum_new_run_count=0
GPU_hours=0
network/SSH/HPC2/Slurm/download=forbidden
repair_budget=at most 2 bounded local engineering loops
Git=preserve the existing three commits; at most one new single-purpose provenance repair commit; no amend/rebase/reset/push
terminal_recipient=019f5149-35f6-7383-b3ec-3376ba413d2b
```

允许修改：

- `configs/loopscope/qwen17_mmlu_phase2_h1_v2.json` 仅更新 schema/versioned implementation hash closure；
- `docs/loopscope_phase2.md`；
- `scripts/loopscope/prepare_qwen17_phase2.py`、`src/tflt/cli.py`、`src/tflt/eval_runner.py`、`src/tflt/loopscope/__init__.py`；
- `src/tflt/loopscope/phase2_schema.py`、`phase2_trajectory.py`、`phase2_analysis.py`、`phase2_reuse.py`；
- `tests/test_loopscope_phase2_*.py`。

`AGENTS.md`、本 control、Phase 1 工件、`wrapper.py/strategies.py/cache.py/config.py` 及其他文件继续只读。不得修改科学集合、创建新 full/probe、访问 label、读取远程 live artifact 或把未知 digest 编造成已验证值。

repair 必须作为一个完整根因闭环完成：

1. **cell-aware producer partition**：精确定义且验证四个允许 Phase 1 reuse 的 full cells（一个 no-loop baseline、三个窗口的 shared K2 anchor）与八个必须由新 lm-eval execution 产生的 cells（六个 fixed-step K3/K4、`12:15` 两个 fixed-horizon K3/K4）。`phase1_immutable_reuse_adapter` 只能用于前四个；`lm_eval_logged_samples_adapter` 只能用于后八个。不得以 K1 等价性把新 cell 或其他窗口映射成 reuse。
2. **真实 Phase 1 reuse source loader**：实现 fail-closed 的 immutable reuse producer/schema，绑定 canonical Phase 1 run manifest、精确 baseline/K2 output mapping、实际 `command_args/env/model_revision/results` 或 exact logged-sample sidecar set、文件 SHA、revision、cell scientific argv 与逐样本 `task/doc_id/doc_hash/raw four-choice score`。Gate A 只用本地 fixture 测试 loader；若 live Phase 1 缺 raw score，状态继续为 `requires_gate_b_live_check`，不得合成、猜测或授权 K2 重跑。
3. **full artifact-root 与 producer-evidence closure**：每个 full envelope 必须声明经 symlink-safe 校验的实际 artifact root 和 exact path+hash producer-evidence tree。新 full sidecar、`command_args/env/model_revision/results` 与 output-owned control evidence必须位于声明的独立 LoopScope write-once root内；reuse 生成的新 sidecar仍落在 LoopScope workspace，只读引用留在旧 Phase 1 immutable root的 source artifacts。
4. **source-aware analysis reload**：analysis input loader/verifier必须解析并限制 full-cell sidecar位于独立 LoopScope workspace，重新读取并验证每个 full producer evidence、cell mapping、revision、samples与 source file hash，然后才允许逐样本重算。不能把 structural envelope validation当作 producer verification。
5. **probe same-root closure**：B1/B2 的 output-owned `command_args/env/revision` 必须是 declared output root 下的精确预期路径；attempt/receipt必须位于受治理的 LoopScope workspace且 packet 中的 output root 与实际一致；input manifest只能指向冻结的 canonical Phase 1 input root。拒绝 `..`、prefix confusion、root 外路径、复制到系统临时目录与 symlink escape。
6. **必需红测与回归**：先复现再修复至少以下反例：新增 K3/K4 冒充 Phase 1 reuse；四个 reuse cell 使用任意 SHA/不存在 source；错误 Phase 1 cell-to-output mapping；伪 lm-eval sidecar缺真实 producer files；root 外或 symlink-escape full ref；full 伪 source进入 source-aware analysis；probe output-owned evidence在系统临时目录但自述 governed root。还必须有合法四-cell reuse fixture、合法 new-cell fixture、合法 probe fixture与 default eval path正例。
7. **最终验证**：Phase 2 targeted、完整 unittest、compileall、所有 Phase 2 CLI help、card self-hash/implementation hash closure、`git diff --check`、protected-path diff、clean worktree和 commit-count检查全部通过。三条现有 exit-0伪造路径必须变为非零/异常拒绝。

若完成修复必须更改未授权文件或科学/资源契约，执行线程立即发送 `BLOCK`。否则最多创建一个 provenance repair commit，保持不 push，并只发送一次新的 `GATE_A_FINAL_AUDIT`；不得进入 Gate B或自行把 card称为已预注册。

### 10.6 Gate A provenance-repair 终态独立审计 PASS_WITH_FIXES 与 raw-file-hash repair（2026-07-14）

同一执行线程以 control SHA256
`bdb3f0ef6aaeb467245c10ee94b2c1219ff581327bcca1884490aae8b8a0cff0`
回传 `SUCCESS_PENDING_PLANNING_AUDIT`。规划线程重新读取 live Git、card、实现与测试，独立确认：

- local `loopscope@ee1638c3bf14adc75b903a08aebae354fe9e3ae4` clean，恰好比
  `origin/loopscope@9e06dd3953ef64e99b6328907fdbd59f479a7ee6` 领先四个未推送 commit；最新 repair
  恰为一个 commit、12 个授权文件，Phase 1 tag、AGENTS、control 与禁止文件未变；
- card file SHA256
  `ba6ddc411b09a26fd2107225682063713816fbe0c0365b026ba6a84e663f9968`、canonical SHA256
  `7d682e8e5ddc9e66b8e07770eaa3a8e83f95ba0161ae3a8d34e629b744fc82be` 与八个
  implementation hash 闭合，冻结 science/trajectory/statistics/decision rules 未漂移；
- 4 个 Phase 1 reuse 与 8 个 new-run cell 的 producer 分区、真实 Phase 1 source loader、
  source-aware analysis reload、full/probe same-root 与 symlink-safe containment 已实现；
- 规划线程独立重跑 Phase 2 targeted `63/63`、full `206/206`、compile、全部 Phase 2 CLI help、
  card/hash closure、freeze-candidate materialization 与 diff checks 均通过；独立反例确认错 producer、
  `..`、prefix sibling 与 symlink escape 已拒绝。

但独立 source-integrity 复核发现一个窄且可复现的 provenance 缺口，因此本次决定为
`PASS_WITH_FIXES`，card 继续保持 freeze candidate，Gate A 尚未关闭：

- `src/tflt/loopscope/phase2_reuse.py::_load_full_evidence_files` 在 raw file SHA256 与 ref
  不等时，对所有 JSON evidence 无差别接受“顶层 `manifest_sha256` 等于 ref”的 fallback，且没有
  验证 canonical self-hash；
- 该 fallback 也覆盖本应严格使用 raw file SHA 的 `command_args/env/model_revision/results`。
  规划线程在 `/tmp` 构造合法 new-full fixture，先验证通过，再只向 `results.json` 增加
  `manifest_sha256=<sidecar 记录的旧 raw SHA>`；文件实际字节 SHA 已改变，
  `verify_full_final_output_artifact` 仍错误接受，复现实测为 `raw_file_hash_bypass=ACCEPTED`；
- 这违反 10.5 的 exact path+file SHA 闭包，但不要求改变任何科学、schema 架构、路径或资源合同。

用户最新指令要求完成 Gate A 后停止，不进入后续 Gate。为完成当前 Gate A，同一 executor
`019f5c1d-7355-7130-81e2-125b4f1e21c0` 获得一次极窄 local-only repair：

```text
gate=A raw-file-hash repair
repository=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope
starting HEAD=ee1638c3bf14adc75b903a08aebae354fe9e3ae4
origin/loopscope=9e06dd3953ef64e99b6328907fdbd59f479a7ee6
expected worktree=clean; ahead of origin by exactly 4 commits
authorized executor=019f5c1d-7355-7130-81e2-125b4f1e21c0
runtime=gpt-5.6-sol/max
maximum_new_run_count=0
GPU_hours=0
network/SSH/HPC2/Slurm/download=forbidden
repair_budget=exactly one bounded local engineering loop
Git=preserve the existing four commits; at most one new single-purpose hash-semantics repair commit; no amend/rebase/reset/push
terminal_recipient=019f5149-35f6-7383-b3ec-3376ba413d2b
```

允许修改且仅允许修改：

- `src/tflt/loopscope/phase2_reuse.py`；
- `tests/test_loopscope_phase2_provenance_repair.py`，必要时同一命名族的一个 provenance regression test；
- `configs/loopscope/qwen17_mmlu_phase2_h1_v2.json` 仅更新
  `phase2_reuse.py` implementation SHA 与 card canonical self-hash。

repair 验收边界：

1. 对 full evidence key 明确区分 hash 语义。`command_args`、`environment`、
   `revision_report`、`results`、`source_command_args`、`source_environment`、
   `source_revision_report`、`source_results` 以及逐样本 source artifact 必须无条件使用实际 raw
   file SHA256，字节不同即拒绝；不得因 JSON 内存在同名 `manifest_sha256` 而放行。
2. 仅 `adapter_request`、`reuse_request`、`attempt_manifest`、`receipt_manifest`、
   `source_run_manifest` 等由 schema 明确定义为 canonical manifest 的 ref，才使用
   `verify_manifest_sha256` 验证 self-hash并与 ref SHA 比较；不能只比较字段字符串。
3. 新红测必须至少覆盖 `results/command_args/env/model_revision` 注入旧 raw SHA 后仍发生字节变化，
   并证明四者全部被拒绝；同时保留合法 canonical manifest refs 与合法 new/reuse full 正例。
4. Phase 2 targeted、full unittest、compileall、全部 Phase 2 CLI help、card self-hash/
   implementation-hash closure、`git diff --check`、protected-path diff、clean worktree和 commit-count检查
   全部通过。

若单轮内无法闭合或需要扩大文件/科学/资源边界，立即发送 `BLOCK`。否则只发送一次新的
`GATE_A_FINAL_AUDIT`，不 push，不进入 Gate B。Gate A 最终审计后，本规划线程必须按用户最新指令停止；
无论 Gate A 结果如何，都不得创建或授权 Gate B executor，直到用户另行明确授权。

### 10.7 用户校准审计重要性、撤销过度加固并决定 Gate A PASS（2026-07-14）

用户明确指出：当前第二阶段以敏捷验证实验假设为目标，不应为了不影响后续实验的小问题追求
完备无懈可击。该最新指令将 10.6 的 raw-file-hash finding 从 blocker 降为 accepted
non-blocking technical debt，并撤销对应 repair 授权。

本项目后续 Gate 审计采用实验重要性分级：

- 只有会实质影响冻结科学参数、样本/label 隔离、正常生产路径、输出数值或统计裁决、实际工件
  identity/revision、可复现性、GPU/Slurm 成本与安全、protected state，或会阻止下一 Gate 正常执行的
  问题，才可判 `BLOCK/PASS_WITH_FIXES`；
- 仅在刻意手工篡改、对抗性伪造、未使用路径、格式完美性或 defense-in-depth 场景出现，且不会改变
  正常实验行为与结果的问题，记录为 technical debt 或 audit note，不阻塞 Gate；
- “存在可构造反例”本身不足以阻塞，规划审计必须同时说明该反例是否能在授权的正常实验路径中发生，
  以及是否会改变科学结论或实际运行状态。

撤权事件已闭环：执行线程在收到 superseding scope change 时只新增了一个未提交红测方法，随后使用
`apply_patch` 精确撤销该方法；规划线程独立核对并收到 `SCOPE_CHANGE_STOPPED`：

```text
HEAD=ee1638c3bf14adc75b903a08aebae354fe9e3ae4
origin/loopscope=9e06dd3953ef64e99b6328907fdbd59f479a7ee6
ahead=4
dirty=clean
new commit=no
implementation/config changes=none
push/SSH/HPC2/GPU/Gate B=not executed
```

因此 Gate A 最终决定为 `PASS`，审计对象为
`loopscope@ee1638c3bf14adc75b903a08aebae354fe9e3ae4`。Gate A 的有效验收证据包括：

- frozen science/trajectory/statistics/decision rules 不漂移，card canonical SHA256
  `7d682e8e5ddc9e66b8e07770eaa3a8e83f95ba0161ae3a8d34e629b744fc82be`；
- 4 reuse + 8 new producer partition、真实 Phase 1 loader、source-aware analysis 与正常路径
  provenance/path closure已实现；
- planning audit 独立重跑 Phase 2 `63/63`、full `206/206`、compile、CLI help、card/hash、
  freeze-candidate 与 protected-state checks 全部通过；
- raw JSON `manifest_sha256` fallback 仅保留为后续可选 hardening，不参与 Gate B admission，未来也不得
  在没有正常路径影响证据时重新升级为 blocker。

按用户此前最新流程指令，Gate A PASS 后立即停止。不得 push这四个提交，不得创建 Gate B executor，
不得 SSH/HPC2/Slurm/GPU；只有用户另行明确授权后，规划线程才能为 Gate B 更新控制面并新建独立 executor。

### 10.8 用户恢复连续推进并授权 Gate B（2026-07-14）

用户最新指令明确恢复 Gate A 后续 Gate，并授权规划/审计线程按
`research-gate-orchestrator` 串行推进，直至完成第二阶段实现与核心实验。该指令取代 10.7 的
阶段停止状态，但不改写 Gate A 历史决定，也不放宽一 Gate 一 executor、事件驱动终态、冻结科学、
protected paths、write-once、资源上限和实质性审计标准。

Gate B 绑定新的独立 executor：

```text
gate=B; subgates=B0→B1→B2 in order
authorized executor=019f5ebd-87d0-79a1-9633-3e67ad93d5b1
title=execute-LoopScope-H1V2-第二阶段-Gate B
runtime=gpt-5.6-sol/max
planning/audit recipient=019f5149-35f6-7383-b3ec-3376ba413d2b
repository=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope
starting HEAD=ee1638c3bf14adc75b903a08aebae354fe9e3ae4
starting origin/loopscope=9e06dd3953ef64e99b6328907fdbd59f479a7ee6
expected worktree=clean; exactly four audited commits ahead
card=H1_ITERATIVE_REFINEMENT_ZONE_V2
card canonical SHA256=7d682e8e5ddc9e66b8e07770eaa3a8e83f95ba0161ae3a8d34e629b744fc82be
remote clone=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
read-only Phase 1 run=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-20260711-053022
Phase 2 workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope
venv=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711; reuse without install/upgrade
B2 budget=one existing A40; at most 18 GPU-hours total including any authorized repair attempt
new full MMLU runs=0
audit-returned repair cycles=at most one bundled material repair; executor-owned repair loops=at most two low-risk loops
terminal event=exactly one GATE_B_FINAL_AUDIT or BLOCK
```

Gate B 允许：

1. **B0**：普通、非 force push `loopscope`；核对 GitHub ref 后在 HPC2 dedicated clone
   `fetch` 并 `merge --ff-only` 到精确审计 commit；保持本地、origin、HPC2 clean closure；复用既有 venv
   做 version/import/unit/compile/CLI help；只读核对 Phase 1 512/14042 identity、四个 reuse cell 的实际
   raw choice/source availability、revision 与 provenance。禁止修改原 reproduction checkout、venv、cache、
   Phase 1 工件或编造缺失 digest。
2. **B1**：只在 B0 `PASS-ready` 后，用冻结 pool 的确定性前四样本执行一个受控四样本 GPU smoke；
   同进程验证 no-loop、三个 K1 admission cells 与 15 个 logical loop cells，`K1` choice equivalence、
   K2/K3/K4 prefix consistency、body-call count、NCA position/boundary/validity、final-choice、restore、
   schema/revision 和 per-cell wall-clock/peak-memory。可执行必要的 limit 工程检查，但 limit accuracy 不得
   改科学矩阵。
3. **B2**：只有 B1 全部 must-pass admission 成立且按 B1 预测不超过预算时，才执行冻结 512 的无标签
   calibration；一个 deterministic no-loop boundary cell 加 15 个唯一 loop cells，向量只在同一受控
   process/session 内短暂驻留，写盘仅限冻结 scalar/norm/validity/final-choice 与 resource/provenance。
4. 所有新 evidence、attempt、receipt、trace、logs 和 outputs 必须进入 Phase 2 workspace 下一个新的
   timestamped Gate B authorization root，保持 exclusive-create/write-once；Phase 1 run 只读引用。
5. 模型、数据与 package 使用现有 cache/venv，运行设置
   `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1` 并 unset `HF_ENDPOINT`；
   cache 缺失必须 `BLOCK`，不得下载或回退网络。
6. B1/B2 可使用现有用户可用的单 A40 `debug` 或 `emergency_gpua40` 调度路径；不得改变 GPU 类型、
   扩 GPU 数或突破 B2 18 GPU-hour 总预算。若资源预测超界，在 B2 前 `BLOCK`。

Gate B 禁止：

- full MMLU、任何 K2 full/trace-only 补跑、Gate C/D、selector、额外窗口、K>4、其他 alpha；
- 在冻结 NCA artifact/hash 前读取 512 label，或根据 B1/limit/B2 结果修改窗口、K、顺序、阈值、
  primary family、H1/NCA 判据；
- 修改 `wrapper.py/strategies.py/cache.py/config.py`、AGENTS、本 control、Phase 1 artifact、原 reproduction
  checkout、venv 或 shared cache；
- rebase、amend、reset、force-push、覆盖/删除旧 evidence、静默 retry/requeue/resubmit。

预授权的低风险修复仅限一个可复现的正常路径工程根因，必须保持 science、identity、资源和外部权限
不变；本地/远端代码修复最多两个 executor-owned loop、一个单一目的 commit并重新普通 push/ff。
每个 B1/B2 子门原则上各一次 scientific attempt；只有 job 未进入科学命令且没有有效输出时，才允许
同一根因的一次 write-once pre-science repair attempt。任何 NaN/Inf、identity/schema、K1/prefix、
restore、OOM、数据或科学结果失败均不得自动重试，必须直接终态 `BLOCK`。

Gate C 保持 `LOCKED`。规划线程收到 Gate B 终态后只做风险成比例的独立审计；正常路径 must-pass
证据闭合且仅余非实质性 hardening 时必须 `PASS`，不得为对抗性或形式完美问题反复退回。

### 10.9 用户冻结 lm-eval 框架并要求 LoopScope 单向适配（2026-07-14）

用户最新明确指令：`lm-eval` 评测框架相关代码与语义必须冻结，LoopScope 必须用自己的实现去适配
框架，而不是修改框架来迁就实验。该要求立即约束正在执行的 Gate B 以及后续 Gate C/full。

冻结范围：

- fixed venv 中安装的 `lm_eval` package/source 与版本 `0.4.11`；不得编辑 site-packages、重新安装、
  upgrade/downgrade 或用本地 shadow package 覆盖；
- lm-eval 的 task/YAML registry、HFLM、TaskManager、`simple_evaluate` 调用契约、renderer、choice request、
  batching/order、filter、`acc,none` metric 与 `results.json`/logged-sample 生产语义；
- 默认未启用 Phase 2 adapter 的 `tflt.eval_runner` 评测行为与既有 Phase 1 结果语义。

允许的适配范围：

- LoopScope 自有 `src/tflt/loopscope/` loader/parser/schema/analysis；
- `src/tflt/eval_runner.py` 中显式 opt-in、默认关闭的 Phase 2 final-output adapter；
- LoopScope 自有 CLI、request/manifest、identity join、sidecar、provenance 与 regression tests。

适配必须读取并诚实解释 lm-eval 的真实输出形态。例如 `lm-eval 0.4.11` 若在逐样本记录中使用
`metrics=["acc"]` 与 top-level binary `acc`，而 aggregate metric 仍为 `acc,none`，可以在 LoopScope
parser 中只接受该精确、无歧义形态并用逐样本 raw choice/top1 交叉验证；不得改写 lm-eval producer、
task registry 或 metric 命名。任何需要改变 lm-eval 框架、评测样本、renderer、batch 语义或 metric
才能继续的情况必须立即 `BLOCK`，不能消耗低风险 repair budget。

Gate B 的新增 must-pass：

1. Git diff/commit 不得包含任何 lm-eval framework/package/task-registry 文件；
2. fixed venv 的 lm-eval version、package tree evidence 与 lock/freeze 在 Gate B 前后不变；
3. default-eval regression 通过，证明未启用 adapter 时 HFLM/TaskManager/simple_evaluate 参数、
   `results.json` 与科学行为不变；
4. 当前及后续兼容修复只能修改 LoopScope-owned adapter/parser/schema/test，并由真实
   lm-eval 0.4.11 artifact fixture/live source 验证；
5. Gate B/C 终态包必须显式列出 framework-freeze proof 和所有适配文件。

该指令不改变 H1 card、窗口、K/alpha、sample、统计 family、GPU 预算或 Gate 顺序。当前 executor
仍可在同一 Gate B 授权内继续其正常路径兼容修复，但必须先核对上述冻结证明；若发现已修改框架，
停止并回传 `BLOCK`，不得补写或覆盖证据。

### 10.10 Gate B 独立审计 PASS 与 Gate C 授权（2026-07-14）

授权 executor `019f5ebd-87d0-79a1-9633-3e67ad93d5b1` 以 control SHA256
`90e0a37379ed2721f772cf1848dfeece9f812f90ef2c94cf14373b50b157df8e` 回传唯一
`GATE_B_FINAL_AUDIT`。规划线程按实验重要性重新读取 live Git、Slurm 与 canonical artifacts，
并由本地 Git、远端 scheduler/provenance、科学工件三条独立只读审计线交叉核对。Gate B 正式决定为
`PASS`，而不是 executor 请求的 `PASS_WITH_FIXES`：终态包没有指出、独立审计也没有发现任何可在
授权正常路径中影响科学输出、identity、统计裁决、protected state、资源安全或 Gate C admission 的缺陷。

审计对象与决定性证据：

- local/origin/HPC2 dedicated clone 均 clean `loopscope@02cd49619f3d42f2dc062be7d7a17855e6d8f756`；
  reproduction checkout 仍 clean `main@3dae7586294789892b22cadb7e69592f294882c2`，Phase 1 rollback tag
  仍 peel 到 `2c3acb34c07362bf82c0f7a0af59e0d435ab44cd`；
- 唯一 Gate B code repair `02cd496` 只修改 LoopScope-owned card/parser/test，科学字段与判读规则不变；
  active implementation-bound card canonical SHA256 为
  `0efb773621f1df7a9e19d86216590934f2ce87f6789153d50fc299224ad30e18`；
- B0 live closure：512 calibration、14,042 full natural-order identity 与 baseline/三 K2 共 56,168 条
  raw four-choice rows全量闭合；Phase 2 targeted `64/64`、full `207/207`、compile/help/hash 与 default-eval
  regression通过；
- B1 job `9977332` 为 `COMPLETED/0:0`；12/12 K1 equivalence、24/24 prefix records、192 body calls、
  132 repeated-step NCA、18 restore 全部闭合，B2 conservative projection `6.1867 < 18` A40 GPU-hours；
- B2 原 job `9977363` 确认 pending-only、0 秒、无节点/分配/科学输出后按授权取消；唯一 scientific job
  `9977369` 为 `COMPLETED/0:0`。15 个 frozen cells 各 512 同序 identity，合计 24,576 body calls、
  16,896/16,896 valid repeated-step NCA、1,536 baseline NCA、8,192 final outputs、0 invalid，label sealed，
  vector 未持久化，15/15 restore；实际 GPU 时间 `568/3600=0.1578` 小时；
- 独立从 raw scalar 重算全部 18 个 10,000/seed 0 bootstrap intervals 与三个 paired
  `12:15-13:16` intervals，逐值匹配；canonical B2 aggregate self SHA256
  `daceb4f188b5994a374248ca4014232f35df83c3b8060ad11f564bcefb4c733d`；
- terminal packet file SHA256
  `3755df1eff77561123afc74783e9ea2251cfb20fef526218fed072451ec21c16`，canonical self SHA256
  `4b69395f94e29c6e9e1ac31c0beed6e6725a4fbd675cc7866400e0112c9e94cd`；直接复核全部引用文件无
  missing/hash mismatch；
- lm-eval 0.4.11 before/after exact match：14,905-entry source tree aggregate
  `6cdd314e66baf463d04db01da44c951abbe7f53b2787c1a9cf76b4e8645185df`，task/YAML/dist-info/RECORD、
  fixed venv freeze 与 lock `f9f8c780f602cd9ae641be56cd5206779a253fc6ace948fce03d8eaf9654a9d2`
  均未变；live source tree重算一致。

接受的非阻断偏差：终态再次 `git ls-remote` 超时，但 B0 已保存 GitHub ref 证明且 local tracking、HPC2
tracking/HEAD 闭合；stderr 只有 Transformers deprecation warning；这些不改变正常实验或 Gate C admission。
Gate B executor 从现在起撤权，只保留历史证据责任，不得进入 Gate C。

Gate C 绑定新的独立 executor：

```text
gate=C
authorized executor=019f5f51-bc44-7e62-93e6-d7043e5bea1b
title=execute-LoopScope-H1V2-第二阶段-Gate C
runtime=gpt-5.6-sol/max
planning/audit recipient=019f5149-35f6-7383-b3ec-3376ba413d2b
repository=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope
starting local/origin/HPC2 commit=02cd49619f3d42f2dc062be7d7a17855e6d8f756; all clean
card=H1_ITERATIVE_REFINEMENT_ZONE_V2
active card canonical SHA256=0efb773621f1df7a9e19d86216590934f2ce87f6789153d50fc299224ad30e18
Gate B run=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/gate-b-h1v2-20260714T040409Z
Phase 1 immutable run=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-20260711-053022
new full configurations=exactly 8
GPU=one A40 per cell; at most 8 concurrently; total scientific attempts <=16 A40 GPU-hours
repair budget=at most 2 low-risk engineering loops and at most one single-purpose project-owned repair commit
terminal event=exactly one GATE_C_FINAL_AUDIT or BLOCK
```

Gate C 唯一科学矩阵：

```text
primary fixed-step continuation:
  11:14 (K=3,alpha=1.5), (K=4,alpha=2.0)
  12:15 (K=3,alpha=1.5), (K=4,alpha=2.0)
  13:16 (K=3,alpha=1.5), (K=4,alpha=2.0)
fixed-horizon control:
  12:15 (K=3,alpha=1.0), (K=4,alpha=1.0)
reuse only:
  Phase 1 baseline + 11:14/12:15/13:16 (K=2,alpha=1.0)
```

Gate C 允许：

1. admission 后为八个 new full cells 与完整 12-cell analysis 创建唯一 timestamped write-once Gate C root；
   在提交前冻结 full manifest、canonical 14,042 identity、cell order、commands、card/control/revision hashes；
2. 使用 frozen `lm-eval 0.4.11` 原框架执行八个 MMLU 5-shot full，LoopScope 只通过显式 opt-in
   final-output adapter精确 join并恢复 canonical order；每项必须保存 14,042 条 raw four-choice/final-choice
   evidence、revision、attempt/receipt、command/env/results/sidecar；full 禁止 residual/NCA；
3. 每 cell 一张 A40，可使用既有 `emergency_gpua40` 或经资源估计可容纳的 `debug` 路径；可用一个
   `0-7` array并在 `1..8` 范围内设置/调整 throttle，调度并发不改变八个科学 cells或总 GPU-hour 上限；
4. 所有运行完全离线：`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
   `HF_DATASETS_OFFLINE=1`，unset `HF_ENDPOINT`；只读复用 fixed venv/shared caches；
5. 八个 new outputs 全部终态、hash/revision/14,042 identity闭合后，冻结 12-cell source tree；此后才允许
   为 512 calibration 创建一次 write-once label-unseal authorization/receipt和 gold-index sidecar，绑定
   active card、B2 baseline+15 sealed artifacts及其 hashes；不得在此前读取 target label；
6. 只执行一次 write-once mechanism analysis：从四个 Phase 1 reuse + 八个 new full source逐样本重算
   full outcome（2,000 bootstrap、seed 20260710）、six-primary exact McNemar/Holm、两项 fixed-horizon Holm、
   四类翻转、entropy/margin/JS与错误过度自信；从已审计 B2 scalar和授权 label sidecar重算 NCA
   10,000/seed 0 subgroup diagnosis。输出一个 H1 candidate label和一个独立 NCA candidate label，再运行
   `verify-phase2-analysis`；executor不得自宣科学结论或 Gate PASS；
7. 若正常路径出现一个可复现、保持 science/identity/resource/framework边界的工程根因，可在授权文件
   `src/tflt/loopscope/*`、显式 opt-in `eval_runner.py`、LoopScope CLI/config/script/tests/docs内做最多两个
   bounded loops和一个单一目的 commit，相关/full tests通过后普通 push并HPC2 ff-only。lm-eval package、
   protected files与 default eval path继续只读。

Gate C 禁止：

- 改模型/revision、task/fewshot/dtype、三窗口、K/alpha、metric/family/seed/阈值/判读规则或八-cell矩阵；
- 重跑 Phase 1 baseline/K2、增加额外 full/window/K、full residual/NCA、HFLM router、修改 renderer/batching
  或 lm-eval package/task registry/framework；
- 根据部分 accuracy、limit、B2 NCA或中途结果改变 cell顺序、删减/扩增矩阵或选择性分析；
- 修改 `AGENTS.md`、本 control、`wrapper.py/strategies.py/cache.py/config.py`、Phase 1/Gate B artifacts、
  reproduction checkout、venv或shared cache；
- 科学命令开始后的失败自动 retry/requeue/resubmit。只有 job 从未开始、无 node/AllocTRES、无科学输出/
  claim时，允许一次 scheduler-only pre-science replacement，必须保留旧 attempt且 scientific body hash不变；
- force push、amend/rebase/reset、覆盖/删除旧 evidence、进入 Gate D或 optional extension。

长时间排队/运行由 Gate C executor 自己创建低频只读 automation监控；按预计时长采用约 10/30/60 分钟
间隔，终态后删除/归档。automation不得写工件、改 throttle以外的科学/资源契约或自动 retry。规划线程不轮询。
Gate C executor必须在完成或材料性阻塞时主动发送唯一结构化终态包，包含 live Git/Slurm、八个 job/cell、
12-cell identity/revision、unseal chain、一次 analysis/hash/verifier、H1/NCA candidate labels、lm-eval before/after
freeze proof、资源/repair/retry/protected-state与 deliberately-not-executed。

### 10.11 Gate C 独立审计 PASS 与第二阶段核心完成（2026-07-14）

授权 executor `019f5f51-bc44-7e62-93e6-d7043e5bea1b` 以 control SHA256
`b062371bb2c109df5c310feb336c899a1ffa3275e011a766d14b643e38400f32` 回传唯一
`GATE_C_FINAL_AUDIT`。规划线程按照实验重要性，分别从本地 Git/repair diff、HPC2 Slurm/provenance、
逐样本科学统计三条独立只读证据线审计。Gate C 正式决定为 `PASS`，而不是 executor 请求的
`PASS_WITH_FIXES`；现存偏差不会改变正常生产路径、任何逐样本结果、统计裁决、protected state 或后续
分支选择。

审计对象与决定性证据：

- local/origin/HPC2 dedicated clone 均 clean `loopscope@96159840525b89240a9b0f70e63764e2110216e2`；
  reproduction checkout 仍 clean `main@3dae7586294789892b22cadb7e69592f294882c2`，Phase 1 rollback tag
  仍 peel 到 `2c3acb34c07362bf82c0f7a0af59e0d435ab44cd`；唯一 repair commit 只修改
  `phase2_analysis.py` 与其 regression test；
- 原 array `9977778` 为 pending-only、0 秒、无 node/AllocTRES；replacement `9977874` 的八个 task
  全部 `COMPLETED/0:0`，时间窗内未发现第二次 scientific retry/requeue/resubmit；累计
  `9889/3600=2.746944` A40 GPU-hours，小于 16 小时；
- terminal packet file SHA256
  `d513ed4f156e66f0bf66dc78371449b57b61086aa1cfeaacaddb82321d3dcef4`，canonical self SHA256
  `98f8cd48e56c33fab0c743956ad3c9ad06268b7a70a1246ee21621f79c636187`。全 run 抽取的 150 个 JSON
  均可解析，106 个 self-hash、29 个 raw-file 引用及 195 个 path+hash 引用均闭合；
- 八个 new full 与四个 immutable reuse sidecar 均恰有 14,042 条样本；十二项逐条 identity 完全同序，
  ordered SHA256 均为 `a863e6dd6d2370f091bd59d63a582a832eba245fe6c93f56288ac6ec05a9759e`；
  四个 raw choice scores 有限且 top1/gold/correctness 自洽，revision 唯一为
  `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`，无 full residual/q/NCA/hidden vector；
- C2 在 `2026-07-14T09:31:53Z` 闭合，unseal authorization 随后创建，最终只有一份 512-label
  sidecar；dataset revision、57 tasks、512 processed identity 与 Phase 1 rendering records 闭合。一次真实
  label access 后的 analysis preflight 暴露 routing bug并诚实停止，repair 后没有第二次 label access或第二份
  label sidecar；analysis 只执行一次，verifier 只执行一次，二者 stderr为空；
- 从十二个逐样本 source 独立重算六项 primary 与两项 fixed-horizon 的 transition counts、accuracy delta、
  exact McNemar、Holm adjusted p，全部精确匹配报告；使用独立 RNG 重算 bootstrap，最大 CI endpoint
  差约 `0.043 pp`，不改变任何标签。`12:15` fixed-step 的 K2→K3 为 `-1.1394 pp`
  (`right→wrong/wrong→right=496/336`, Holm `p=1.61e-7`)，K3→K4 为 `-1.0041 pp`
  (`395/254`, Holm `p=1.61e-7`)，按冻结优先级确定 `H1=PERTURBATION`；
- 九个 primary NCA cells、三个 paired `12:15-13:16` intervals 与三个 native-fidelity subgroup独立重算
  一致。虽然三个 paired difference均为正，但 `12:15` 自身 K2/K3/K4 NCA intervals全部显著为负，
  native-fidelity支持数为 0，因此确定 `NCA_NONDISCRIMINATIVE` 与
  `native_fidelity_check=EVALUATED_NOT_SUPPORTED`；
- lm-eval 0.4.11 在 Gate C before/after 的 11 个比较键完全一致；RECORD 中 14,914 个 hashed entries
  全部匹配、0 missing/mismatch，六个关键源码 hash、dist-info与 lock
  `f9f8c780f602cd9ae641be56cd5206779a253fc6ace948fce03d8eaf9654a9d2` 不变。Gate B 与 Gate C 的
  full-tree aggregate字符串不可直接比较，因为清单明示使用 `sorted` 与 `globally sorted` 两种聚合配方；
  该配方差异不是 framework mutation。

接受的非阻断 provenance 偏差：冻结 card 中 `phase2_analysis.py` implementation SHA256仍为
`817b24ec50658c39ab1f9610bb1c005491e251faf98cdd2757d5d2d0788a232b`，实际报告 runtime 是
commit `9615984` / file SHA256
`cf60e24ad4c6e6d6987e1aa108ade11221fd3ba45536208c20329c0d36329415`。diff 只在 exact-key验证后剥离
routing-only `cell_id` 再调用原 path/hash loader；未修改 source、bootstrap、McNemar、Holm、H1/NCA
classifier或阈值。旧代码只会 fail-fast，不会产生另一套结果；独立重算又不依赖项目统计函数。因此不
回写已解封 card、不重跑 analysis、不退回 repair。当前完整 test suite唯一失败正是该已知 hash mismatch；
后续新研究若复用代码，必须绑定实际 runtime commit或另立新 card，不能把旧 implementation hash冒充当前
runtime。analysis attempt/receipt的命名也仅记为非阻断文档债务，实际执行由唯一 report/stdout/verifier/hash
独立证明。

科学与流程结论：H1 V2 在当前 `Qwen3-1.7B-Base × MMLU 5-shot` cell 中得到完整负向结果——增加
fixed-step test-time compute 持续显著降低准确率，且 NCA 不能提供有用的方向判别。Gate `PASS` 表示
实验契约和证据完整，不表示 H1 得到支持。按照预注册互斥矩阵，`H1=PERTURBATION` 时停止 selector
扩展；第二阶段 A–C 核心实现与实验到此完成，Gate D/H2 均不授权。Gate C executor从现在起撤权，只
保留历史证据责任；任何新任务、模型、指标或消歧实验必须由用户选择后另立 hypothesis card。

### 10.12 用户授权 fixed-horizon 逐步机制补充实验与 Gate D（2026-07-16）

用户明确授权把 Gate C 后的补充研究冻结为一张敏捷、描述性 H2 card，并要求规划/审计线程按
`research-gate-orchestrator` 创建独立执行线程完成实验。该授权不推翻 Gate C 的
`H1=PERTURBATION + NCA_NONDISCRIMINATIVE`，也不恢复 selector 扩展；它只回答：为什么
`12:15` fixed-horizon 在 K2 达到当前最高 full accuracy，而 K3/K4 随后下降，以及第一阶段和第二阶段
无标签指标在每次真实 block 调用中如何演化。

#### 10.12.1 H2 card 与科学口径

```text
card=H2_FIXED_HORIZON_STEPWISE_PROFILE_V1
role=post_hoc_exploratory_mechanism_profile; not confirmatory selector evidence
model=Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1
task=mmlu; num_fewshot=5; dtype=float16
window=12:15 inclusive; width=4
protocol=fixed_horizon
iteration_mode=block; strategy=damped_euler
k=1,2,3,4; alpha=1.0; step h=1/K; total horizon=1
beta=0.0; cache_strategy=last; decode_mode=bypass
new_trajectory_scale=Phase 1 frozen validation 512 in exact natural order
full_scale=read-only reuse of the audited Gate C/Phase 1 K1-K4 14042 sidecars
maximum_new_full_run_count=0
new_GPU=one A40; GPU-hours have no fixed upper cap; completion of the exact frozen experiment is the resource priority and actual usage must be recorded
```

对每个 K 和 zero-based body-call `t=0..K-1`，定义：

```text
x_t      = 第 t 次进入 12:15 block 的 latent state
z_t      = G(x_t)，四层 block 的 raw exit
u_t      = z_t - x_t，raw residual
h        = 1/K
x_(t+1)  = x_t + h*u_t，实际 damped-Euler applied state
tau_entry=t/K; tau_applied=(t+1)/K
```

跨 K 比较必须同时按 body-call `t` 与归一化 horizon `tau` 报告；不得把不同 K 的同一 ordinal
误称为同一迭代时间。K1 的 raw/applied 相同；K1 没有 repeated-step q/cosine/NCA，缺值必须保持显式
`null/not_applicable`。

512 新采集必须区分以下指标，不得覆盖或改写 Phase 1 原字段：

1. **raw activity/direction**：每轮 `||u_t||`、`r_t=||u_t||/||x_t||`、相邻
   `q_t=||u_t||/||u_(t-1)||`、`cos(u_t,u_(t-1))`、既有 NCA 与 validity；
2. **applied update**：`h||u_t||`、`||h*u_t||/||x_t||`、逐步和累计 applied path length；
3. **统一 choice-lens**：只用 frozen final norm 与 A/B/C/D choice-token output weights，在 answer
   position 计算四个 choice scores/probability。对 entry、raw exit、applied state 分别保存 entropy、top
   margin、top1；这些字段必须命名为 `intermediate_lens_*`，只作辅助，不能冒充模型 final output；
4. **逐轮 entropy drop**：同时报告
   `H(x_t)-H(z_t)` 与 `H(x_t)-H(x_(t+1))`；
5. **逐轮 entropy flatness**：对 raw call 内的五个边界
   `x_t, L12(x_t), L13(...), L14(...), z_t` 的 choice entropy 曲线计算
   `-population_std`；临时 forward hooks 必须在每个 cell 后完整移除；
6. **逐轮 KL drop**：reference 固定为同一样本既有 native K1/no-loop final direct-probe choice
   distribution，分别计算 entry→raw 与 entry→applied 的 KL-to-reference drop；不得改用各 K 自己的
   final distribution作为 reference；
7. **逐轮 effective rank**：在同一 512 answer-position cohort 上，沿用 Phase 1 estimator v2
   （单位归一化、跨样本中心化、squared-singular-value spectrum），分别计算 entry、raw exit、applied
   state 的 eRank、raw/applied delta 与相对 x0 的累计 delta。完整 hidden/vector 只可在内存中短暂存在，
   不得持久化；
8. **final/output adjudication**：512 仅可把 direct-probe top1/correctness 明确标记为 development
   diagnostic。真实 MMLU accuracy、RR/RW/WR/WW、mean entropy/top-margin/correct-margin/JS、top1
   retained 与错误过度自信必须从既有 14,042 canonical K1-K4 sidecars重算/复用，不能把两个 score
   namespace 混合或逐样本 join。

512 trajectory 标量在 gold 隔离状态下先闭合；之后 D4 可只读复用 Gate C 已授权的 512 label sidecar，
按 K1→K2、K2→K3、K3→K4 报告 direct-probe correctness/flip subgroup 与逐步指标分布。该 subgroup
分析是 post-hoc mechanism diagnosis，不得拟合 selector、阈值或新 composite score。新 scalar 的描述性
bootstrap 固定为 2,000 次、seed 20260716、95% percentile；既有 NCA 保持原 10,000/seed 0，既有 full
paired outcome 保持 2,000/seed 20260710。eRank 默认报告确定性 point estimate；小 subgroup eRank
不是 must-pass。

#### 10.12.2 Gate D executor 与顺序

```text
gate=D; ordered subgates=D0→D1→D2→D3→D4
authorized executor=019f668f-e98d-7033-a28a-36d55d0cf5ac
title=execute-LoopScope-FixedHorizonK-第二阶段-Gate D
authorized runtime=gpt-5.6-sol/high; manually set by the user and not a blocker
planning/audit recipient=019f6634-0369-7832-a617-d9f7e225345f
repository=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope
starting local/origin commit=96159840525b89240a9b0f70e63764e2110216e2; expected clean
remote clone=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
Gate B source=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/gate-b-h1v2-20260714T040409Z
Gate C source=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/gate-c-h1v2-20260714T064922Z
Phase 1 source=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-20260711-053022
new run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/gate-d-fixed-horizon-stepwise-<UTC timestamp>
```

执行顺序：

- **D0 本地冻结与实现**：建立 versioned H2 card、独立 LoopScope-owned probe/analysis module、standalone
  script、最小 tests 与 runbook。优先新增文件，避免修改 H1 implementation-bound files；不注册/修改
  lm-eval。实现临时 layer hooks、raw/applied lens metrics、eRank accumulator、write-once scalar schema与
  source-aware atlas assembler。
- **D1 remote CPU + 4-sample GPU smoke**：普通 push、HPC2 dedicated clone ff-only；复用固定 venv，
  离线 import/unit/compile/help。四样本在一个 model session 中跑 K1-K4，验证调用数、`h=1/K`、
  raw/applied identity、五边界 hook count、hook removal、K1等价、K2-K4 与既有 B2 final-choice/q/cos/NCA
  在冻结数值容差内一致、restore、finite 与无 tensor persistence。
- **D2 512 scalar profile**：只有 D1 pass-ready 后，使用一个 A40、一个 model process/session 对冻结
  512 精确 natural order 执行 K1-K4，共 `512*(1+2+3+4)=5120` 个 block body calls；只写逐样本
  scalar/choice-lens/validity与 cohort eRank summary，label保持隔离。
- **D3 source closure**：确认 512/K1-K4 全部完整，读取而不修改 Gate B/C/Phase 1 canonical evidence；
  绑定既有 K1-K4 full source、revision、identity与已授权 label sidecar。
- **D4 一次 write-once atlas**：只执行一次分析，生成逐 K、逐 t、逐 tau、raw/applied/cumulative 表，
  512 development subgroup表，以及 full K1-K4 absolute/adjacent表；输出 canonical JSON、中文 Markdown
  报告和独立 verifier receipt。不得用四个 K 点做夸大的总体相关/因果或 selector claim。

#### 10.12.3 允许、禁止与敏捷验收

允许新增或最小修改：

- `configs/loopscope/` 下本 H2 card；
- 新的 `src/tflt/loopscope/phase2_fixed_horizon*.py`；
- 新的 `scripts/loopscope/*fixed_horizon*`、`tests/test_loopscope_phase2_fixed_horizon*.py`；
- `docs/loopscope_phase2.md` 只追加 Gate D 实际命令/工件说明；
- 必要的同名模块导出，但优先用 standalone module/script，避免改变 H1 implementation hash。

允许 ordinary non-force push、HPC2 dedicated clone `fetch + merge --ff-only`、SSH/Slurm、一个 A40、固定
venv/shared cache只读复用，以及 executor-owned 约 10 分钟低频只读 automation；终态后必须删除/归档
automation。所有模型/数据运行必须设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
`HF_DATASETS_OFFLINE=1` 并 unset `HF_ENDPOINT`。

禁止：

- 新 full MMLU、其他窗口、K>4、其他 alpha/strategy、selector/ranking/threshold/composite score；
- 修改 `lm_eval` package/source/task/YAML/HFLM/TaskManager/simple_evaluate/renderer/batching/metric，或修改
  `src/tflt/eval_runner.py` 的现有 H1/default eval 路径；
- 修改 `wrapper.py/strategies.py/cache.py/config.py`、旧 H1 card、AGENTS、本 control、Phase 1/Gate B/C
  工件、reproduction checkout、venv或shared cache；
- 保存完整 hidden/residual/native-continuation tensors；覆盖、删除、移动旧 evidence；force push、
  amend/rebase/reset；在 D2标量闭合前用 label 调整字段/口径；进入 post-Gate-D工作。

必须通过的材料性验收只有：

1. scientific argv、K/h/body-call、512 identity、revision、raw/applied定义和 source namespace正确；
2. D1 证明 instrumentation 不改变既有 K1-K4 outputs/核心 q-cos-NCA，hooks完整恢复；
3. D2 恰有 512×K1-K4 完整有限标量、5120 body calls、完整 cohort eRank、无持久化向量；
4. D4 只执行一次且完整呈现用户指定的第一阶段动态指标、第二阶段指标与既有 full accuracy/flip结果；
5. local/origin/HPC2 Git闭合，lm-eval与protected state不变，实际 GPU time完整记录且无未授权 retry；
6. functional tests、targeted tests、compile/help与新 card self/hash closure通过。Gate C 已知的旧 H1
   `phase2_analysis.py` implementation-hash drift可原样保留，不得把它当成本 Gate blocker，也不得为修它
   回写旧 card。

非材料性命名、图表美化、通用 schema、未使用路径 hardening、小 subgroup eRank 与对抗性手工篡改防护
不构成退回理由。执行线程最多两个低风险工程 repair loops、一个主要实现 commit与一个必要的单一目的
repair commit。D1 smoke最多两个 fresh write-once attempts；D2只有一次 scientific attempt。job 从未开始、
无 node/AllocTRES/输出时允许一次 scheduler-only replacement；D2一旦产生任何有效样本/科学 claim，失败
不得自动 retry/requeue/resubmit，必须发送 `BLOCK`。

执行线程最终只向规划线程发送一次 `GATE_D_FINAL_AUDIT` 或 `BLOCK`。终态包至少包含 runtime核对、
control/card hash、Git commits、tests、jobs/GPU time、run/artifact paths与hash、D1 noninterference、D2
5120-call closure、D4 atlas关键表、accepted deviations、protected/lm-eval proof和 deliberately-not-executed。
executor不自宣 Gate PASS。本规划线程不轮询。

#### 10.12.4 GPU-hours 上限撤销（2026-07-16）

用户最新明确指令撤销 Gate D 原 `≤2 A40 GPU-hours` 硬上限，并要求以完成本卡冻结实验为第一目标。
因此资源预测与实际耗时只作调度、监控和终态报告依据，不再构成 admission/block 条件；executor 可为
同一单 A40 scientific job 选择足以完成 512×K1-K4 与规定指标的 walltime，并允许其正常运行至终态。

本次 scope change 只移除 GPU-hours 上限，不自动扩大其他权限：仍是一个 A40、同一 512 样本、同一
12:15/K1-K4/alpha=1 矩阵、0 个新 full；D1/D2 attempt、write-once、retry/requeue/resubmit、protected
code、lm-eval freeze 与后续 Gate 禁令全部不变。实际 elapsed、AllocTRES 与 GPU-hours仍必须诚实记录。

#### 10.12.5 执行线程 runtime 覆盖（2026-07-16）

用户已手动把当前 Gate D executor 设置为 `gpt-5.6-sol/high` 并明确要求它继续执行本计划。该最新指令
撤销初始 handoff 中“必须为 max，否则 BLOCK”的 admission 条件；`gpt-5.6-sol/high` 从现在起是本 Gate
合法且最终的 runtime binding，不得因其不是 max 停止。除此之外，Gate D 科学、资源、文件、重试、
protected-state 与终态路由边界均不变。

#### 10.12.6 Gate D 终态路由恢复（2026-07-16）

Gate D executor 已完成 D0→D4，并在其自身线程生成完整的 `GATE_D_FINAL_AUDIT`。首次跨线程投递失败的
唯一原因是本 control 仍指向已经归档的旧 planning/audit thread
`019f5149-35f6-7383-b3ec-3376ba413d2b`；该失败不是执行、工件、Slurm、科学结果或跨线程工具权限失败。

用户授权立即修复路由并让同一 executor 重新发送一次既有终态包：

```text
successor planning/audit thread=019f6634-0369-7832-a617-d9f7e225345f
historical archived predecessor=019f5149-35f6-7383-b3ec-3376ba413d2b
authorized executor=019f668f-e98d-7033-a28a-36d55d0cf5ac
authorized action=one delivery-only resend of the existing GATE_D_FINAL_AUDIT
```

该恢复不创建第二个 Gate outcome，也不授权重跑 D0–D4、重算 atlas/verifier、修改 Git/control 以外文件、
SSH/HPC2/Slurm/GPU、覆盖工件或进入 post-Gate-D 工作。executor 必须原样重投递既有终态包，并在包头明确
这是对首次 archived-recipient 失败的路由修复；既有 artifact/job/hash/provenance 保持不变。旧 planning
thread 继续归档并只作历史来源，不改写既有 handoff 或工件中的 immutable provenance。

重投递已确认成功：sender 为授权 executor `019f668f-e98d-7033-a28a-36d55d0cf5ac`，recipient 为 successor
`019f6634-0369-7832-a617-d9f7e225345f`，包头为 `ROUTE_REPAIR_RESEND`，并绑定本节授权时的 control
SHA256 `e22ea5302594eeaa54565cb03ef3ca051b450d41710aae310dd3e12208c5a311`。executor 已恢复 idle，未执行任何
实验、工件、Git、SSH、HPC2、Slurm、GPU 或 post-Gate-D 动作；该事件随后触发 10.12.7 的独立审计。

#### 10.12.7 Gate D 独立审计 PASS（2026-07-16）

规划/审计线程对成功重投递的同一终态包作了材料性校准的独立只读审计，Gate D 决定为 `PASS`：

- sender、successor recipient、执行期/路由恢复 control SHA 与 executor binding 一致；
- local、origin 与 HPC2 dedicated clone 均 clean `loopscope@1465c8019867560eeccc0f7e2845b36b50979ef7`，
  commit 只包含六个授权 Gate D 文件；`eval_runner.py`、wrapper/strategy/cache/config、AGENTS 与 lm-eval
  protected paths 对 Gate C base 无 diff；
- Slurm jobs `9986389` 与 `9986595` 分别 `COMPLETED/0:0`、49 秒与 268 秒、各一张 A40，当前 queue 空；
- D1/D2/D3/D4 关键文件 raw SHA256 全部与终态包一致；D2 summary 独立读取为 512 样本、ordered identity
  `7afc0fc1091c82f998e2984a4be45241d20bb51ca52f340ac9c869d0b8d874e8`、5120/5120 body calls、
  K1/K2/K3/K4 eRank steps=1/2/3/4、`vectors_persisted=false`；
- 14,042 full K1–K4 的 accuracy、entropy、top/correct margin、retention、三组 paired CI/transition 与
  wrong-overconfidence 直接读取后均匹配终态包；本地 targeted `10/10` 与 H2 card closure
  `5ff663d175937d9aa5de5c07f306a8b251aa2bad5ef90a866a114c63e21513ba` 通过；
- canonical repair verifier receipt 明确 `atlas_successful_write_count=1`，五个 source-recomputed output
  canonical SHA 与落盘值逐项相等，repo/card/atlas/science 均未修改。

D4 的额外计算在 materialization 前因 existing directory 失败，没有生成或覆盖任何工件；stock verifier
失败来自 JSON roundtrip 后 K key 的 integer/string 表示差异，修复 verifier 已证明科学值完全相同。因此
两项均是正常结果解释不受影响的工程偏差，不触发 `PASS_WITH_FIXES/BLOCK`。cosmetic EOF blank line与
Gate C 已接受的旧 H1 hash drift同样不阻塞。

科学结论保持探索性：fixed-horizon `12:15` 的 full accuracy 在 K2 达到观测峰值
`63.124911%`，K3/K4 降至 `62.925509%/62.704743%`；同时 entropy递增、margin递减，K 内后期
raw/applied entropy drop衰减并轻微反向，而 eRank继续增加。它支持“短暂精炼后过度精炼/扰动”的机制
描述，并进一步说明持续 residual、稳定 cosine 或更高 eRank并不足以保证额外 loop有益；不构成 selector、
跨窗口排名或因果证据。Gate D executor 从现在起撤权并关闭，Phase 2 补充实验完成，后续需另立 card。

## 11. 终态与审计契约

每个 Gate handoff 必须绑定 control SHA、executor ID、base commit、允许/禁止动作、最大新 run 数、write-once root、retry/repair budget 和终态接收线程。

执行线程只发送一次：

```text
decision requested: GATE_<X>_FINAL_AUDIT
result: SUCCESS_PENDING_PLANNING_AUDIT | BLOCK
control SHA256 and hypothesis card:
branch/base/final/dirty:
exact commands and exit codes:
jobs/run roots/artifacts/hashes:
observed result and scientific label candidate:
deviations/errors:
deliberately not executed:
```

规划线程重新只读核验 live Git、Slurm、样本 identity、配置/hash 与不可变工件后，才给 `PASS / PASS_WITH_FIXES / BLOCK`；executor 不自宣 PASS。

## 12. 决策记录

| 日期 | 决策 | 说明 |
|---|---|---|
| 2026-07-11 | Phase 1 closed | Gates A–E 与 workspace/runtime governance 全部 `PASS`；代码基线 `2c3acb3` |
| 2026-07-11 | Initial Phase 2 draft | 提出多模型信号竞赛；未提交、未授权、未执行 |
| 2026-07-12 | Phase 2 retargeted | 用户要求以周报 7.12 与线程后段“可迭代精炼区”假设为核心；改为 A–C focused mechanism cycle，旧大计划降为 optional backlog |
| 2026-07-13 | H1 V2 / NCA integrated | 用户授权在冻结前加入 NCA 作为非决定性方向代理；三窗口、两条轨迹和八个 full 上限不变；完整 NCA selector 延后为 Gate C 后独立 H2，Gate D 改为条件分支 |
| 2026-07-13 | H1/NCA decision rules concretized | 在冻结前补齐 six-contrast Holm family、标签优先级、NCA validity/interval 与 Gate D 穷尽分支；当前仍为未提交、未执行草案 |
| 2026-07-13 | Phase boundary pushed | annotated tag `loopscope-phase1-complete-20260711` 固定 `2c3acb3`；Phase 2 governance 以单独 commit `a81fba1` 推到 `origin/loopscope` |
| 2026-07-13 | Trace adapter scope clarified | stable policy commit `9e06dd3` 仅允许 `eval_runner.py` 显式 opt-in sidecar 接线，默认路径不变；`wrapper/strategies/cache/config` 继续只读 |
| 2026-07-13 | Gate A authorized | 绑定 executor `019f5c1d-7355-7130-81e2-125b4f1e21c0`，仅允许本地最小增量实现；Gate B 与全部 HPC2/GPU 动作仍锁定 |
| 2026-07-14 | Gate A BLOCK | 终态 sender 与 control binding 正确；独立审计确认 full residual/sample identity 架构未闭包，B1 producer、closed-world schema 与逐样本 provenance analysis 不完整；保留 dirty 草案，等待用户选择修复路线，同一 executor负责 Gate A repair |
| 2026-07-14 | Gate A repair authorized | 用户选择尺度分离：512 calibration 负责 residual/NCA，14,042 full 只负责 final-choice/gold outcome；同一 executor获得 local-only bounded repair，Gate B 仍锁定 |
| 2026-07-14 | Gate A repair audit BLOCK | 独立审计确认 local implementation 与测试大幅收敛，但 fixed-horizon family、完整判读规则、actual-run provenance、sealed root、B1→B2 admission、source-aware analysis verifier 与 canonical source binding仍未闭合；同一 executor获得第二次 local-only bounded repair，Gate B继续锁定 |
| 2026-07-14 | Gate A second-repair audit BLOCK | 独立审计确认 card/统计/identity/封存契约已收敛，但 full producer kind 可跨 cell 冒充 Phase 1 reuse、合法 reuse 无实际 source loader、analysis 不重验 full producer、probe evidence 缺同根路径闭包；card 不预注册，不自动授权下一次 repair，Gate B 继续锁定 |
| 2026-07-14 | Standing progression + Gate A provenance repair authorized | 用户授权规划/审计线程按既定 Gate 顺序持续推进、无需逐次确认；同一 Gate A executor获得最多两个 local engineering loops修复 cell-aware/source-aware/same-root provenance closure，仍禁止 push/SSH/GPU/Gate B |
| 2026-07-14 | Gate A provenance repair audit PASS_WITH_FIXES | 4+8 producer/source/path 主闭包已通过，但 raw-file loader 可被伪 `manifest_sha256` 绕过 exact file SHA；同一 executor 获得一次极窄 hash-semantics repair。用户要求 Gate A 完成后停止，Gate B 保持锁定 |
| 2026-07-14 | Audit materiality recalibrated; Gate A PASS | 用户要求以是否影响正常实验与科学结论为验收尺度；raw-file 对抗性加固降为 non-blocking debt，repair 已撤权并恢复 clean `ee1638c3`。Gate A关闭，card hash由本审计冻结/预注册；流程停止，Gate B 等待新授权 |
| 2026-07-14 | Continuous progression resumed; Gate B authorized | 用户恢复 Gate A 后续 Gate 并授权串行推进至第二阶段核心实现与实验完成；新 executor `019f5ebd-87d0-79a1-9633-3e67ad93d5b1` 绑定 B0→B1→B2，Gate C 仍待 Gate B 审计 PASS |
| 2026-07-14 | lm-eval framework frozen | 用户要求冻结 lm-eval package/framework/task/metric/eval semantics，所有兼容只能由 LoopScope-owned opt-in adapter/parser/sidecar完成；Gate B/C 增加 framework-freeze must-pass，无法单向适配则 BLOCK |
| 2026-07-14 | Gate B PASS; Gate C authorized | 独立审计核对 live Git/Slurm、B0 reuse、B1 K1/prefix、B2 sealed 512/NCA bootstrap与 lm-eval freeze，未发现材料性缺陷；新 executor `019f5f51-bc44-7e62-93e6-d7043e5bea1b` 获授权执行冻结八-cell focused full与一次机制分析，Gate D继续锁定 |
| 2026-07-14 | Gate C PASS; Phase 2 core complete | 独立核验八个 new full、四个 reuse、label-unseal、逐样本统计、NCA、Slurm、Git与 lm-eval freeze；科学结果为 `PERTURBATION + NCA_NONDISCRIMINATIVE`，按冻结矩阵停止 selector 扩展；routing-only implementation hash漂移记为非阻断 provenance偏差，不回写 post-unseal card或重跑分析 |
| 2026-07-16 | H2 fixed-horizon stepwise profile; Gate D authorized | 用户选择一次 12:15、fixed-horizon、K1–K4 的 512逐步机制补充实验，动态测量 raw/applied r-q-entropy-KL-eRank 与第二阶段方向/输出指标，并只读合并既有 14,042 full；独立 executor `019f668f-e98d-7033-a28a-36d55d0cf5ac` 绑定 D0→D4，0 个新 full、一个 A40；后续用户移除 GPU-hours 硬上限并以完成冻结实验为资源优先，selector与后续扩展继续锁定 |
| 2026-07-16 | Gate D routing repaired and PASS | successor planning thread 接收原 executor 重投的同一终态包；独立核对 Git/Slurm/5120-call/identity/hash/full statistics/verifier/protected state 后决定 PASS。额外 no-write recomputation 与 JSON key verifier false negative 为非材料偏差；Phase 2 补充实验关闭，后续新研究需独立 hypothesis card |
