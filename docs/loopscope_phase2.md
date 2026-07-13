# LoopScope Phase 2：可迭代精炼区 H1 V2 实验 Runbook

本文件说明第二阶段轻量 Gate 的执行对象、证据与禁止项。它不是当前授权；动态状态只看 `../../.planning/loopscope_phase2_control.md`，长期规则只看 `../AGENTS.md`。

## 1. 核心问题

Phase 2 首先不做通用自动选窗，而检验：

> `12:15` 是否属于“可迭代精炼区”——block 在普通单次前向中已经参与任务决策，但重复调用时 residual 不会迅速消失，并且额外计算产生的是持续有益修正而非持续扰动。

H1 V2 拟在 Gate A 把 `Native-Continuation Alignment (NCA)` 冻结为无标签方向代理；只有 Gate A 审计 `PASS` 后才可称为预注册。NCA 只检查 loop residual 是否与原模型自身的下游延续方向一致；它不是正确性标签，也不在 H1 中排名、筛窗或决定部署。

三个局部窗口形成最小对照：

- `11:14`：`q≈1`，K=2 小幅正收益；
- `12:15`：`q≈1`，baseline entropy 适度下降，eRank 增长，K=2 当前最好；
- `13:16`：同样 `q≈1`、eRank 增长，但 K=2 明显负收益，是“方向有益”条件的关键 falsifier。

`4:7、15:18、22:25` 只复用现有 K=2 工件作为反例，不默认新增 K sweep。

## 2. 全局路径与保护

```text
local repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
remote repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
read-only reproduction repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction
Phase 2 workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope
Phase 1 input=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/inputs/loopscope-qwen17-mmlu-phase1-20260711-043615
Phase 1 run=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-20260711-053022
reusable venv=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711
shared caches=/hpc2hdd/home/xhuang225/shared/{hf_home,datasets,uv}
```

Phase 1 run 永久只读。所有新 attempt、trace、results 和 analysis 使用 Phase 2 独立 workspace 下的 timestamped write-once root。

## 3. 测量定义

### 3.1 Native-Continuation Alignment

对 inclusive window `w=[a,b]`，`B_j` 表示 decoder layers `0..j-1` 后的边界状态；`p` 是由冻结 renderer/tokenizer 规则确定的 final pre-answer prompt token。对样本 `i`：

```text
c_(w,i) = B_N[i,p,:] - B_(b+1)[i,p,:]
NCA_(w,i,t) = cos(delta_(w,i,t)[p,:], c_(w,i))
```

- body-call index 固定为 zero-based `t=0..K-1`，其中 `t=0` 是 initial call；baseline NCA 使用普通单次前向的 window update `B_(b+1)-B_a`；actual-loop NCA 只汇总真实重复调用的 `t=1..K-1` residual，二者必须分别标记 provenance A/B。`K=1` 没有 repeated-step NCA，只用于等价性基线；
- 向量在设备内以 float32 临时计算，只序列化 per-sample NCA、`||c||`、`||delta||`、validity mask、位置/边界 identity 和汇总值；
- `||c||` 或 `||delta||` 近零、NaN/Inf、shape/dtype/device、token 位置或 boundary identity 不一致时必须 invalid/fail-fast，不得用普通零分掩盖；
- 窗口汇总先在每样本声明的 step set 内取中位数，再跨样本取中位数；置信区间使用 10,000 次 sample-index bootstrap、seed 0；
- NCA 正值只表示 native-path alignment。它可能保持正确 baseline，也可能强化错误 baseline，因此最终方向是否有益仍由 final-output 和 gold-label paired analysis 裁决。

H1 的 mandatory NCA 范围是冻结 512 校准池与 `11:14、12:15、13:16`。若同一次 baseline pass 已得到所有冻结候选窗口的边界状态，可额外输出其他窗口的 baseline-only NCA scalar，但必须标记 exploratory，且不得据此改变 Gate C 窗口、K 或 full 顺序。

### 3.2 实际 loop residual 轨迹

对窗口 block 映射 `G`，第 `t` 次实际调用前状态为 `x_t`：

```text
delta_t = G(x_t) - x_t
q_t = ||delta_(t+1)||_2 / ||delta_t||_2
```

同时记录：

- `||delta_t|| / ||x_t||`；
- `cos(delta_t, delta_(t+1))`；
- 实际 operator body-call index 与总调用数；
- NaN/Inf、shape、dtype、device 和恢复状态。

只在设备上计算 answer-position scalar 后写入；本 H1 的 residual、`q_t`、相邻方向与 NCA 只在冻结 512 calibration 上采集。14,042 full 不含任何 residual/NCA 字段，也不研究 full residual–答案翻转个体相关性。完整 residual/hidden/native-continuation tensor 在两个尺度都不得持久化。

`q_t≈1` 只说明 residual 强度持续，不说明方向有益。更新方向是否有益必须由最终输出和 gold-label paired analysis 判定。

### 3.3 每个 K 的最终输出轨迹

对完整模型前向结束后的四个候选答案 choice scores 归一化为 `P_K`，计算：

```text
final_entropy(K) = -sum_c P_K(c) log P_K(c)
top_margin(K) = score(top1) - score(top2)
js_to_k1(K) = JS(P_K, P_1)
top1_retained(K) = [argmax P_K == argmax P_1]
```

这些指标来自最终模型输出，不需要把中间 block hidden state 当成可直接解码表示，是 H1 的主要置信度/决策证据。

两个尺度的 score provenance 必须分开：512 direct probe 使用 final pre-answer token 上、冻结 A/B/C/D choice token 的 next-token log probability，字段值固定为 `direct_probe_next_token_log_probability_over_frozen_choice_token_ids`；14,042 full 使用 lm-eval `acc,none` 实际消费的四个 raw per-choice log-likelihood，字段值固定为 `lm_eval_acc_none_raw_per_choice_loglikelihood`。前者不得冒充后者；跨尺度不做 identity bridge 或逐样本 join。

### 3.4 Gold-label paired 分析

card 与无标签字段冻结后才解封 label：

- `wrong→right`：额外计算纠正错误；
- `right→wrong`：额外计算破坏正确答案；
- `right→right` / `wrong→wrong`；
- `correct_margin = score(gold) - max score(non-gold)`；
- 错误样本 entropy 下降且错误 top1 margin 上升的“错误过度自信”比例。

同时计算累计（相对 baseline）和 incremental（相邻 continuation 点）净纠错率。六个三窗口 prospective incremental contrasts（每窗口 `K2→K3`、`K3→K4`）构成唯一 primary confirmatory family，使用 two-sided exact McNemar 与 Holm step-down、family-wise `alpha=0.05`。`12:15` fixed-horizon controls 构成独立、内部 Holm 校正的 secondary control family，且仅含 `K2→K3` 与 `K3→K4`；`K1→K2` 只属于 B1 等价性与 cumulative context，不进入 Holm family。cumulative-vs-baseline / `K4-vs-K2` 只作 secondary sign guard/context，不能单独触发 `REFINEMENT_SUPPORTED`。净纠错率与 accuracy delta 数值相关，但四类转移和条件分布揭示机制，不能只报告 aggregate accuracy。

冻结 512 NCA 的公式、位置、validity、汇总和 artifact hash 后，才可解封该池 label，作 baseline-correct/wrong 与四类转移的 development-only 条件分析。不得用这些 label 调 NCA 阈值或重新选窗。full paired outcome 继续使用 2,000 次 bootstrap、seed 20260710，不得和 NCA 的 10,000/seed 0 混成同一统计 family。

### 3.5 预注册科学判读规则（Gate A 冻结候选）

Gate A 必须把本节规则、优先级和实现 hash 写入 versioned config；审计 `PASS` 后才成为预注册判据。这里的 K2/K3/K4 只指 primary fixed-step `(2,1)/(3,1.5)/(4,2)`。`U24` 是 `12:15` 的 d24 使用 full paired bootstrap（2,000、seed 20260710）所得 95% percentile interval 上界。令：

```text
d23(w) = acc(w,K3) - acc(w,K2)
d34(w) = acc(w,K4) - acc(w,K3)
d24(w) = acc(w,K4) - acc(w,K2)
```

`p_Holm` 是六项 primary family 的 Holm-adjusted two-sided exact McNemar p-value。工件、identity 或统计字段不完整时 Gate `BLOCK`，不分配科学标签；完整数据按以下优先级只取第一个匹配项：

1. `PERTURBATION`：`12:15` 的 `d23` 或 `d34` 为负、对应 `p_Holm<0.05`，且该 contrast 的 `right→wrong > wrong→right`。
2. `REFINEMENT_SUPPORTED`：未命中 `PERTURBATION`；`d23(12:15)>0` 且 `p_Holm<0.05`；`d24(12:15)>=0`。
3. `TRANSIENT_ONLY`：未命中前两项，Phase 1 已知 `K2-vs-baseline>0`，但 `U24<0`；仅 d24 点估计为负而 interval 跨 0 时不得使用本标签。
4. `SUGGESTIVE`：未命中前三项；`d23(12:15)>0`、`d24(12:15)>=0`，但 `12:15` 的 `d23` 未达到 `p_Holm<0.05`。
5. `INCONCLUSIVE`：其余完整但矛盾或低于当前统计分辨率的轨迹。

`q`、JS、entropy、margin、错误过度自信、fixed-horizon family 与 cumulative contrasts 仍全部报告，但只作机制解释、反证定位或 secondary guard，不能增加临时主判据。邻窗继续作预注册 controls/falsifier，但“12:15 显著、邻窗不显著”不是窗口间显著差异；H1 outcome 不声称 `12:15` 唯一或显著优于邻窗。

K4 saturation 单独报告：若 `d34>0` 且对应 `p_Holm<0.05`，表示截至 K4 仍在改善、尚未观察到饱和，不算 H1 失败，也不自动授权 K>4；未显著也不能在没有预设 equivalence margin 时宣称已饱和。

NCA diagnosis 不依赖 H1 outcome，必须独立输出。对 fixed-step primary cells `s∈{(2,1),(3,1.5),(4,2)}`，每个 sample-cell 只有声明的全部 repeated steps 都 valid 时才 valid；先对这些 steps 的 NCA 取中位数，再跨样本取中位数。用 10,000 次、seed 0 的同组配对 sample-index bootstrap 计算 `12:15` 中位数及 paired-valid intersection 上 `NCA(12:15)-NCA(13:16)` 的 95% percentile interval，lower bound 记为 `L`。native-fidelity 条件分析还用同一预算的 stratified bootstrap 计算 `right→right` 组的 NCA interval，以及 `median_NCA(wrong→right)-median_NCA(wrong→wrong)` 的 interval；只有 `right→right、wrong→right、wrong→wrong` 三个子组都至少有 10 个 valid samples 时该 cell 才合格。完整数据按以下优先级只取第一个匹配项：

1. `NCA_INCONCLUSIVE`：任一三窗口 × 三 primary cell 的 valid fraction `<0.95`，任一 primary paired-valid intersection `<0.95`，或 primary overall/difference interval 缺失/非有限。
2. `NCA_NATIVE_FIDELITY_ONLY`：三个 `12:15` cell 均有 `L_NCA>0`；至少两个合格 cell 同时满足 `L_NCA(right→right)>0` 与 `U_[NCA(wrong→right)-NCA(wrong→wrong)]<=0`。
3. `NCA_DIRECTION_SUPPORTED`：未命中第 2 项；三个 `12:15` cell 均有 `L_NCA>0`；三个 paired `12:15-13:16` interval 中至少两个 `L_diff>0`。
4. `NCA_NONDISCRIMINATIVE`：其余有效但不能同时证明正向、相对 harmful 窗口区分力与净纠错一致性的情况。

若 native-fidelity 合格 cell 少于两个，另报 `native_fidelity_check=NOT_EVALUABLE`；它不把 primary NCA 诊断改成 `NCA_INCONCLUSIVE`，继续按第 3–4 项判定。NCA intervals 均为 pointwise、未校正的 secondary diagnostic intervals，不构成独立 FWER-controlled confirmatory rejection，也不能代替 H1 的 gold-label primary family。

### 3.6 Baseline layer probe 的限制

Phase 1 的 entropy drop、KL-to-final 和 eRank 来自普通单次前向的边界 `B_a → B_(b+1)`，不是 loop 后轨迹。它们只作为 H1 的起始状态描述。

若 Gate A 额外采集中间每轮 hidden state 并经 final norm/lm_head 解码，字段必须命名 `intermediate_lens_*`，单独验证并只作辅助；不得与 `final_entropy(K)`、`top_margin(K)` 混合。

## 4. Core experiment card

```text
card=H1_ITERATIVE_REFINEMENT_ZONE_V2
model=Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1
task=mmlu
num_fewshot=5
dtype=float16
windows=11:14,12:15,13:16
iteration_mode=block
strategy=damped_euler
beta=0.0
cache_strategy=last
decode_mode=bypass
primary_continuation=(k,alpha)=(2,1.0),(3,1.5),(4,2.0)
primary_step_size=alpha/k=0.5
fixed_horizon_control=(k,alpha)=(1,1.0),(2,1.0),(3,1.0),(4,1.0)
calibration=Phase 1 frozen validation 512
nca_position=final_pre_answer_prompt_token
nca_continuation=B_N-B_(b+1)
nca_bootstrap=10000
nca_seed=0
nca_role=prospective_secondary_direction_proxy; never selector in H1
full_identity=Phase 1 ordered 14042 samples
bootstrap=2000
seed=20260710
```

当前 damped Euler 实现使用 `step=alpha/k`。主 continuation 轨迹把 step 固定为 Phase 1 的 `0.5`，用 K/alpha 成对变化来延长迭代时长；fixed-horizon control 固定 `alpha=1`，只改变子步分辨率。两条轨迹必须分别命名和分析。width、window、模型、任务、样本、decode/cache/strategy 均不能因 limit/full 结果而调整。

每个 config、output dir、claim 和 analysis key 必须同时编码 `protocol、window、k、alpha`，不得只用 K 命名。

## 5. Gate A — 轻量轨迹/NCA 实现与 H1 V2 冻结

### 目标

只补齐 H1 V2 所需的最小 residual、NCA、final-output 与 paired-analysis 能力，不建设通用多模型或全窗口 selector 平台。

### 必做

1. 版本化 H1 V2 config/card、schema、上述 H1/NCA 判读优先级与 deterministic hash。
2. 实现冻结 final pre-answer token、inclusive boundary 与 native continuation `B_N-B_(b+1)` 的最小 NCA 采集路径。
3. 最小方式采集每个实际 body call 的 residual norm、相邻 residual ratio/cosine、NCA 和调用序号；只保存 scalar/norm/validity，不保存完整向量。
4. 在 512 保存 direct-probe final choice，在 full final-output adapter 保存 evaluator raw choice；两者分别绑定独立 canonical identity namespace、card/revision/cell 与 attempt/receipt provenance。
5. 实现逐样本四类翻转、entropy/margin/JS 条件分析、paired bootstrap/McNemar，以及独立 NCA diagnosis。
6. 只读审计 Phase 1 baseline/K=2 与 512 probe 是否包含可复用 final choice/boundary 字段，输出 reuse matrix；缺字段只报告，不自动补跑。
   本地只能确认 Phase 1 schema 不持久化 Phase 2 per-step NCA；512 live identity、14,042
   ordered identity 与 raw four-choice availability 均保持 `requires_gate_b_live_check`，不得把
   “schema 可描述”写成“live artifact 已验证”。
7. 单元测试必须覆盖：
   - inclusive window 与 body-call count；
   - final pre-answer token、`B_(b+1)`/`B_N` boundary identity 和 NCA 方向；
   - NCA float32 cosine、near-zero/non-finite invalid、shape/dtype/device mismatch fail-fast；
   - baseline NCA 与 actual-loop NCA provenance 不可混淆；
   - `(K=1,alpha=1)` 与普通 operator 的等价语义 fixture；
   - fixed-step 与 fixed-horizon config 不可混淆；
   - `q_t` 零分母/NaN fail-fast；
   - final-output 与 intermediate-lens 字段不可混用；
   - ordered identity mismatch 拒绝分析；
   - frozen pool exact schema、sealed-label guard、B1 first-four→B2 source binding；
   - actual argv/attempt/receipt/command/env/revision/results provenance 与 new-root 写入；
   - per-cell wall-clock/peak-memory/restore proof schema；
   - B2 aggregate identity/baseline/15-cell path+hash closure 与 source-aware report 重算；
   - wrong/right 四类计数和 margin 公式；
   - six-contrast Holm family、H1/NCA 标签优先级与边界条件；
   - Phase 1 schema 向后兼容。

### 允许与禁止

- 允许：`src/tflt/loopscope/`、`configs/loopscope/`、`scripts/loopscope/`、tests、CLI/docs 的最小增量。
- `wrapper.py` 若确需新增只读 audit payload，必须在 Gate A handoff 中列出精确最小 diff、保持 collector=None 路径行为不变并加回归测试；未授权不得改。
- 禁止：SSH、模型/数据下载、GPU、Slurm、修改 Phase 1 工件、实现全窗口 selector/no-loop verifier 或 16-window/多模型框架。

### 本地验证

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m compileall -q src tests scripts/loopscope
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tflt.cli --help
```

退出包必须包含 commit/diff、full tests、card/config/schema hashes、reuse matrix 和未执行事项。

## 6. Gate B — HPC2 CPU + smoke/limit + bounded NCA calibration

### Gate A 实现接口（freeze candidate）

Gate A 的本地实现提供五个显式接口，默认 Phase 1/eval 路径不启用任何
Phase 2 行为：

```bash
python -m tflt.cli prepare-phase2-h1 \
  --card configs/loopscope/qwen17_mmlu_phase2_h1_v2.json \
  --output-dir <new-local-output>

python -m tflt.cli probe-phase2-trajectory \
  --probe-mode b1-smoke --max-examples 4 \
  --model qwen3-1.7b-base \
  --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --card configs/loopscope/qwen17_mmlu_phase2_h1_v2.json \
  --input-jsonl <frozen-probe-pool.jsonl> \
  --input-manifest <frozen-probe-pool-manifest.json> \
  --attempt-manifest <write-once-attempt.json> \
  --receipt-manifest <write-once-receipt.json> \
  --output-dir <new-write-once-output> \
  --dtype float16 --device cuda

python -m tflt.cli probe-phase2-trajectory \
  --probe-mode b2-calibration \
  --b1-admission-proof <audited-b1-admission-proof.json> \
  --model qwen3-1.7b-base \
  --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --card configs/loopscope/qwen17_mmlu_phase2_h1_v2.json \
  --input-jsonl <frozen-probe-pool.jsonl> \
  --input-manifest <frozen-probe-pool-manifest.json> \
  --attempt-manifest <write-once-attempt.json> \
  --receipt-manifest <write-once-receipt.json> \
  --output-dir <new-write-once-output> \
  --dtype float16 --device cuda

python -m tflt.cli validate-phase2-trace \
  --trace <scalar-envelope.json> \
  --card configs/loopscope/qwen17_mmlu_phase2_h1_v2.json \
  --identity-manifest <canonical-identity-manifest.json>
python -m tflt.cli analyze-phase2-h1 --input <complete-analysis-input.json> \
  --output-dir <new-write-once-analysis>
python -m tflt.cli verify-phase2-analysis \
  --input <complete-analysis-input.json> \
  --report <new-write-once-analysis/phase2_h1_analysis.json>
```

`probe-phase2-trajectory` 是 Gate B 才可执行的 remote-only producer；Gate A
只实现并用 fake/pure-Python fixture 测试。`b1-smoke` 在一个 model process 中对四个
样本实际执行 no-loop、三个 K1 admission cells 与 15 个 logical cells，写出 K1 choice
equivalence 及 K2 对 K3/K4 的 state/residual prefix scalar proof；只有该 proof 经审计后，
`b2-calibration` 才在同一进程内对冻结 512 先执行一次 no-loop boundary pass，再执行
15 个 logical cells。B1 proof 绑定同一 Phase 1 frozen 512 source manifest、确定性前四
identity、renderer subset、card/revision 与实际 producer；B2 admission 必须逐项核对这些
绑定，不能用任意四样本 proof 解锁。三个窗口的 `B_N-B_(b+1)` 始终只保留在内存。
写盘内容限于 identity、norm、ratio、cosine、NCA validity/scalar 与 final four-choice
scores/probabilities；aggregate 另以 closed-world schema 记录每 cell wall-clock、peak GPU
memory 和 wrapper restore proof。完整 hidden、residual 和 native-continuation vectors
禁止持久化。

`python -m tflt.eval_runner` 另提供显式 opt-in
`--phase2-final-output-manifest <versioned-json>`，只从 lm-eval completed logged samples 生成独立
`phase2_final_outputs.json`，不修改 `results.json`。参数未出现时使用
`argparse.SUPPRESS`，因此既有 `command_args.json`、`LoopConfig.audit_collector=None`、
HFLM/TaskManager/simple_evaluate 和结果语义保持不变。adapter 不接 wrapper collector，
也不按 callback/batch 顺序关联；它把 evaluator 任意顺序的 logged rows 与 canonical
full `task/doc_id/doc_hash` manifest 作唯一、完整、无额外项的 exact join，再恢复 natural
order。因此 `batch_size=auto` 本身不是 blocker；缺 identity、duplicate/extra row 或缺四个
raw choice scores 才 fail-fast。Phase 1 full artifacts 是否具备 raw choice/doc_hash 仍属于
`requires_gate_b_live_check`，不能由本地 schema 宣称已验证。

adapter request 使用 v2 path+hash refs 实际读取 card、full identity、attempt 和 receipt；运行前
把 requested cell 与真实 model/revision/task/fewshot/dtype 及 active loop argv 逐字段核对，
baseline 的 inactive loop defaults 不冒充 active science。运行后再读取 write-once
`command_args.json、env.json、model_revision.json、results.json`，核对 model/tokenizer revision
closure，并由这些实际文件构造 producer hashes。adapter 模式要求新 output root，所有上述
文件与 sidecar 都使用 exclusive-create；未启用 adapter 的默认 eval 路径保持原行为。
attempt/receipt 还必须把真实 `output_root` 绑定到 card 冻结的独立
`training_free_looped_transformers_loopscope` workspace；resolved path 落入旧 Phase 1 workspace、
`/tmp` 或经 symlink 逃逸都会在 model load 前失败。

`complete-analysis-input.json` 只能列出 card、两个 canonical identity manifest、sealed B2
aggregate、baseline/15 calibration cells、12 个 full final-output cells、authorized calibration gold-index
sidecar及 unseal authorization/receipt 的 path+hash；禁止提供预聚合 delta、p-value、Holm
或 NCA summary。B2 aggregate 的 artifact root、identity/baseline/15-cell refs 必须与实际加载
路径和 hash 完全一致；probe pool 递归拒绝 `gold_index/correctness`。calibration identity 绑定
Phase 1 frozen pool manifest、renderer subset 与 dataset revision；full identity 绑定
`cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`、5-shot natural order 与 Phase 1
run/input root。source-provenance v2 不只核对 source manifest 元数据：calibration 必须从
`rendering_records[task_name,target_doc_id,target_doc_sha256]` 逐项派生 512 identity；full 必须
从逐文件哈希的 immutable Phase 1 `gate-e-full/baseline-full/results.json` 中
`samples` 映射逐项派生；若该历史结果不内嵌 `samples`，则按 Phase 1 loader 契约使用同目录
完整、排序且逐文件哈希的 `samples_*.jsonl` 集合。两条路径都按 task 名排序、task 内保留
logged-sample 顺序，派生 14,042 identity，并与
canonical manifest 完全同序。尚未在 Gate B live 核验的 digest 只能保持
`requires_gate_b_live_check` 和 null，不能虚构。analysis producer重新从逐样本 sidecar计算
全部统计，并以 same-directory atomic exclusive-create + fsync 写出报告；随后必须运行
`verify-phase2-analysis`，从相同 source 重新确定性计算并要求报告逐字段及 manifest hash 相等。
analysis input 的 authorization/attempt/receipt 均为实际 path+hash refs，三者绑定 card、完整
source-ref tree、executor 与独立 workspace output root；CLI write-once 生成自己的
`command_args.json/env.json`，verifier 重新读取并核对后才重算报告。

B1/B2 probe 的 attempt/receipt 同样使用 exact versioned schema，绑定 probe mode、4/512
尺度、card/revision、Phase 1 input manifest、executor 与 output root。aggregate 与 B1 proof
保存实际 attempt/receipt/input/command/env/revision path+hash；B2 admission 还会从 512 pool
重新计算 natural-order 前四记录的 selected/render hashes，不能用任意四样本 proof 解锁。
合法 renderer `target={source,split,subject}` 仅作无标签 provenance；任何 target answer、
`gold_index`、correctness 或 evaluator accuracy 仍被 sealed schema 拒绝。

错误过度自信是非决定性机制诊断：冻结候选将分母明确为 final-wrong pairs
（`wrong→wrong + right→wrong`），条件为 final entropy 下降且 raw top margin 上升，
并分别报告两个 subgroup。该指标不参与 H1 或 NCA 标签。

Gate B 在一个 executor 内按 B0→B1→B2 顺序执行；前一子门失败不得进入下一子门。

### B0 CPU/provenance

- fast-forward 同步 Gate A 已审计 commit；
- local/origin/HPC2 clean closure；
- venv/lock/version/import/unit/compile/help；
- model/tokenizer/dataset revision 与 frozen 512/14042 identity；
- 只读确认 Phase 1 reuse matrix。

### B1 最小 GPU/limit

1. 四样本覆盖三窗口的主 continuation `(2,1)/(3,1.5)/(4,2)`，以及 fixed-horizon `(1,1)/(2,1)/(3,1)/(4,1)`，验证 body-call count、实际 step、residual/NCA trace、final outputs、restore、无 NaN/OOM/cache 错误。
2. 证明 `(K=1,alpha=1)` 与普通 baseline 在同样 prompt 上 final choice scores allclose、top1 和 sample identity 一致；不成立即 `BLOCK`。
3. 对 primary continuation 验证 prefix consistency：在内存中比较 K=3/4 前两轮 residual/state tensors 与 K=2 allclose，只把 max-abs/scalar proof 写入工件；不成立则不能称为延长同一轨迹并立即 `BLOCK`。
4. 验证 NCA 的 token position、inclusive boundary、float32 scalar、validity mask、baseline/actual-loop provenance 与 10,000/seed 0 bootstrap schema。
5. limit 工程覆盖两条轨迹，验证 command/results/trace schema、ordered identity 与 revision closure；limit accuracy 不得用于改 card。`alpha=2` 若产生有限但有害输出属于科学证据，NaN/Inf/schema 失败才是工程 blocker。
6. 记录每配置峰值显存和墙钟，给 B2 与 Gate C 计算资源上限。

### B2 冻结 512 无标签 NCA calibration

只有 B1 工程审计闭环后才可执行：

1. 使用 Phase 1 frozen validation 512 的精确 ordered identity，label 保持 sealed；
2. mandatory actual-loop NCA 只覆盖 `11:14、12:15、13:16` 与五个唯一设置 `(2,1)、(3,1)、(4,1)、(3,1.5)、(4,2)`；共享 `(2,1)` execution 只运行/引用一次，`(1,1)` 只作 baseline equivalence；
3. 先执行一次可复用的 deterministic no-loop boundary pass，再写入 timestamped write-once NCA root；不得覆盖 Phase 1 probe；
4. B2 最多 `1` 个 no-loop boundary cell + `3 windows × 5 unique loop settings = 15` 个 loop cells，使用一个现有 GPU，总计不超过 18 GPU-hours；B1 预测超界时在 B2 前 `BLOCK`，不得静默缩样本、改设置或扩资源；
5. 在 label sealed 状态输出 per-sample scalar/norm/validity 与 final choice scores/answer，并输出 bootstrap interval、activity/restore/revision/provenance 和资源记录；解封后只能按冻结规则做条件分析；
6. B2 的 NCA、limit accuracy 或任何 exploratory baseline-only window 不得改变 Gate C 三窗口、K、full 顺序或八个 full 上限。

### 禁止

- full MMLU；
- K=2 全量补跑；
- 改 windows/K/指标；
- 在 NCA 规则与 artifact hash 冻结前读取 512 label；
- 执行全窗口 actual-loop shortlist、selector/no-loop verifier 或 window-identity permutation；
- 自动下载、改 venv、静默换 partition/GPU 科学契约；
- 根据 limit accuracy 或 NCA 调整 Gate C 顺序。

## 7. Gate C — 三窗口 focused full

### Admission

- Gate B 独立审计 `PASS`；
- H1 card、reuse matrix、full manifest、统计脚本与 write-once root 已冻结；
- `(K=1,alpha=1)` equivalence 成立；
- Phase 1 baseline/`(K=2,alpha=1)` 的实际复用字段已明确。
- B2 512 NCA artifact 已审计；其结果无论正负都不得回改 Gate C 矩阵。

### 默认最大新计算

```text
primary continuation:
  11:14 at (K=3,alpha=1.5), (K=4,alpha=2.0)
  12:15 at (K=3,alpha=1.5), (K=4,alpha=2.0)
  13:16 at (K=3,alpha=1.5), (K=4,alpha=2.0)
fixed-horizon control:
  12:15 at (K=3,alpha=1.0), (K=4,alpha=1.0)
```

原则上共八个新 full 配置。`(K=1,alpha=1)` 使用经 Gate B 证明等价的 baseline；`(K=2,alpha=1)` 使用 Phase 1 immutable results。任何 K=2 trace-only/full 补跑、额外窗口、K>4 或其他 alpha 都需 control 新授权。

### 分析顺序

1. 完成所有新结果并核对 Slurm/job/log/revision/sample identity；
2. 合并 Phase 1 reuse data，不复制覆盖旧工件；
3. 只执行一次 write-once mechanism analysis；
4. 分别输出 fixed-step continuation 与 fixed-horizon control，报告 paired CI/McNemar/Holm、四类翻转与 full entropy/margin/JS；residual/q/NCA 机制解释只引用独立、已审计的 B2 512 evidence；
5. 对 `REFINEMENT_SUPPORTED / SUGGESTIVE / TRANSIENT_ONLY / PERTURBATION / INCONCLUSIVE` 给出 H1 证据矩阵；
6. 独立给出 `NCA_DIRECTION_SUPPORTED / NCA_NONDISCRIMINATIVE / NCA_NATIVE_FIDELITY_ONLY / NCA_INCONCLUSIVE` 诊断矩阵；executor 不自宣科学结论或 Gate PASS。

### 完成边界

Gate C 结束后不自动进入多模型、MMLU-Pro、selector 或 Gate D。规划线程先独立审计 H1 outcome 与 NCA diagnosis，再由用户选择条件分支。

## 8. Gate D — Gate C 后的可选条件分支

Gate D 不是核心完成条件，也没有预授权。Gate C 审计后按以下互斥矩阵只选择一个方向：

- **H1=`TRANSIENT_ONLY/PERTURBATION`，任意 NCA**：停止 selector 扩展；NCA 若 inconclusive 只记为诊断缺口，不改变 H1 negative 路由；
- **H1=`SUGGESTIVE/INCONCLUSIVE`，任意 NCA**：不得自动扩展；用户只能停止或另立一张有固定预算的最小 H1 消歧卡；
- **H1=`REFINEMENT_SUPPORTED` 且 NCA=`NCA_INCONCLUSIVE`**：停止或另立最小 NCA 消歧卡；
- **H1=`REFINEMENT_SUPPORTED` 且 NCA=`NCA_DIRECTION_SUPPORTED`**：另立 `H2_NCA_CERTIFICATE` card，才允许全窗口 baseline ranking、top-3、唯一胜者/no-loop verifier、window-identity permutation 和独立预算；
- **H1=`REFINEMENT_SUPPORTED` 且 NCA=`NCA_NONDISCRIMINATIVE/NCA_NATIVE_FIDELITY_ONLY`**：同模型一个新任务，或同任务一个新模型，做最小机制确认。

规划线程为被选分支新建独立 card，重新冻结 calibration/eval split、真正未见的 held-out 对象、最小窗口/K、统计功效和最大 full-run 数。Phase 1 已看过 label 的窗口只能作 development controls，不得冒充 held-out；不得默认展开旧版三模型×两任务矩阵。

## 9. 新 hypothesis card 的敏捷模板

```text
CARD_<ID>
Question:
Why existing artifacts cannot answer it:
Model/task/window/K:
Metric source: baseline layer | actual loop | final output | gold adjudication
Primary observable:
Falsifier/counterexample:
Existing artifacts to reuse:
Minimum new smoke/full computation:
Maximum new full-run count:
Frozen analysis and decision labels:
Files/code allowed:
Forbidden expansion:
Terminal audit recipient:
```

只读分析卡可直接进入一次审计 Gate；新增字段先本地/小样本；只有无法用已有工件回答时才 full。

## 10. 终态包

```text
decision requested: GATE_<X>_FINAL_AUDIT
result: SUCCESS_PENDING_PLANNING_AUDIT | BLOCK
control SHA256 and card ID:
executor identity:
branch/base/final/dirty:
files changed:
exact commands and exit codes:
jobs/run roots/artifact hashes:
reuse versus newly produced evidence:
sample/revision/freeze closure:
observed result and candidate scientific label:
deviations/errors/repair loops:
deliberately not executed:
```

执行线程不自宣 PASS。规划线程收到终态事件后重新读取 live Git、Slurm 与工件，再决定 `PASS / PASS_WITH_FIXES / BLOCK`。
