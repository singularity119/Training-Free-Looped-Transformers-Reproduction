# LoopScope Phase 5 post-terminal Gate I：Adjacent Angular-Distance Diagnostic 冻结合同

```text
CONTRACT_ID=LOOPSCOPE_PHASE5_GATE_I_ADJACENT_ANGULAR_DISTANCE_V1
GATE=I
STATUS=FROZEN_FOR_EXECUTOR_BINDING
CLAIM_SCOPE=POST_TERMINAL_DIAGNOSTIC_ONLY_RETROSPECTIVE_MECHANISM_ANALYSIS
PLANNING_THREAD=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
EXECUTOR_THREAD=019fa464-6734-7ec0-a57a-e07a33223331
EXECUTOR_TITLE=execute-LoopScope-Angular-Trajectory-第5阶段-Gate I
AUTHORIZED_BASE_COMMIT=97de998504a9bad9867688f307a8bfc6fffdfb81
AUTHORIZED_RUN_ROOT=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-i-angular-trajectory-20260727T161957Z
```

## 1. 科学定位与禁止外推

本 Gate 只回答一个描述性机制问题：在两个已经完成的
`Qwen3-Base × MMLU 5-shot` cell 中，final answer position 上相邻 decoder
boundary 的 raw residual-stream 表示平均旋转幅度如何随深度变化。

本 Gate：

- 不是新的 selector、eligibility、ranking 或 loop-window Gate；
- 不读取或产生 MMLU test、gold、label、correctness、accuracy、gain、flip 或 loop outcome；
- 不允许用 angular-distance knee 修改 Phase 3/5 历史窗口、K、结论或 artifact；
- 只把 Kneedle 输出称为 `descriptive candidate boundary`；
- 即使曲线与历史窗口视觉重合，也只能报告 retrospective mechanism coherence。

## 2. 两个冻结 cell

共同任务合同：

```text
dataset=cais/mmlu
dataset_revision=c30699e8356da336a370243923dbaf21066bb9fe
task_group=mmlu
num_fewshot=5
fewshot_split=dev
trajectory_split=validation
population=1531 exact identities / 57 subjects
sampling=none
renderer=existing natural lm-eval 0.4.11 MMLU renderer
prompt_terminal=Answer:
position=final non-padding prompt token
forward=native no-loop
use_cache=false
output_hidden_states=true
loop_insertions=0
```

共同 renderer/identity closure：

```text
validation_projection_file_sha256=439e41113458ccfbd51d211ab7fae4e91ecc02723c7b14ad6c1969803aa61a6a
renderer_manifest_file_sha256=48ba67c204249b5f2f9a6cf711f8678711b5e606398d35ba4d919d4eee70e6e7
renderer_manifest_internal_sha256=e1baaa92c8ab3804c75a5e415a324d00fed0277d17c26b16d12b8c99aafbe949
renderer_source_sha256=105ff765d068bd246387249d3941a697c964070834d92d7a3268056b7a68a8a9
template_sha256=15dea12df4d7dc69b4d9d425a51bc14976f560b3e5ed52a319895cde79f2dabb
render_contract_sha256=eff9612c2984c7f2668e5cfb11d9eec89a5468c6f227ff94e4a226843043d6f4
dataset_fingerprint_sha256=c3f97b35254b7953d20cffc65414e1c78a792128f215805962c0db5d4d185e11
canonical_identity=(task,doc_id,doc_hash)
canonical_order=subject/task/doc_id/doc_hash lexicographic
```

输入 pool 必须只读复用：

```text
pool_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3b-20260715T223304Z
pool_file=validation1531_pool.jsonl
pool_manifest=validation1531_pool_manifest.json
source_manifest=source_manifest.json
```

执行线程必须在 smoke/formal 前从上述 canonical artifacts 重算并记录实际 file/internal
SHA256；数量、身份或 prompt hash 任一不闭合即 `BLOCK`，不得重新加载数据集生成替代 pool。

### 2.1 Qwen3-1.7B-Base

```text
model=Qwen/Qwen3-1.7B-Base
revision=ea980cb0a6c2ae4b936e82123acc929f1cec04c1
tokenizer_revision=same
dtype=float16
decoder_layers=28
boundaries=B_0...B_28
transitions=T_0...T_27 where T_l=B_l->B_(l+1)
model_snapshot=/hpc2hdd/home/xhuang225/shared/hf_home/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1
bootstrap=57-subject-stratified joint; R=2000; seed=20260716
```

### 2.2 Qwen3-4B-Base

```text
model=Qwen/Qwen3-4B-Base
revision=906bfd4b4dc7f14ee4320094d8b41684abff8539
tokenizer_revision=same
dtype=bfloat16
decoder_layers=36
boundaries=B_0...B_36
transitions=T_0...T_35 where T_l=B_l->B_(l+1)
model_snapshot=/hpc2hdd/home/xhuang225/shared/hf_home/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
bootstrap=57-subject-stratified joint; R=2000; seed=20260722
```

两模型的 bootstrap stream 必须独立；不得共享 draws 或直接比较未经归一化的绝对层号。

## 3. Raw residual boundary 合同

`B_0` 是 embedding output；`B_j` 是第 `j` 个 decoder layer 后、FinalNorm 前的 raw
residual-stream hidden。主指标不得使用 final-normalized lens-space 表示。

Qwen3 的 `outputs.hidden_states[-1]` 是 FinalNorm 后状态，不能直接作为 raw `B_L`。
最小合法实现为：

1. 使用 `outputs.hidden_states[:-1]` 取得 raw `B_0...B_(L-1)`；
2. 对 frozen `model.model.norm` 注册一个临时 forward-pre-hook，捕获它唯一一次调用的输入，
   作为 raw `B_L`；
3. hook call count 必须精确等于 1，并在 `finally` 中移除；
4. `FinalNorm(captured_raw_B_L)` 必须与 `outputs.hidden_states[-1]` 数值闭合；
5. 只在内存中保留本次 forward 的 hidden；约化完成后立即释放，禁止序列化。

不得把现有：

- `hidden_cosine_to_final`；
- `hidden_cosine_distance_to_final`；
- FinalNorm 后的 `z_l`；
- loop residual 或 repeated-call cosine

冒充本 Gate 的相邻层 raw angular distance。

## 4. Primary metric

对 identity `i`、transition `T_l=B_l->B_(l+1)`，在冻结 answer position 取 raw vectors
`h_(i,l)` 与 `h_(i,l+1)`：

```text
dot_i,l = sum_d float32(h_i,l,d) * float32(h_i,l+1,d)
norm_i,l = ||float32(h_i,l)||_2 * ||float32(h_i,l+1)||_2
cos_i,l = dot_i,l / norm_i,l
cos_clamped_i,l = min(1,max(-1,cos_i,l))
adjacent_angular_distance_i,l = acos(cos_clamped_i,l) / pi
```

数值合同：

- dot/norm reduction 使用 float32；
- `acos` 使用 Python/NumPy float64 scalar；
- 任一 zero norm、non-finite dot/norm/cosine/angle 立即 fail closed；
- scalar 必须位于 `[0,1]`；
- clamp 必须在 `acos` 前且只能 clamp 到 `[-1,1]`；
- 不允许先对 hidden/FinalNorm、对 cosine 求均值后再 `acos`，也不允许
  `1-cosine` 替代 angular distance。

Primary estimator 是 1,531 个 per-identity scalar 的 transition-wise micro mean。
每个 transition 同时报告：

```text
n=1531
subject_count=57
mean
sample_sd_ddof_1
median
q25
q75
bootstrap_ci95_low
bootstrap_ci95_high
```

median/Q25/Q75 使用 ascending sort 后 `q*(n-1)` 线性插值。CI 复用各自冻结的
subject-stratified joint bootstrap，所有 transition 共用同一 model-local replicate
index map；percentile 使用 ascending sort 后 `q*(R-1)` 线性插值。

Transition 的 normalized depth 冻结为 `l/L`，其中 `l=0...L-1`。该字段只用于跨模型
横轴对齐，不改变统计值。

## 5. Sanitized record 合同

每个模型必须写 1,531 条 sanitized per-identity records。允许字段仅包括：

```text
schema_version
model_key/model_repo/model_revision
identity{task,doc_id,doc_hash}
subject
split=validation
prompt_sha256
renderer_provenance hashes
answer_position
sequence_length
loop_insertions=0
boundary_count
transition_count
adjacent_angular_distance[length=L]
raw_residual_before_final_norm=true
```

禁止持久化：

```text
prompt text
target/gold/label/correctness/accuracy/gain/flip/outcome/test
raw hidden tensors
normalized hidden tensors
raw/full logits
full-vocabulary probabilities
choice probabilities
token ids or input ids
loop residuals
```

Verifier 必须 closed-world 检查未知字段和 forbidden token，并从 formal scalar records
独立重算 aggregate CSV、bootstrap CI、Kneedle summary 和 figure source-data。

## 6. Descriptive Kneedle

每模型分别在 unsmoothed transition mean curve 上运行 forward 和 tail-reversed 两次：

```text
x_l=l/L
curve=convex
direction=decreasing
interp_method=polynomial
polynomial_degree=2
online=true
S=1.0
implementation=kneed.KneeLocator
required_version=kneed==0.8.5
```

反向运行使用 ascending `x'_j=j/L` 与 `y'_j=y_(L-1-j)`；结果映射回原 transition
`l=L-1-j`。

点结果与稳定性：

1. 对同一方向的 degree-2 fitted curve，二阶导必须严格为正，且在完整 x-grid 上一阶导
   必须严格为负；否则 `NO_STABLE_KNEE_SHAPE_ASSUMPTION`。
2. `all_knees` 必须恰有一个与 x-grid 在 `1e-12` 内闭合的 knee；否则
   `NO_STABLE_KNEE_NON_UNIQUE`。
3. 对各自 2,000 个 frozen subject-stratified bootstrap mean curves 重新完整运行相同
   Kneedle；shape invalid、无 knee 或多 knee 均按该 replicate 无 winner，但仍计入总分母。
4. point mapped transition 的 exact bootstrap frequency 必须 `>=0.80`；等于 0.80 通过，
   否则 `NO_STABLE_KNEE_UNSTABLE`。

映射：

- forward knee at `T_l` -> candidate encoder/recursive boundary `B_(l+1)`；
- reverse knee mapped to `T_l` -> candidate recursive/decoder boundary `B_l`。

只有 forward/reverse 都稳定且 `B_(forward_l+1) < B_(reverse_l)` 时，才发布一个
descriptive recursive-block interval。否则 model-level status 为 `NO_STABLE_KNEE`，
并保留逐方向原因。raw decreasing-step fraction、quadratic coefficients、bootstrap
winner histogram 和 frequency 都只作诊断。

若 audited environment 缺少 `kneed==0.8.5`，允许 executor 在 Gate run root 的
`deps/` 中从国内 PyPI mirror 做一次 pinned、hash-recorded 的 Gate-local 安装；不得升级
共享 venv。安装失败则 `BLOCK`，不得静默改写算法。

## 7. Figure contract

单张两面板 composite figure：

- Panel A：x 为原始 transition index `l`，同时画 1.7B 与 4B mean；
- Panel B：x 为 normalized depth `l/L`，叠加两模型 mean；
- 实线和 hollow-circle marker 是未经平滑的真实 mean；
- 阴影是各模型独立 subject-stratified 95% CI；
- degree-2 fit 只用同色细虚线；
- 只对稳定 Kneedle candidate 画同色竖虚线；不稳定时标注 `NO_STABLE_KNEE`；
- 1.7B 使用 `#C98022`，4B 使用 `#3775BA`；
- 字体、spine、线宽、白底、SVG/PDF font contract 复用现有 Phase 5 trajectory figure；
- 不叠加 accuracy、gain、historical loop window 或 outcome 信息。

冻结尺寸与文件名：

```text
figure_size=7.2 x 3.4 inches
png_dpi=600
base=phase5_gate_i_adjacent_angular_distance_dual_model
formats=svg,pdf,png
source_data=angular_distance_composite_source_data.csv
```

## 8. Write-once outputs

Canonical Gate root：

```text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-i-angular-trajectory-20260727T161957Z
```

至少包含：

```text
qwen3_1p7b/sanitized_angular_distance_records.jsonl
qwen3_4b/sanitized_angular_distance_records.jsonl
angular_distance_by_transition_qwen3_1p7b.csv
angular_distance_by_transition_qwen3_4b.csv
angular_distance_kneedle_summary.json
angular_distance_composite_source_data.csv
phase5_gate_i_adjacent_angular_distance_dual_model.svg
phase5_gate_i_adjacent_angular_distance_dual_model.pdf
phase5_gate_i_adjacent_angular_distance_dual_model.png
angular_distance_summary_zh.md
angular_distance_verifier_receipt.json
phase5_gate_i_manifest_receipt.json
```

允许在所有 remote must-pass 闭合后，将报告、source-data 与 figure 三种格式只读复制到新的
本地 sibling：

```text
报告/figures/phase5_angular_trajectory/
报告/LoopScope_第五阶段GateI相邻层角距离诊断.md
```

不得覆盖任何既有 report/figure。

## 9. Acceptance 与科学停止规则

Must-pass：

1. exact cards、model revisions/snapshots、pool/renderer/identity hashes 闭合；
2. 每模型 4-identity smoke 证明 boundary/transition count、answer position、zero loop、
   FinalNorm-prehook raw `B_L` closure、finite range 和 forbidden-field absence；
3. formal 分别精确为 1,531 unique identities / 57 subjects，missing/duplicate/extra=0；
4. 每 identity 只发生一个 native forward；无 loop modules/wrappers；
5. 1.7B 每行 28 angles，4B 每行 36 angles，全部 finite `[0,1]`；
6. aggregate、model-local bootstrap CI、Kneedle 与 figure source-data 可由 verifier 重算；
7. SVG/PDF/PNG/source-data 均存在非空且 figure 不含 outcome/window overlay；
8. information barrier 字段明确为 false/zero；
9. independent verifier=`PASS`。

Kneedle 是否找到 stable knee 不是 Gate PASS 的先验要求；合法
`NO_STABLE_KNEE` 是可接受科学结果。任何对模型、数据、split、position、raw-vs-normalized
metric、population 的改变都必须 `BLOCK`。

Gate I terminal 后停止；不授权 Gate J、selector 修改、loop/outcome 或额外模型。
