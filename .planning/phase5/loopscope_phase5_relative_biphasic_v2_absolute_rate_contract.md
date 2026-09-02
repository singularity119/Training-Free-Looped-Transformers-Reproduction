# LoopScope Phase 5：RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE 冻结合同

```text
CONTRACT_ID=LOOPSCOPE_PHASE5_RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE
STATUS=FROZEN_AND_AUTHORIZED
GATE=H
PLANNING_THREAD=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
AUTHORIZED_EXECUTOR=019f8fe5-0880-7173-9650-7c4c76ee475d
CLAIM_SCOPE=RETROSPECTIVE_DUAL_MODEL_METHOD_DEVELOPMENT_ONLY
OUTCOME_ACCESS=FORBIDDEN
MODEL_FORWARD=FORBIDDEN
GPU_CUDA_SLURM=FORBIDDEN
```

## 1. 科学问题与边界

本合同只利用已经闭合的两份 MMLU validation no-loop 逐样本逐层 Choice Entropy /
KL-to-final trajectory，对以下两个 cell 离线重算窗口选择：

- `Qwen/Qwen3-1.7B-Base × MMLU 5-shot`；
- `Qwen/Qwen3-4B-Base × MMLU 5-shot`。

V2 完整继承 V1 的候选域、boundary/window 语义、subject-stratified joint bootstrap、
shape 检测和 admission：

```text
NewEligible =
  Scorable
  AND NetPositive
  AND SoftRelativeStable
  AND BiphasicStable
```

V2 只替换 `NewEligible` 内的最终排序和稳定发布规则。`SoftRelativeStable` 仍依赖
overlap/flank references；因此 V2 只消除了 reference 对最终排名的影响，并没有把
admission 变成完全 reference-free。

两个 cell 都参与了规则形成，只能提供 retrospective method-development、coherence
和 falsification evidence。`selected` 只表示本合同内部得到唯一且稳定的窗口，不表示
positive gain，也不是 prospective validation 或通用自动选窗器证据。

## 2. 冻结输入与候选域

### 2.1 本地 V1 实现

| Artifact | SHA256 |
|---|---|
| `loopscope-tflt/configs/loopscope/phase5_relative_biphasic_reversal_card.json` | `8a23bbe1d0de4f0421a67c62e7243f01aa242761c80913eee4393836f7de86e2` |
| `loopscope-tflt/src/tflt/loopscope/phase5_relative_biphasic.py` | `d61dee7070f2283805362ebf3db4974b4fba46186d4668761a228641da71b200` |
| `loopscope-tflt/scripts/loopscope/run_qwen_phase5_gate_g.py` | `465570ad29fdd1b0360cb06179665a7e5b842a27e20bb525576ef6202f4d1636` |
| `loopscope-tflt/tests/test_loopscope_phase5_relative_biphasic.py` | `caeedd5ba38e51f6e7abce6feed2120eec3ef3e84cc0f3adf3e20368d7b443d5` |

### 2.2 Canonical trajectory / legacy 输入

1.7B：

- trajectory:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3b-20260715T223304Z/validation1531_no_loop_trajectories.jsonl`
  SHA256 `fb7ae66a1a2696f46727c7ca5069ade2e2f9d6e181db03ed1d005cb2eed1525f`;
- trajectory manifest:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3b-20260715T223304Z/validation1531_no_loop_trajectory_manifest.json`
  SHA256 `75a568c6f6595f74d34d1e86efc0f716934e3a2a1cf972daab7874e4afe89cf0`;
- legacy table:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3f-dual-20260716T121430Z/strict_consensus_98_table.json`
  SHA256 `74332060f0b1da99202a7e647bc5b18a6733a992ed152889043c77b5c3cffb00`.

4B：

- trajectory:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-b-repair-b36-native-20260722T035452Z/formal/validation1531_phase5_trajectories.jsonl`
  SHA256 `aa4894e632f557ddc4614e88d06ec1fc4c525463be7531aef69fa3c28671edf9`;
- trajectory manifest:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-b-repair-b36-native-20260722T035452Z/formal/validation1531_phase5_trajectory_manifest.json`
  SHA256 `343689e4d2c4d8bf40f2237233a5333f1a7d53cef219e226e56d017d5aad21c6`;
- legacy table:
  `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-e-multiwidth-mid40-20260723T100836Z/phase5_multiwidth_mid40_scores.json`
  SHA256 `a4bd8b7be3b531b89961667b1cbfba9db0f4020e01bfa8efc97a3ccd43826e44`.

Canonical V1 output root:

```text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-g-relative-biphasic-dualmodel-20260723T151428Z
```

关键 V1 输出：

- 1.7B candidates JSON SHA256
  `a88b2827e5786263a1a0681f29e9a9f000a0ffd0e0ad1a911eabfcd9f1564c3d`;
- 4B candidates JSON SHA256
  `08a2f88da57f7318c851e9887e39aa59b9650d2bf67b6062713c79f3303adaa7`;
- cross-model summary SHA256
  `b1f3c2feb112b4f4644ec5925823a596cb561a1f63d7c7423b0477198194cb15`;
- verifier receipt SHA256
  `85f8a255b52566dbf62be78b1ba79d38d185d5ffdc630ab1ca851ee02aef3b2d`;
- manifest receipt SHA256
  `66cc03c5aca73fb48697bba0d3db77e04ebe6288b5ac5a9b3e21d7ac7392662d`.

两份 trajectory 均必须闭合为 `1,531` 个 unique validation identities、`57`
subjects，边界数分别为 `29` 和 `37`。

候选域精确继承 V1：

| Model | Central blocks | w3 | w4 | w5 | w6 | Total |
|---|---|---|---|---|---|---:|
| 1.7B, L=28 | `9…18` | starts `9…16` | `9…15` | `9…14` | `9…13` | 26 |
| 4B, L=36 | `11…24` | starts `11…22` | `11…21` | `11…20` | `11…19` | 42 |

不得因 V2 分数、已见 outcome 或开发窗口调整候选域。

## 3. V1 admission 与 backward compatibility

对每个候选，V2 必须复算并完整发布 V1 的：

- candidate identity、width、start、inclusive window 和 boundary path；
- `G_H/G_K`、`O_H/O_K/F_H/F_K`、四个 z 和 `S_REL`；
- legacy strict、`Scorable`、`NetPositive`、`SoftRelativeStable`；
- 全部 shape pair、最佳 `q/tau`、12 margins、`S_SHAPE`、simultaneous bounds；
- `T_seg/T_core`、pair frequency、`pi`、boundary support；
- `BiphasicStable`、failure reasons 和 `NewEligible`。

候选 identity/count、布尔、枚举、failure reasons 必须逐项精确一致；浮点绝对误差
不得超过 `1e-10`。任何不一致均为 `BLOCK_V1_BACKWARD_COMPATIBILITY_MISMATCH`，
不得重解释旧结果或继续 V2 排名。

V1 的 percentile CI、shape max-t、pair/window frequency 字段和阈值不得被 V2 改写。

## 4. 逐样本绝对平均变化率

对 `W(s,w)=[s,s+w-1]`，boundary path 为 `B_s -> B_(s+w)`。先逐样本计算：

```text
G_H,i(s,w) = (H_i,s - H_i,s+w) / w
G_K,i(s,w) = (D_i,s+w - D_i,s) / w
```

点估计为 `N=1531` 条样本的 micro mean：

```text
G_H = mean_i G_H,i
G_K = mean_i G_K,i
```

单位均为 `nats/boundary`。`ABSOLUTE_RATE` 表示中心窗口自身的 reference-free rate，
不是对数值取绝对值；任一方向非正必须失败。

只有 `G_H`、`G_K` 均 finite 且严格大于 0 时定义：

```text
S_RATE = sqrt(G_H * G_K)
```

`S_RATE` 是唯一主排名分数，只允许在同一 model-task cell 内比较。禁止改为算术平均、
调和平均、最小值、总变化，引入 lambda 或 width bonus/penalty，或混入
`S_REL/S_SHAPE`。

## 5. RateStable

分别复用两模型既有 subject-stratified joint bootstrap：

| Model | R | Seed | PRNG/API | Draw stream |
|---|---:|---:|---|---|
| 1.7B | 2000 | 20260716 | `random.Random/randrange` | V1 model-local frozen stream |
| 4B | 2000 | 20260722 | `random.Random/randrange` | V1 model-local frozen stream |

同一 replicate index map 必须联合重算全部层、H/K、window 和 width。两模型之间不共享
draw；不得新增 seed 或重新抽样。标准差使用 `ddof=1`，分母下限
`epsilon=1e-12`。

对每个窗口单独计算：

```text
SE_m = sd({G_m^(b)}_(b=1..R), ddof=1),  m in {H,K}

M_b = max_m [
  (G_m(point) - G_m^(b)) / max(SE_m, 1e-12)
]

c95 = max(0, linear_quantile({M_b}, 0.95))
LCB_m = G_m(point) - c95 * max(SE_m, 1e-12)

RateStable = (LCB_H > 0) AND (LCB_K > 0)
```

这里固定使用 `point - bootstrap`，因为目标是 one-sided lower confidence bound；若使用
`bootstrap - point`，在偏斜分布下对应的是另一侧误差，不能严格构成所声明的下界。
这是规划审计对建议公式方向的数学裁决，不是根据当前结果调参。

`linear_quantile`：对有限值升序排序，在位置 `q*(R-1)` 取相邻 order statistics
线性插值。`c95`、SE、LCB 和输入 rate 任一非 finite，属于分析无效并
`BLOCK_NONFINITE_RATE_ANALYSIS`，不能伪装成 ABSTAIN。`LCB=0` 不通过。

该 max-t 只对同一窗口内的 H/K 两维做同时覆盖，不是 across-all-windows 或
across-model simultaneous coverage。点分数和不确定性严格分离。

## 6. V2 排名、频率与 ABSTAIN

```text
C_RATE = {
  W : NewEligible(W) AND RateStable(W)
}
```

在 `C_RATE` 内只按 point `S_RATE` 降序选择。width/start 只允许用于确定表格的稳定
展示顺序，不得解决科学 top tie。

point top tie：若至少两个候选的 `S_RATE` 与最高值满足
`math.isclose(rel_tol=1e-12, abs_tol=1e-12)`，则不发布窗口。

selection frequency 使用固定的 point `C_RATE`：

1. 每个 bootstrap replicate 对每个固定候选重算 `G_H^(b),G_K^(b),S_RATE^(b)`；
2. 任一 replicate rate 非 finite 为分析无效并 BLOCK；任一 rate `<=0` 时该窗口在该
   replicate 不可胜；
3. 该 replicate 无可用窗口，或 replicate top 在相同 tolerance 下并列，则无 winner；
4. 无 winner 仍计入总分母 `R=2000`；
5. point top 的 exact-window win count 除以 2000 得到 selection frequency；
6. frequency `>=0.80` 才发布，等于 0.80 通过。

终态：

- `C_RATE` 为空：`ABSTAIN_NO_RATE_STABLE_ELIGIBLE`;
- point top 不唯一：`ABSTAIN_NO_UNIQUE_TOP1`;
- point top selection frequency `<0.80`：
  `ABSTAIN_RATE_RANK_UNSTABLE`;
- 否则：`SELECTED_WINDOW`.

`eligible`、`ranking_candidate`、`rank-1`、`selected`、`positive gain` 是五个不同
概念。本 Gate 不读取 gain/outcome，因此绝不输出 positive-gain 判定。

## 7. 诊断字段

以下继续完整输出但不进入 V2 排名：

- V1 `S_REL`、O/F、四个 z；
- `S_SHAPE`、q/tau、12 margins、`T_seg/T_core`、pair frequency 和 `pi`;
- `balance_ratio=min(G_H,G_K)/max(G_H,G_K)`，仅在两者严格正且 finite 时定义，不设阈值；
- `total_entropy_change=w*G_H`;
- `total_kl_change=w*G_K`.

使用 per-layer rate 是为了预注册地避免长窗只因长度累计更多变化而获益，但它可能偏好
尖锐短窗。看到结果后不得改用 total change。

## 8. 输出 schema

每个模型的完整候选行至少包含：

```text
model, width, start, end, window, boundary_entry, boundary_exit,
G_H, G_K,
O_H, O_K, F_H, F_K, z_OH, z_OK, z_FH, z_FK, S_REL,
legacy_strict, Scorable, NetPositive, SoftRelativeStable,
best_q, best_tau, boundary_supported, external_side,
shape_pairs, shape_margin_count, S_SHAPE,
T_seg_H, T_seg_K, T_core_H, T_core_K,
tau_pair_frequency, pi, BiphasicStable,
v1_failure_reasons, NewEligible,
rate_SE_H, rate_SE_K, rate_c95, rate_LCB_H, rate_LCB_K,
RateStable, S_RATE, balance_ratio,
total_entropy_change, total_kl_change,
ranking_candidate, display_rank, point_top_tie,
model_rate_selection_frequency, v2_decision, v2_failure_reasons
```

模型级 summary 至少包含 candidate/NewEligible/RateStable/C_RATE counts、完整
`C_RATE` 排名、point top、selection frequency、selected 或 ABSTAIN、bootstrap
seed/draw digest、V1 compatibility digest。跨模型 summary 只并列报告，不比较 raw
`S_RATE`。

fresh write-once root：

```text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-h-relative-biphasic-v2-absolute-rate-20260723T164940Z
```

必须生成：

- `qwen17_v2_absolute_rate_candidates.json/.csv`;
- `qwen4_v2_absolute_rate_candidates.json/.csv`;
- `relative_biphasic_v2_cross_model_summary.json`;
- `relative_biphasic_v2_summary_zh.md`;
- `relative_biphasic_v2_verifier_receipt.json`;
- `relative_biphasic_v2_manifest_receipt.json`.

## 9. 最小测试与证伪

必须先通过 focused synthetic tests：

1. 单项趋近或等于 0 时不定义合法主分数；
2. 负 rate 失败；
3. point top tolerance tie 导致 ABSTAIN；
4. 在另一项固定且正时，`S_RATE` 对任一 rate 严格单调；
5. 短窗尖峰与宽窗平缓按 per-layer rate 而非 total change 排序；
6. point rate 均正但任一 simultaneous LCB 不正时 `RateStable=false`;
7. bootstrap top-1 不稳定时 ABSTAIN；
8. replicate rate 非正时该窗口不可胜，无 winner 仍进入分母；
9. 固定 admission 后修改 reference 不改变 `S_RATE`;
10. V1 backward compatibility 全量通过。

之后只允许对两个 canonical cell 做一次正式离线 analyze 和一次 fresh-process
independent verify。即使当前 4B 排出 `15:18`，也只能报告 retrospective coherence，
不得称为验证成功或有效增益。

## 10. 信息屏障与失败状态

analyzer/launcher/card 不得接受 outcome、accuracy、test、gold、correctness、prediction、
gain、flip 或 loop-result 路径/字段。科学工件中出现这些值即 BLOCK。

分析无效 BLOCK：

- `BLOCK_INPUT_HASH_OR_POPULATION_MISMATCH`;
- `BLOCK_V1_BACKWARD_COMPATIBILITY_MISMATCH`;
- `BLOCK_NONFINITE_RATE_ANALYSIS`;
- `BLOCK_OUTPUT_INFORMATION_BARRIER_VIOLATION`;
- `BLOCK_VERIFIER_MISMATCH`.

科学 ABSTAIN：

- `ABSTAIN_NO_RATE_STABLE_ELIGIBLE`;
- `ABSTAIN_NO_UNIQUE_TOP1`;
- `ABSTAIN_RATE_RANK_UNSTABLE`.

## 11. 明确禁止

- 模型、tokenizer 或数据集加载；
- 任何 forward、loop/full-accuracy/outcome；
- GPU、CUDA、Slurm；
- 打开或解析已有 outcome/accuracy 工件；
- 根据 outcome 或本轮结果修改公式、阈值、候选域或 seed；
- 修改 canonical V1 card/code/output；
- public push；
- 自动启动 Gate I 或任何后续实验。
