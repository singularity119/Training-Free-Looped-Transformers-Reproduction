# LoopScope Phase 6 — MMLU 5-shot Prefix Trajectory Follow-up

## 1. 研究问题与唯一变量

在已闭合的 MMLU 0-shot Gate H/I/J 之后，追加一个隔离的 5-shot follow-up，检验加入标准
dev demonstrations 后，同一 `Qwen/Qwen3-4B-Instruct-2507` 在 prefix 最后一个有效 token
上的逐层 choice/hidden trajectory 是否仍产生稳定的 V2 窗口。

与 0-shot 相比，唯一科学变量是：

- `num_fewshot: 0 -> 5`；
- `fewshot_split: null -> dev`；
- 因 demonstrations 改变而必须 fresh acquisition 的 rendered prompt projection/hash。

模型、数据 revision、validation population、choice-loglikelihood、plain prompt、probe position、
37 个 boundaries、choice entropy/KL、hidden diagnostics、V2 candidate set、bootstrap、TFLT
配方与信息屏障全部保持不变。

## 2. 冻结科学合同

- model：`Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`；
- dataset：`cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`；
- evaluator：lm-eval 0.4.11，standard MMLU choice-loglikelihood，`num_fewshot=5`，
  `fewshot_split=dev`，plain prompt，`apply_chat_template=false`，
  `fewshot_as_multiturn=false`，generation=false；
- validation population：1,531 identities / 57 subjects，canonical order 与 0-shot 相同；
- continuation surfaces：exact ` A/ B/ C/ D`，沿用已审计单-token IDs；
- probe：rendered 5-shot prefix 的最后一个非 padding token；每 identity 一次
  zero-loop native forward，`use_cache=false`，`output_hidden_states=true`；
- boundaries：raw `B0...B36`，B0...B35 final norm 一次，B36 不 double norm；
- selector metrics：四选一 choice entropy `H` 与 `KL(p_l||p_36)`；
- diagnostic-only：Hidden RMS-L2、Hidden cosine、Hidden cosine-distance、
  adjacent angular distance；
- selector：`RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE`，central blocks 11...24，
  widths 3/4/5/6，共 42 candidates；subject-stratified joint bootstrap 2,000 次，
  seed 20260801，frequency threshold 0.80；
- TFLT identity：bfloat16、K=3、block Euler、step=1/3、horizon=1、cache=first、decode=full；
  仅用于解释建议窗口，本 follow-up 不运行 loop outcome。

允许 renderer 读取 dev demonstrations 及其示例答案；严禁读取 validation gold、test split、
correctness、accuracy、gain、flip 或任何 loop outcome。0-shot trajectory/selector 只作历史对照，
不得混入 5-shot selector 样本或 bootstrap。

## 3. Gate 表

| Gate | 目标 | 最早真实实验 | PASS/终止条件 | 后续权限 |
|---|---|---|---|---|
| K | 冻结 5-shot renderer、fresh prompt projection、card/schema 与最小 producer/verifier 适配 | 本地 focused tests + HPC2 outcome-safe CPU projection | 1531/57、5 dev demonstrations、prompt/order/hash、choice-token 与 closed schema 全部闭合 | 解锁 L |
| L | HPC2 CPU admission 与四 identity 真实 GPU smoke | `debug` partition，A40/合适 debug GPU，<30 min | 4 records、37 boundaries、36 transitions、one native forward、metrics/final-norm/verifier PASS；无 generation/loop/outcome | 解锁 M |
| M | validation-1531 正式 5-shot trajectory、V2 selector 与 follow-up 终审 | 同 commit/launcher/env 的 debug preflight PASS 后，最高合法优先级 A800 formal array | 1531/57 完整、fresh verifier PASS，selector 单次执行并输出合法 selected window 或 ABSTAIN | Phase 6 follow-up terminal |

## 4. 资源、监控与失败恢复

- Gate K 不授权 GPU/Slurm；仅允许 bounded read-only HPC2 provenance/cache/source 检查和
  outcome-safe CPU prompt projection。
- Gate L 的所有 smoke/debug 只能使用 `debug` partition；只有任务稳定 RUNNING 约 60 秒后
  才创建唯一 smoke 10 分钟 heartbeat，提前结束则不创建。
- Gate M 正式任务冻结 A800，使用普通用户合法的最高优先级 A800 partition/QoS；提交取得
  真实 job ID 后立即创建唯一 full 60 分钟 heartbeat，包括 PENDING 阶段。
- job 失败是 attempt-level event。executor 必须在冻结合同内继续定位、低风险修复并使用
  fresh write-once root 重试；只有科学选择、信息屏障、安全/破坏性边界或无法取得材料性进展
  才发送 Gate-level `BLOCK`。
- executor-owned 低风险工程修复次数不限；每个 Gate 只允许其自身 executor，下一 Gate 必须
  由 planning/audit task 验收 PASS 后新建并显式授权。

## 5. 阶段终点

Gate M 的终点是 5-shot validation-1531 的正式逐层 trajectory、fresh verifier 以及一次冻结
V2 selector 的 `SELECTED_WINDOW` 或合法 `ABSTAIN`。不运行 test-14042，不读取 outcome，
不据此启动任何 full loop 实验。

