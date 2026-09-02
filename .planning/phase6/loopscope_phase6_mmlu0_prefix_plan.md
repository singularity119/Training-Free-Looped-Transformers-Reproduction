# LoopScope Phase 6 — MMLU 0-shot Prefix Trajectory Gate Plan

## 1. 目标与冻结合同

本分支检验：在 `Qwen/Qwen3-4B-Instruct-2507` 的标准 MMLU 0-shot
choice-loglikelihood 评测中，prefix 最后一个有效 token 的逐层 choice/hidden trajectory
是否出现可复现的良性 entropy--KL 窗口特征，并据此得到冻结的建议窗口或合法弃权。

冻结配置：

- model：`Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`；
- task：`cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`；
- evaluator：lm-eval 0.4.11，plain prompt，0-shot，no chat template，no multiturn；
- scoring：exact continuation ` A/ B/ C/ D` choice loglikelihood，不生成 CoT/答案文本；
- probe：rendered prefix 最后一个非 padding token，一次 zero-loop `use_cache=false` native forward；
- boundaries：raw `B0...B36`，B0...B35 final norm 一次，B36 不 double norm；
- selector fields：choice entropy `H`、`KL(p_l||p_36)`；
- diagnostics：Hidden RMS-L2、Hidden cosine、Hidden cosine-distance、raw adjacent angular distance；
- selector：`RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE`，中央 blocks 11...24，
  width 3/4/5/6 共 42 candidates，2,000 次 subject-stratified bootstrap，seed 20260801；
- loop：bfloat16、K=3、block Euler、step=1/3、horizon=1、cache=first、decode=full；
- selector pool：MMLU validation-1531 / 57 subjects；本分支不读取 test split、不运行 loop outcome。

此前 MMLU-Pro pre-answer D-2/D-2P 全部为历史分支，不得复用其 trajectory、selector、
panel 或 outcome 作为本分支科学输入。

## 2. Gate 表

| Gate | 目标 | 最早真实实验 | PASS/终止条件 | 后续权限 |
|---|---|---|---|---|
| H | 冻结 0-shot renderer、identity、choice token、schema 和最小 producer/verifier | 本地 synthetic + HPC2 bounded provenance read | exact contract、单 token choice surface、validation closure 和 focused tests 通过 | 解锁 I |
| I | HPC2 CPU admission 与 4-identity 真实 GPU smoke | `debug` partition `<30min` | 同一 forward 闭合 B0...B36、choice/hidden metrics、final norm/LM-head、无生成/无 outcome | 解锁 J |
| J | validation-1531 全量 trajectory、V2 selector 与阶段终审 | 同 commit `debug` preflight 后，最高合法优先级 formal | 1531/57 完整、fresh verifier PASS，输出合法 selected window 或 ABSTAIN | Phase 6 terminal |

## 3. Gate H 执行计划

最小实现只做当前实验必需的合同层：

1. 复用 Phase 5 的 MMLU identity/renderer/choice-space producer 与 Phase 6 的 36-layer
   Instruct model provenance；将 fewshot 精确改为 0，禁止 chat template/generation。
2. 冻结 validation-1531、57 subjects、renderer source、prompt terminal、
   continuation surfaces/token IDs、model/tokenizer revision 与 dependency hashes。
3. 定义 closed trajectory schema：identity/subject、37 个 choice distributions、H/KL、
   37 个 hidden-to-final diagnostics、36 个 adjacent angular diagnostics；不持久化 logits、
   full vocabulary distribution、hidden tensor、gold、label、correctness 或 outcome。
4. 实现最小 CLI dry-run、producer pure helpers、selector adapter 与独立 verifier；只跑 focused
   tests、compile/help/dry-run/self-test，不做全历史 suite。
5. 允许 bounded read-only HPC2 provenance/tokenizer/task-source closure；禁止模型 forward、
   CUDA/Slurm、正式 acquisition、selector 正式运行和任何 outcome read。

Agility stop rule：合同、关键 focused tests 与 verifier self-test 通过后立即提交/push 并回传；
不得扩展成新通用框架、重写 Phase 5/6 历史模块或增加与真实主线无关的防御性矩阵。

## 4. 信息屏障与监控

- H/I/J 全程不读任何 gold/correctness/outcome 或 test split；J 只读 validation sanitized
  trajectory。本分支不创建 sealed outcome、panel 或后续 unseal Gate。
- debug 任务只有真实 RUNNING 稳定约 60 秒后才建立 heartbeat；不足 60 秒即结束则不建。
- formal 大任务取得真实 job ID 后立即建立唯一 heartbeat，即使仍 PENDING；probe/full cadence
  分别 30/60 分钟。失败 attempt 唤醒 exact executor 继续低风险修复，不自动终止 Gate。
