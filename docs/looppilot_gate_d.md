# LoopPilot Gate D：固定 limit 工程语义闭合

Gate D 仅验证固定 20 个 MMLU test docs 的工程语义，不选择阈值、不进行 signal 排名，也不形成准确率或计算节省结论。

## 冻结范围

- 模型：`Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`
- lm-eval：`0.4.11`
- tasks：`mmlu_abstract_algebra`、`mmlu_anatomy`、`mmlu_astronomy`、`mmlu_business_ethics`
- 每个 task 固定 test doc indices `0..4`，共 20 docs
- `num_fewshot=5`、`fewshot_seed=1234`、`batch_size=1`、`dtype=float16`
- loop arm：`[12,15]`、K=2、block、damped Euler、`alpha=1.0`、`beta=0.0`、last cache、bypass decode、显式 `always_loop`
- baseline arm：同一 allocation、同一模型加载、未 patch baseline

`configs/looppilot/gate_d_limit.json` 是唯一允许的 Gate D 配置。任一 task、limit、revision、batch、window、K、solver、controller 或非科学使用 flag 漂移都会在模型加载前失败。

## Metadata 与泄漏边界

lm-eval `Instance.task_name/doc_id/idx` 是 request metadata 的权威来源。adapter 对每条 request 重算并核验 renderer-bound doc key：

```text
task | task_version | renderer_hash | doc_hash | occurrence_id(test:<index>)
```

同题四个 request 分别具有 choice index `0..3`，但 controller probe 只收到当前题 shared context 上的 `r0_median`、`c0`、`n0_median`。choice continuation 仅由 lm-eval 自身用于 loglikelihood 评分，不进入 controller probe，也不写入 signal/decision 工件。gold、prediction、margin、correctness、full logits 和 raw hidden 同样禁止进入 signal/decision 工件。

## Write-once 工件

每次 run 使用全新目录：

```text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot/gates/gate_d/<UTC>_<commit8>
```

核心结果为：

```text
results/baseline_samples.jsonl
results/loop_samples.jsonl
results/signal_records.jsonl
results/decisions.jsonl
results/join_report.json
results/run_summary.json
```

另外保存 baseline/loop 原始 lm-eval results、renderer/revision/runtime 工件、完整 argv、环境白名单、Slurm 记录、source venv 前后只读证明，以及 preflight/results SHA-256 manifests。所有 JSON/JSONL 通过 exclusive create 写入，存在即失败。

## 验收

离线 verifier 必须同时证明：20/20 baseline、loop 和 decisions；80/80 signals；四个集合的 doc key 完全一致；每 doc 恰有 `{0,1,2,3}` 四个 choice；四条 request 的 decision id/action/reason 和 shared-context signals 一致；`operator_body_calls == k_used == 2`；signal finite 且满足 Gate C 数值范围；wrapper restore、revision closure、禁止字段与非科学使用 flags 全部闭合。

Gate D 的 `AlwaysLoop == legacy K2` 依据是已接受 Gate C results manifest，Gate D 不新增第三个科学 arm。Gate D 完成后只请求规划/审计线程进行 `GATE_D_FINAL_AUDIT`，不得进入 Gate E。
