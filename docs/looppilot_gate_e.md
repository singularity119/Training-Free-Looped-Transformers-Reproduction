# LoopPilot Gate E

Gate E 在冻结的 Qwen3-1.7B-Base、完整 MMLU test、5-shot、batch size 1 上顺序运行 `baseline_a`、`baseline_b` 和 `always_loop`。GPU arm 不运行候选阈值 controller；候选规则只在全部 shard 完成后离线选择。

CPU preflight 递归读取 lm-eval 0.4.11 的权威 `mmlu` group YAML，冻结 57 个 leaf task、全部 test docs、无标签 hash split 和 8 个确定性 task shards。split salt、24-cell candidate grid、subject cluster bootstrap、matched random null、comparator 和终态顺序均在配置中预冻结。

GPU 只允许两次顺序 array：A=`0-3%2`，逐项终态并验证后才允许 B=`4-7%2`。所有 8 shard `COMPLETED/0:0` 且 manifests 通过后，只允许一次 `i64m512u` CPU analysis job。任一 submit 回执不明确时必须按 `AGENTS.md` 的 ambiguous-submit guard 多源核验，禁止直接重提。

最终建议仅为 `PASS_TO_ONLINE`、`PIVOT_TO_PHASE3`、`PIVOT_TO_COARSE` 或 `BLOCK`。Gate E 后不自动进入第二阶段或新实验。
