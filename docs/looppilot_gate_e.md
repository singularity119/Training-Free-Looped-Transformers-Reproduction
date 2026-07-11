# LoopPilot Gate E

Gate E 在冻结的 Qwen3-1.7B-Base、完整 MMLU test、5-shot、batch size 1 上顺序运行 `baseline_a`、`baseline_b` 和 `always_loop`。GPU arm 不运行候选阈值 controller；候选规则只在全部 shard 完成后离线选择。

CPU preflight 递归读取 lm-eval 0.4.11 的权威 `mmlu` group YAML，冻结 57 个 leaf task、全部 test docs、无标签 hash split 和 8 个确定性 task shards。split salt、24-cell candidate grid、subject cluster bootstrap、matched random null、comparator 和终态顺序均在配置中预冻结。

GPU 只允许一次正常 array：`0-7%N`。`N=min(8, scheduler-visible free A40)`，必须紧邻提交前由 `/opt/slurm/bin/sinfo`、`scontrol show node -o` 和 `squeue` 的 write-once 原始快照计算；仅统计属于 `i64m1tga40u`、配置 A40 且不处于 DOWN/DRAIN/FAIL/MAINT/NOT_RESPOND 的节点。`N=0` 时只允许低频重做容量快照，不得提交或退回固定并发。调用唯一一次 `sbatch` 后，无论回执是否明确，都进入 `AGENTS.md` 的 ambiguous-submit guard，禁止第二个正常 array。

所有 8 个 array elements `COMPLETED/0:0` 且 manifests 逐项通过后，exclusive-create `gpu-array-terminal-verified.txt`，才允许一次 `i64m512u` CPU analysis job。

最终建议仅为 `PASS_TO_ONLINE`、`PIVOT_TO_PHASE3`、`PIVOT_TO_COARSE` 或 `BLOCK`。Gate E 后不自动进入第二阶段或新实验。
