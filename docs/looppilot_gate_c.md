# LoopPilot 第一阶段 Gate C：四题 GPU probe

Gate C 只验证真实 Qwen3-1.7B 张量上的 signal、loop 语义、body-call 与 restore。它不运行
limit/full MMLU，不计算准确率，不选择阈值，也不进入 Gate D。

冻结配置位于 `configs/looppilot/gate_c_probe.json`：MMLU 5-shot、batch size 1、窗口
`[12,15]`、K=2 damped Euler、float16、cache last、decode bypass。四题固定为
`mmlu_abstract_algebra`、`mmlu_anatomy`、`mmlu_astronomy`、
`mmlu_business_ethics` 的 test split doc index 0。

## 执行顺序

1. local/origin/HPC2 必须 clean 且同一 `looppilot` commit，固定基点必须为祖先。
2. 本地与 HPC2 先完成 full tests、`python -S`、compile、CLI、validator 和 collector tests。
3. 在全新 UTC timestamp + commit8 的 run root 中运行 CPU preflight。preflight 只读共享
   cache，冻结 model/tokenizer revision、lm-eval 版本、四个 task YAML、renderer、无标签 doc、
   question mask、probe config 与 schema 哈希。
4. `submit_gate_c_probe.sh` 只提交一个非 array `debug/gpu:a40:1` job。GPU 脚本必须重建并
   精确比对 preflight doc/renderer/mask，任何差异立即失败。
5. 只审计该 JobID；失败、超时、OOM、cache miss 或工件不闭合均不得重提。

远端唯一提交形式：

```bash
scripts/looppilot/submit_gate_c_probe.sh \
  /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_looppilot/gates/gate_c/<UTC>_<commit8>
```

成功工件至少包括 `control/preflight/preflight_manifest.json`、`results/documents.jsonl`、
`results/action_records.jsonl`、`results/probe_summary.json`、`results/revision_closure.json`、
Slurm stdout/stderr、JobID 和 SHA-256 manifest。不得保存 raw hidden、full logits、gold、
correctness 或 choice continuation。
