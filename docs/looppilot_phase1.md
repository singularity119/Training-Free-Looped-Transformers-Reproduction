# LoopPilot 第一阶段本地契约

Gate A 只建立本地、无模型依赖的工程骨架。冻结范围见
`configs/looppilot/qwen17_mmlu_phase1.json`；model 与 tokenizer 的 HEAD revision 由
`configs/looppilot/resolved_revisions.json` 固定。正式 signal run 前仍必须从实际
lm-eval renderer 生成 authoritative renderer artifact/hash，并把 batch size 冻结为显式整数。

## 执行语义

- `BASELINE/NeverLoop` 返回第一次窗口调用的 `y0`，不是 `x0`。
- `LOOP_K2/AlwaysLoop` 复用 `y0 - x0`，总 operator body call 为 2。
- controller 为 `None` 时沿用 legacy `run_loop`，不产生 controller 事件。
- MMLU 决策单位是 doc，同题四个 choice request 必须共享 action。
- per-row masking 若仍对全 batch 计算第二次 operator，不报告真实 FLOPs 或墙钟节省。

## Signal 与工件

`signals.py` 提供 R0/C0/N0/Q1 的纯 Python reference 聚合；mask 只允许题目/共享
context token。NaN、Inf、空 mask、shape mismatch 都直接失败。`schema.py` 的 JSON/JSONL
writer 使用 exclusive create，拒绝覆盖。Q1 与 residual cosine 只作第二次调用后的诊断。

Gate A 的 `--looppilot-signal-jsonl` 只完成 CLI/argv/schema 接口。任何实际执行会在创建
工件或加载模型前失败；真实 tensor collector、renderer closure 与四题 probe 属于 Gate C，
不得由 Gate A 越权实现或运行。

## 本地验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m tflt.cli --help
PYTHONPATH=src python3 -m compileall -q src tests
python3 scripts/looppilot/validate_phase1_config.py
```
