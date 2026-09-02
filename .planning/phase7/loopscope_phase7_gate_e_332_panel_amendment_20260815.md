# LoopScope Phase 7 Scientific Amendment — Gate E 3/3/2 Top-Available Panel

```text
AMENDMENT_ID=LOOPSCOPE_PHASE7_GATE_E_TOP_AVAILABLE_332_V2
DATE=2026-08-15
PHASE=7
AUTHORITY=USER_EXPLICIT_APPROVAL_AFTER_BLOCK_PANEL_NOT_FORMABLE
SUPERSEDES_PANEL_FORMATION_RULE=.planning/phase7/loopscope_phase7_gate_e_outcome_amendment_20260815.md
GATE_E_ADMISSION=SATISFIED_AFTER_EXACT_EXECUTOR_BINDING_AND_HANDOFF
PLOTTING_CHECKPOINT=PASS
OUTCOME_ACCESSED_BEFORE_AMENDMENT=FALSE
```

## 1. 修订原因

原 Gate E 要求每模型都提供三个 finite width-4 `S_RATE_TURN` 窗口。Gate D 封存候选表显示
Qwen/Llama/Gemma 的 finite width-4 数量为 4/3/2，因而触发预注册的
`BLOCK_PANEL_NOT_FORMABLE`。在没有读取 MMLU test、gold 或任何 outcome 的前提下，用户明确
批准最小科学修订：Qwen/Llama 保留 top-3，Gemma 使用现有全部 top-2。

本修订只改变 Gate E panel 大小，不重算 Gate D，不改 V3.1 公式、阈值、候选域、
seed、模型或 prompt，也不将 Gemma `ABSTAIN_COMBINED_RANK_UNSTABLE` 改写为 selection。

## 2. 冻结窗口与 cell 数

| 模型 | 冻结 width-4 windows（半开） | windows | cells |
| --- | --- | ---: | ---: |
| Qwen2.5-3B | `[14,18)`, `[19,23)`, `[16,20)` | 3 | 13 |
| Llama-3.2-3B | `[10,14)`, `[11,15)`, `[9,13)` | 3 | 13 |
| Gemma-2-2B | `[10,14)`, `[12,16)` | 2 | 9 |

每模型包含一个 no-loop baseline；每个冻结窗口运行
`k={2,3} × cache={first,last} × mode=block`。因此总数为
`13 + 13 + 9 = 35 cells`。排名依据仍是 Gate D finite `S_RATE_TURN` 降序及冻结 tie-break；
不使用 non-finite candidate、其他 width、fallback score 或替代模型。

## 3. Gate E 信息与执行边界

- 数据仍为 `cais/mmlu` test 全量 14,042 identities / 57 subjects、标准 plain non-chat
  5-shot、dev demonstrations。
- 除用户指定的 window、`k`、cache 外，loop 非变量固定为
  `strategy=euler`（TFLT `damped_euler` 等价别名）、`alpha=1.0`、`beta=0.0`、
  `decode_mode=full`、`batch_size=16`，与现有标准 MMLU 5-shot
  full-sequence outcome 路径保持一致。
- 35/35 cells 及共同 identity/completeness closure 完成前，不得读取部分 accuracy/gain
  或据 outcome 改 panel。完整后只做一次 combined analysis 与 fresh verifier。
- Gemma 两窗结果必须标记为 ABSTAIN 后的 post-ranking exploratory outcome；三模型均不得
  把未经冻结多重比较程序的 nominal 单 cell 改善声称为 confirmatory 通用选窗成功。
- 正式任务前必须完成同 immutable decision-critical path 的 HPC2 `debug` `<30min`
  preflight；正式资源按 normal-user 合法最高优先级与显存/runtime/queue/time-to-result
  选择，使用唯一 60-minute full heartbeat。

## 4. 治理

Gate E canonical title 修订为
`execute-LoopScope-TopAvailable-Outcome-第7阶段-Gate E`，以准确描述该 Gate 解决的问题。
Gate E 使用新的独立、用户可见 executor，继承 Codex 配置中的默认模型与 reasoning，
不显式覆盖。执行线程严格遵守 `research-gate-orchestrator`，仅能发送唯一
`GATE_E_FINAL_AUDIT` 或真正的 `BLOCK`。Gate E planning `PASS` 后，绘图/报告更新与
consolidated Phase 7 audit 由 planning thread 亲自完成，不创建 Gate F。
