# LoopScope Phase 7 Gate F V3 Top-2 Outcome Amendment

```text
AMENDMENT_ID=LOOPSCOPE_PHASE7_GATE_F_V3_TOP2_OUTCOME_V1
DATE=2026-08-16
USER_AUTHORIZED=true
PHASE=7
GATE=F
INTERPRETATION=POST_TERMINAL_V3_POINT_RANK_TOP2_OUTCOME_EXPLORATORY
```

## 用户新授权

用户在 Gate E 已完整解封并完成 Phase 7 原终审后，明确要求把三模型“选窗前两名”
另行跑全量 loop outcome，并继续遍历 `k={2,3}`、`cache={first,last}`、`mode=block`。
该指令显式扩展 Phase 7，取代原“Gate E 后不创建 Gate F”的停止分支。

“前两名”绑定为 Gate D 封存的**模型内全 width V3.1 point rank 前二**，不是 width-4
子表，也不是 Gate E outcome 排名。窗口在本 amendment 落盘时已固定，不得据 Gate E
或 Gate F outcome 改变。

## 冻结 panel

```text
qwen25_3b rank1=[14,17) score=0.2375409511067389
qwen25_3b rank2=[12,15) score=0.22794108469005608
llama32_3b rank1=[11,14) score=0.060057980866983884
llama32_3b rank2=[10,13) score=0.037028676844756976
gemma2_2b rank1=[13,16) score=0.07844678208305271
gemma2_2b rank2=[10,13) score=0.0780351750863574
k={2,3}
cache_strategy={first,last}
iteration_mode=block
strategy=euler
alpha=1.0
beta=0.0
decode_mode=full
new_loop_cells_per_model=8
new_loop_cells_total=24
```

三个 no-loop baseline 必须复用已验证 Gate E 同模型、同 revision、同 MMLU test-14042
membership、同 renderer/evaluator/runtime 的 baseline records，不重跑。Gate F 最终分析 panel 因此为
`3 retained baselines + 24 new loop cells = 27 cells`。Qwen/Llama 继续 batch 16，Gemma 继续
已由 full-shape OOM recovery 冻结的 batch 8。

## 不变量与声明边界

- 模型/revision、`cais/mmlu` test-14042/57、plain 5-shot、lm-eval 0.4.11、BF16、无量化不变。
- 不重算 Gate D，不改 V3 分数、eligibility、frequency、候选域或 Gemma ABSTAIN 终态。
- 24/24 新 loop cells 与 baseline-reference membership 闭合前，不得读取或汇报 partial Gate F
  accuracy/gain/correctness。闭合后只做一次 Gate F combined analysis 与 fresh verifier。
- 因 panel 是在 Gate E outcome 已知后追加，不声称 prospective 或 confirmatory selection success；统一标为
  `POST_TERMINAL_V3_POINT_RANK_TOP2_OUTCOME_EXPLORATORY`。

Canonical panel manifest:
`资产/报告/phase7/phase7_gate_f_v3_top2_panel_manifest.json`。
