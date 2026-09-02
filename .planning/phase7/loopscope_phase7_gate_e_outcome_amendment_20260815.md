# LoopScope Phase 7 Scientific Amendment — Gate E Full Outcome

```text
AMENDMENT_ID=LOOPSCOPE_PHASE7_GATE_E_FULL_OUTCOME_V1
DATE=2026-08-15
PHASE=7
AUTHORITY=USER_EXPLICIT_SCOPE_EXTENSION
CURRENT_GATE_UNCHANGED=B
FUTURE_GATE=E
FUTURE_GATE_STATE=LOCKED_UNTIL_GATE_D_PASS_AND_PLANNING_PLOTTING_QA_COMPLETE
STANDING_CONTINUATION=GRANTED_THROUGH_GATE_E_AND_PLANNING_PHASE_END_AUDIT
```

## 1. 用户确认的科学扩展

Phase 7 不再以 Gate D 后的绘图为终点。Gate D `PASS` 后，planning thread 仍先独立完成原定
trajectory/V3 绘图、三格式 QA 与 outcome-blind 中文报告；只有该 plotting checkpoint 完成并冻结
Gate E panel 后，才创建新的独立 Gate E executor。

Gate E 在三个 Phase 7 模型上运行 `cais/mmlu` test 全量 14,042 identities / 57 subjects、标准
plain non-chat 5-shot outcome panel。每模型固定 13 cells：

```text
1 × no-loop baseline
3 × width-4 V3 score top-ranked windows
2 × k={2,3}
2 × cache={first,last}
mode=block
total=1+3*2*2=13 cells/model
three-model total=39 cells
```

每模型在 Gate D 完整 candidate table 中先过滤 `width=4`，再按 V3.1 最终 point score
`S_RATE_TURN` 从高到低取前三；精确相等时沿用 frozen V3 candidate/tie-break 顺序。panel membership
与 Gate D 的 `SELECTED_WINDOW/ABSTAIN` 终态解耦：即使模型为合法 `ABSTAIN`，仍直接采用 width-4
有限可评分候选的前三名，不修改 eligibility、threshold、candidate domain、seed、metric 或 prompt。
若某模型少于三个有限可评分 width-4 候选，Gate E admission 为 `BLOCK_PANEL_NOT_FORMABLE`，不得
改宽度、补其他 ranking 或替换模型。

## 2. 信息顺序与解释边界

1. Gate A–D 继续保持 validation-1531、outcome-blind、zero-loop trajectory/V3 合同。
2. Planning plotting checkpoint 只能读取 Gate D 已验证 aggregate/source-data，不得读取 test、gold、
   prediction、correctness、accuracy 或 gain；先完成图表和原定 outcome-blind报告，避免 outcome 反向
   改图或改 selector 叙述。
3. Gate E 在 panel manifest、test-14042 identity closure、同路径 debug preflight 和 pre-outcome
   verifier 全部通过后，才执行正式 loop/no-loop outcome panel。
4. 39 cells 全部完成、identity/completeness closure 通过前不得做部分 outcome 排名或据结果删减、
   追加、重排 cells。全部闭合后只做一次 combined outcome analysis 与 fresh verifier。
5. Gate E 报告标准 MMLU accuracy、相对同模型 no-loop baseline 的 paired delta，并保留 per-subject/
   paired evidence。由于每模型检验 12 个 loop cells，结果按 fixed exploratory panel 解释；未经另行
   冻结的多重比较程序，不把 nominal 单 cell 改善表述为 confirmatory selector success。
6. 对 Gate D 为 `ABSTAIN` 的模型，Gate E top-3 结果只能称为 post-ranking exploratory outcome，
   不能改写为该模型被 V3 prospective selected。

## 3. Gate E 资源与治理

- Gate E title 预注册为
  `execute-LoopScope-Top3-Outcome-第7阶段-Gate E`。
- Gate E 只在 plotting checkpoint `PASS` 后创建并绑定独立 executor；Gate B/C/D executor 均无
  Gate E 权限。
- 正式任务前必须在 HPC2 `debug` partition 完成同 immutable commit、launcher、environment、
  model/data/runtime、loop/cache path、output schema 与 verifier 的 `<30min` end-to-end preflight。
- 正式 job 使用 normal user 合法可用的最高优先级 partition/QoS，并按模型显存适配、实测 runtime、
  当前可用性、排队估计和 time-to-result 选择 GPU，不预先锁死卡型或使用 privileged QoS。
- 正式 job ID 核实后由 Gate E executor 维护唯一 60-minute full heartbeat，覆盖整个 Gate E job set，
  包括 PENDING；heartbeat 仅观察并唤醒 exact executor，不拥有 retry、unseal 或 Gate decision 权限。
- failed/partial attempts 保留在 write-once roots；合同内 launcher/OOM/resource/serialization 等低风险
  工程问题可 fresh retry，只补 invalid/missing cells。不得覆盖旧 evidence 或按已见 outcome 改 panel。

## 4. Phase terminal

Gate E 三模型 39/39 cells、共同 test membership、一次 combined analysis 与 fresh verifier 全部闭合，
并由 planning 判定 `PASS` 后，planning 更新中文报告的 outcome 章节与结果表，完成一次 consolidated
Phase 7 audit，然后才把 Phase 7 标记为 terminal。Gate E 后不自动创建 Gate F。
