> 2026-09-05 接续更新：[Phase 8 control](.planning/phase8/loopscope_phase8_control.md) 已激活，用户授权规划任务逐 Gate 推进；当前 Gate A，尚无模型实验权限。下文 2026-09-02 的 Phase 7 “当前”叙述是归档快照。

# LoopScope 项目记忆与接续入口

更新日期：2026-09-02。本文是交接摘要，详细证据保留在 [.planning](.planning/README.md)。
本次核对了本地代码、远端 Git 分支和现有 control/审计文档；没有重新查询 HPC 作业或重算实验。
实验状态和数值均是已归档的审计记录，不代表本次重新复现实验。

## 1. 开始工作时先读什么

1. 阅读原有 [AGENTS.md](AGENTS.md)，然后阅读本文。
2. 读取 [Phase 7 control](.planning/phase7/loopscope_phase7_control.md) 顶部状态区块；这是当前最新阶段的控制入口。
3. 按任务读取相应 contract、handoff、方法规则和 [阶段 runbook](.planning/README.md)。不要一次性加载全部历史记录。
4. 新设备首次接续先完成 [新设备说明](docs/new_device_handoff.md)，确认 checkout、分支、兼容链接及本机访问环境。

用户最新明确指令优先；事实核对使用对应代码、配置和原始工件。授权范围由适用 control/handoff 确定。
本文和 README 不能覆盖当前 control，旧 handoff 中的授权也不能自动重新生效。
迁移保留了原 `AGENTS.md`；其中旧设备路径、初始化语境和过时阶段叙述须结合新设备说明理解。

## 2. 研究问题和当前阶段

LoopScope 研究：能否只利用冻结模型一次普通前向产生的内部信号，在运行 loop outcome 前预测有益的 loop window。
重点是无标签选窗的证据，而不只是找到一个有正收益的固定窗口；目前不能宣称通用自动选窗器已得到验证。

- 当前最新归档状态：`PHASE7_TERMINAL_PASS_AFTER_GATE_F`。
- `ACTIVE_GATE=NONE`，`AUTHORIZED_EXECUTOR=NONE`；历史 Gate A–F 均已关闭。
- [Gate F 与 Phase 7 重新终审](.planning/phase7/loopscope_phase7_gate_f_and_phase_reclose_audit_pass_20260816.md) 是最近的阶段结束依据。
- 没有已激活的 Gate G 或 Phase 8；复制仓库、恢复文档或更换设备不会启动新实验。
- 新实验需要明确问题、冻结配方、成功/反证标准和新的 control/handoff；不要从旧线程 ID 恢复已撤销的执行权限。

## 3. 各阶段保留下来的主要结论

| 阶段 | 实验对象与主要发现 | 必须保留的限制与依据 |
| --- | --- | --- |
| Phase 1 | Qwen3-1.7B-Base × MMLU 5-shot，完成 width-4、K=2 的初始 grid，形成后续研究锚点 | 已看过 outcome，属于开发背景。[control](.planning/phase1/loopscope_phase1_control.md) |
| Phase 2 | 对重复循环的“可迭代精炼区”做机制检验；H1=`PERTURBATION`，NCA=`NCA_NONDISCRIMINATIVE` | 不能把 native-path alignment 当成朝正确答案移动；后续 fixed-horizon profile 是单独的探索性补充。[control](.planning/phase2/loopscope_phase2_control.md) |
| Phase 3 | 在 Qwen3-1.7B 上恢复并保留 `12:15`；width-4 主线为 `WIDTH4_RANKING_SIGNAL_ONLY` | blind enrichment 未成立；targeted variable-width candidates 未超过 `12:15`；单层 15 的 K=2/3/4 也未超过该锚点。[control](.planning/phase3/loopscope_phase3_control.md) |
| Phase 4 | Qwen3-4B-Instruct-2507 × MMLU-Pro 5-shot CoT，prefix 全词表 entropy/KL 选窗后做 full decode | selector 仍为 `ABSTAIN`；High–Low enrichment 为正不代表选窗成功；六个 blind gains 均为负，已知 `15:18` 是 panel-local best。只支持 same-population outcome-blind transductive 表述。[control](.planning/phase4/loopscope_phase4_control.md) |
| Phase 5 | Qwen3-4B-Base × MMLU 5-shot，主线终态 `H5_ABSTAIN_WITH_RANKING_RESULT`；后续方法/几何诊断扩展完成至 Gate I | ranking enrichment、absolute gain、known comparator 与 prospective selected success 必须分别报告，扩展不追溯覆盖原主线结论。[control](.planning/phase5/loopscope_phase5_control.md) |
| Phase 6 | Qwen3-4B-Instruct-2507 × MMLU 5-shot，V3 selected inclusive `14:16`；完成五-cell 固定 panel | selected gain `+0.0926 pp`，CI 跨零；`15:18` gain `+0.5697 pp`，95% CI `[+0.1496,+0.9970] pp`，是该 panel 唯一 CI 全正的 cell。[control](.planning/phase6/loopscope_phase6_control.md) |
| Phase 7 | Qwen2.5-3B、Llama-3.2-3B、Gemma-2-2b × MMLU 5-shot，完成验证轨迹、V3.1、Gate E 和追加 Gate F outcome | 没有确证正向准确率收益；Gate F 为已知 Gate E outcome 后追加的探索性实验。[终审](.planning/phase7/loopscope_phase7_gate_f_and_phase_reclose_audit_pass_20260816.md) |

Phase 6 原 MMLU-Pro pre-answer D-2 分支被用户终止，仅完成 6/8 shards；partial 结果不是完整正式证据。
不得自动补齐、解封、与后续 MMLU 分支合并，或将 partial 分析当作 selected-window 结论。

## 4. 最新方法版本与 Phase 7 结果

方法演进为 entropy–KL CONSENSUS → relative biphasic V1 → absolute-rate V2 → aggregate common-turn V3.1。
版本与差异见 [选窗规则索引](.planning/selector_rules/README.md)，当前已使用的定义是
[AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE@3.1.0](.planning/selector_rules/aggregate_common_turn_v3_absolute_rate.md)。

- 输入是同一 identity 的逐层 `H` 与 `D=KL(p_l || p_L)`，按 category-macro 聚合。
- V3 eligibility 要求 `Scorable AND NetPositive AND RateStable AND AggregateCommonTurn`。
- 点排名分数是 `sqrt(G_H * G_K * Q_H * Q_K)`；combined-rank frequency 门槛为 `0.80`。
- 正式 V3 bootstrap 为 2,000 次、seed `20260801`；hidden geometry 只作诊断，不进入选窗。
- 合法 `ABSTAIN` 是科学终态；不得为获得窗口而调门槛，也不能把工程错误伪装成弃权。

Phase 7 的候选域按真实模型深度生成，width 为 3/4/5/6；三个模型有 42/26/26 个候选。
[Gate D 审计](.planning/phase7/loopscope_phase7_gate_d_audit_decision_pass_20260815.md) 记录：

| 模型 | selector 结果 | 后续 outcome 范围 |
| --- | --- | --- |
| Qwen2.5-3B | `SELECTED_WINDOW [14,17)`，frequency 0.973 | Gate E 取 width-4 top-3；Gate F 取全候选 point-rank top-2 |
| Llama-3.2-3B | `SELECTED_WINDOW [11,14)`，frequency 1.000 | Gate E 取 width-4 top-3；Gate F 取全候选 point-rank top-2 |
| Gemma-2-2b | `ABSTAIN_COMBINED_RANK_UNSTABLE`，winner frequency 0.467 | Gate E 取仅有的两个 finite width-4 候选；Gate F 仍只是弃权后的探索 |

Gate E 是 35 cells（13/13/9），Gate F 是 24 个新增 loop cells 加三个已验证 Gate E baselines，共 27 cells。
两组均用独立于 validation-1531 的 MMLU test-14042；完整闭合后才解封 outcome。
Gate E 的微小正点估计 CI 跨零；Gate F 的 Qwen/Llama contrasts 全为负，Gemma 仅一个配置有约 `+0.0142 pp` 且 CI 跨零。
这两组结果必须保留 exploratory、nominal single-cell、non-confirmatory 限定。

## 5. 容易重复犯的解释和实施错误

- **工程 PASS 不等于科学假设成立。** score、eligibility、selected、accuracy、gain 是不同对象。
- 窗口写明 notation：旧 CLI `12:15` 为 inclusive，等价于半开 `[12,16)`；Phase 7 表中的 `[14,17)` 不能抄成 inclusive `14:17`。
- boundary `B_s → B_(e+1)` 对应 inclusive blocks `s:e`。Choice Entropy 与全词表 entropy 是不同输入空间。
- 解释差分先确认符号：报告常用 exit-minus-entry；旧卡的 entropy drop 可用 entry-minus-exit，不能直接混用。
- Euler 的 `alpha` 是 total horizon，单步系数为 `alpha/K`。fixed-step 与 fixed-horizon 是不同实验。
- final norm 只应用一次；FinalNorm 前 raw boundary 与 post-norm `hidden_states[-1]` 不能混用。
- 全量 choice-loglikelihood 与 cached one-token decode 路径不同；不能把某次 cache/full/bypass 等价外推到所有评估。
- 正式显存预算要覆盖真实模型、batch 和长序列峰值；小样本或不同 GPU 的 smoke 不足以证明 formal 资源可行。
- 计算节点 launcher 需固定解释器并自包含；不能假定登录节点的 Git、Modules 或 PATH 都存在。
- 保留失败/取消/partial 记录；重试必须遵守当前 Gate 语义并用 fresh run root，不能覆盖已有证据。

## 6. 尚未形成新实验结论的后续想法

2026-08-30 的导师沟通提出借鉴 SFA：检验额外循环 residual 是否在主导方向上持续集中/累积，
比较不同窗口的 residual spectrum、能量集中、方向稳定和投影增强；机制得到支持后，
再比较 residual 主方向 soft damping 与统一降低 `alpha`。
这个方向连接 “where to loop” 与 “how to loop”，未来才可能扩展到 “when to stop”。

来源是历史沟通任务 `01a050e2-d74b-7183-aa84-5c4ce8a1681c` 的摘要，属于记忆导入；
本次没有在项目 `.planning` 中核到对应新 Gate 合同或结果，不能称为已证实机制或已授权实验。
在新设备继续这一方向前，先向当前项目任务核对进展，再形成最小可证伪 hypothesis card。

## 7. 代码、环境与实验工件入口

- 核心实现：[src/tflt/loopscope](src/tflt/loopscope)；冻结配置：[configs/loopscope](configs/loopscope)。
- Phase 7：[runbook](docs/loopscope_phase7.md)、[trajectory contract](configs/loopscope/phase7_trajectory_contract.json)、[Gate E card](configs/loopscope/phase7_gate_e_outcome_card.json)、[Gate F card](configs/loopscope/phase7_gate_f_outcome_card.json)。
- 轨迹、V3、outcome 的 launcher/verifier 都在 [scripts/loopscope](scripts/loopscope)，先读合同再选择入口。
- HPC 环境：[envs/hpc2](envs/hpc2/README.md)、[remote runbook](docs/remote_hpc2.md)。历史已审计解释器为专用 checkout 下 `.venv-loopscope-cu121-20260711/bin/python`，使用前核对现场状态。
- HPC2 专用代码目录：`/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope`。
- HPC2 实验根：`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope`，包含 `inputs/runs/staging/artifacts`。
- Phase 1 canonical run 仍在旧实验根 `training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-20260711-053022`，不能迁移或改名。
- Gate E canonical analysis：上述新实验根下 `runs/phase7-gate-e-20260816T041724Z-formal-remaining24-gitless/analysis/combined_analysis.json`。
- Gate F canonical analysis：上述新实验根下 `runs/phase7-gate-f-20260816T073546Z-formal/analysis/combined_analysis.json`。
- Gate C/D 的具体 sealed/scoring roots 从 [Phase 7 control](.planning/phase7/loopscope_phase7_control.md) 及相应审计文件检索，不猜路径。
- 中文报告与图表保留在旧工作区外层 `资产/报告/phase<N>/`、`资产/figures/`，没有随本次 Git 迁移上传；需要展示原图时单独迁移。

## 8. 维护方式

只在阶段终态、关键结论、方法版本或主要下一步发生变化时更新本文；每个结论附证据入口。
新增阶段后更新本文当前 control 链接和 `.planning/README.md` 的索引，旧 Gate 记录按历史保留。
不要追加逐命令日志，不将个人 Codex memory 或聊天全文复制进项目；旧设备路径和线程 ID 仅作来源线索。
新设备接续时以 Git 实际已提交内容为准；迁移范围和历史路径映射见 [迁移说明](docs/new_device_handoff.md)。
