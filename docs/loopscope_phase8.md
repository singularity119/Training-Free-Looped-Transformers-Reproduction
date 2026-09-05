# LoopScope 第八阶段：双模型固定窗口的谱软衰减实验计划

> 更新：2026-09-05。Gate A/B 已验收 PASS，双模型 debug 已完成；当前授权 Gate C 正式512题校准与8方向拟合，test和accuracy评估尚未运行。
> [总体计划](../.planning/phase8/loopscope_phase8_plan.md) 描述研究方案，[control](../.planning/phase8/loopscope_phase8_control.md) 决定当前权限，[科学合同](../.planning/phase8/loopscope_phase8_contract_v1.md) 固定参数，[Gate C handoff](../.planning/phase8/loopscope_phase8_gate_c_handoff.md) 指导当前执行任务。

## 1. 本阶段要回答什么

固定模型和循环窗口后，在每轮更新中削弱一个主要残差方向，能否提高 MMLU 5-shot 准确率？主要比较是同一个模型、同一个窗口、同一批题目下，加入谱软衰减后的循环与原版 Euler 循环之间的 acc 差异。

导师分享的 SFA 论文提供了“寻找主要方向，再减去该方向投影”的思路。这里把它迁移到冻结语言模型的循环残差上。原论文是训练期随机特征增强；我们的首版方案是独立校准后固定方向的推理期软衰减，因此称为 **SFA-inspired residual spectral damping**。目前没有第八阶段结果，不能先认定主方向有害。

## 2. 已确定的模型、窗口和原版循环

| 模型 | 实际执行的层 | 对应层间边界 | cache mode |
| --- | --- | --- | --- |
| Qwen3-4B-Base | 12、13、14、15 | B12→B16 | first |
| Qwen3-4B-Base | 13、14、15、16 | B13→B17 | first |
| Qwen3-1.7B-Base | 12、13、14、15 | B12→B16 | last |
| Qwen3-1.7B-Base | 6、7、8、9 | B6→B10 | last |

窗口编号指代码中的层索引。B12 是第 12 层的输入，B16 是第 15 层的输出；窗口 12–15 包含四层，不能误写为 B12→B15，也不需要再减一。

普通循环已由用户确认采用原阶段的 Euler 步：

$$
\Delta_t=G(X_t)-X_t,\qquad X_{t+1}=X_t+\frac{\alpha}{K}\Delta_t.
$$

代码名称是 `damped_euler`，`euler` 是它的别名。K 包含首次调用；K=2 表示窗口总执行两次。两种方法使用完全相同的 K、alpha、窗口、cache 与评测流程，比较只改变残差处理的影响。

已按用户要求固定 **alpha=1，所有 K 的总时长保持为 1**。K=2 时每步为 1/2，K=4 为 1/4；增加 K 研究的是同一总时长下更多、更小的 Euler 步。本阶段不采用 alpha 随 K 增大的方案。已确认 K=2 为主实验、K=4 为补充，两者都完整运行。

不同 K 的步长不同，因此中间状态和残差轨迹一般也不同。后续校准不能把 K=4 的前两步直接当作 K=2 的轨迹；因此每个模型、窗口、K 分别拟合方向，K=2 只用自己的方向，K=4 同理，共 8 个固定方向。

## 3. 谱软衰减怎样处理更新

每道题的完整残差形状是 token 数×hidden dimension。为了先做最小可解释实验，只取生成答案前的最后一个有效 prompt token：每道题就得到一个 d 维向量。512 道校准题各提供一行，形成 512×d 的矩阵。

在无干预循环中，用第一次重复调用（t=1）的这些向量拟合能量最大的单位方向 v。每个模型、每个窗口、每个 K 各自拟合一个 v，不能跨模型或跨 K 直接共用向量。正式评测时方向固定，不从测试题或当前 batch 重新计算。

只在 t>=1、上述 token 位置，把残差改为：

$$
\delta'_t=\delta_t-\lambda v(v^\top\delta_t).
$$

已确认 rank=1、lambda=0.5，即将该方向的分量减半，其他正交分量保持。例如残差分成“沿 v 的 4 个单位”和“其他方向的 3 个单位”，干预后变为 2 和 3，而不是一起减半。之后仍按同样的 Euler 步长加回隐藏状态。

这个操作能减小当前残差范数，但不能保证最终模型更稳定或答题更准确。主方向可能包含有用信息，也可能仅仅是共性更新。因此必须用真实配对答题结果判断。

## 4. 已确认的数据分工

任务固定为 MMLU plain 5-shot，使用多选答案的 log-likelihood 判分，不生成 CoT。五个示例来自 dev split。

用 validation 的固定 512 题无标签拟合方向，正式在完整 test 的 14,042 题上比较。512 题按 subject 比例分层抽样，固定 seed=20260905，所有模型/窗口/K 使用同一题目清单。每个配置只在自己的无干预轨迹上拟合，不使用 test 或校准题正确答案。

拟合集与评测集按样本身份隔离；两个模型共享划分，以便比较。历史 MMLU 已经被本项目使用，所以新实验可以提供受控干预证据，但不能包装为从未见过的 benchmark 验证。

## 5. 怎样看 acc 变化

每个模型/窗口分别给出：原版 acc、SFA acc、差值（百分点 pp）、题目总数、错→对题数、对→错题数和配对 95% 置信区间。主指标是 SFA−原版；同时报告 native no-loop，判断是否超过原生模型。

正式 panel 共 18 个配置：四个模型/窗口组合 × 两个 K × 原版/SFA 两种方法，共 16 个循环配置，再加两个模型各一个 native。每配置覆盖 14,042 题，共 252,756 个样本评测；8 组方向拟合共 8×512=4,096 条配置-样本轨迹。具体耗时由 debug 测量。

本轮按确认的最小比较范围，不增加同范数、随机方向或 Lower-alpha 正式对照。若出现提升，只能说明谱软衰减整体方案有效；还不能据此证明定向处理优于所有同等幅度的普通阻尼。此限制会写入结果报告。

四个 K=2 的 SFA−Loop 构成主比较组；更大的 K 不替换主结果。点估计为正但区间跨零时应说“趋势为正、证据仍不确定”。无论结果正负，均保留全部预定窗口，不报告最佳窗口代表全体。

## 6. 分 Gate 推进

| Gate | 执行工作 | 规划验收关注点 |
| --- | --- | --- |
| A：配方与最小核心 | HPC 只读环境核对，四窗口映射，投影/同范数数学参考，提出实际 hook 接入方案 | 配置正确、关键数学测试通过、下一步改动明确 |
| B：干预接入与 debug | 按已冻结合同接入真正应用的残差，两个模型端到端 smoke 与资源测量 | 关闭干预恢复原路径，处理位置/步次正确，评分与原版一致 |
| C：无标签校准 | 采集校准残差，拟合并冻结八个方向，记录有限跨步诊断 | 方向来自校准集，数据身份正确，数值有效 |
| D：配对准确率 | 运行完整冻结 panel，配对统计和报告 | 配置/样本完整，结果可复核，结论与证据一致 |

Gate D 后由规划任务做一次跨 Gate 终审。两个模型都在主范围内，不要求 1.7B 先取得正收益才能做 4B。工程 PASS 只说明可以信任当前产物；负结果和不确定结果都可以完成阶段。

## 7. HPC 执行与任务协作

每个 Gate 对应一个独立执行任务，本任务负责规划、授权和验收。只有当前 Gate 有执行权限。验收 PASS 且下一门条件满足后直接下发下一 Gate；涉及未确认科学选择时才停下来决定。

Gate A 不提交模型作业。以后 debug/smoke 一律在 debug partition、每次少于 30 分钟；正式运行前必须通过同版本、同运行路径的端到端 preflight。GPU 型号和有界并发按现场显存、上尾输入和吞吐测量决定，不为利用率改变 batch 或科学计算。

长作业由执行任务持有一个监控：smoke 每 10 分钟、probe 每 30 分钟、full 每 60 分钟。正常运行保持安静，终态唤醒原执行任务完成验证，再把终态包送回规划任务。失败尝试保留，在授权范围内修复后仅重跑无效或缺失部分。

## 8. 面向导师的简述

第八阶段准备固定 Qwen3-4B-Base 和 Qwen3-1.7B-Base 的四个窗口，在相同 MMLU 5-shot 配方下比较原版 Euler 循环与加入谱软衰减的循环。方法是先用独立校准题识别重复窗口残差的主要方向，再在推理中削弱该方向的更新。我们先验证干预确实进入原计算路径，再看配对准确率和错对翻转。本轮先完成原版与 SFA 的直接比较，同范数对照可作为后续增量。目前还没有效果结论，目标是明确这种处理在固定窗口上是否有稳定收益，以及无效时是机制还是工程路径的问题。

## 9. 留给后续阶段的动机

逐 prompt 在线估计方向已经讨论，但不纳入第八阶段。后续可研究同一 prompt 的多 token 或跨步残差，估计输入特定方向再衰减。本阶段先完成离线共享方向的明确配方，避免同时改变方向来源和干预方式。

## 10. Gate B 运行入口与工件

以下入口仅服务当前 Gate B；是否已实际通过双模型 debug，以 Gate B evidence 和规划验收为准。
在专用 HPC clone、现有解释器与离线缓存环境下生成共享池：

```bash
PYTHONPATH=src .venv-loopscope-cu121-20260711/bin/python scripts/loopscope/build_phase8_debug_pool.py --cache-dir /hpc2hdd/home/xhuang225/shared/datasets --output-dir <fresh-phase8-gate-b-input-dir>
```

该步骤仅生成 512 个 validation 身份和 8 个 debug prompt，加载对应版本 tokenizer，
不运行模型。`debug_pool.json` 的前四题用于 SMOKE_ONLY 拟合，后四题为双 tokenizer
最大长度排名选出的上尾验证题；`calibration_identities.json` 仅保存正式拟合候选身份清单。

完成 clean commit/推送/专用远端 clone 的 fast-forward 核对后，以 `/opt/slurm/bin/sbatch`
提交 `scripts/loopscope/phase8_debug.sbatch <model-index> <pool-json> <fresh-run-root> <checked-commit>`。
model-index 0 为 4B、1 为 1.7B。脚本固定 debug/29分钟/1GPU/8CPU/128G；可用第五参数 cell-index
0–3 按窗口再 K 顺序拆分小作业，须保持已有效结果不重跑和 Gate 的最多两作业并行上限。
提交前仍需按实际上尾长度与资源完成少于30分钟的估计；本入口本身不证明资源准入。

运行后验证：

```bash
PYTHONPATH=src .venv-loopscope-cu121-20260711/bin/python scripts/loopscope/verify_phase8_debug.py --run-root <completed-run-root>
```

每个 run root 保存 `command_args.json`、`env.json`、`native.json`、`summary.json`；
各 `w<start>-<end>-k<K>/` 保存 `basis.json`、仅含 answer-position t1 行的 `residual_t1.pt`
及 `debug.json`。basis 明确标记 SMOKE_ONLY，不能作为 Gate C 正式 512 方向。
得分只用于原 HFLM、None/zero 与有限性核查，不读取目标标签或计算准确率。
峰值显存、scoring 吞吐和真实 job/partition 位于运行证据中，不能从 dry-run 推断。

## Gate B 验收与 Gate C 入口

Gate B 的两模型debug作业已完成，原版/关闭干预/零强度scores一致，8个窗口/K组合的语义检查通过。这是工程有效性证据，还不是acc提升结果。Gate C复用B保存的512身份清单，正式按每模型/窗口/K拟合方向；新增采集入口经过debug后运行完整校准。输出包括8个basis、512身份与有限无标签谱/跨步诊断；不读取test正确答案。

## Gate C 执行入口

`build_phase8_calibration_pool.py --gate-b-identities <B calibration_identities.json>`
以原安全 renderer 重建512题并逐项核对原身份顺序。正式采集入口
`run_phase8_calibration.py --model-index <0|1> --cell-index <0..3> --pool <calibration_pool.json> --run-root <fresh cell> --commit <source> --scope <PREFLIGHT_ONLY|FORMAL_CALIBRATION>`。
每cell只保存native答案前残差、positions、basis、无标签diagnostics和资源summary；不保存choice scores。

`phase8_calibration.sbatch <model> <pool> <fresh group> <commit> <scope> <packing 1..3> <cell indices...>`
按固定packing分波启动独立进程，失败保留有效cell；partition/time由授权sbatch参数指定。
同版本双模型debug先覆盖512内首两题和最长两题。正式scope拟合512行，debug scope不得用于D。
`verify_phase8_calibration.py --cell-roots <roots...> --pool <pool> --scope FORMAL_CALIBRATION --require-all --output <fresh verification.json>`
在独立进程检查8配置身份、张量与Rayleigh/eigen residual（相对容差1e-4），不重复完整SVD。
