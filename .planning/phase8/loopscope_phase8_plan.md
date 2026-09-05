# LoopScope 第八阶段总体计划：双模型固定窗口的残差谱软衰减

> v0.2，2026-09-05。用户已授权本规划任务逐 Gate 下发独立执行任务并验收，直至阶段完成。
> 动态权限以 [control](loopscope_phase8_control.md) 为唯一来源；本文不独立授权后续 Gate。
> 原始机制推导和候选设计保存在 [v0.1 历史设计](loopscope_phase8_design_v0_1.md)。其中旧模型迁移、17 配置和 1019 评测方案不再是当前默认合同。

## 1. 当前研究问题

在用户固定的两个 Base 模型、四个窗口上，比较加入受 SFA 启发的残差谱软衰减前后，MMLU 5-shot 的配对 accuracy 变化。工程可行不代表方向有害，正、负和无法分辨的结果都可完成阶段。不再把第二模型的运行条件设为第一个模型必须取得正收益，也不加入 Qwen2.5。

| 模型 | inclusive 层窗口 | 层间边界 | cache |
| --- | --- | --- | --- |
| Qwen/Qwen3-4B-Base | 12:15 | B12→B16 | first |
| Qwen/Qwen3-4B-Base | 13:16 | B13→B17 | first |
| Qwen/Qwen3-1.7B-Base | 12:15 | B12→B16 | last |
| Qwen/Qwen3-1.7B-Base | 6:9 | B6→B10 | last |

层索引从 0 开始。B_s 是进入 layer s 前的 residual stream；layer s 的输出是 B_(s+1)。例如 12:15 执行代码层 12、13、14、15，不额外减一。

用户已确认普通循环使用原计划 damped_euler，所有 K 固定 alpha=1，h=1/K、总时长 K·h=1；euler 是代码别名。K 包含首次调用，t=0 是首次，t>=1 是重复。当前主方案 K=2、alpha=1、block、beta=0、batch=1、decode=bypass。K 范围和评测 split 正等待用户回复，Gate A 不依赖这两项，也不运行真实模型。

K=2 时 h=0.5，K=3 时 h=1/3，K=4 时 h=0.25。不同 K 比较的是同一总时长下的离散步数变化，不能解释为增加总演化时长。本阶段不运行 K=4/alpha=2 的固定步长分支。由于步长不同，K=2 与 K=4 的中间状态/残差前缀一般不同；不能直接截取 K=4 前两步充当 K=2 校准轨迹。若纳入多个 K，须在 Gate B 前明确 basis 的拟合 K 与复用范围，并在工件标注，不继承旧固定步长方案的前缀复用假设。

## 2. 方法与公平比较

设每步原始残差 Delta_t=G(X_t)-X_t。对校准集每个样本，只取最终非 padding 的 pre-answer prompt token，形成 N×d 的残差矩阵。这里 N 是校准题数，不是运行 batch size。

主方案是在无干预轨迹的 t=1 残差上拟合每个模型、每个窗口独立的单位 top-1 右奇异方向 v，使用未中心化、未做逐行归一化的矩阵。方向在评测前冻结，不看校准 gold，也不从评测 batch 在线估计。

干预只作用在该位置、t>=1：

$$
\delta'_t=\delta_t-\lambda v(v^\top\delta_t),\qquad X_{t+1}=X_t+h\Delta'_t.
$$

默认提案 rank=1、lambda=0.5，t=0 和其他 token 沿原路径。它是 SFA-inspired 方法；不是原论文少步随机幂迭代加完整投影删除的逐字复现。用户的 SFA/K 确认尚待回复时，不把提案写成已冻结事实。

重要工程边界：原始低精度 residual 与 FP32 测量差分分开记录；只观察 audit callback 不能代替改变实际应用更新。关闭干预时必须走原算法路径，不能经代数重写声称浮点输出必然相同。cache first/last 按模型保留；若全序列 choice scoring 中 cache 设置不改变分数，也如实记录，不能为了让 cache 生效而改任务。

## 3. 数据与统计方案

MMLU 为 plain 5-shot、多选 choice log-likelihood accuracy，不生成 CoT。五个示例使用 dev split，与目标题分开。拟合建议使用 validation 的固定 subject-stratified 512 题；方案一在完整 test 14042 题评测，方案二沿用 validation 剩余 1019 题，待用户选择。禁止在此选择明确前读取目标 outcome 或提交模型实验。

所有窗口共享相同的拟合和评测 sample identities，但按模型/窗口分别拟合方向。历史 MMLU 已被项目使用，不能宣称 untouched benchmark。5-shot renderer、token 位置和四个 choice scoring 须与现有 evaluator 对齐；不能拿未经校验的自定义评分替代原 baseline。

主指标为每个模型×窗口同 K/alpha 下：DeltaAcc=Acc(Spectral)-Acc(Loop)，以百分点 pp 报告。完整配对输出同时给出正确数、总数、wrong→right、right→wrong、subject macro 和 paired bootstrap 95% CI。计划冻结 10000 次 subject-stratified bootstrap、seed=20260905；四个 K=2 主 contrasts 如给出显著性，采用 exact McNemar + Holm family correction。更大 K 只作事先声明的补充，不能用最佳 K 替换主结果。

最小正式 panel：两个模型 native no-loop，以及四个模型/窗口 cell 的 Loop 与 Spectral，共 10 个配置（K=2）。若加入同位置同范数 Uniform 控制则为 14 个，用于区分定向收益与单纯减小残差；该控制和 K=4 的正式预算由 Gate B 前合同冻结。本阶段不默认运行旧 17 配置网格、随机种子大 panel 或额外模型。

## 4. Gate 路线与终态

| Gate / 执行任务 topic | 目标与最小证据 | admission |
| --- | --- | --- |
| A / Recipe-and-Spectral-Core | HPC 只读 provenance、四窗口映射、纯 Python 投影/同范数参考与关键测试、实际干预接入点方案 | 当前用户阶段授权 |
| B / Intervention-Debug | 最小 opt-in 真正残差干预、两个模型的 debug 端到端 smoke、原版本关闭路径等价、资源测量 | A PASS；待定科学选择解决，运行合同冻结 |
| C / Residual-Calibration | 无标签校准采集、拟合四个方向、跨步诊断与冻结 basis | B PASS；同版本 debug preflight 有效 |
| D / Paired-Accuracy | 对冻结完整 panel 采集与配对统计，输出全部模型/窗口的 acc 差异 | C PASS；方向和评测身份冻结，正式预算明确 |

每 Gate 一个独立用户可见任务，标题为 execute-LoopScope-<topic>-第8阶段-Gate <letter>。只有当前 Gate 绑定 executor。规划每 Gate 只验收 1–3 个决定性事实；最后做一次跨 Gate 阶段审计。工程 PASS 不表示 accuracy 必须提高，机制弱或结果为负也须完整报告。无法计算有效方向或数据身份/评分路径不成立时先解决材料性问题；不能静默调参数绕过。

阶段终态为：D 的冻结 panel 完整、配对统计和报告可解释，规划终审给出支持/不支持/不确定的结论；或者记录无法在当前科学约定内解决的材料性阻塞。K=4 若获授权，在 D 内预定完成，不按 K=2 的结果临时选择是否运行。

## 5. HPC 与协作

使用 hpc2-hkustgz 的 LoopScope 专用 clone 与 workspace。Gate A 仅 SSH 只读及 CPU/import/config 检查，不提交 Slurm、不运行模型、不读 outcome。后续 debug/smoke 一律 debug partition、每次 <30 分钟；正式工作须有同 commit/launcher/env/模型运行路径/producer 的成功端到端 preflight。

正式资源由执行任务按现场合法优先级、显存和吞吐证据选择；不固定 GPU 型号。保持 batch 与科学数值不变，验证代表性上尾长度后再采用有界同卡 packing。失败尝试保留，允许授权范围内只重试无效/缺失工作，禁止覆盖历史结果。

长作业由执行任务持有一个 heartbeat：smoke 10 分钟、probe 30 分钟、full 60 分钟；正常等待保持安静。终态暂停 monitor 并通过工具唤醒原 executor；executor 完成后把 GATE_X_FINAL_AUDIT 或真正 BLOCK 发回规划任务并保留可见确认。规划不轮询。

阶段连续执行授权已经成立，但每个 Gate 的 GPU、写路径、protected-code 变更与 outcome 权限仍须在 handoff 明确授予。遇到普通工程权限不足，由规划按既有项目 operational delegation 判断并补最小授权；不擅自改变科学配置或处理凭据。
