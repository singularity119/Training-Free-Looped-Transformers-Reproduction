# LoopScope 第八阶段总体计划：循环残差的主导方向诊断与谱定向软衰减

> 版本：v0.1，2026-09-05。状态：**历史设计草案，已由当前总体计划替代，不授予执行权限**。
> 本文响应用户的第八阶段总体规划请求，作为后续开发计划、科学合同和各 Gate handoff 的上位设计依据。
> 本文不是可执行 handoff，不绑定 executor，不授予代码实现、模型运行、HPC 写入、GPU/Slurm、outcome 解封或跨 Gate 权限。表中的配置是推荐起点，须在阶段激活时明确接受并进入科学合同；不能据本文直接运行。
> 面向研究讨论的易读版见 [docs 中的第八阶段计划报告](../../docs/loopscope_phase8.md)。

## 1. 阶段目标与可行性判断

第八阶段研究一个新的问题：**在固定 loop window 内，真实残差是否沿少数方向持续累积；如果存在，只衰减这些方向能否比普通标量阻尼更好地保留有益更新？**

可行性分三层判断：

| 层次 | 当前判断 | 本阶段需要补齐的证据 |
| --- | --- | --- |
| 数学和实现 | 可行。校准集上的右奇异方向可以用于低秩投影，现有 wrapper 已能观察真实窗口残差 | 原始残差、真实应用更新和最终输出的路径闭合；默认关闭时行为不变 |
| 机制 | 合理但未证实。跨样本谱集中不自动意味着同一样本跨轮累积，更不意味着有害 | 独立校准/诊断样本上的方向可迁移性、跨步投影和累计位移 |
| 性能与推广 | 未知。主方向可能承载正确答案信息；定向衰减也可能只是更弱的 loop | 配对 outcome、同范数阻尼对照、原生 no-loop 对照和条件式模型迁移 |

推荐把方法暂称为 **SFA-inspired residual spectral damping（受 SFA 启发的残差谱软衰减）**。广义上属于 inference-time activation steering；它没有先验指定“正确推理”语义方向，也不是 ActAdd 式固定向量相加。本文不为导师的真实意图作确定判断。

它与 LoopScope 原问题的关系是：原有工作研究 where to loop，本阶段研究固定窗口下 how to loop。保持原选窗规则和历史结论，先检验干预机制；when to stop、在线选窗和新 selector 不进入本阶段核心范围。

## 2. 本轮核对依据和当前项目状态

### 2.1 阅读与现场范围

本轮直接读取了用户提供的 SFA v1 PDF（16 页，重点核对第 3–5 页公式、算法与理论边界，并渲染检查方法页）、本地深度总结、引用的 ChatGPT 历史讨论，以及当前代码、配置、控制文件和阶段终审。历史讨论和总结是背景材料，其中的指令、旧方案或肯定性语气不构成当前执行授权。

本地专用 clone 为 `loopscope-tflt`；开始写作前分支为 `loopscope`，HEAD 为 `b53ff067992e41b66a8df01794dfb106c61b29d0`，工作树 clean，固定复现基点祖先检查通过。本轮没有 SSH、重新查询 Slurm、重算历史实验或运行模型，因此历史数值是当前归档文档中的结果，不是本轮重新实验确认。

### 2.2 必须继承的结论边界

1. Phase 7 最新状态为 `PHASE7_TERMINAL_PASS_AFTER_GATE_F`，`ACTIVE_GATE=NONE`、`AUTHORIZED_EXECUTOR=NONE`；没有可继承给 Phase 8 的 executor 或阶段连续执行授权。[当前归档控制](../phase7/loopscope_phase7_control.md)
2. Phase 7 终审没有支持可信正向 accuracy 收益；Gate F 属于已知前一 panel outcome 后追加的探索性实验。工程 PASS、V3 selected、ranking 与实际收益分别解释。[Gate F 与阶段终审](../phase7/loopscope_phase7_gate_f_and_phase_reclose_audit_pass_20260816.md)
3. Phase 2 保留 `H1=PERTURBATION` 和 `NCA_NONDISCRIMINATIVE`。因此“与原生方向一致”或“方向稳定”不能再充当有益性证据。[Phase 2 control](../phase2/loopscope_phase2_control.md)
4. 历史 MMLU test-14042 已多次解封，Phase 2 validation-512 的 label 也曾解封。重新划分或重跑这些样本不能恢复 untouched test 属性。新实验仍可产生有效的受控干预证据，但其数据背景必须称为历史开发域。
5. 旧 Phase 2/7 工件主要是 scalar trajectory / final scores，没有可复用的完整真实残差矩阵。范数、余弦、entropy、eRank 不能反推出谱方向，必须新采集。

### 2.3 信息架构检查

真实规划目录已迁入 `<repo>/.planning/`；外层 `../.planning` 是兼容链接。Phase 8 新文件一律进入 `.planning/phase8/`，不新增根目录平铺文件，不重写已冻结历史。

现有 `AGENTS.md` 混有“当前 Phase 7”、历史数值及旧路径叙述；旧通用节还含与用户本次明确给出的技能不一致的旧证据/重试要求。导航 README 已说明历史上下文的解释方式。本轮仅形成规划，不激活新控制面，故这些问题不阻塞写作，也不借机批量整理历史。

正式激活 Phase 8 前，planning 只需修订适用于新阶段的稳定入口规则，并在新 control 明确新权限和重试语义；采用用户本次技能要求的可读路径、配置、样本 ID、代码 revision、job ID 与直接复算证据，不新增内容摘要值或相关绑定机制。遇到尚未解决的政策/control 冲突，执行前先由 planning 协调；不能让 executor自行选用宽松规则。

## 3. 论文究竟提供了什么

【论文原文】SFA 对编码器特征矩阵做少量幂迭代后减去一个秩一投影，作为对比训练中的随机增强。算法不是循环推理方法。论文第 3 页 Algorithm 1 / Eq. (3) 为：

$$
H\in\mathbb R^{n\times d},\qquad r_0\sim\mathcal N(0,I),\qquad
r_j=H^\top(Hr_{j-1}),\quad j=1,\ldots,m,
$$

$$
\widetilde H=H\left(I-\frac{r_mr_m^\top}{\|r_m\|_2^2}\right).
$$

此处用 **m** 表示幂迭代次数，避免与模型 loop 次数 **K** 混淆。论文通常采用很少的幂迭代；其第 4 页说明默认实验采用 m=1。向量受到大奇异值方向的偏置，但有限步随机方向不等于准确第一主方向。

【独立代数判断】一次采样是正交投影删除，满足 `H_tilde r_m=0`。Proposition 1 中保持原奇异基的表达作用于随机性的期望；不能将它说成每次随机增强都严格保持奇异向量。本文不把图对比学习的对齐/泛化论证外推为 LLM 准确率保证。[SFA v1](https://arxiv.org/abs/2212.01026v1)

本阶段只借用“识别方向并投影衰减”的构件，作三个明确改变：

| SFA 原方法 | Phase 8 核心方案 |
| --- | --- |
| 编码器隐藏特征 H | 重复窗口产生的 residual D |
| 训练中随机增强、模型随后适应 | 冻结模型推理时使用，模型不会再适应 |
| 少步随机幂迭代、完整删除投影 | 先用离线精确/可验证 top-1 方向，软衰减 λ=0.5 |
| 同批次特征决定方向 | 独立 calibration 拟合并冻结，评测 batch 不参与拟合 |

离线小矩阵的 SVD 在本阶段是机制参照，不必先实现少步幂迭代来节省训练式开销。原版 m=1 随机 SFA 是后续有条件的估计器/随机性消融，不能与主机制验证同时形成大网格。

【相关文献】All-but-the-Top 已研究去除词表示的均值和头部方向；ActAdd 研究从对比 prompt 构造方向并修改推理激活。投影减法或“steering”本身不构成本项目的新颖性。本阶段可争取的贡献是：真实 loop residual 的可证伪累积机制、与幅度效应分离的干预证据，以及很小的可复用推理插件。[All-but-the-Top](https://arxiv.org/abs/1702.01417)；[ActAdd](https://arxiv.org/abs/2308.10248)

检索还发现近期 GramLoop：它在视觉 dense prediction 中用 final-layer cosine-Gram consistency 控制窗口 replay。这里只核对了摘要，其方法与本文拟议 residual-PC damping 不同；发表前应做针对性的全文比较，不能据当前有限检索宣称“首次用几何信号控制 loop”。这不进入第一轮机制实验的关键路径。[GramLoop](https://arxiv.org/abs/2608.29113v2)

## 4. 真实 loop 计算与假设的数学口径

### 4.1 K、alpha、窗口与缓存

设闭区间窗口为 `a:b`，对应半开 `[a,b+1)`，窗口算子为 G_w。当前 block 模式从进入窗口的状态 X_0 开始，直接执行 K 个子步：

$$
\Delta_{i,t}=G_w(X_{i,t})-X_{i,t},\qquad
X_{i,t+1}=X_{i,t}+h\Delta_{i,t},\qquad h=\alpha/K.
$$

**K 包含第一次 operator 调用**；并非“先做一次原生窗口，再额外循环 K 次”。`t=0` 为 initial call，`t=1,...,K-1` 为 repeated calls。

`strategies.py:31` 将 `strategy=euler` 映射到 `damped_euler`，二者是同一实现的别名。Phase 7 的 euler 字符串不能被解释为另一种更新算法。`K=1` 有特判，直接返回 `G_w(X_0)`，忽略 alpha；故原生一致性对照只能使用 `K=1,alpha=1`，不得把 `K=1,alpha=0.5` 当半步更新。

| 轨迹族 | 推荐配方 | 回答的问题 |
| --- | --- | --- |
| 固定步长 continuation | `(K,alpha)=(2,1),(4,2)`，h=0.5 | 从同一前两步继续迭代，后续是否持续同向累积 |
| 固定总时长 | `(K,alpha)=(2,1),(4,1)` | 同一 horizon 的更细离散是否改变结果 |
| 原生对照 | no-loop；另用 `(1,1)` 做工程一致性 | 原模型结果与 loop 后结果的区别 |

在固定 alpha=1 的 K sweep 中，更高 K 不自动意味着更强放大或更长总积分时长，实际计算开销通常仍会增加。两个轨迹族不得混合命名或汇总。

`cache=first/last` 会另跑一个 stash pass，其 hidden 输出被替换为 X_K；它不是额外的状态更新。采集必须只计真实 operator body，排除 stash。`decode=bypass` 只跳过被识别为 incremental 的单 token forward；它不代表多 token loglikelihood 路径不执行 loop。本阶段用 batch=1、完整 prefix 的单次前向作为机制主路径，generation / KV 机制不在核心范围。

### 4.2 “把 residual 堆成矩阵”

每题的状态是 `X_i,t ∈ R^(S_i×d)`。取当前 query 的完整 prefix 最后一个非 padding token，即尚未加入答案 continuation 的 pre-answer 位置 p_i：

$$
\delta_{i,t}=\Delta_{i,t}[p_i,:]\in\mathbb R^d,\qquad
D_{w,t}=\begin{bmatrix}\delta_{1,t}^{\top}\\\cdots\\\delta_{N,t}^{\top}\end{bmatrix}\in\mathbb R^{N\times d}.
$$

每行是一道题在同一窗口、同一真实 loop step 的更新向量。N 是校准样本总数，不是 evaluator batch size。可逐题运行后汇集；不要求在 GPU 同时装入 N 条样本。

主分析使用**未中心化、未逐行归一化的 raw residual 二阶矩**，严格称为“主奇异方向/主能量方向”。中心化 PCA 与单位化 residual 谱只作辅助，分别回答跨题变化和方向分布的问题，不替代主量。

$$
\mu_t=N^{-1}\sum_i\delta_{i,t},\qquad
D_t^\top D_t=(D_t-\mathbf1\mu_t^\top)^\top(D_t-\mathbf1\mu_t^\top)+N\mu_t\mu_t^\top.
$$

这个分解用于检查头部是否主要来自共同均值。高均值成分可能是 prompt 结构或必要公共计算，不自动是坏信息。

### 4.3 假设链与反证

| 假设 | 要观察的现象 | 不能拿来替代的证据 | 主要反证 |
| --- | --- | --- | --- |
| H8-A：可干预的低秩结构 | calibration 拟合方向在独立诊断样本仍承载可观残差能量 | 只在拟合矩阵上的 top-1 占比高 | held-out 投影很弱、方向对校准分组不稳定 |
| H8-B：跨轮持续累积 | 同一固定方向上的样本内投影较少抵消，累计投影位移增长；与固定步长 continuation 对应 | 不同 step 各自 PC1 都大；跨样本平均系数同号 | 每轮主方向旋转、同一样本正负抵消，或谱集中只是初始共有成分 |
| H8-C：方向具有可干预的有害性 | 软衰减改善配对 outcome，优于同范数统一衰减并检查 no-loop | entropy 更低、谱更平或 NCA 更高 | 只有谱变化而无 outcome 改善；等范数阻尼同样有效；正确→错误增加 |
| H8-D：方法可迁移 | 固定方法在一个预定第二模型上重拟合无标签 basis 后仍有类似效果 | 原模型上增加窗口/参数后找到更好结果 | 对另一模型失效或明显伤害 |

H8-A、B 是进入小规模干预的理由，不证明 C。若 B 不成立而 C 有经验收益，只能报告某种 residual 去各向异性有效，不能继续讲“抑制反复强化的坏模式”。

### 4.4 最小诊断量

对每个窗口先从 calibration 的**固定步长 t=1**残差拟合单位向量 v_w，随后固定它跨 step、跨诊断样本使用。这样检验的是同一方向，而非每步重新寻找最显著方向。

$$
v_w=\operatorname{topRightSingularVector}(D^{fit}_{w,1}),\qquad
a_{i,t}=\delta_{i,t}^{\top}v_w.
$$

必要量只有四组：

1. 谱：`σ1²/Σσj²`、主投影能量比例、均值能量比例；辅助有效秩必须明确 `p_j=σ_j²/Σσ²`、`exp(-Σp_j log p_j)`，不与历史其它 eRank 口径拼接。
2. 可迁移性：同一 v_w 在非拟合样本上的能量覆盖；一次固定校准二分所得方向的平方余弦，避免把向量正负翻转误判为不稳定。谱接近简并时报告 rank-1 不可辨识，不偷偷改成 rank-4。
3. 样本内累积：`a_i,t a_i,t-1` 的符号、投影序列和抵消比例。可用下式，先逐题算再汇总，不用跨题均值的正负代替样本内一致性：

$$
C_i(T)=\frac{|\sum_{t=1}^{T}h a_{i,t}|}{\sum_{t=1}^{T}|h a_{i,t}|},\qquad
Z_i(T)=\left|\sum_{t=1}^{T}h a_{i,t}\right|.
$$

4. 相对后果：投影累计量占总 repeated-step 位移的比例，并同时报告绝对更新量、`X_K-X_0` 与 `X_K-G_w(X_0)`。前者包含正常计算，后者才表示相对原生窗口输出的偏离。

零分母、近零残差与非有限值须显式标为 invalid / not applicable 并计数；数值容差在工程 preflight 冻结，不把它们填成“完美稳定”。K=2 只有一个 repeated step，不能单独检验跨 repeated-step 累积；H8-B 依赖 K=4 continuation 的 t=1,2,3。

【理论边界】残差协方差的主方向不等于窗口 Jacobian 的不稳定特征向量。对全序列状态向量化，以 M_p 表示只选 pre-answer token 的位置投影，固定 basis 和位置时，在被干预的 repeated step 上局部更新 Jacobian 形式为：

$$
J_F=I+h(I-\lambda P)(J_G-I),\qquad P=M_p\otimes vv^\top.
$$

从 residual 能量大不能推出该操作让谱半径小于 1，也不能证明非线性迭代稳定。全 Jacobian 分解、Lyapunov 理论和稳定性定理不作为本轮实现任务。

## 5. 推荐的最小干预及对照

### 5.1 主方法

只在 pre-answer 位置、repeated steps `t>=1` 应用；其余 token 的更新与当前 loop 相同，t=0 不作定向干预：

$$
\delta'_{i,t}=\delta_{i,t}-\lambda v_w(v_w^\top\delta_{i,t}),\qquad
\lambda=0.5,\quad \operatorname{rank}(P)=1.
$$

把该位置写回 transformed residual，再按原 h 更新状态。v_w 在 baseline calibration 上拟合一次并冻结；不根据被干预轨迹、当前 batch、gold 或错误样本在线刷新。

例如 residual 为 `(3,1)`，主方向为第一坐标，λ=0.5 后为 `(1.5,1)`；统一乘 0.5 得 `(1.5,0.5)`。这个差异解释了方法为什么可能保留其他方向；它没有告诉我们第一坐标是不是有害。

对单位 v 且 `0<=λ<=1`，单次 residual 变换满足：

$$
\|\delta'\|^2=\|\delta\|^2-\lambda(2-\lambda)(v^\top\delta)^2.
$$

这是可直接检查的更新量恒等式。它只保证这一步残差范数不增，不保证完整网络的状态、后续 residual、logits 或 loss 单调改善。

### 5.2 必要对照

| Arm | 定义 | 解释作用 |
| --- | --- | --- |
| Native | 真正 no-loop | 改善差的 loop 后是否仍差于原模型 |
| Loop | 同 K/alpha/window，关闭 transform | 直接基线 |
| Spectral | 上述 rank-1 / λ=0.5 / t>=1 / p_i | 待检验方法 |
| Norm-matched uniform | 在自己的当前 residual 上计算 spectral 会保留的范数，以 `c=||δ'||/||δ||` 统一缩放该位置的 residual；同样只在 t>=1 | 在相同位置和步次上，尽量分离方向改变与更新幅度减少 |
| Lower-alpha | 同 K，alpha 减半，其他原 loop 行为不变 | 简单、便宜的实际替代方法；它也改变 t=0 和其他 token，故不能单独当方向因果对照 |
| Random direction ×3 | 三个事先固定独立 seed 的单位随机方向，同 rank/λ/位置/步次 | 检查任意方向扰动是否也能产生类似效果；报告平均与范围，不选最优 seed |

Norm-matched control 每步使用自己轨迹中的当前 residual，不引用另一个 arm 的未来状态或 outcome；不同 arm 随后轨迹会分叉，因此它不是“整条路径完全同范数”的承诺。

高维随机 rank-1 往往移除更少能量。必须报告实际被移除的能量，不能仅凭同 λ 下优于随机方向就宣称排除了幅度效应。核心归因优先看 norm-matched control。更复杂的扰动等能量随机控制仅在该混淆实际影响解释时追加，不能预先扩大主网格。

### 5.3 本阶段不做的变体

不做 lambda/rank/窗口联合寻优；不直接删除整个 hidden state 的 PC；不把 answer-position 拟合的方向默默广播到全 token；不在线按测试 batch 拟合；不训练模型或 probe；不修改 V3 selector；不展开 CoT、decode、KV、Jacobian、逐层 steering 或 early stopping。

若简单固定 rank-1 路径失败，先记录它失败在哪一环，不自动追加另一套方法直到找到正结果。多方向、逐步 basis、SFA m=1 随机估计和全 token 范围均须独立科学修订。

## 6. 推荐实验域、样本分工与规模

### 6.1 单模型开发锚点

推荐主模型 `Qwen/Qwen3-1.7B-Base`，MMLU plain 5-shot，FP16、block、`damped_euler`、beta=0、cache last、decode bypass、机制 producer batch=1，无量化。精确模型/tokenizer/data revision 和现有环境由 Gate A 只读核对后冻结，不照抄历史路径当现场存在。

选择小模型的原因是它已有固定步长/固定 horizon 轨迹基础、运行便宜，方便机制定位。它是**历史开发锚点**，不继承已关闭 Phase 2 的权限。主窗口 inclusive `12:15`（半开 `[12,16)`）；固定邻窗 inclusive `13:16`（半开 `[13,17)`）作为位置敏感性对照。邻窗不预先命名为“坏窗”；本阶段不根据新 outcome 改窗。

### 6.2 数据分工

推荐对 canonical MMLU validation-1531 做基于可读 identity、subject 分层、固定 seed 的确定性划分：

| 集合 | 数量 | 可用内容与用途 |
| --- | --- | --- |
| C_fit | 512 | 无标签 prompt/choice；拟合各窗 v_w、数值范围、方向稳定辅助检查 |
| C_mech | 从其余 1019 中预定 256 | 不拟合方向；Gate C 无标签机制诊断 |
| E_dev | C_fit 之外全部 1019，包含 C_mech | Gate D 固定 panel 的配对 outcome；包括已诊断样本是已声明设计，不称为机制选择之外的全新确认集 |

划分仅按 identity/subject/seed，不按长度、正确性或 gain 挑“合适样本”。长序列资源 preflight 可另外固定少量代表样本，但不得取代正式集合。dev split 的五个 demonstration answers 可供标准 renderer 使用；当前 query 的 target gold 不进入方向拟合。

该划分提供**拟合与评估隔离**，没有洗掉整个项目对 MMLU 的历史暴露。本阶段主要支持受控机制和开发域干预结论。若后续要作未见任务上的 confirmatory 主张，需要另行冻结真正未参与开发的评测任务/集合；不能把历史 test 重跑包装成新测试。

Gate C 前只存无标签 residual/诊断量。Gate D panel、对照、分析脚本和阈值冻结后才能使用 E_dev gold；该 panel 全部完成并通过关键 identity/completeness 检查后统一分析。不得边看部分 accuracy 边改 lambda、窗口、模型或对照。

### 6.3 规模上限与复用

Gate C 推荐仅三条无干预轨迹：主窗 `(K=4,alpha=2)`、邻窗 `(4,2)`、主窗 `(4,1)`；各在 C_fit 和 C_mech 上采集。固定步长 K=4 的前两步 residual 可以在一致性验证后用于 K=2 机制参照，但完整 K=2 下游 outcome 仍需要实际完成相应 forward。

Gate D 推荐 **18 个分析角色，17 个唯一执行配置**，每个唯一配置覆盖 E_dev 1019：

| 子 panel | 配置 | 分析角色数 |
| --- | --- | ---: |
| 原生 | no-loop，共用一个 | 1 |
| 主问题 | 主窗 K=2/alpha=1：Loop、Spectral、Norm-matched、Lower-alpha、Random×3 | 7 |
| 更长迭代 | 主窗 K=4/alpha=2：Loop、Spectral、Norm-matched、Lower-alpha | 4 |
| 固定 horizon | 主窗 K=4/alpha=1：Loop、Spectral、Norm-matched | 3 |
| 邻窗 | 邻窗 K=2/alpha=1：Loop、Spectral、Norm-matched | 3 |
| 合计 | 1+7+4+3+3 | 18 |

这里 Lower-alpha 明确相对所在行的原 alpha 减半。主窗 `(K=4,alpha=2)` 的 Lower-alpha 与固定 horizon `(K=4,alpha=1)` 的 Loop 是同一配置，必须复用同一份 write-once 结果。执行完整性按 **17 个唯一配置**检查，分析表保留 18 个角色引用；不得把共享结果当两次独立实验。该 panel 共需 17×1019=17,323 个完整样本 forward，另计 C/preflight/必要无效尝试；一次 forward 内的 operator 调用成本随 K 和 cache 路径变化。

Gate C 可提前结束本阶段。若进入 Gate D，规模不得因结果趋势临时扩大。Debug 测量后由 executor 给出 model forward 数、预计 GPU 时间、内存和磁盘量；本计划不把未经测量的“几小时”写成保证。

条件式 Gate E 只做一个迁移 cell 域：`Qwen/Qwen2.5-3B × MMLU`，预定 Phase 7 窗口 `[14,17)`，对应 inclusive `14:16`，K=2/alpha=1，保留 rank=1、λ=0.5、位置和步次规则，在同样 C_fit identities 上按该模型重新拟合方向，在 E_dev 评估 Native / Loop / Spectral / Norm-matched / Lower-alpha 五个 arms。dtype 推荐 BF16，须在 Gate A 将这项模型特定差异写入合同。方法迁移不要求把 1.7B 的向量直接传给另一模型，也不重新运行 selector 或扫窗。

## 7. 分析、证据强度与科学停止条件

### 7.1 Outcome 的口径

主指标为 canonical four-choice scorer 的逐题 correctness 和配对 accuracy 差，单位 pp；同时给 micro accuracy 与按 subject 等权 macro，主次在合同冻结。推荐 micro 为主，macro 为稳健性描述。若使用单 token choice scoring，必须核实四个 continuation surfaces 的真实分词和原 scorer 一致；不默认各模型都成立。

每个 contrast 保留配对四格表与 `wrong→right / right→wrong`。补充正确答案 log-prob/margin、错误样本置信度、choice entropy；不得用 entropy 更低替代 correctness。

第一个主问题固定为“主窗 K=2 的 Spectral 相对 Loop、Norm-matched、Lower-alpha 和 Native”。这是预先固定的四个 contrasts，不把事后最优窗口或 K 提升为主结果。报告配对 bootstrap 95% CI，若作显著性主张，对这四项使用 exact McNemar + Holm 控制 family-wise 0.05；正向点估计和原始四格表必须同时给出。Bootstrap 次数、seed 和采样单位在 Gate A 冻结；建议 10,000 次 subject 内重采样、固定 subject membership，不逐 cell 改统计方法。

K=4、固定 horizon、邻窗和随机方向属于预先声明的次要机制/敏感性分析。随机 seed 不是独立的数据集重复；同一模型的三条 random arms 不可拿来夸大样本数。只出现次要正结果时，结论是新假设线索，不自动进入 confirmatory 叙事。

1019 条样本对小幅收益可能不足以排除零；CI 跨零写“不确定”，不写“已证明无效”。不确定性不自动授权加样本或重跑历史 test；扩大规模需明确新的科学修订与预算。

### 7.2 Gate C 的科学分支

不为谱集中度随意设一个通用分数线。Planning 查看一份预定诊断图/表，判断是否存在足以测试的稳定方向与真实同向累积；判断依据必须在 outcome 读取前写明：held-out 投影、校准分组方向一致性、t=1..3 的样本内累积、初始共有成分对照。

- **MECHANISM_PLAUSIBLE**：存在可重复测量的候选方向与样本内累积，进入已计划 Gate D 的小规模因果对照。
- **MECHANISM_UNSUPPORTED**：候选方向无法迁移、主要抵消或 rank-1 不可辨识；Gate C 可以工程 PASS，阶段以负机制结论结束，不启动 D。
- **MECHANISM_INCONCLUSIVE**：证据混合但工程路径有效；默认停止大规模扩展。若一个已定义、有限 Gate D 能直接解决当前不确定性，planning 可在阶段授权范围内说明该因果检验的必要性后准入；不得悄悄改方向估计器。

这是一项有限专业判断，不是无限寻找可疑模式的自由。所有推荐配置须在 C 前固定，不能在不同窗口/谱处理方法之间寻找最适合过门的图。

### 7.3 Gate D 后的科学分支

| 结果 | 允许的解释 | 后续 |
| --- | --- | --- |
| 谱被削弱，outcome 更差或无改善 | 头部能量可能有用；当前干预失败或证据不足 | 阶段结束，保留结果 |
| 优于 Loop，但不优于 Norm-matched / Lower-alpha | 更新幅度减少已足以解释收益，未证实方向优势 | 阶段结束或记录普通阻尼结果，不继续包装 spectral 优势 |
| 优于 Loop 和同范数对照，但仍差于 Native | 找到较好的 loop 纠偏方式，但尚未让 looping 胜过原模型 | 记录局部机制结果；默认不自动扩展 |
| 主 contrasts 支持方向优势，且至少未观察到相对 Native 的退化 | 有理由检验方法迁移；是否称为“无退化”仍需正式区间/等效性设计，不能仅凭不显著 | Gate E 可按预设条件与权限准入 |
| 仅次要 K/邻窗出现改善 | 局部探索线索，主假设未通过 | 作为下一科学修订候选，不重排主次 |

推荐 Gate E 的严格默认准入为：主窗 Spectral 相对 Loop 与 Norm-matched 的配对收益得到预定主检验支持，且相对 Lower-alpha 和 Native 的点估计为非负；后两项区间证据不足必须保留限制。这个准入只控制是否值得投入一次迁移，不等同于完成全部效果证明。若用户希望更强的门槛，在 Gate A 冻结前调整，不能读完 D 再改变。

### 7.4 阶段终态

允许的科学标签为：`MECHANISM_UNSUPPORTED`、`MECHANISM_INCONCLUSIVE`、`DAMPING_ONLY`、`LOCAL_DIRECTIONAL_EFFECT`、`TRANSFER_SUPPORTED`、`TRANSFER_UNSUPPORTED`。这些是报告分类，不是新数值评分系统。正式控制面另记工程 `PASS/PASS_WITH_FIXES/BLOCK`。

本阶段终点为：C 的科学停止、D 的科学停止，或条件 E 完成后的 consolidated phase audit。任何一种完整负结果都可以使阶段工程 PASS；材料性实现错误才进入修复/BLOCK，不把不喜欢的科学结果当工程故障。

## 8. 最小实现边界

### 8.1 可复用接口与必要新采集

现有 `wrapper.py:162–167` 在真正 body 完成后给 audit collector 发送 `g_minus_x` 的 before/after；`phase2_trajectory.py` 已有 sample identity、answer position、单样本 forward 和 finally restore 模式。Gate A 先新增 Phase 8 collector/analysis，复用此观察接口，不先重构 wrapper。

Phase 2 的持久化规则不能直接承载本次 basis 拟合。本阶段需单独授权在 HPC 专属新根保存 **C_fit/C_mech 的 answer-position residual 矩阵**：每 row 绑定 model、window、轨迹族、t、readable sample ID、p_i；禁止完整 `[N,S,d]` hidden dump，不把矩阵或 basis 提交 Git。若存储量为 `4*N*T*d` bytes 的 FP32 矩阵，应按实际 N/T/d/窗口数量预估并冻结预算。

方向/basis 是新数据衍生工件，必须记录拟合集合、raw/centered 口径、dtype、求解算法和参数；只把配置、schema、代码和小型测试纳入仓库。E_dev 正式 outcome 只需最终四选项 scores 与少量标量诊断，不要求全量保存 residual tensor。

### 8.2 干预接口

最小接口建议：`config.py` 增加默认 None 的 residual transform / context；`strategies.py` 在 `r=G(x)-x` 与 `x+h*r` 之间显式调用。数学、basis 加载和 position mask 位于 `src/tflt/loopscope/phase8_*` 旁路模块。初步无需修改 `cache.py`；`wrapper.py` 只有在正常路径最小复现证明缺少必要 context 时才申请窄修改。

这些文件属于现有 protected paths。**新增 transform 的实现须在对应 Gate handoff 中经 planning 批准具体最小 diff 与回归证据，本总体计划没有批准直接修改。** Collector 回调返回值被忽略，禁止通过原地篡改 audit tensor 偷偷实现干预。

机制主路径使用新 direct producer、batch=1 明确传入 p_i。若未来切换到 lm-eval full producer，则需独立的 identity/position/scoring 一致性证明和准确 adapter 权限；不能把旧 final-output sidecar 的权限扩大成 HFLM residual router。

### 8.3 测量与数值

区分三种对象：原模型 dtype 下实际计算的 raw residual、转换到 FP32 的测量值、应用 transform 后实际进入加法的 residual。旧 collector 的 `float(out)-float(in)` 与先做低精度减法再转 FP32 未必相同，报告字段不可混用。

建议投影内积用 FP32，写回时恢复原 dtype；disabled / λ=0 分支直接返回原 residual，不增加无意义 cast/matmul。Gate B 必须证实默认关闭与原路径一致，并用小样本给出 λ>0 后真实状态更新，而不是根据未干预公式重建 applied-state。

## 9. Gate 总体路线及职责分离

全部 Gate 目前为 **PLANNED / UNASSIGNED / NOT_AUTHORIZED**。Gate 字母在标题中不重复 Phase 前缀。

| Gate | canonical executor title | 目标与 admission | 最小输出与退出条件 |
| --- | --- | --- | --- |
| A | `execute-LoopScope-Residual-Probe-第8阶段-Gate A` | 阶段执行授权与新 control 已建立；冻结核心科学卡，做最小只读残差 collector/CLI，拟定 protected diff | 推荐配置变成明确合同；真实旧接口可用；局部 CPU 测试；B 的窄实现权限已可审查 |
| B | `execute-LoopScope-Spectral-Debug-第8阶段-Gate B` | A PASS；protected config/strategy 修改已被 planning 精确授权 | 实现 opt-in transform 和 control arms；debug 分区四题真实端到端 preflight，原生/关闭/λ=0/位置/步次/更新恒等式闭合 |
| C | `execute-LoopScope-Residual-Mechanism-第8阶段-Gate C` | B PASS，代表路径 preflight 仍有效，fit/mech 新数据权限和资源明确 | 三条 baseline 轨迹、拟合 basis、held-out 诊断；H8-A/B 分支和 D admission；无 gold outcome |
| D | `execute-LoopScope-Directional-Intervention-第8阶段-Gate D` | C 科学分支准入；17 个唯一配置及 18 个分析角色、analysis 和 gold 权限已冻结 | E_dev 全 panel、一次配对分析、归因对照、成本；科学停止或 E admission |
| E | `execute-LoopScope-Model-Transfer-第8阶段-Gate E` | D 达到预定准入且阶段授权包含条件 E；新模型 preflight 有效 | 一个预定模型域、五 arms；方法迁移结果和最终解释边界 |

Phase-end audit 由 planning 自己完成，不为写终审再建立一个实验 executor。未来 Gate 的标题预先列出，exact thread ID 只在当前 Gate 授权时绑定，不提前创建所有 executor。

### 9.1 每个 Gate 的 Agility budget

| Gate | 最小增量 | executor 的必要工程验证 | planning 的 1–3 项轻量验收 | 停止继续加固的条件 |
| --- | --- | --- | --- | --- |
| A | 一条 residual collector 与最小 schema/CLI | identity/step/shape、torch-free 小矩阵公式、CLI dry-run | 配方与权限闭合；真实观察接口；最小测试输出 | 能进入 B 即停止，不建通用探测框架 |
| B | 单位置、重复步的 transform 与有限对照 | 真实四题 forward，restore，K/alpha 前缀一致性，默认不变；资源长样本形态单独取证 | 一次有效 debug；真实更新恒等式；producer/scorer 位置正确 | 正常路径通过后立即准备 C，不扩充假想分支 |
| C | 既有路径批量采集和离线 basis/诊断 | 三轨迹 membership、有限值、fit/held-out 分离、方向口径 | basis 来源；held-out 累积图；科学分支理由 | 机制结论足以决定 D 即停，不做更多选窗竞赛 |
| D | 相同 producer 上执行固定 arms | panel 完整性、配对 identity、原生 scorer closure、关键分析重算 | 主 contrasts；norm-matched 对照；完整 panel 与冻结参数 | 一次完整分析后裁决，不因非显著重试 |
| E | 相同方法迁移一个模型 | 新模型分词/位置、preflight、五 arms 完整性 | 冻结方法未漂移；迁移主结果；成本 | 完成后进入阶段终审，不再追加模型 |

普通局部修复在同 Gate 的冻结科学、路径和资源范围内持续进行；保持失败尝试并使用 fresh paths。作业失败是 attempt event，不自动是 Gate BLOCK。Planning 审计返回的材料性问题一次集中给同 executor 修复；普通 Gate 一轮，第二轮只处理同根因未解决或前次修复直接引入的材料性回归。

不得因写法、命名、未使用路径、通用 schema 美观或可构造的手工篡改情形拖延真实实验；主线测试通过后不反复跑完整 suite。全套回归只有一个具体广泛影响机制时才加入，不能沿用旧阶段“全绿”条款机械重复。

## 10. 后续开发计划与 handoff 的必填项

本文解决研究路线；后续 `loopscope_phase8_development_plan.md` 负责把它变成文件级实现顺序。科学参数的 canonical owner 是 `loopscope_phase8_scientific_contract.md`；动态授权与决策只放 `loopscope_phase8_control.md`。按用户 2026-09-05 的明确要求，`docs/loopscope_phase8.md` 已先形成面向人阅读的阶段计划报告，命令/schema/运行方式随实现补充。报告是本文的阅读入口，不另立科学或授权来源；方案变更时同步其摘要，详细约束通过链接复用。

激活阶段时，planning 应先完成以下工作，不能留给 executor自行猜测：

1. 明确用户授权是只到某 Gate，还是包含条件 E 的阶段持续推进；记录终点与停止分支。本次仅规划，不假定已经有持续执行授权。
2. 核实本任务 exact planning ID；新 control 绑定当前 Gate 的独立用户可见 executor ID 与 canonical title。不得借用 Phase 7 的旧 planning/executor ID。
3. 接受或修改第 6 节推荐配方，并冻结模型/tokenizer/data revision、prompt、choice surfaces、sample membership/seed、窗口 notation、K/alpha、rank/λ、intervention token/step 范围、随机 seed、dtype、batch 和统计口径。
4. 冻结 Gate C 科学分支判据、D 主 contrast family、E admission 与科学修订边界；不在 outcome 后补写。
5. 写具体代码/配置/脚本/测试路径与 exact commands。本文中尚未实现的 `phase8_*` 名称是开发建议，不伪装成现成可运行命令。
6. 对 protected core 变更、answer-position residual/basis 持久化、gold 解封、remote writes、Git commit/push、GPU/Slurm 和 fresh retry 分别授予当前 Gate 所需的最小权限。
7. 记录 Git base/branch/dirty、可写路径、受保护历史、唯一 run root、资源预算、repair 边界和 terminal 收件人。

建议文件归属：

| 层次 | 路径 | 本次状态 |
| --- | --- | --- |
| 稳定项目规则 | `AGENTS.md` | 已读；本轮不改历史规则 |
| 总体研究计划 | `.planning/phase8/loopscope_phase8_plan.md` | 本文件 |
| 具体开发计划 | `.planning/phase8/loopscope_phase8_development_plan.md` | 后续规划产出，尚未创建 |
| 冻结科学合同 | `.planning/phase8/loopscope_phase8_scientific_contract.md` | 激活前完成 |
| 唯一动态控制 | `.planning/phase8/loopscope_phase8_control.md` | 尚未创建，无当前 Gate 授权 |
| 当前 Gate handoff | `.planning/phase8/loopscope_phase8_gate_<letter>_handoff.md` | 只在授权当前 Gate 时创建 |
| 阶段计划报告及后续 runbook | `docs/loopscope_phase8.md` | 按用户指定路径建立易读计划报告；可执行部分随最小实现补充 |
| 新代码/配置/测试 | `src/tflt/loopscope/phase8_*`、`configs/loopscope/phase8_*`、`scripts/loopscope/*phase8*`、`tests/test_phase8_*` | 待 Gate 授权 |
| 人类报告与图 | 外层 `资产/报告/phase8/`、`资产/figures/` | 有实验结果后产出，不在 repo 镜像 |
| 大型证据 | HPC2 专属 workspace 下新的 `phase8-*` run/artifact roots | 尚未生成 |

## 11. 远端资源、长任务和终态回传

HPC2 使用现有专用 clone `/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope` 和专属 workspace `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope`。现有复现 checkout 只读，历史 runs/cache 不覆盖、不迁移。每 Gate handoff 明确自己的 host/path/read/write/network/resource 权限；规划线程只做有目的的只读验收，不替 executor实施或日常轮询。

推荐沿用技能的基本资源规则：

- debug/smoke 只在 `debug` partition；正式长任务之前必须有同 commit、launcher、环境、模型/runtime、hook、producer/verifier 的 `<30min` 端到端 preflight。关键路径改变后重做最小 preflight。
- 功能路径与资源形态分别验证；四题测试不自动覆盖长序列峰值。debug 无法复现 formal 资源形态时，先用最小、可保留的 formal canary，闭合后再扩展。
- 正式 GPU 依当前显存、吞吐、排队与 time-to-result 选择，使用普通用户合法最高优先级；不默认锁定 A800。以避免 OOM 为硬边界，实测后采用有界同卡 packing，记录 concurrency、峰值内存、throughput、OOM 次数；不为占显存改科学 batch 或共享模型状态。
- 本阶段函数式基线和干预 arms 均 batch=1，不为提速偷偷改变；可用独立进程有界 packing。失败保留，资源修复后的重试进入新目录，只补无效/缺失工作，不重跑已解封科学结果寻找更好数字。

每个长任务由 exact executor 持有唯一 heartbeat：smoke 10 分钟、probe 30 分钟、full 60 分钟。正式非 debug 作业核实真实 job ID 后即监测 PENDING；短 debug 作业稳定 RUNNING 约一分钟后才需要 heartbeat，提前结束则无需创建。

作业到终态后 monitor 做一次有限只读检查，先暂停/删除自身，再向 exact executor 投递 `AUTOMATION_TERMINAL_RESUME` 并保留成功确认；未确认则向 planning 发 `AUTOMATION_RELAY_REQUIRED`。失败作业唤醒 executor做范围内诊断/修复，不能仅发“可以继续”后闲置。

每 Gate handoff 首个机器可读区块须包含 `COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR`，同时绑定 planning/executor、同名标题、权限问题回 planning、唯一 heartbeat、terminal route 和 fallback。Executor 最终只发送一个 `GATE_X_FINAL_AUDIT` 或真正的 `BLOCK`；只有 planning 裁决 `PASS/PASS_WITH_FIXES/BLOCK`。直接投递保留目标 ID 与工具成功回应；未确认时输出 `TERMINAL_DELIVERY_UNCONFIRMED` 的可转发同包。

本轮只读调查子代理属于 planning 的有限资料核对，不是 Gate executor。未来每 Gate 使用一个新的独立用户可见任务；executor可在自己 Gate 内按技能使用最多三个并发窄子任务，但不转移 Git integration、远端写入、GPU/Slurm 和 terminal 责任。

若用户以后授权整阶段持续推进，planning 在当前 Gate PASS 且下一 admission 满足时，撤销旧 executor权限，创建并绑定下一独立 executor，投递并确认 handoff后简要解释本 Gate 目标/步骤/证据/边界，然后等待终态事件，不持续轮询。无阶段持续授权时不能据此自动推进。

## 12. 阶段结束时必须交付什么

一份数据驱动报告即可承载主要结果：残差谱/均值分解图、固定方向跨轮投影与抵消图、固定步长和固定 horizon 的配对 outcome 图、对照表和成本表。所有图有 source data、样本域、真实 K/alpha/位置口径，机制量与正确率分开展示。

Planning 只做一次 consolidated phase audit：核对最终论断是否由接受的 Gates 支持，核心配方/样本分工是否保持，最终必要 cells 与 jobs 是否闭合，结果和控制状态是否一致。复用已验收证据，不重演每个测试和日志。

报告应回答四个朴素问题：方向是否存在、是否持续累积、削弱后是否比普通阻尼好、是否值得推广。如果只完成前两步，明确只有机制诊断；若只改善失败的 loop 而未超过 Native，明确没有证明 loop 总收益；若迁移失败，保留模型特定结果，不汇总成通用方法成功。

可用于导师沟通的本阶段定位：**准备在 LoopScope 中检验重复窗口产生的残差是否沿少数稳定方向累积，先用小模型和固定窗口补齐机制证据，再比较主方向软衰减、降低步长和同范数统一阻尼。只有方向效应和任务结果同时支持时，才进入一次模型迁移；目前尚未确认该机制成立或能够提高准确率。**

## 13. 主要来源与代码入口

- 目标论文：[SFA arXiv v1](https://arxiv.org/abs/2212.01026v1)，PDF 第 3 页 Algorithm 1 / Eq. (3) / Proposition 1，第 4–5 页分析，第 6–7 页实验。
- 用户提供的本地 [原始 PDF](</Users/huangxutao/Desktop/科研/paper reading/hrm:looped transformer/文献/Spectral_Feature_Augmentation/2212.01026v1.pdf>) 与 [深度总结](</Users/huangxutao/Desktop/科研/paper reading/hrm:looped transformer/文献/Spectral_Feature_Augmentation/Spectral_Feature_Augmentation_论文深度总结/Spectral_Feature_Augmentation_论文深度总结.md>)；总结中的代数边界已对照原论文关键公式，其他复现实验提议不直接继承。
- 用户提供的 [历史讨论](chatgpt-conversation://6a872dfb-880c-83ee-bf76-a9b3daadfd76)：用于恢复问题背景，不作为当前代码或授权来源。
- [TFLT 原论文](https://arxiv.org/abs/2605.23872)：循环窗口、阻尼子步的研究背景；本地计算语义以代码为准。
- [策略实现](../../src/tflt/strategies.py)：31 行别名，32–33 行 K=1 特判，59–63 行实际 Euler 更新。
- [wrapper](../../src/tflt/wrapper.py)：146–180 行 block operator、collector、looped 与 stash；289–318 行 incremental bypass；425–429 行 audit callback。
- [配置](../../src/tflt/config.py)：36 行窗口 notation，51 行 audit collector，53–68 行窗口校验，75–84 行窗口访问器。
- [Phase 2 producer](../../src/tflt/loopscope/phase2_trajectory.py)：138–252 行 collector，1543–1587 行 direct forward / restore，2262–2277 行 answer-position FP32 测量。
- [Phase 2 fixed-horizon](../../src/tflt/loopscope/phase2_fixed_horizon.py)：325–389 行 hooks 与未干预 applied-state 口径，不可直接冒用为干预后状态。
- [Phase 7 outcome card](../../configs/loopscope/phase7_gate_f_outcome_card.json) 与 [scalar trajectory contract](../../configs/loopscope/phase7_trajectory_contract.json)。
- [项目政策](../../AGENTS.md)、[规划索引](../README.md)、[项目接续摘要](../../PROJECT_MEMORY.md)。

后续首先完成第 10 节的开发计划和激活合同；在此之前，本文件始终是研究规划，没有正在运行的第八阶段 Gate。
