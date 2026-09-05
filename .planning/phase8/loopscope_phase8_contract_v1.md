# LoopScope Phase 8 科学合同 v1

2026-09-05。用户明确确认：validation 512 拟合方向、完整 test 14042 评测、K=2 主实验＋K=4 补充、全部 alpha=1、rank=1、lambda=0.5。该确认解除 Gate B 的科学选择等待；运行权限仍由当前 control/handoff 分 Gate 授予。

## 1. 冻结 cell 与运行语义

| 模型与 revision | dtype | inclusive windows / boundaries | cache |
| --- | --- | --- | --- |
| Qwen/Qwen3-4B-Base @ 906bfd4b4dc7f14ee4320094d8b41684abff8539 | bfloat16 | 12:15 / B12→B16；13:16 / B13→B17 | first |
| Qwen/Qwen3-1.7B-Base @ ea980cb0a6c2ae4b936e82123acc929f1cec04c1 | float16 | 12:15 / B12→B16；6:9 / B6→B10 | last |

两模型采用历史运行 dtype，而非从 checkpoint 的默认 dtype 自动推断。tokenizer 使用对应模型同 revision。以上 revision 是代码/模型版本身份，不要求内容摘要。

K={2,4}，alpha=1，beta=0，strategy=damped_euler，iteration_mode=block，batch_size=1，decode_mode=bypass，eval 模式、无梯度、不训练。K 包含首次 t=0，h=1/K，总时长=1。K=2 与 K=4 是固定 horizon 的不同离散步数，不声称增加了总演化时长。cache stash 不计入 t，也不采集或干预其 residual。保持原 HFLM full-sequence choice scoring，不通过 cached generation 强行使 cache first/last 产生影响。

## 2. 数据、评分与样本身份

Dataset=cais/mmlu @ c30699e8356da336a370243923dbaf21066bb9fe（历史项目版本，B 必须确认缓存来源；不静默换版本）。57 subjects；dev first_n 五个 demonstrations，plain prompt 结尾 Answer:，不开 chat template/multiturn，不生成 CoT。lm_eval 0.4.11 的标准 MMLU choice-loglikelihood acc，保留 A/B/C/D 原顺序与完整 token continuation 求和；不以 length-normalized acc 替代。

校准 C_fit：validation1531 中固定 subject-stratified 512 个身份。seed=20260905。按各 subject 数量比例分配512名额，先向下取整，余量按小数余数降序、subject 名字升序打破并列；subject 按字典序，使用一个 random.Random(20260905) 依次从原始 doc_index 列表抽样，选中后按 subject/doc_index 排序。身份为 dataset revision + split + subject + 原始 doc_index，并保留直接可核对的 question/choices；不新增内容摘要。两模型、窗口、K 共用同一512题清单。目标 gold 不参与校准或方向拟合。

E_test：test 全部14042题/57subjects，使用同一来源的 canonical subject/doc_index 顺序。与 C_fit 依 split 隔离，不删除长题、不随结果换样本。test 不进入方向拟合、lambda/K/window 选择。历史 benchmark 已多次被本项目使用，最终表述为历史开发域中的受控干预比较，不声称 untouched test。

exact HFLM tokenization、context/continuation 切分、左侧长度截断、右侧 padding 与得分 token 对齐必须保留。pre-answer 位置定义为 evaluator 实际输入中最后一个 context token；只有验证四候选为单 token continuation 时，才可用最终非 padding 位置简化。不把 gold answer 放进 prompt。发现长题截断/多 token continuation 时应保持原 evaluator 语义、正确映射位置，不能悄悄丢题或换评分。

## 3. 离线方向拟合和干预

每个 model×window×K 独立拟合一个方向：2模型×2窗×2K=8个 basis。每个 basis 只用该配置无干预 Euler 轨迹在 t=1 的 pre-answer residual，形成 D∈R^(512×d)。K=2 的方向应用于 K=2，K=4 的方向应用于 K=4；不同步长不共用轨迹前缀。K 对比代表各 K 下分别校准的方法，不单独隔离 basis 改变的影响。

每个身份在每个t只计一行；四个候选评分请求或cache stash不得重复计入校准矩阵。

D 的行是实际 native-dtype `out-x` 残差转为 FP32 的值，未中心化、未做逐行归一化。拟合用 CPU float64 reduced SVD 提取 top-1 右奇异方向，规范符号为最大绝对分量为正，保存该向量及拟合 identity/K/window/model/原始 dtype/算法。运行时使用其 FP32 单位方向 v。非有限值、无非零能量或无法定义有效单位向量是数值问题，显式报告，不静默使用零方向。报告头部谱占比及前两奇异值间隔作诊断，不用谱大小筛选窗口或取消某个 cell。

只在 t>=1、pre-answer token 处应用：

$$
\delta'_t=\delta_t-0.5v(v^\top\delta_t),\qquad X_{t+1}=X_t+\frac{1}{K}\Delta'_t.
$$

先计算原始 native-dtype residual。投影和此位置的减法在 FP32 中执行，再将该位置结果 cast 回 residual dtype；其他位置直接保留原 residual。Euler 标量和加法仍走原 dtype 路径。t=0 不改变，basis 在 test 全程固定，不逐 prompt 或逐步更新。测量 `out.float()-x.float()` 不等同于实际 `out-x` 后转 FP32，不得偷换。

干预关闭及强度0检查必须保留原数值路径；config callback 默认 None。K=1 原行为保持但不进入正式 loop panel。方向和科学参数不受 test 标签/得分影响。512×d 的校准 answer-position residual 与8个 basis 仅存HPC专属新工件，禁止持久化完整 N×S×d hidden/residual 或将数据/basis 提交Git。

## 4. 正式 panel 与统计

当前批准的最小完整 panel：
- 每模型一个 native no-loop，共2个。
- 每 model×window×K 的原版 Loop 与 Spectral，共2×2×2×2=16个。
- 合计18个唯一完整评测配置，每个14042题，共252756个样本评测；这不是精确 operator 次数，token forward 和 cache stash 开销另计。校准为8×512=4096个配置-样本轨迹，debug与失败尝试另计。

同范数 Uniform 数学核心保留，但不扩本轮正式 panel；无 Lower-alpha、Random、额外 K/window/model 搜索。因缺少同范数正式对照，若 Spectral 获益，只能证明整体干预方案改善，不能单独归因为方向特异性优于普通阻尼。后续增量需要新的明确范围决定。

主分析：四个 model×window 的 K=2 Spectral−Loop。逐题配对 micro acc 差（pp）、原始正确数/N、wrong→right、right→wrong、subject macro。95% paired bootstrap：固定57subjects，各subject内成对重采样，10000次，seed20260905，汇总micro差。exact McNemar双侧 + Holm 对四个主 contrasts 控制 family α=0.05。K=4 的同类四个 contrasts 为单独的次要 family，同样报告，不能用其最佳结果替换主分析。Spectral−Native、Loop−Native 为上下文比较，明确探索性。报告所有预定cell、nominal CI及校正p，不能将单个CI当作多重校正通过。

全部18配置的身份/完整性闭合后在 Gate D 一次性计算 outcome。B/C 不读目标 labels/accuracy；D 采集阶段仅做完整性/数值/配置检查，闭合后才结合test gold计算统计，不按中途acc调参。正、负、不确定均为合法科学结果，不以效应正号作为工程 PASS。

## 5. 阶段边界

逐 prompt 在线估计方向记录为后续阶段动机，本阶段不实施、不调参。当前 Gate B 仅实现与debug，Gate C 才拟合正式512方向，Gate D 才正式test。具体资源上限和路径在各 handoff 下发；正式作业必须先有同版本/launcher/env/runtime/producer 的 debug 成功证据，并以实际峰值/吞吐确定安全 packing。
