> 2026-10-02执行更新：科学规模以contract_v2为准（68独立/86展示、三策略、MMLU59新9复用）；A/B已PASS，当前C具体资源和信息授权见control/C handoff。所有非正式任务debug，正式最高合法优先级A800；下文原始预算与旧分区讨论只保留设计历史，不授予权限。

> 调度更新（2026-10-02）：用户已指定非正式全部debug、正式A800最高合法优先级，覆盖下文历史A800预检例外；当前执行以control及gate_b_partition_supplement为准。

> 2026-10-02 current_t修订：当前科学规范为[contract_v2](loopscope_phase10_contract_v2.md)，Gate A执行补充为[current_t amendment](loopscope_phase10_gate_a_current_t_amendment.md)。新增每轮当轮估计并当轮衰减（含t0），每数据集68独立/86展示，MMLU9复用59新，ARC68新；7个迁移lambda，统计families为14/126/45。下文v2总体规划的旧数量和双策略描述保留为修订前背景，冲突以新科学合同为准；Gate顺序与其余操作边界保持。

# LoopScope 第十阶段总体规划

日期：2026-10-02。状态：总体规划 v2，已获逐Gate执行授权；用户最新明确移除 Matched-norm 对照；用户明确要求全面对标第九阶段，在指定三组配置内同时消融 fixed_t0 与 lag1，选择 ARC-Challenge、25-shot 标准答案文本评分（主 acc_norm，附 acc）。本文不授予实现、作业或执行线程权限。当前授权唯一入口为同目录 control。

## 1. 已核对的依据与目标

- 本地专用 clone：`loopscope-tflt`，分支 `loopscope`，起始 HEAD `ee9feb4dbee1caca53a2fc23f02527c2bbe05106`，规划前 clean，保护基点为其祖先。
- Phase9 contract_v2、control、主线终审、Gate E 验收；外层 `资产/报告/phase9/` 主线和追加结果。
- 用户 Notion 报告：https://app.notion.com/p/3e47d1c8d46f8030bd22cb0ab4537670 。本次直接读取：表内已有逐轮结果，但正文仍说 Gate E 未完成；此处以本地已完成的验收和实现为事实依据，不改写该外部页面。
- 用户提供的 `Training-Free Looped Transformers.pdf`，重点核对 §2–3、Table 1/3/6、Appendix C/D/P，Table 6 已渲染检查。论文明确包含 ARC-Challenge 25-shot；论文正文“无调参”与 Appendix P 搜索流程表述存在差异，不能把其概括当作本阶段协议。本文沿用第九阶段真实运行语义，不因论文中 alpha 记号差异改变 Euler 总时长。
- `phase9_runtime.py` 与 `phase9_gate_e_runtime.py` 已支持 strength；`phase9_accuracy.py` 将任务、lambda=0.5 和旧面板锁死，`run_phase9_accuracy.py` 也固定传入 0.5。`phase9_adapter.py` 只允许 ABCD、并检查候选答案保留前缀一致；不能直接将 ARC 数据塞入旧入口。

目标：在用户指定的高绝对性能配置上，寻找在线残差衰减的强度响应，并检验原配置和 MMLU 选定强度能否迁移到 ARC-Challenge。高绝对性能不等于衰减已有效：旧 fixed_t0 lambda0.5 的 Online−Loop 分别为 +0.0214、−0.0071、+0.1353 pp，不能当作已确认收益。

## 2. 实验矩阵

| 模型 | inclusive 窗口 / 边界 | K | dtype / cache |
|---|---|---:|---|
| Qwen/Qwen3-4B-Base | 15:18 / B15→B19 | 2 | BF16 / first |
| Qwen/Qwen3-4B-Base | 15:18 / B15→B19 | 3 | BF16 / first |
| Qwen/Qwen3-1.7B-Base | 12:15 / B12→B16 | 2 | FP16 / last |

模型及 tokenizer revision 继承 Phase9 contract_v2。block damped_euler、alpha=1（总时长）、步长1/K、beta=0、batch1、decode bypass、eval/no_grad、rank1、FP32 exact reduced SVD/CUDA gesvd 均保持。无训练、量化、生成 CoT、新窗口或新 K。

lambda/strength = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}，表示移除方向分量的比例，不是保留比例，也不是 Euler alpha。

每种策略仅跑 Online 定向衰减；按用户最新指令不运行或复用 Matched-norm 对照：

- fixed_t0：每题 D0[S,:] 不中心化、不逐行归一化，估 v0；t>=1 使用 v0。
- lag1：t>=1 使用上一轮未施加当前轮干预前的有效 prompt 残差所估 v_(t-1)，在自身轨迹更新下一轮方向；不得改成使用当前轮方向的另一种方法。
- 两者 t0 不干预，只改真实最后 context 位置 p：delta' = delta − lambda v(v^T delta)。S 只含实际保留 prefix 的有效 token，绝不含候选 continuation。
- K2 两策略科学计算等价：唯一干预使用 v0。A/B 先证明两个调用路径输出等价，随后将 lag1-K2 标为同一结果的别名，不重复评分、不当作独立重复实验。若路径不等价，先修实现或报告原因，不静默合并。

每数据集独立配置数：两个 K2 cell ×9档lambda=18；一个 K3 cell ×2方向策略×9档lambda=18；3个Loop+2个Native=5；合计41。Native每模型只需一份。lambda=0用于工程等价检查，正式基线直接用Loop；lambda=1不在正式扫描。

完整展示矩阵为3 cell×2方向策略×9 lambda+5基线=59行；其中18行lag1-K2在A/B验证等价后作为fixed_t0-K2的别名，预计独立计算配置为41。报告保留两个分支全部行并标注共享结果，不把59行称为独立运行。“全面消融”按用户给定三个cell及最新移除Matched-norm的范围展开，不自动恢复其他窗口、K4或离线Shared方向重拟合。

MMLU：41逻辑独立配置，优先复用旧Native2、Loop3、fixed_t0 lambda0.5三个、lag1 K3 lambda0.5一个，共9；预计新增32×14042=449344题级评分记录。历史复用必须核验逐题原始分数、模型/数据/评分配方，不能仅复用acc。不可复用时由规划另行决定同配方补跑，不自动扩大预算。

ARC：41新配置；按官方test1172计为48052题级评分记录。两数据集合计82个独立配置，预计新增73个配置、497396条题级评分记录；每条包含全部候选分数，不是单token/单候选调用数。最终数量以冻结数据revision身份清单为准。

## 3. 数据与评分（ARC 口径已由用户确认）

MMLU 完整保留 Phase9：cais/mmlu 的原 revision、test14042/57subjects，dev first_n5，plain5shot，lm_eval0.4.11 完整答案字母 continuation loglikelihood；不丢长题、不按结果筛题。

ARC 用户确认口径：`allenai/ai2_arc` / `ARC-Challenge` / 全test，25-shot 数量参照论文 Table 6，train 为示例池、validation 为工程样本池。按固定 lm_eval0.4.11 原生任务渲染与完整答案文本 loglikelihood，主指标 acc_norm，原始 acc 同时报；不在看结果后切换主指标。数据 revision、fewshot seed（建议20261002）、固定示例 identities、max_length 与精确 tokenization 由 A 在正式前冻结。

官方 task 依据：
- https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/v0.4.11/lm_eval/tasks/arc/arc_challenge.yaml
- https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/v0.4.11/lm_eval/tasks/arc/arc_easy.yaml
- https://huggingface.co/datasets/allenai/ai2_arc/raw/main/README.md （test1172、validation299、train1119）

原生 ARC prompt 是 Question/Answer，候选为答案文本；方向的 S 因而不包含当前题的候选文本，这属于任务模板差异，须报告。它是 Base 模型上的新迁移实验，不能声称复现论文 Instruct 模型对应准确率。必须支持实际选项数和数字/字母标签，不强制四选项。

对多 token continuation，仍只干预真实答案前 p，后续答案 token 不参与方向拟合或直接干预。A/B 核对完整 continuation 评分和因果 prefix 一致性。用无标签长度检查验证所有候选截断后保留共同 prefix；若25-shot导致候选依赖截断，先提出统一前缀长度/任务口径修订，不能擅自缩短 continuation、删题或降低 shot。Native/Loop/各方法必须使用完全相同的已冻结 task 配方。

## 4. 分析与迁移设计（建议冻结稿）

MMLU 已反复使用，且本阶段 cell 按历史表现挑选，因此全部强度响应属于开发域探索。完整报告所有 lambda、负结果、正确数/N、micro accuracy、MMLU subject macro、净纠错、错→对/对→错和开销；best-of-grid 不称为无偏收益。

迁移参数选择规则在 MMLU 新结果前冻结：对4个非重复的 cell×方向策略组合，各取 Online 的 MMLU micro acc 最大 lambda，精确并列取较小 lambda。选定后在 ARC outcome 前记录4个 lambda*；即便 MMLU 全部不如 Loop 也按规则执行并报告负结果，不因结果差取消 ARC。Native/Loop 仍是明确的部署回退候选，不能强迫宣称干预有效。

ARC 完整41配置均执行，全部分数闭合后才读取 target gold/outcome。在任何 ARC outcome 被查看之前，迁移 lambda* 清单必须固定。报告分两层：

1. 直接迁移主比较：4个预选 Online 与对应 Loop、Native 比较，共8项一个 Holm family，alpha0.05，exact 双侧 McNemar，方向为正才支持增益。
2. 完整九档响应：36个非重复 Online 配置 ×2对照，共72项探索 family，每数据集各自 Holm；不得把每个 lambda 拆成独立 family。另列9项 K3 lag1−fixed_t0（Online九档lambda）一个次要family。lambda0.5旧值只是历史锚点，不当新独立验证。

MMLU subject内配对 bootstrap10000次、ARC题级配对 bootstrap10000次，seed20261002，nominal95%CI；nominal CI 不代替多重校正决策。ARC 次指标 acc 仅描述性，主推断用 acc_norm。如选择其他 ARC 评分格式，此处在结果前同步改为其预定主指标。

结论分开：Online>Native 说明总方案优于原模型；Online>Loop 才说明额外干预收益；本阶段未设Matched-norm，因此不能区分收益来自方向选择还是单纯缩小更新，也不作方向特异性机制结论。lag1>fixed_t0 需要其直接比较支持。ARC仅1172题，一题约0.0853pp，对MMLU上0.02–0.14pp量级差异可能检验力不足；不显著不等于等效。完整网格的ARC最佳lambda只称当地探索最优，不能替代直接迁移结论。

## 5. Gates 与最短关键路径

以下为前瞻规划，所有 executor 尚未创建/绑定。每个 Gate 一个新的独立可见执行聊天；后续具体 handoff 才冻结 commands、base commit、可写文件和资源权限。

| Gate / prospective topic | 准入 | 执行目标与交付 | 规划轻量验收 |
|---|---|---|---|
| A / Lambda-and-ARC-Adapters | 科学口径确认并获执行授权 | 新phase10配置/入口/清单/分析；复用现有runtime，接通lambda与方向策略；ARC适配；冻结数据和统计方案 | 面板41身份、关键评分边界、lambda0/0.5及K2等价的定向证据 |
| B / End-to-End-Preflight | A PASS | 两模型、K2/K3、两策略、lambda0.1/0.5/0.9的代表性真实preflight；测长prefix显存、并发加载和吞吐；核验旧9配置可复用 | 同源码/环境端到端成功；候选prefix语义；可行资源预算 |
| C / MMLU-Lambda-Sweep | B PASS及C资源包络 | 59新配置、9复用；正式分片计时后全量闭合，一次分析；固定7个迁移lambda* | 68配置完整性、无漏题/串参数、选参规则正确 |
| D / ARC-Transfer-Sweep | C PASS且7个lambda*已记录 | 三cell三策略九档68独立配置；全量闭合再分析；分开迁移14比较和探索扫描 | ARC样本/口径闭合、lambda*未依据ARC改动、关键配对结果 |
| E / Report-and-Phase-Closure | D PASS | 中文报告、完整表和强度曲线；资源成本与局限；结束monitor | 一次跨Gate综合审计，判断结论是否被完整证据支持 |

标题格式 `execute-LoopScope-<topic>-第10阶段-Gate <letter>`。用户最新指定 gpt-6.1-sol/high；每执行线程最多同时3个subagent，创建首条消息与handoff均必须要求实际读取协作skill。未预建未来线程。

Agility budget：只新增phase10必需模块、配置、launcher、定向测试；保持Phase8/9冻结校验和结果。A不做框架重构；B不重复512全量几何诊断；关键正常路径验证完成后立即进入正式实验，不追求一般性验证平台。主要测试覆盖lambda实际送达、t0无干预/非p位置不变、K2策略等价、lambda0回到Loop、lambda0.5兼容旧结果、ARC多token/可变选项/标签映射/截断边界及清单不覆盖。

## 6. 资源、权限与终态

当前已进入Gate派发阶段；具体实现、Git、SSH、资源与数据权限以当前control和exact handoff为准。

未来host限 `hpc2-hkustgz`，项目专用source与workspace遵守AGENTS。正式Gate允许的源代码和写入路径需在handoff列明，protected wrapper/strategies/cache/config默认不改。使用新timestamp run路径，保留失败、只补缺失/无效工作，禁止覆盖旧数据。

预检<30分钟，正式使用相同关键源码、环境、launcher、runtime和producer路径；任何关键修复后重做最小预检。当前AGENTS有用户历史明确的A800最高合法优先级分区例外，包含preflight；与skill通用debug分区默认有文字差异。正式派发前需在control/handoff明确采用何者，不能由executor自行解释。卡型和预算尚未授权，不把第九阶段配额继承到新阶段。

建议并发上限先按2 GPU设计，GPUh硬预算在B测量后提出。历史18个新主线配置约23.94GPUh，32个同规模MMLU配置粗略线性量级约43GPUh，但lag1、K组成、候选长度、重复SVD和ARC25-shot均使其不能充当承诺或预算。B给出按cell类别测量的time-to-result及上限。默认4B先packing1；更高packing必须实际验证并发加载峰值及长prefix，同卡利用率不能优先于无OOM和稳定性。

执行者每个Gate只发送一个最终 `GATE_X_FINAL_AUDIT` 或真正的 `BLOCK`，规划负责PASS/PASS_WITH_FIXES/BLOCK。低风险工程失败保留尝试并在既定权限内修复，不能把单job失败直接当阶段失败。每Gate一次普通审计退修，只有同根因或修复引入回归时允许第二次。科学选择、信息边界、预算外成本需规划/用户决定。

未来长作业由executor独占一个heartbeat：smoke10min、probe30min、full60min；无变化静默，终态先停监控再唤醒exact executor，executor完成后可见交付planning。沿用用户最近的executor-only监控路由偏好，最终需在新阶段授权时绑定确切线程及fallback；规划不轮询。阶段终态是完整报告与综合审计或明确科学停止，工程完成不要求正收益。用户已授权planning逐Gate派发并连续推进至第十阶段终态。

## 7. 信息架构检查与待冻结项

外层 `.planning` 是 `loopscope-tflt/.planning` 的兼容链接，实际新文件只写Git根内phase10。AGENTS/PROJECT_MEMORY/README保留旧阶段状态叙述，但较新顶部明确历史优先级；不借本次规划大规模清理历史。Phase9末态无活动executor，未发现需要接管的跨Gate任务。本次不重新核验HPC终态或重新审计旧PASS。

待冻结：数据revision/示例/长度细节；科学统计建议转为执行合同；分区规则冲突的明确处理；实际执行授权、线程身份和资源预算。两方向分支、三组配置、九档lambda、ARC数据集及25-shot标准文本评分已由用户明确，后续不重复询问。未决项不影响本轮总体规划完成，但不得被草案自动转化为执行权限。
