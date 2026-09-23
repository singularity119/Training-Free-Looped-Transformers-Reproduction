# Gate E：逐轮滞后一轮方向探索实验

按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，实际读取skill及references/protocol.md；HPC操作实际读取hpc2-hkustgz-ssh。监控仅向本executor自身汇报，沿用monitor_routing_amendment，无planning fallback。

Planning=01a07ff5-79b1-79f0-809a-4d0971475686；Executor=01a0cc03-0181-7a53-b305-c32637a00315；model=gpt-5.6-luna/max。科学依据为gate_e_lag1_contract及较新的gate_e_direction_policy_amendment（实现策略补充优先）；A800分区以gate_e_a800_priority_supplement为准；Phase9 contract_v2提供被复用的模型、数据与老臂语义，旧A–D均已完成。用户确认前一轮全部有效prompt token原生残差、仅K3/4、含新Matched对照、沿用MMLU test并标注事后探索。不得自行改成同轮方向或仅答案向量。

## 最短执行路径

1. 在精确loopscope branch/专用clone增量实现Lag1-Online与Lag1-Matched以及12-cell新manifest、无标签producer/verifier和冻结分析入口；与原Phase9 runtime并行存在，Phase8/原Phase9核心不覆盖。对每个prompt保留必要的前一方向，在t>=1使用上一方向衰减当前答案行，再基于当前D_t有效prefix拟合下一方向；只需K−1次SVD，debug验证v来源、t0原样、t1/t2/t3时序、跨题释放与K2旧法等价。首选新路径 `src/tflt/loopscope/phase9_gate_e_*`、`scripts/loopscope/*phase9_gate_e*`、`configs/loopscope/phase9_gate_e_*`、`tests/test_phase9_gate_e_*`、`docs/loopscope_phase9_gate_e.md`；原Phase9 scorer/adapter若无法复用，先提供最小失败例和变更说明给planning，再获补充授权。不要大重构。
2. 只读旧Phase9 test pool/score/artifact与模型缓存。新producer的test prompt/choices允许读，gold和旧逐题correctness在全部12新cell原始分数闭合之前不得读。模型/任务/revision/评分固定；结果运行标识与真实git HEAD分别记录。
3. 定向本地测试、CLI/bash-n、两模型两臂K3/4短debug真实模型/输出/verifier；同commit/launcher/runtime通过后才入正式。正式优先A800，先一GPU可保留代表性canary并实测不同长度/峰值/SVD时间，之后在160GPUh与同时2GPU内批量完成12×14042；每次新launcher或producer修复先单child验证。必须监控当前运行GPU数，避免Phase9 C发生的3GPU超限；前任务全终态/释放才提交会越上限的新任务。同卡packing1起，packing2须实测两模型同时加载与长prefix峰值且至少15%显存余量，不能根据单进程推断。吞吐不改善就保持1。允许局部工程故障按原配方新路径只补无效/缺失，保留OOM/失败日志；绝不删长题或改SVD。
4. 完成12新cell每个14042唯一身份、57subjects、四候选finite原始分数/配置/revision一致的无gold闭合，并记录时点。复用原47结果需逐题原始分数身份/配方核验，不只拿acc。全部闭合才直接按合同接gold执行一次预先指定12项Holm及CI/翻转分析；因为同test已看过，报告统一标“事后探索”。生成外层 `资产/报告/phase9/LoopScope_第九阶段GateE逐轮方向实验报告.md` 和完整新cell/contrast附表，不改写已结项主报告结论。

## 资源、Git与路径

总累计160GPUh（含debug、canary、失败/重试），同时最多2GPU，单正式job至多24h且服从partition限制。初期8CPU/128GiB per GPU；法律/队列限制与实际需求允许在24CPU/256GiB内调整，不超节点规则。若预测无法在剩余预算完成，带实测数回报BLOCK，不超支。

本地唯一写repo=`/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt`，branch精确loopscope，保护祖先4f59bd93eca4da3cbf458a93508f91c5b23912bc。每次写文件/提交/远端运行先核status、branch、HEAD、root、祖先。不回退其他人编辑。允许普通commit/push origin loopscope，禁止force/rebase/reset。远端原专用clone当前历史上被C切至phase9-gate-c分支；先只读核对，并使用**另一个明确独立路径**的工作副本/源码快照执行本Gate，不在原路径内切分支来冒充独立快照；必须确保运行源码与本地Git提交可追溯。新增远端输入/输出仅在 `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope` 下 `phase9-gate-e-<timestamp>`，缓存仍原shared/固定环境，不擅自升级。

SSH strict BatchMode/StrictHostKeyChecking=yes/UpdateHostKeys=no/ClearAllForwardings=yes/ConnectTimeout10。只可取消/重试本Gate的明显失败或超预算任务，不碰旧job/其它任务；pending不反复取消重提。冻结科学、gold边界、GPU上限改变需planning/用户新授权。最多3 bounded子agent且明确ownership，不能独立调度/远端写入/Gate终态；你不是唯一工作者，不回退他人改动。

## 终态

最小验收是12新cell完整、旧47原始分数复用核验、一次冻结探索性统计、报告准确且完整披露资源偏差。正结果不是PASS必要条件。无真实job不建monitor；真实长job只有你自己的一个heartbeat，debug稳定RUNNING后10min，full60min从PENDING观察；终态pause后仅自唤醒，健康静默。完成pause monitor，主动向planning发送唯一GATE_E_FINAL_AUDIT或真正BLOCK并确认送达，含源码/运行身份、测试/作业、闭合/解封时点、统计、报告路径、累计GPUh/并发/峰值/OOM、偏差。你不自行PASS、不进入下一Gate。
