# Gate D 验收与第八阶段综合终审

2026-09-06。Planning `01a06fe9-c7bb-7d72-a906-234f301de317`。
Gate D：**PASS**。Phase 8：**TERMINAL PASS**（完成合同内实验，不表示假设成立）。
科学结论：**NO_RELIABLE_SPECTRAL_GAIN_IN_FROZEN_PANEL**。
exact executor `01a071e9-d133-7742-8826-e0cf5993a1f1` 的 GATE_D_FINAL_AUDIT已收到。
D producer `2dd3dfd59caf547a3abbc0a1516bc4a3207d9af0`；验收final源码/证据HEAD `8bf0748c7d4576da4d1708c72a645dc5574a9148`，loopscope clean且已push。

## Gate D 三项决定性验收

1. planning直接读HPC closure与fresh pre_outcome_verification：FULL_PANEL_CLOSED，54分片、18配置、每配置14042题、57subjects、252756配置-样本评分，target_gold_loaded=false。pre closure保存13:16:36.606559 UTC早于gold读取13:16:38.081938。读取producer/verifier/分析入口确认先全量身份、配置与有限分数闭合，再读取gold，无中途acc路径。
2. planning直接读取HPC analysis，对18个correct/N与24项配对结果运行独立标量算术检查：treatment_correct-reference_correct=wrong_to_right-right_to_wrong，差值除14042乘100与pp相等。fresh_count_verification为VERIFIED。检查统计代码中的subject内配对multinomial bootstrap、exact McNemar和两个四项Holm family，符合冻结合同；bootstrap重算复用同一算法，不夸称第二独立实现。中文报告全部18cells/24contrasts与结果及解释一致。
3. sacct现场确认最后12652622/12652623及batch/extern均COMPLETED0:0，9103/5061秒；monitor本地配置PAUSED。D全运行资源合计15617GPU秒=4.33805556GPUh，OOM0，最终显存余量47.57%/38.00%。A40排队取消为用户资源迁移且零GPU时间；同arm三段硬件分片安排一致。Git范围无D新增protected runtime修改。

W=`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope`。
闭合：`W/artifacts/phase8-gate-d-20260906T060000Z-closure-a1/verification.json`。
分析及fresh pre-outcome记录：`W/artifacts/phase8-gate-d-20260906T131600Z-analysis-a1/`。
正式pool/panel：`W/inputs/phase8-gate-d-20260905T142900Z-a1/`。
所有原始逐题scores、basis和校准残差保留HPC，Git只保存源码/配置/执行文档与证据摘要。

## 一次跨A-D综合审计

| Gate | 已验收对象 | 本阶段依赖闭合 |
|---|---|---|
| A | 数学核心、层/B边界、固定alpha语义；producer3fe34c4 | 用户后续确认完整科学合同v1，未使用历史草案网格 |
| B | 最小residual opt-in及原HFLM评分；producerc7e4e1f | None/zero路径与原版一致；实际native residual只在t>=1 pre-answer处理 |
| C | producer6d6214f，8正式basis、共同有序validation512 | 各model/window/K独立t1未中心化SVD，无test拟合，smoke basis未进入D |
| D | producer2dd3dfd，18全test配置、一次统计与报告 | K2/K4均alpha1、lambda0.5/rank1、方向固定，全部14,042身份；未扩arm或调参 |

本次复核现有A/B/C验收和合同连接，不重复已通过GPU/SVD/全套测试。模型revision、dtype、窗口cache、K对应basis、plain5shot和原continuation求和一致；固定horizon与K-specific basis的解释被保留。用户要求的A800切换有明确补充授权和真实canary，不以A40短debug估计长期显存。C显存预测偏差已在D重测并解决，未丢弃有效结果。

监控接续曾需要用户提醒，但最终exact executor主动交付已确认；无数据缺失或越Gate授权。四个独立executor均保持单Gate责任。没有材料性返修项；停止进一步hardening。

## 科学结果与边界

| 模型/窗口 | K2 Spectral−Loop pp | K4 Spectral−Loop pp |
|---|---:|---:|
|4B /12:15|+0.0214|+0.0142|
|4B /13:16|+0.1709|-0.0641|
|1.7B /12:15|-0.0641|+0.1282|
|1.7B /6:9|-0.0499|+0.0285|

八项nominal95%CI全部跨0，K2/K4各自Holm后无显著项。最大K2正点估计是4B13:16多对24题，CI[-0.0641,+0.4059]pp、Holm p0.6888。结论仅限固定rank1/lambda0.5、答案位置、离线共享方向和本历史开发benchmark；不是严格等效证明，也不否定逐prompt在线或其它谱方法。校准谱集中/同号累积不证明主方向有害。没有matched-norm正式对照，不能证明方向特异性优于普通阻尼；不以K4或Native探索对比替换主结论。

## 关闭

撤销D executor执行/调度/修改权限，保留任务只读；全部A-D PASS。ACTIVE_GATE=NONE、AUTHORIZED_EXECUTOR=NONE，唯一monitor保持PAUSED。第八阶段用户授权目标已完整结束，无自动Phase9/新增Gate/额外实验。逐prompt在线方向继续作为后续研究动机，需未来单独科学计划授权。

人类报告：外层`资产/报告/phase8/LoopScope_Phase8_配对结果报告.md`，同目录cells.csv/contrasts.csv/analysis.json。docs/loopscope_phase8.md继续保存可读计划与结果导航。
