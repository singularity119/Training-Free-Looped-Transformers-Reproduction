# Phase 9 冻结科学合同 v1

2026-09-22。用户授权planning逐Gate推进至第九阶段结束，模型、测试集和loop配置对标第八阶段最终实验。本文取代总体报告原K2-only建议，作为科学配方唯一规范来源；执行权限仍由control/exact handoff授予。

## 模型、数据与计算

- Qwen/Qwen3-4B-Base revision 906bfd4b4dc7f14ee4320094d8b41684abff8539，BF16/cache first，零基inclusive 12:15、13:16、15:18，边界B12→B16、B13→B17、B15→B19。
- Qwen/Qwen3-1.7B-Base revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1，FP16/cache last，12:15、6:9，边界B12→B16、B6→B10。tokenizer同对应model revision。
- 全部五窗口K=2,3,4；K包含t0；alpha1、步长1/K、总时长1、beta0、block damped_euler、batch1、decode bypass、eval/no_grad、不训练、不量化、不生成CoT。
- cais/mmlu revision c30699e8356da336a370243923dbaf21066bb9fe。plain5shot，dev first_n5示例，lm_eval0.4.11原完整continuation loglikelihood评分和tokenization/截断语义。
- 诊断复用第八阶段validation512，固定subject/doc_index及seed20260905；只读sanitized prompt/choices，不读目标gold。正式全test14042/57subjects；不删长题、不换split。允许按冻结规则从每条test自身prompt在线估计方向，但不跨test聚合拟合、不看label/score选择方向或参数；这是Phase9对旧“test不拟合方向”规则的明确方法范围更新。

## 方法

D_t必须先按原dtype求G(X_t)−X_t，再转精度。t0不干预，S取实际保留context中非padding、非tokenizer特殊token的位置，包含demonstrations/题干/选项/Answer:，排除continuation。p为实际最后context位置，不能用最后模型输入位置代替。

Online-t0：对D0[S,:]不中心化、不逐行归一化，FP32 reduced SVD（CUDA使用gesvd，CPU精确SVD）取得top1右奇异方向，规范最大绝对分量为正并FP32单位化。不使用随机低秩近似或幂迭代。t>=1仅p处 delta'=delta−0.5*v*(v^T delta)，FP32投影后cast回原dtype，其余位置及Euler原加法不变。同题内v固定，跨题释放。零能量、非有限或SVD失败显式报错，不静默退化。

Matched-norm：在该臂自身轨迹，使用该题t0方向和当前delta计算s=||delta−0.5*v*(v^T delta)||/||delta||，应用s*delta，零delta保持零。K2干预前状态一致，构成严格同状态范数对照；K3/K4干预后轨迹分化，只称各自当前状态上的自适应等范数规则对照，不声称与Online真实轨迹逐步严格匹配。两臂均保持t0不变。

允许每候选重新估计作为最小实现；同prefix方向/投影须一致。任何缓存仅限exact prompt/model/window，不能跨题污染；需验证后再优化。同题候选context截断不一致则显式处理评分边界问题，不得静默用候选特异方向。

Shared-t0与Shared-t1完整保留Phase8实现、basis及其FP64拟合语义；原方向不覆盖、不重拟合、不改Phase8脚本配置。Native与Loop同第八阶段。Phase9以独立模块复用现有residual_transform接口，禁通用重构。

## Panel、复用和统计

最终逻辑面板：五窗口×K2/3/4×[Loop,Shared-t1,Shared-t0,Online-t0,Matched-norm] + 两个Native =77配置，合计1,081,234配置—样本记录。新增30个在线/等范数配置，共421,260评分；原47配置优先复用经身份/配方/原始分数核验的历史结果。不能只复用acc，也不把77逻辑配置称为77次新跑。旧结果不可验证时由planning允许原配方补跑，禁止换配方。正式新评分全闭合后再统一接入旧逐题outcome和gold分析；A/B只可核查旧配置/路径/schema，不能读旧逐题correctness来设计方法。

主要：K2五窗口Online−Loop和Online−Matched-norm共10项Holm family。次要K3同10项一个family；K4同10项一个family，分别报告，不替换K2主结果。Online−Shared-t0与Online−Shared-t1共30项组成一个探索性family。Native比较描述性。exact双侧McNemar，family alpha0.05；subject内配对bootstrap10000次、seed20260922，nominal95%CI。报告correct/N、micro acc、pp、错→对、对→错、净纠错、subject macro、全部负结果和资源开销。

只有同cell Online−Loop及Online−Matched-norm均经所属family通过，才支持该cell内方向选择的额外价值（K>2保留轨迹分化限制）。不显著不等于等效；MMLU是已使用的开发域，不能宣称全新benchmark验证。

## 诊断、数值与资源

B在未干预Loop轨迹收集E0、A_t、C0t、谱间隔、实际范数比、token长度和SVD耗时。v_t仅为诊断不用于干预；零分母null注明原因。FP32与CPU FP64参考比较投影，正常谱间隔样本相对投影误差目标<=1e-3；近重根单列谱/投影稳定性，不按几何或acc删题，材料差异报planning。无标签几何不决定保留哪些cell。

debug/smoke一律debug、preflight<30min；正式前同commit/launcher/env/runtime/producer须通过。卡型不冻结，采用普通用户合法最高优先级资源，按实测长序列峰值/吞吐选择packing。GPU总预算和并发由B证据后C handoff限定；不能把CPU参考SVD静默替代主算法或改变batch/dtype以填显存。保存失败、只补无效缺失工作，新路径写入。

## 终态与扩展

A增量实现，B真实debug及512诊断，C全量闭合与一次统计，D报告；planning在D末做一次跨Gate综合审计。工程PASS不要求正向结果。未含新窗口、rank/lambda搜索、t1在线重估、全token干预、新benchmark或训练，扩展需用户新科学范围。第八阶段F交付收尾保持独立，不由Phase9自判其PASS。
