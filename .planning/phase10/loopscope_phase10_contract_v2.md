# Phase10 科学合同 v2：加入 current_t

2026-10-02。用户新增当轮估计、当轮立即衰减消融，包含t0。本文取代v1科学合同；v1与原始handoff保留历史，冲突处以本文及current_t amendment为准。执行仍限当前control。

## 固定范围

三组cell：Qwen3-4B-Base inclusive15:18 K2/K3，Qwen3-1.7B-Base inclusive12:15 K2。模型/tokenizer revision、dtype/cache、alpha1总时长、步长1/K、beta0、batch1、block damped_euler、decode bypass及原Phase9数值语义保持。lambda为0.1至0.9共九档；无Matched-norm。Native与Loop对照。

MMLU原revision、完整test14042/57subjects、plain5shot、dev示例和原完整字母continuation评分。ARC固定revision210d026faf9955653af8916fad021475a3f00453、全test1172、25-shot原生文本评分，主acc_norm附描述性acc；seed20261002，具体输入binding仍由A提供、planning接受后B准入。保持无标签prefix边界，不改shot/丢题/缩短答案。

## 三策略

所有策略以其自身当前状态计算D_t=G(X_t)-X_t，原dtype残差后FP32拟合。S是实际保留prompt的非padding/非special位置，不含continuation。方向均为不中心化、不行归一化的FP32 exact reduced SVD top1，CUDA gesvd。干预仅p=真实最后context位置：delta'_t=delta_t-lambda*v*(v^T delta_t)，cast回原dtype后原Euler更新；其余位置不变。

- fixed_t0：t0估v0但不干预；t>=1使用固定v0。
- lag1：t0估v0但不干预；t>=1用上一轮干预前残差估出的v_(t-1)。
- current_t：每轮t=0,...,K-1先从当轮尚未干预的D_t[S,:]估v_t，立即用于该轮p处衰减，再Euler更新进入下一轮。包含t0，不在已经衰减的D'_t上反复拟合，不借用其他臂轨迹，不额外调用窗口求方向。

K2仅fixed_t0与lag1等价，经实现验证可共享；current_t-K2不等价，必须独立运行。同题候选方向只在完全相同前缀和科学轨迹保证下复用，跨题/配置不共享。current_t改变干预起点及方向时序，其对比是组合策略消融，不能单独归因于t0或当前方向；本阶段不额外引入隔离因素的第四分支。

## 数量与复用

每数据集：原36个Online配置+current_t三cell×九lambda=27，合计63干预配置；再加Loop3、Native2=68独立配置。展示3cell×3policy×9+5=86行，其中18行为K2历史两策略别名。

MMLU仍可核验复用9个历史配置，新增59；ARC68全新。两数据集合计136配置、预计新增127。新题级记录MMLU59×14042=828478、ARC68×1172=79696，合计908174。历史复用需同配方逐题原分数核验，不能只用acc。

## 分析

每个非重复cell×policy共7组，MMLU各自Online micro acc最高lambda作为迁移参数，并列较小lambda；ARC outcome前固定7个lambda*，不根据ARC修改。

ARC预选迁移7×[Loop,Native]=14项一个Holm主family。每数据集完整63×2=126项扫描比较一个探索Holm family。策略比较每lambda共5项：两个K2 cell各current_t对fixed_t0/lag1的共享基线；K3三策略两两比较三项。九档合计45项一个次family，K2别名不重复检验。exact双侧McNemar、alpha0.05，bootstrap10000次seed20261002，MMLUsubject内配对、ARC题级配对，nominal95%CI。acc_norm是ARC主指标，原acc描述。

无Matched-norm，不作方向特异性结论。报告全网格和负结果，区分MMLU开发、预选ARC迁移、ARC当地最优。其余信息隔离和停止条件沿用既有计划；任何科学配方变动由planning处理。
