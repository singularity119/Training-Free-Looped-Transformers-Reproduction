# Phase 9 追加 Gate E：逐轮滞后一轮方向

2026-09-23 用户在 Phase 9 主线已结项后明确追加一个独立 Gate E 实验。本合同仅约束新增探索性实验；已完成的 contract_v2、47-cell panel、原报告及 A–D PASS 保持历史事实，不追溯修改。

## 科学问题

原 Online-t0 对同一题一次拟合 D0 有效 prompt token 残差第一主方向，并在全部后续轮固定使用。新 Lag1-Online 在每次 t>=1 调用时，用该臂自身前一次窗口调用的原生、干预前残差 D_(t-1)[S,:] 的第一右奇异方向 v_(t-1) 衰减当前 D_t 答案前位置；t0不干预。t1使用v0，t2使用v1，t3使用v2。先从D_t原生残差计算下一轮方向，再仅用v_(t-1)处理当前答案前位置，不使用同轮v_t干预，也不能读未来轮。t=K−1无需估计未使用的下一方向。估计方向来自每轮完整有效prompt token残差矩阵，而非单条答案位置向量。每题/候选prefix边界、排除padding/special/continuation及跨题释放沿用Phase9原规则。

λ=0.5、α=1、Euler步长1/K。原dtype计算G(X_t)−X_t后转FP32，用不中心化、不逐行归一化的精确reduced SVD，CUDA gesvd，规范符号和单位化；只在t>=1答案前位置衰减。非有限/零能量/SVD失败显式失败。历史方法实现保持只读。新增 Lag1-Matched-norm 对照：在该臂自身轨迹同样逐轮从前一轮完整残差估计v，当前答案前位置的残差按Lag1定向处理产生的范数比例缩放，零向量保持零；K>=3不同臂轨迹会分化，只称自身规则的等范数对照。

K=2 时Lag1-Online与Online-t0算法相同、Lag1-Matched与原Matched-norm算法相同，故不新增K2全量运行；真实debug须验证数值等价。新配置为3模型窗口组合×K3/4×2新臂=12 cell，MMLU test每cell14042题，共168504新评分。旧47配置全部保持，可经原始身份/配方/分数验证后复用；延展展示为59逻辑cell，共828478配置-样本记录，不声称59次新运行。

## 模型与结果性质

沿用contract_v2两模型及revision、4B13:16/15:18与1.7B12:15、dtype/cache、MMLU revision与原plain5shot完整continuation loglikelihood、batch1、decode bypass、模型eval/no_grad、固定first_n5 demos。旧test14,042/57 subjects已被Phase9查看；本实验是明确的事后探索性比较，任何p值/CI均不称为新held-out确认。方向仅从当前题prompt在线拟合，不使用gold/accuracy选方向、窗、K或参数。

新增原始分数完整闭合前不读gold/逐题correctness；闭合后核验旧配置原始分数并做一次预先指定分析。首要探索性family为六个cell各Lag1-Online减原Online-t0、Lag1-Online减Lag1-Matched-norm，共12项，exact双侧McNemar、一个Holm family α0.05；subject内paired bootstrap10000、seed20260923、nominal95%CI。Lag1-Online对Loop、原Matched-norm、Native只作描述性背景，不从其显著性宣称方向更新价值。只有同cell两项均通过12项family，才称该cell有支持“逐轮方向更新额外价值”的探索性证据；K>2两新臂自身轨迹分化限制必须注明。所有结果、负值、算力/SVD开销均报。

## 工程和资源边界

新增phase9_gate_e_*配置/模块/脚本/测试，复用稳定Phase9 scoring接口；不得覆盖Phase8或Phase9原runtime、basis、score、analysis和报告。真实debug在debug分区<30min覆盖两模型、两臂、K3/4、长prefix与K2等价。正式A800若可用，先可保留、代表性单进程canary实测长prefix和SVD峰值/吞吐，再按实际并发加载峰值扩展。总预算160GPUh含失败，最多同时2GPU；同卡packing初始1，进一步扩展须同卡、同模型、同K真实代表性并发加载/运行实测且OOM余量充分，不能把单进程canary或别模型结果外推。保持batch1，不为显存使用改变科学。异常保留，只有无效/缺失分片可在新路径重试；禁止越并发上线。

Gate E完成后由本planning验收并做追加实验简报；不自动授权Phase10或新Gate。
