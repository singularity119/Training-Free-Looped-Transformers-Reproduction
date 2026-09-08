# Phase 8 Gate E 科学追加：4B K=3

2026-09-08，用户明确要求新增独立Gate补做4B K3实验组，其他配置不变。原A-D合同v1、18配置结果和2026-09-06终审保持冻结；本文件仅定义后续追加，不能追溯改写原主/次分析。

模型Qwen/Qwen3-4B-Base revision906bfd4b4dc7f14ee4320094d8b41684abff8539、对应tokenizer、BF16；窗口inclusive12:15(B12→B16)、13:16(B13→B17)，cachefirst。K=3包括t0，alpha1，h=1/3，总时长1；beta0、damped_euler、block、batch1、decodebypass、plain MMLU5shot及评分均沿用v1。

使用C已有同一有序validation512身份及prompt；每窗口重新采集K3无干预轨迹t0,t1,t2，t1答案前位置实际native residual转FP32，不中心化/不逐行归一化，CPUfloat64 reducedSVD取rank1。共2新basis、1024配置-样本轨迹、3072 residual行。不能复用K2/K4 basis或轨迹前缀。单位方向规范符号和FP32运行方式同v1。

在原D test pool完整14042题/57subjects上运行4新配置：2窗口×原版Loop/Spectral。Spectral只t1,t2的pre-answer位置衰减lambda0.5；t0/其它token不变。共56168配置-样本评测。1.7B不扩展，K2/K4不重跑；原D 4B Native仅作已知、已审计的探索上下文，可复用其保存结果，不增加NativeGPU评测。

4配置全部身份/配置/有限性闭合后，才统一读取gold分析新增结果。原D结果已知，因此本组明确标为post-outcome exploratory supplement，不声称独立确认。两项K3 Spectral−Loop为单独K3_EXPLORATORY family：microacc、correct/N、错→对/对→错、subjectmacro；subject内paired bootstrap10000次seed20260905、nominal95CI，exact双侧McNemar、Holm仅两项。不能把K3误归入K4_SECONDARY或重新合并旧family。与Native及旧K2/K4的展示仅描述性，不选择最佳K冒充预定主结果。正负不确定均合法。

后续逐prompt方向、额外rank/lambda、其它模型窗口、训练、test驱动调参均不授权。Gate E结束后planning接受结果并对追加范围简洁重结案，原A-D终审保留。
