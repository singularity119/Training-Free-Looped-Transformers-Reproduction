# LoopScope 第八阶段总体计划：双模型固定方向谱软衰减

> v0.5，2026-09-06。Gate A/B/C/D与阶段终审均PASS，阶段已结束；固定谱衰减配方未显示可靠acc收益。
> [科学合同v1](loopscope_phase8_contract_v1.md) 是完整参数/样本/方法的唯一来源；[control](loopscope_phase8_control.md)是动态授权唯一来源。原推导保存在[v0.1历史设计](loopscope_phase8_design_v0_1.md)，旧网格不执行。

## 1. 目标与已确认范围

比较固定window的原版Euler循环与加入SFA-inspired residual spectral damping后的MMLU 5-shot配对accuracy。两个模型均必做，不以第一模型正收益作为第二模型admission。

| 模型 | inclusive层窗口 | residual边界 | cache | dtype |
| --- | --- | --- | --- | --- |
| Qwen3-4B-Base | 12:15、13:16 | B12→B16、B13→B17 | first | BF16 |
| Qwen3-1.7B-Base | 12:15、6:9 | B12→B16、B6→B10 | last | FP16 |

K=2主实验、K=4补充；所有K固定alpha=1，h=1/K，总时长1。保留block、batch1、decodebypass和标准HFLM plain5shot choice-loglikelihood评分。窗口是代码层索引，12:15恰为四层，不是B12→B15。

## 2. 方向拟合与干预

从validation1531按固定subject分层seed20260905选择512题，无标签拟合方向。在完整test14042上评测，dev提供五个demonstrations。历史benchmark背景不改写为untouched test。

每model/window/K分别在无干预t=1的pre-answer residual上拟合top1，合计8个固定basis。矩阵512×d，未中心化、不逐行归一化；实际native residual转FP32，CPUfloat64 reducedSVD拟合。各K轨迹不共用前缀，不从test或当前prompt重新估计方向。

推理t>=1只在pre-answer token做delta'=delta-0.5*v*(v^T*delta)，FP32计算后cast回residualdtype，随后原h更新。t0、其他tokens、cache stash与关闭路径保留。主要方向可能有用，不能先认定有害。

## 3. 配置规模与判断

正式18配置：两个native，加四个model/window在两个K下各Loop/Spectral共16。每配置14042题，共252756个样本评测；校准8×512=4096条配置-样本轨迹。实际operator/token成本及walltime由debug测量，不承诺未经测量的耗时。

四个K2 Spectral−Loop为主family，K4为次要family；逐题配对acc(pp)、正确数/N、错→对/对→错、subjectmacro、10000次分层paired bootstrap95CI、exactMcNemar+Holm按family报告。最终全18配置闭合后分析，不用partialacc挑窗/调参。

本轮不增加同范数/随机/Lower-alpha正式arms；对应数学参考可保留。若得到提升，结论属于整体干预方案，缺少matched-norm正式对照不能证明方向特异性。正、负、不确定均合法。

## 4. Gate路线

| Gate / 独立任务topic | 目标 | admission |
| --- | --- | --- |
| A / Recipe-and-Spectral-Core | 只读provenance、数学核心、窗口映射、最小接入方案 | 已PASS，commit3fe34c4 |
| B / Intervention-Debug | 真正residual opt-in接入、双模型8题内debug、评分/关闭路径一致 | 已PASS，producer c7e4e1f |
| C / Residual-Calibration | 同一512身份采集8配置无干预残差、拟合冻结8basis及有限跨步诊断 | 已PASS，producer6d6214f，8正式basis/512身份闭合 |
| D / Paired-Accuracy | 完整18配置test采集、一次配对统计与报告 | 已PASS，18配置完整评测及一次统计/报告，阶段终审完成 |

每Gate创建首条消息和handoff显式强调读取research-gate-orchestrator。只有当前Gate独立executor获得权限，不能由子agent代替。PASS后规划立即推进满足admission的下一Gate；阶段末完成一次跨Gate终审，不要求正acc才能结束。

## 5. HPC和后续动机

debug/smoke一律debugpartition、每job<30分钟。正式需同commit/launcher/env/runtime/producer的成功debug，资源按现场合法优先级、上尾长度和显存吞吐决定，禁止为利用率改变batch/科学。executor持有一个smoke10分钟/probe30分钟/full60分钟heartbeat；正常安静，attempt终态暂停并唤醒exactexecutor，最终GATE_X_FINAL_AUDIT或真正BLOCK送planning并确认。planning不轮询。

逐prompt在线方向作为后续阶段动机：可从同prompt多token或跨步residual自适应估计，但本阶段不实施。本阶段结果和实现限制将用于决定后续是否值得研究该方向，而非提前扩大当前panel。
