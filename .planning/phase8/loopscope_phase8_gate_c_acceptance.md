# Gate C 轻量验收：PASS

2026-09-05；planning `01a06fe9-c7bb-7d72-a906-234f301de317`。
已收到 exact executor `01a070f0-158e-72f2-9d06-5ba47c02e2bd` 的 GATE_C_FINAL_AUDIT。
生产源码 `6d6214f6ad912b1984b1e0994015270acf627f59`；最终证据提交 `8c8ef434a4be2025f9603ce7690b90d4abb2f330`。

规划独立核对三项决定性事实：

1. HPC sacct 显示 formal 12649531/12649532 及 batch/extern 均 COMPLETED 0:0，耗时386/176秒。
2. 直接读取 closure verification.json：FORMAL_CALIBRATION、8/8配置、各512身份、正确模型/窗口/K/hidden width。fresh verifier 最大 eigen residual relative error 为3.452e-8，小于1e-4。另直接比较所有8 basis.source_identities 与正式 pool 的完整有序512清单，相等，rank1/lambda0.5/row_count512均一致。
3. 读取各cell collector：4096条配置-样本轨迹、12288 residual rows、duplicate_calls0；源码改动范围和终态包符合授权。唯一C monitor已PAUSED。

工件共同根：`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope`。
closure：`artifacts/phase8-gate-c-20260905T100000Z-closure-a1/verification.json`；pool：`inputs/phase8-gate-c-20260905T100000Z-a1/calibration_pool.json`；basis分别在`runs/phase8-gate-c-20260905T100000Z-formal-q4-a1`和`-formal-q17-a1`下cell-0..3。

4B正式packing3峰值46346MiB、余量约4.5%，高于短debug预测；作业成功、OOM0，因此保留有效校准，不要求重跑。Gate D必须重新测量长程和test上尾长度，不能沿用三进程显存结论。这是下一Gate资源约束，不是方向拟合失败。

谱集中及同号累积只描述校准集，尚无准确率或有害主方向结论。无test/outcome解封。
Gate C PASS、executor撤销执行权、保留只读；八个正式basis准入Gate D。其余跨Gate解释和结果一致性在阶段末一次综合审计。
