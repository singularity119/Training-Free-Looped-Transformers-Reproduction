# Gate E 追加实验验收与终态

Date: 2026-09-24
Planning thread: 01a07ff5-79b1-79f0-809a-4d0971475686
Executor: 01a0cc03-0181-7a53-b305-c32637a00315
GATE_E_AUDIT_DECISION=PASS
EXTENSION_STATE=COMPLETE

## 决定性证据

规划线程独立以strict SSH直接读取HPC2 `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase9-gate-e-20260923T031256Z-exec` 中的 `FULL_NEW_12_SCORE_PANEL_CLOSED.json`、`REUSED_NEW_18_RAW_SCORES_REVERIFIED.json`、`REUSED_HISTORICAL_29_RAW_SCORES_REVERIFIED.json`、`analysis-gate-e-20260924-a1/pre_outcome_verification.json` 与 `analysis.json`。新增12cell、24shard、168504条原始分数和57subjects；旧18+29cell原始分数分别252756+407218条复核，三者同一冻结test pool且闭合时 `target_gold_loaded=false`。pre-outcome 2026-09-24 07:06:52 UTC，gold开始07:06:59 UTC；分析59逻辑cell、30项比较（12主+18描述），预设10000次bootstrap seed20260923。

规划独立根据全部12项主比较翻转计数重算exact双侧McNemar、净纠错/百分点和12项Holm，均与`analysis.json`一致；0项校正通过。最强nominal点估计4B13:16 K3 Lag1-Online减Lag1-Matched +0.249252pp，95% nominal CI[+0.0570,+0.4415]，Holm p0.176261；同cell减原Online-t0为−0.021364pp。没有cell同时满足两个预设对照，故无逐轮方向更新额外价值的探索性支持；同一已看过的MMLU test，不宣称新held-out确认。外层中文报告及59/12/18行附表存在，表行数与关键结果已读回核对。

执行任务报告本地`loopscope` HEAD `782df42d49e428a47d8574126b0f4eec361b226f` clean且已推送，运行源码快照为`4e0c239962c41dd737aba4dbf570972527fdc748`；后续775a827 verifier及782df42分析导出修复不改变评分worker。10项定向测试、py_compile和git diff --check通过。规划核对本地branch/HEAD/clean及保护祖先。原Phase8/Phase9主线实现、47cell结论和报告未被覆盖。

规划直接读取Slurm sacct：12839160/12839161 debug COMPLETED 0:0；旧12839277 canary取消前有1秒launcher缺jq失败；12839298 canary与12840467 remaining均COMPLETED 0:0，A800 emergency_gpu。执行账本累计33.099722GPUh/160，最大并发2GPU、packing1、科学batch1、OOM0；因未实测同卡双进程长prefix，保持packing1。失败及用户指定分区迁移记录保留，没有无效分片混入有效闭合。监控已暂停并只向executor回报。

Gate E PASS评价的是执行和证据闭合，不代表方法有效。追加实验结束，Gate E executor执行权限撤销；无新的计算、Gate或下一阶段授权。第九阶段A–D主线终态保持原样，本追加终态独立记录。
