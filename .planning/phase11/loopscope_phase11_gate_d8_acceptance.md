# Phase 11 Gate D-8 规划验收（2026-10-21）

GATE_D8_AUDIT_DECISION: PASS

验收对象：executor 01a15760-371e-7e61-af51-d8375a503bdc/local；producer 316e467d1a0129ff46509637dd1a90b4e941604e；交付证据 6c161939aadd194a9a50fe70652172c8299b9b75。PASS 表示本 Gate 的执行及冻结评分交付完成，不表示方法有效或显存准入达标。

## 轻量核验

采用已交付 final_audit、completion_evidence、analysis aggregate 和 terminal_delivery；本次未重新 SSH 或逐条重审远端 raw。核对 10 个 exact jobs（13231104_0..2、13231534_0..6）完整 accounting、45 个子进程完成记录、1344/1344 闭合与预测封存早于 gold 的时间顺序。规划独立以二项组合尾概率复算两项 exact McNemar p、净正确数变化及正式 66.987778 GPUh，均一致；两项 Holm family 的调整值及判定一致。monitor PAUSED 已有执行方回读证据。

|配置|正确/448|准确率|较 Native|Holm p|
|---|---:|---:|---:|---:|
|Native|183|40.8482%|—|—|
|current_t K2 λ0.5|164|36.6071%|−4.2411 pp|0.061386|
|fixed_t0 K3 λ0.9|154|34.3750%|−6.4732 pp|0.017470|

仅 K3 在冻结 exact/Holm 标准下显著下降。K2 nominal percentile bootstrap CI 不跨零，但不能替代冻结的显著性判据。S2 no_match 为80/140/126；格式失败影响综合准确率，不能把下降全归因为推理能力下降。配置来自同一 GPQA 题集的历史选择，且只有单次采样；这是 checkpoint/thinking 配方迁移的探索性结果，不能分离 checkpoint 与 thinking 的因果效应或估计采样方差。

## 接受的已完成运行偏差与后续约束

实际整卡峰值81031/81920 MiB，仅1.09%余量，没有达到预定约20%余量。此项资源准入偏差明确保留，不改写为达标。已有记录全量有效、无OOM、无非零退出，未发现其影响科学输出的证据；因此不为弥补资源估计而重跑已完成面板。未来不得把本次成功视为 packing2 安全依据，须在新的明确授权下重新选择/验证有足够显存余量的并发配置。保留失败debug及修复记录、SSH可见性中断记录和全部原始工件。

## 权限收口

D8 关闭；executor 的采集、GPU/Slurm、下载、重评分及监控执行权限撤销，仅保留验收回执。monitor 保持 PAUSED。不创建下一 Gate，不接管 D7，不宣称 Phase11 整体完成。原有 staged/unstaged AGENTS/control/handoff 变更保留；本次未提交或推送它们。
