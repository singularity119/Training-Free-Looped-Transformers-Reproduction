# Phase 11 D-8 GPQA thinking 接续说明

动态权限以[control](../.planning/phase11/loopscope_phase11_control.md)和
[D8 handoff](../.planning/phase11/loopscope_phase11_gate_d8_handoff.md)为准。
exact executor、source、jobset和监控绑定集中在
[replacement execution](../.planning/phase11/gate_d8_replacement_execution.json)。
本文是操作说明，不授予新Gate权限；D7保持原独立executor和monitor。

## 冻结配方

- `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`，原生thinking模板。
- GPQA Main原448题及既定选项顺序，三组Native、current_t K2 λ0.5、fixed_t0 K3 λ0.9，共1344条fresh。
- `max_new_tokens=32768`，BF16、batch1、native EOS；sampling temperature0.6/top_p0.95/top_k20/min_p0/repetition_penalty1。
- 每题seed=`20261018+canonical_index`，三组同题同seed；工程恢复不能改变采样流。
- 两方法沿用inclusive15:18、cachefirst、decodefull；不新增配置或更改方向拟合语义。

## 当前接续与资源

2026-10-20的GPU运行源码为`316e467d1a0129ff46509637dd1a90b4e941604e`，
使用已提交tree的`source-r3`导出与原pinned Python。
权重来自获准的国内HF镜像，固定revision的config、generation_config和权重索引
已与官方缓存直接比对；tokenizer复用同revision官方缓存。下载未向镜像发送凭据，
临时代理隧道已关闭；下载证据保留在机器绑定指向的artifact root。

真实debug `13230943_0`于2026-10-20完成，`COMPLETED 0:0`、75秒，
`debug-r3-path-verified.json`确认三组真实thinking/final区间、S2、decode/cache和方向生命周期。
旧`13230892_0`失败记录及三条synthetic raw保留，消耗136秒；不覆盖历史尝试。

三条可保留正式canary各用canonical_index344（最长prompt2819 tokens），packing1。
2026-10-20用户直接授权提交其余1341条：真实debug通过且已有容量证据时，
允许结合整卡/进程peak memory与保守完整KV预算提交全量，无需等待全部canary终态。
各组剩余计划都排除344，成功/在途canary均保留；完整registry严格为三组各448。
短算术debug不能用于推定32768预算的正式显存。预留约20%显存余量，禁止OOM导向的盲目packing。
每次提交核对普通用户最高合法A800分区/QoS；整个Phase11与D7合计最多8张A800，
每GPU最多8CPU。成功正式记录保留，仅缺失/无效项可在新路径恢复。

## 验证器与评分边界

`check_phase11_d8_gpu.py`对每条raw调用原`validate_bundle`，重算冻结S2并核对
identity、seed、实际thinking/final文本、增量decode、cache和方向生命周期。
合法生成可能得到S2 `no_match`；它是正式计错结果，不能仅因此判producer失败。
smoke至少一个真实S2 positive仅覆盖解析正路径，不授权重新抽样、挑seed或放宽S2。
2026-10-20的verifier修复只改该检查及其测试，producer、S2和正式输入均未改变。

主评分仅解析首个生成`</think>`之后的final区间。未闭合thinking、final为空、
no_match或conflict均计错，截断不排除，始终保留每组448的分母。
不从thinking中的临时答案挽救预测。

全部1344条raw/identity/预测闭合后，先运行：

```bash
python scripts/loopscope/analyze_phase11_d8.py --close --registry REGISTRY --output-dir NEW_CLOSURE
```

该步骤保持gold-free并固化预测。随后才使用既有独立gold：

```bash
python scripts/loopscope/analyze_phase11_d8.py --closure NEW_CLOSURE/closure.json --gold GOLD --output-dir NEW_ANALYSIS
```

两方法对Native组成一个两项Holm family，exact双侧McNemar；
题级paired bootstrap10000次、seed20261018，CI为名义95%。
这是同题集配置选择后的跨checkpoint/thinking探索性迁移，不能称独立题集确认，
也不能把差异单独归因于thinking或checkpoint。

## 监控与交付

只使用`phase11-d8`一个executor-owned heartbeat：debug10分钟，正式60分钟，
PENDING即允许监控。observer从jobset读取exact IDs，只读squeue/sacct及完成计数；
健康无变化静默。必须由sacct确认每个exact ID，退休队列或Invalid job id不能证明完成。
当前observer导出路径见机器绑定，不能假设旧source-r2包含observer模块。

终态先暂停同一monitor并保存lifecycle，再触发exact executor继续；若executor已在活跃
回合中检出终态，则直接保存暂停证据并接续，避免重复唤醒。无法确认投递时向planning
发送`AUTOMATION_RELAY_REQUIRED`。attempt失败先诊断并在原权限内修复，不自动判Gate BLOCK。
最终由executor向exact planning投递唯一`GATE_D8_FINAL_AUDIT`或真正`BLOCK`并保留确认；
只有planning决定PASS。

原题、生成token/文本、逐题预测和gold留HPC。人类聚合报告放仓库外层
`资产/报告/phase11/`，不把原始数据或模型加入Git。
