# Gate D-3 — GPQA length pilot

2026-10-12；用户明确授权按已讨论路线派发。Executor01a12e14-57d6-7e51-9cfa-588f4d038352/local，gpt-6.1-sol/high；planning01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local。按照research-gate-orchestrator技能及protocol协作，实际读AGENTS。Base c825b2edf8ff5bc3f49bde30f1800b84e27f9d77，nested loopscope-tflt/loopscope。保留并可随本Gate提交未提交D2 acceptance/control。

## 科学边界

仅改变max_new_tokens到8192及必要的输入总长度准入检查；原2048工件、合同与评分不覆盖。固定模型revision/BF16、GPQA448输入/seed/选项顺序、native chat template/0shot CoT、greedy、native EOS、batch1、window15:18、K2/K3、原18cell方法不变。cache_strategy=first、decode=full及prefill方向冻结复用不变；禁止缓存优化、强制答案、简短提示或新停止规则。8192必须验证原生context容量满足完整prompt+8192，不截prompt、不改RoPE/模型配置。无法满足则BLOCK。

在读取新输出/长度/准确率前，用Python random.Random(20261012).sample(range(448),64)冻结64个canonical indices及顺序；不能按旧正确率/截断/类别挑题。Native运行全部64题；其余17cell统一运行该列表前16题，共64+17×16=336条。这样以较小预算覆盖所有策略，避免挑既有赢家；16题组不是精确尾部率估计。先保存样本清单及完整18cell绑定，再运行。

只做label-free完成率/格式/长度/成本分析，不读GPQA gold或旧正确性向量，不计算新准确率。原S1整行规则作为格式成功指标，S0辅助，所有cell使用同一规则。保存新原始输出在远端私有结果目录，不向本地导出原题/正文/候选预测。每条运行到native EOS或8192；报告EOS累计<=2048/4096/8192比例、尚未结束比例、各预算前缀的S0/S1格式成功率（明确这是前缀诊断，不等同独立重跑）、长度分位数、各组n及实际GPUh/吞吐/显存。8192上限仍未结束者为右删失。描述性重复/退化指标可预先定义并报告，禁止以正确率挑长度。

## 工程与资源

允许最小独立extension配置/脚本及必要的max_new_tokens参数化；保护原runtime/wrapper/cache/strict parser和旧输出。不同长度是新科学分支，不复用原2048结束记录冒充本pilot。使用新write-once phase11-gate-d3-*目录，真实Git source checkout与原pinned环境。SSH hpc2-hkustgz及既有项目workspaces范围可读/新staging与runs可写，旧固定复现checkout只读。不得更改SSH信任/DNS系统设置；遇主机密钥异常按hpc2-recover-host-key技能。

本Gate是非正式probe，所有GPU作业必须debug分区且每job<30min（AGENTS最新规则优先）。先同immutable commit/launcher/环境/producer的最小synthetic端到端预检，通过后才采集pilot。至少验证多个实际cached decode步、EOS/8192配置与写出路径；8192尾部资源需直接测量，不能继承2048 packing6。初始每卡1进程，按长序列实测留约20%余量才提高packing；max8GPU是上限非目标，普通用户权限。1GPU资源CPU申请遵守现场8CPU/GPU约束。debug GPU类型按可用与fit选择并明确成本外推限制。若单条8192无法在debug时限内完成，保留失败证据并报告准入BLOCK，不能自行改名formal或投正式分区。

本Gate最多336条成功pilot，debug+pilot+失败实际与预留合计上限16 GPUh（本非正式预检独立边界，不更改用户历史正式任务无总GPUh上限政策）；达到边界回planning，不自行扩样。首次最多2GPU并发，容量及吞吐核定后可增到8；禁止盲cancel/requeue已有任务。低风险同范围工程修复允许保留证据新路径重试，不重复成功记录。

## 交付和后续

最小目标：336条闭合或明确准入阻碍，三档完成率/格式/成本表及长序列资源实测，给出4096/8192候选成本。上限由planning基于label-free结果确定，不由executor自动进入全量。先偏好4096，但不预设其足够；16题干预组小样本不支持精确阈值结论。保持后续完整448×18及S1主/S0辅方案待准入，属于已见数据的后续探索。

人类报告外层资产/报告/phase11，机器证据.planning/phase11；允许targeted测试、单目的commit正常push，禁止模型/原数据入Git。验收只核样本/长度机制、资源及完成率闭合三项。监控按probe单一30min heartbeat，短debug未稳定运行约1分钟则不创建；终态暂停并恢复exact executor继续汇总。允许既有用户授权派发/修复/终态消息；完成发送GATE_D3_FINAL_AUDIT或真实BLOCK到planning，不自行PASS/下一Gate。MMLU-Pro无访问/运行权限。
