# Gate D-6 — GPQA CoT 与 decode 衰减对照

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
GATE=D-6
PLANNING_THREAD=01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f
PLANNING_HOST=local
EXECUTOR_THREAD=01a13d93-a681-7d20-ab05-434dc91e1097
EXECUTOR_HOST=local
CANONICAL_TITLE=Gate D-6-execute-LoopScope-CoT-Decode-Ablation-第11阶段
MODEL=gpt-6.1-sol
REASONING=high
```

2026-10-15用户授权两组对照并交给新独立执行线程；方法仅前述两套最优配置。本handoff是D6科学与权限合同，覆盖D5关闭后等待条款。问题与普通工程权限直接报告planning，不绕过planning向用户停摆。实际读取research-gate-orchestrator及references/protocol.md；executor实现/验证/调度/交付，planning裁决。可使用最多3个有界subagent，但它们无Git整合、remote-write、GPU/Slurm或terminal权限，不再委派。唯一60min heartbeat及终态self-wake见末节。唯一GATE_D6_FINAL_AUDIT或真正BLOCK必须投递exact planning并有工具确认；否则输出TERMINAL_DELIVERY_UNCONFIRMED。

## 冻结科学设计

同一Qwen/Qwen3-4B-Instruct-2507（沿用D5 exact revision/tokenizer/env）、GPQA Main作者源56686c06f5e19865c153de0fdb11be3890014df7的448题，原canonical顺序和选项排列/input seed20261008不变；0-shot、原生chat template、BF16、batch1、greedy、native EOS、max_new_tokens8192。全部12cells fresh，不复用D4/D5预测；统一实际运行RNG seed20261015（Python/NumPy/torch/CUDA均生效，不重建输入排列），bootstrap seed20261008。新prompt仅改变提示指令，不改变题目/选项及原system/user角色结构；不重新选窗/选lambda。

| 提示 | 配置 | 衰减作用阶段 | cells |
|---|---|---|---:|
| 原CoT | Native、普通Loop K2、普通Loop K3 | 无衰减 | 3 |
| 原CoT | current_t K2 λ0.5、fixed_t0 K3 λ0.9 | prefill+decode | 2 |
| 直接作答 | 同上五配置 | 方法为prefill+decode，对照无衰减 | 5 |
| 原CoT | current_t K2 λ0.5、fixed_t0 K3 λ0.9 | 仅prefill | 2 |

合计12×448=5376条；每个提示内Native共享，两组对照共享CoT full两方法，不重复运行。每cell新进程/方向状态/独立输出，不共享可变模型状态。原CoT逐字保留D5 frozen rendered inputs，其指令为：
`Reason step by step, then end your response with exactly one line in the form Final answer: (X), where X is the letter of the correct option.`
直接作答仅将该指令替换为：
`Do not provide reasoning or explanation. Respond with exactly one line in the form Final answer: (X), where X is the letter of the correct option.`
在读新outcome前保存两份可读prompt配方和输入身份映射。若历史指令与上述源码不闭合，向planning报告exact差异，不能覆盖历史。直接作答仍有8192同上限，不额外early stop/强制单token或特殊decoder；模型不遵循直接作答也按原样计入，并报告格式和长度，不宣称成功关闭内部推理。

所有Loop保持inclusive15:18、block damped_euler、alpha1 beta0、cachefirst、decode_mode=full；方法沿用FP32 exact reduced SVD rank1、自身prefill轨迹拟合、decode不拟合。**prefill-only仅禁止decode residual damping，不能用decode_mode=bypass关闭普通Loop**。两种scope在prefill完全相同；prefill-only decode每个t仍执行同K普通Loop/first-KV，仅残差不投影衰减。prefill首个输出token logits可以受衰减，必须在报告明确。

## 分析先冻结，完整后统一评分

沿用D4/D5 S2整行提取语法、重复一致归并；冲突/无匹配计错，不修改parser、不语义救答案。全部5376 raw及S2预测/identity/配置闭合后才读取gold统一分析；不看局部分数决定续跑。不假称历史未见gold。冻结的最优配置来自同一GPQA的事后选择，本Gate也是探索性机制检验，不是新题集确认。

主分析4项（一个Holm family，alpha.05）：
1–2. 每个K分别检验交互 D=[Acc(method_full,CoT)−Acc(Loop,CoT)]−[Acc(method_full,direct)−Acc(Loop,direct)]。逐题d_i由同一题四臂正确性差计算，报告pp点估计、题级paired bootstrap10000名义95%CI；双侧paired sign-flip randomization test，100000 draws、seed20261008，p=(1+尾部计数)/(100001)。不把“一边显著另一边不显著”当交互证据；sign-flip为可交换性假设下的探索性检验，说明限制。
3–4. CoT下每个方法full−prefill-only的逐题配对正确性差，exact双侧McNemar及同题bootstrap CI。与前两项合为4项Holm校正。
辅助family：两提示×两方法×[Native,同KLoop]=8个full方法比较，外加CoT两prefill-only方法各对同KLoop=2，共10项exact McNemar+Holm；普通Loop−Native及其他跨prompt变化仅描述性，不能挑选显著检验。报告12行正确数/accuracy，所有delta单位pp，名义CI和调整p邻列。不得将各臂或历史重复视作独立题集、不能扩大样本量。
长度/行为同时报告：全448题生成token均值/中位数/p90、EOS与length-limit计数、S2成功/无匹配/冲突、各方法相对同KLoop的配对长度差及CI；EOS-only均值作为条件性描述，不能当统一分母效率结论。记录walltime/GPUh/packing/peak memory；不按格式成功子集替换主accuracy。负结果也可工程PASS。

## 实施、授权和验证

仓库loopscope-tflt，branch loopscope；派发HEAD0e9b34cd490a2cca16f0e19406e49aac5e617b31，保留现有planning README/control/D3/D4 handoff、D4/D5验收和观察证据dirty。仅stage自身变更，正常commit/push origin/loopscope；不force/reset/rebase/main。原source/历史results/输入只读。
允许最小D6 plan/launcher/producer/analysis及定向tests/docs增量；允许phase11_data/accuracy/adapter/runtime/checkpoint/panel的向后兼容opt-in prompt和damping-scope参数，默认保持历史full行为。最小runtime改动可在__call__的applied条件加scope判断并记录真实applied与scope，prefill拟合与callback/KV次序不改。wrapper/strategies/cache/config/eval_runner等共用核心protected路径只读，确需变动先报告planning。不得改历史S2评分或输入构建默认。不要造通用框架，不新增digest、长度pilot、其他lambda/K/模型/任务。
定向验证：prompt差异只在指令且题目/选项身份一致；full默认结果路径不变；两scope prefill相同；prefill-only decode无damping、仍同K Loop及cachefirst、decode_fit_count0；实际runtime seed生效；12cells身份无缺重和S2规则沿用。CPU单元/最小真实路径验证足够后前进，不重审旧Gate。必要同immutable commit/launcher/env/model/producer/verifier端到端debug <30min，非正式debug全部debug分区（项目用户规则优先skill A800例外）；不额外跑长度预检。工程debug仅synthetic/现有gold-free工程输入，不读新accuracy，不能将失败测试改名formal绕过。

HPC2先读hpc2-hkustgz-ssh skill；读source/env/cache/input/scheduler获授权。主机hpc2-hkustgz，原只读source及venv=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope，workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope。可写workspace下inputs/staging/runs/artifacts/phase11-gate-d6-<timestamp>新roots，immutable已提交source staging；不升级依赖/下载模型、不改VPN/hosts/路由/SSH信任。hostkey异常走恢复skill。连接失效保留jobs，恢复后先核对原ID。
正式总max8 A800、每GPU最多8CPU且尊重现场更紧限制；用普通用户合法最高优先级A800分区/QoS，现场验证资格，预计emergency_gpu。无累计GPUh硬上限，记录预计/实际/失败成本；使用少量长任务、合法最大TimeLimit。先有效debug，再最小可保留正式首批验证新scope资源；沿用相同GPQA8192已有packing5证据但检查新增scope峰值，不盲外推，保持实际余量。正式首批计入5376不重复；吞吐稳定后铺剩余。每题checkpoint/恢复，成功不覆盖不重跑。允许同科学范围低风险修复、仅缺失/无效项fresh retries；只可取消本executor自身确证错误/重复/失控D6 exact jobs，不为排队盲取消、不碰其他Gate。MMLU-Pro、ARC、Base/MMLU未授权。

## 交付、验收和停止点

初次正式获得真实job IDs后立即启用唯一executor-owned full60min heartbeat（含PENDING）；健康无变化静默。terminal attempt先pause，再send_message_to_thread唤醒exact executor并保留确认；失败也恢复同executor诊断/授权内修复，不能自动等同Gate BLOCK；无法投递则向planning发AUTOMATION_RELAY_REQUIRED。用户授权派发、修复、terminal及monitor self-wake消息路由。planning不轮询/不代跑。
完成向planning发送唯一GATE_D6_FINAL_AUDIT，含源码commit、定向tests/退出码、jobs/roots、12cells/5376闭合、prompt与scope实证、结果/统计/长度、资源成本、偏离、monitor暂停证据。原题/输出/gold保留远端，不上Git/不下载Mac；聚合中文报告到外层资产/报告/phase11/gate-d6-cot-decode-<date>，机器证据到.planning/phase11。工程问题可自主低风险修复，无进展或科学/权限材料边界报planning BLOCK。planning只用1–3决定性事实轻量验收，验收不要求正收益；一次合并材料退修为默认。D6完成后等待用户讨论，不进入下一Gate或MMLU-Pro。

## 2026-10-15 debug排队监控补充

planning批准exact executor为已提交debug job13073061_0建立唯一10min smoke heartbeat，允许从PENDING(MaxJobsPerAccount)开始，不必等RUNNING稳定60秒。只读观察原job，不因排队取消/重投或改资源；RUNNING后沿用同一monitor，正式任务时更新为full60min。terminal attempt先pause再self-wake exact executor并确认投递，无法确认则relay planning。健康/排队无材料变化静默，其余科学、调度及权限不变。这是本Gate对debug monitor启动时机的限定例外。
