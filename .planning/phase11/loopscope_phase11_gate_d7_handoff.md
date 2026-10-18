# Gate D-7 — MMLU-Pro 最优配置与基线

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
GATE=D-7
PLANNING_THREAD=01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f
PLANNING_HOST=local
EXECUTOR_THREAD=01a142a7-562a-7e83-b085-70999fbbc780
EXECUTOR_HOST=local
CANONICAL_TITLE=Gate D-7-execute-LoopScope-MMLUPro-BestConfig-第11阶段
MODEL=gpt-6.1-sol
REASONING=high
```

用户2026-10-16明确推进MMLU-Pro，只做前述最优配置及Native/普通Loop，普通Loop最后；随后明确采用8192＋S2。此授权解除本Gate五配置范围MMLU-Pro暂停，不重开D6或旧18臂矩阵。实际读取research-gate-orchestrator SKILL.md及references/protocol.md、AGENTS与control。问题/普通操作权限只报告exact planning，不绕过planning向用户停摆；唯一executor负责实现/Git/调度/终态，planning裁决。允许最多3个有界subagent仅做本Gate窄实现/只读检查，无独立Git整合、remote-write、GPU/Slurm、terminal权限、不再委派。终态为唯一GATE_D7_FINAL_AUDIT或真正BLOCK，向planning投递成功才算交付，否则完整TERMINAL_DELIVERY_UNCONFIRMED。唯一heartbeat的pause+self-wake路由见下。

## 科学面板与用户指定执行顺序

模型Qwen/Qwen3-4B-Instruct-2507，model/tokenizer revision cdbee75f17c01a7cc42f958dc650907174af0554；TIGER-Lab/MMLU-Pro revision b189ec765aa7ed75c8acfea42df31fdae71f97be，test完整12032题、14类别。保留Gate A canonical题目/选项顺序与5个同类别validation CoT示例及顺序；示例来自A已绑定的每类别first_n五条，不能重抽样。读入目标仅question_id/question/options/category/src，target cot_content/答案不进入producer。允许原冻结示例答案。

| 批次 | 配置 | 记录数 |
|---|---|---:|
| 第一批 | Native无Loop；current_t K2 λ0.5；fixed_t0 K3 λ0.9 | 3×12032=36096 |
| 最后一批 | 普通Loop K2；普通Loop K3 | 2×12032=24064 |

合计5cells、60160条fresh生成。第一批三配置可并行；它们全部raw身份/完整性闭合后，才提交/运行最后两组普通Loop正式生成（包含其正式canary），不用调度优先级冒充顺序。不在第一批完成后先看准确率。最后两组可并行；全部五组完成后一次分析。synthetic工程检查可覆盖普通Loop，但不计正式题、不提前采集其test输出。

方法仅GPQA挑出的两套、全部prefill＋decode衰减；不加prefill-only、direct-answer或lambda sweep，也不在MMLU-Pro重新挑最优。所有Loop inclusive15:18、block damped_euler、alpha1 beta0、步长1/K、cachefirst、decodefull、batch1 BF16、无训练。各题自身prefill轨迹FP32 exact reduced SVD rank1拟合，decode固定复用、decode_fit_count0；Native不装Loop。完整原生chat template、单user及现有5-shot CoT prompt，保留Reason step by step指令、答案格式与示例归一化，不改system/角色/stop。greedy/do_samplefalse、native EOS、max_new_tokens8192，无额外字符串stop/答案补救/截prompt或shot。最大prompt2860加8192在既有context262144内，仍核对实际tokenization，无静默截断。统一runtime seed20261016，Python/NumPy/torch/CUDA在模型/计算开始前实际生效；输入/演示构建不被运行seed改变。

旧Gate C的2048小样本只作只读来源/资源参考，不能混入本次8192 fresh面板或据其预测选样。全部5cells均重新前向；共享只读权重/输入cache，独立进程/输出/方向状态。

## S2-MMLU评分适配与信息屏障

用户选S2语义：整行Final answer语法、大小写归一、括号可选、沿用S1标点容忍；多行均为同一合法字母时接受，不同字母冲突或无匹配计错，不从正文找答案。现有phase11_s2.extract_s2/S1_PATTERN硬编码A-D，禁止原样用于MMLU-Pro。新增MMLU专属opt-in parser，不修改GPQA历史函数默认/旧结果；使用同语法识别A-Z单字母，依据每题实际options数量映射A…最后合法选项，不能假设每题恒十项。冻结：任何符合答案行语法但超出本题合法范围的字母，标记out_of_range并提取失败；合法范围内多行不一致为conflict；一致则接受；无答案行为no_match。分别报告这些类别，均保持完整12032分母。此为S2语义对多选项的预先适配，不与GPQA四选项逐字parser声称相同。

在新正式生成/任何新outcome之前，保存parser版本与清晰规则，并测试A/D/E/J、不同选项数、超范围、大小写/括号/标点、多条一致/冲突/无匹配；只解析新生成文本，不能把输入few-shot中的答案行带进提取。答案字母→index按本题选项数，gold mapping不重排。debug与首批只读运行/长度/identity，不读gold或accuracy；全部60160 raw及保存S2预测/身份/无缺重闭合后，才一次读取原封存gold统一分析。不可因前3组结果调整普通Loop或停止；不得中途看分数/正确数/翻转。

## 预先冻结的分析

主指标为完整12032题micro accuracy（含提取失败计错）；附14类别每类n/accuracy与category macro描述，不能按结果换主指标。主family4项：两方法各对Native、同K普通Loop，exact双侧McNemar＋Holm alpha.05。基础family2项：Loop K2/K3各对Native，独立Holm。类别内分层paired bootstrap10000、seed20261008，主CI为按原类别样本量加权micro差值的名义95%CI。不把每类别探索差异当未校正显著结论，不做事后选最佳类别。报告5行accuracy/正确数、deltaNative/deltaLoop及CI/p，输出EOS/length-limit、S2各类别、token均值/中位数/p90、walltime/GPUh/峰值和packing。

方法由GPQA事后选定，本次是固定配置在另一任务上的迁移评测；不称MMLU-Pro任务内最优、不把曾用过该数据的项目历史说成完全未见独立验证、无Matched-norm不证明方向特异性。负结果/格式失败/截断如实保留，不设必须涨点的工程PASS条件。

## 实施授权与资源

本地nested loopscope-tflt branch loopscope，派发HEAD4da2a007e07addda7cc0dedd27f8e9ee14e781e4；保留planning README/control/D3/D4 handoff与D4/D5/D6验收、观察、派发等dirty，禁止stage他人文件。正常commit/push origin/loopscope，不force/reset/rebase/main。最小D7 plan/launcher/producer/parser/analysis及定向tests/docs；允许Phase11专属脚本/模块的兼容opt-in改动用于MMLU路由、runtime seed、8192与输出恢复。protected共用wrapper/strategies/cache/config/eval_runner只读；需要材料变更先报planning。原输入/历史run/source/shared caches只读，不升级依赖/下载模型。不要通用框架、digest、防御测试扩张或重新审计旧Gate。

HPC2访问先读hpc2-hkustgz-ssh，信任异常走hpc2-recover-host-key；不改VPN/hosts/路由/SSH信任。host hpc2-hkustgz。原source/venv只读 /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope；可写workspace /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope 的 inputs/staging/runs/artifacts/logs 下 phase11-gate-d7-<timestamp>新roots，部署已提交immutable source。原A输入与sealed gold按信息权限读。SSH失联保留jobs，恢复先检查原ID，不能盲重投。

正式总最多8 A800、每GPU<=8CPU且服从更紧现场配额；普通用户合法最高优先级A800 partition/QoS（当前预计emergency_gpu，必须核对）。少量长任务与合法最大TimeLimit，沿用阶段无累计GPUh硬上限，但预估/实际/失败成本必须记录。本面板60160条，比GPQA大得多；旧MMLU小样本资源受4题长度选择和2048删失影响，只能作粗略参考。首次retainable正式小批完成后，用真实新路径耗时/长度/packing报告总GPUh及8卡墙钟范围，不把GPQA估计直接当MMLU实测。若材料性不可行/资源异常先报planning，不改变长度/题数/方法。

所有工程debug/smoke严格debug分区<30min（项目用户规则优先skill A800例外）；正式前同immutable关键路径的端到端debug通过，验证真实模型/5shot样式synthetic输入/CPU parser/存盘/校验/loop方向与cache生命周期、种子有效。修改launcher/producer后做最小针对性检查，不重跑全套历史测试。不是额外长度pilot、不调8192；debug无法准入报planning，不能改名formal绕过。

随后第一批3cells各取4个只依prompt长度分位q=.25/.5/.75/1选定的canonical身份（排序按token长度、同长canonical index，floor((N-1)*q)），共12条可保留正式首批；保存membership后运行，不据输出选择。以真实5shot8192下的峰值和吞吐支持packing（不要从GPQA直接外推5），保留约20%设备显存余量、batch1及独立模型状态。首批成功计入各12032，不重跑；资源闭合后推进第一批剩余，最后再Loop两组。最长prompt提前EOS不当作8192最坏峰值证明，结合KV上界/真实设备证据保守定packing；不加独立调参pilot。逐题持久checkpoint，成功不覆盖；恢复只跑缺失/无效项、新路径保留失败。允许同科学范围低风险launcher/env/序列化/scheduler修复；只可取消本executor自己确证错误/重复/失控D7 exact jobs，不因排队盲取消，不动其他Gate。

## 监控、交付及停止

debug真实jobID后允许唯一10min smoke heartbeat从PENDING开始；正式真实ID后更新同一monitor为full60min（含PENDING）。健康无变化静默；terminal attempt先pause，再向exact executor发送AUTOMATION_TERMINAL_RESUME并保留工具确认；失联/失败恢复本executor诊断，不能自动判Gate BLOCK；投递失败向planning发AUTOMATION_RELAY_REQUIRED。用户standing授权本Gate派发/询问/修复/terminal/self-wake路由。planning不重复轮询。第一批完成的self-wake必须推动普通Loop最后批，而非把第一批当Gate结束。

终态唯一GATE_D7_FINAL_AUDIT，附5cells/60160闭合、第一批→Loop顺序证据、原输入/8192/实际seed/parserfreeze/方向生命周期、gold时间屏障、6项配对统计、资源与失败记录、source/命令/tests/jobs/roots、monitor PAUSED与投递确认。聚合中文报告到外层资产/报告/phase11/gate-d7-mmlupro-<date>，机器证据.planning/phase11；原题/生成/逐题预测/gold留远端不上Git不下载Mac。executor不自行PASS；planning核对1–3决定性事实，材料问题一次合并退修。完成后等待用户讨论，不启动新实验或下一Gate。

## 2026-10-16 MMLU-Pro gold来源澄清与隔离adapter许可

planning核对phase11_data.py：原Gate A仅GPQA构建sealed sidecar，MMLU-Pro原输入路径没有由该构建器生成的gold sidecar。因此覆盖“原封存gold”这一不准确路径假设；不要求搜寻或提前打开历史gold。允许executor现在实现并用合成数据测试CPU-only adapter，但只有全部5cells/60160 raw＋S2预测及身份闭合、记录GOLD_UNREAD后，才允许实际运行adapter读取目标答案。
唯一原始答案来源为 /hpc2hdd/home/xhuang225/shared/hf_home/hub/datasets--TIGER-Lab--MMLU-Pro/snapshots/b189ec765aa7ed75c8acfea42df31fdae71f97be/data/test-00000-of-00001.parquet 。仅投影question_id/answer_index/category，原parquet行顺序保留source row index；按原A规则重建 TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be:test:{row_index}:{question_id}，与已冻结canonical pool的全部12032 identities一一join并核对category、无缺重、answer_index在该题choice_labels范围内，不重排选项。实际schema可先只读元数据；字段不一致报planning，不静默取别的gold字段或revision。
在D7新artifacts root的sealed子目录生成write-once JSONL，字段identity/label_index/category，沿用现有分析入口。保存首次答案读取前时间与完整closure引用、读取后receipt、源revision/path/投影列/总数/join结果；该adapter即本次首次gold读取，不能只记录随后JSONL读取时间。只在远端进行，不打印或下载逐题gold；失败保留证据、同来源低风险CPU修复仍在原最终分析范围，不调parser。该许可不改变评分、统计、GPU顺序或任何采集权限。

## 2026-10-16用户强调：沿用Loop窗口KV cache处理

用户明确要求loop window的KV cache处理逻辑也沿用。D7必须沿用D6/Phase11已验证的同一cache_strategy=first实现：窗口内K轮不持久累计临时KV；各轮读取对应历史KV、临时计算后回退，完成Loop后以进入窗口前的原始表示额外执行窗口并stash该遍各层KV，K轮最终表示传递到后续层。decode对当前token只持久追加一次，不能把K轮各自KV都写入、切换last或以全序列重放代替增量路径。prefill同样沿用原first stash语义。prefill方向bank与KV cache分别管理，方向在decode冻结不重拟合。
这不是新增实现或重跑要求：现有debug若已直接验证body_call_count、stash_pass_count=1、stash_cache_strategy=first、decode缓存单次增长与相同wrapper源码，则引用该决定性证据即可。不得为“沿用”重写共用wrapper/cache或额外跑已通过的完整套件。若发现实际路径漂移，停止正式准入并向planning报告材料差异，保留现有job/证据。

## 2026-10-18用户授权两组已完成结果的单次提前评分

用户明确“已完成的两个拉一下acc”：授权exact D7 executor核实Native与current_t K2 λ0.5各完整12032条后，建立独立write-once两cell snapshot，先固定全部24064条S2预测/identity与完整性，再按既定gold adapter从原revision投影答案、exact join，计算这两组accuracy/正确数及相对Native差值。若任一未完整不得冒充全量。此为原全5cell gold屏障的限定单次用户例外，不授权看K3或Loop局部分数，不改变其运行顺序/题集/parser/参数，不按结果停止/调整后续。先前gold未读事实保留，首次访问必须如实标注为本次中途snapshot；最终全5cell分析不得再声称gold首次读取。原raw及正式closure均不覆盖，snapshot用D7新interim-<timestamp>路径。主请求为两行acc，无需新增显著性检验或GPU。其余采集继续原计划，向planning回报聚合结果及快照/评分时序证据，非Gate终态。

## 2026-10-18用户追加K3完成后提前评分

用户要求确认K3是否跑完，若完成也拉acc与前两组比较。授权exact D7 executor一次有界核查fixed_t0 K3 λ0.9的完整12032条（含canary）；仅完整后先独立write-once封存全组S2预测与identity，再以此前已授权读取、相同revision/identity绑定的gold评分。新K3快照不覆盖原两组或正式raw/closure；只新增K3正确数/acc/格式失败及相对Native描述差值，不看普通Loop outcome、不调parser、不重跑GPU/新增显著性检验。若未完整仅报完成量，不算部分acc。最终报告须披露两组与K3提前评分时序，不声称最终才首次gold；采集仍按原计划，第一批raw闭合后推进普通Loop。
