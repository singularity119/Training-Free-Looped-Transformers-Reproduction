# Gate G：4B 15–18窗口完整实验组

必须按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，实际读取skill及references/protocol.md。
Planning=01a08024-3556-75c1-8209-64e89345b940。
Executor=01a08443-1e4f-7623-aa1b-69d3dbae406e。
Title=execute-LoopScope-4B-Window15-18-第8阶段-Gate G。
本地专用loopscope-tflt clone，branch loopscope，基点 46d71b07c2f56162aec989792639672ad4780ec9；保护祖先4f59bd93eca4da3cbf458a93508f91c5b23912bc。mutation前检查path/branch/HEAD/dirty/ancestor，保留他人编辑，禁止main/force/rebase/reset。

## Admission和目标
用户2026-09-09明确新增该窗口并要求立即独立执行。原B–E已验收，F完成分析且planning直接核验过36cell与7job终态，F正式终态交付仍独立收尾。G不依赖F outcome进行选择，无需等待F行政闭合；只有G获新实施/模型/调度权，F仅原终态交付。G负责检验15–18窗口下原Euler与两种谱方向拟合的准确率，不扩大到其他模型窗口、不启动Phase9。已知旧结果后的探索性追加，不称前瞻确认。

## 冻结科学
Qwen/Qwen3-4B-Base revision906bfd4b4dc7f14ee4320094d8b41684abff8539；BF16/cache first；inclusive零基层15:18共四层，残差边界B15→B19。K=2,3,4，alpha=1,h=1/K,total horizon1,beta0,block damped_euler,batch1,decode bypass。三arms=Loop、Spectral-fit-t1、Spectral-fit-t0，9新cell，每cell同一MMLU test14042/57subjects，共126378新评分。
其余模型/tokenizer/data revisions、lm_eval0.4.11 plain5shot、dev first_n5 demonstrations、完整A/B/C/D continuation loglikelihood，与contract_v1和F路径一致。cais/mmlu revisionc30699e8356da336a370243923dbaf21066bb9fe。无训练/聊天模板/CoT/生成/量化/参数搜索。
原固定validation512身份顺序/seed20260905直接复用C pool，不重抽样、不读目标gold。新窗口必须新采无干预K2/3/4轨迹；每K取t1拟合3份方向，从K2轨迹t0额外拟合1份跨K2/3/4共享方向，总4basis，不重复拟合t0。512×d答案前位置native残差→FP32→CPUfloat64 reduced SVD，不中心化/行归一化，top1 signpivotpositive，FP32unit。rank1/lambda0.5。不得拿12:15或13:16方向代替新窗。
干预仍仅t>=1的pre-answer残差，FP32投影减法后cast原dtype；t0及其他token不干预。t0元数据明确source_k2/applies_to_k[2,3,4]。K变化不改alpha。
Native复用D实测10262/14042=73.0808%，仅表格上下文，不重复跑、不用旧acc代替新Loop。报告表9cell完整correct/N/acc，按3行K展示Native/Loop/t1/t0及差值。

## 统计和信息
9新cell全部身份唯一完整/finite/config闭合后，允许一次gold读取和统一分析，禁止partialacc/outcome调参重跑。比较t1−Loop、t0−Loop、t0−t1，每类3项K构成单独探索Holm family，三family明确列出；不并入/重写旧D/E/F family。microacc、correct/N、pp、错对翻转、subjectmacro、学科内配对bootstrap10000次seed20260905 nominal95%CI，exact双侧McNemar及Holm。Native差值如列只描述，不宣称显著；多family分别控制而非9项统一错误率，报告此范围。不挑最佳K后宣称普遍收益。

## 实现/路径/权限
立即授权最小新增G专属phase8模块、scripts/config/tests；必要的既有phase8辅助模块小改以显式新窗参数/opt-in实现，保持旧配置默认不变。不能直接扩改旧冻结model_windows/manifest冒充历史合同；G独立配置可承载新窗。protected wrapper/strategies/cache/config只读，必要改动先报planning最小原因/diff。
允许单目的普通commit/push origin loopscope；不上传模型数据basis原始分数/大日志。已有GitHub连接/认证问题只记录，不改key/凭据/网络绕过；producer可经已授权严格SSH将本地已提交版本放到G固定源码快照，记录Git版本/路径，不因文档push失败重跑实验。
HPC alias hpc2-hkustgz，实际读取hpc2 skill，严格hostkey/BatchMode/ClearAllForwardings。source专用clone=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope，只读原source以避免旧owner漂移；venv其下.venv-loopscope-cu121-20260711，shared hf_home/datasets只读，不改依赖/revisions。W=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope。允许W/{inputs,runs,artifacts,staging}/phase8-gate-g-<timestamp>-<purpose>-a<N>新工件和固定提交源码快照，write-once。只保存必要答案位置残差，不全token hidden。旧E/F/C/D只读，不接管jobs/monitor。
同Gate低风险工程/launcher/resource修复次数不限但新路径只补无效缺失工作，保留失败，不能改科学/越预算/解封前越界。planning集中返修最多一次，第二次仅同材料根因或回归。最多3 bounded子agent，不嵌套/不调度，executor负责Git/Slurm/终态，不回退他人编辑。

## GPU与最早实验
所有debug/smoke用debug partition，<30min（29min上限），同immutable commit/launcher/env/runtime端到端通过才能正式。覆盖新窗K2/3/4轨迹、4basis保存加载、实际t0跨K一致性、t0不干预及后续干预、三arm评分/序列上尾/写出/verifier；微样本basis标PREFLIGHT，不作正式方向。
正式A800普通用户合法最高优先级，单job<=6h，每GPU8CPU128G起始，同时<=2GPU，累计<=24GPUh含debug/retry。packing1–3依据实际本窗上尾峰值/吞吐，至少15%瞬态余量，不改batch/dtype/共享tensor。先保留各cell前512 canary验证，再扩余量；已知参考峰值不替代新资源形状检查。不因PENDING取消重提，允许仅本G失败/超预算/OOM风险job取消后原范围新路径修复。

## Agility budget / 验收
只实现新窗配置准入、4basis/9cell/统计及报告，不重构框架。确定性检查git diff --check、针对G unittest（PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase8_gate_g*.py'，0tests不算通过）、新launcher bash -n、CLIhelp；重点新窗边界/4basis映射/干预时机/三项family。命令和exit写一份G evidence，关键检查通过即debug/实验，不重复A–F全套。
planning只接受3项：新窗配方和4方向来源；9×14042闭合及统计；终态jobs/资源和报告结论。阶段末复用历史验收做一次简洁综合结案，不重算全阶段。

## monitor与终态
独立唯一G executor-owned heartbeat；debug实际稳定RUNNING约1分钟后smoke10min（短job无需），formal真实jobID后立即启用含PENDING，calibration probe30min/full60min，转换类型更新同一个。健康检查安静；attempt终态先暂停，再send_message_to_thread主动唤醒exact G executor，AUTOMATION_TERMINAL_RESUME包含job/root/证据/下一授权验证修复动作。无法确认给planning发AUTOMATION_RELAY_REQUIRED，不能只说ready。未知连接状态保留原job先查，不重提。
报告外层资产/报告/phase8/gate_g/，独立中文导师易读报告、cells/contrasts.csv、analysis.json；不改Notion，由planning验收后更新。完成暂停monitor，唯一GATE_G_FINAL_AUDIT或真正BLOCK送planning01a08024-3556-75c1-8209-64e89345b940：身份/起止Git/dirty/push、files/命令exit/jobs/roots/4basis/9cell/统计资源偏差、申请PASS/PASS_WITH_FIXES/BLOCK，保留成功送达响应；无确认明确TERMINAL_DELIVERY_UNCONFIRMED，不自PASS、不进入下一Gate。
