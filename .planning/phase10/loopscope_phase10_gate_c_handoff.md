# Phase10 Gate C：MMLU-Lambda-Sweep

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PLANNING_THREAD=01a0fae2-12df-74e0-a874-190518ef0501
EXECUTOR_THREAD=01a0fba5-4ee0-7262-83d4-5bde515f3a94
EXECUTOR_HOST=local
EXECUTOR_TITLE=execute-LoopScope-MMLU-Lambda-Sweep-第10阶段-Gate C
MODEL=gpt-6.1-sol
REASONING=high
MAX_CONCURRENT_SUBAGENTS=3
```

必须实际阅读 /Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md 及 references/protocol.md，不能只读间接引用。依次读AGENTS、PROJECT_MEMORY、planning README、live control、contract_v2、B acceptance/evidence/resources及本handoff。普通操作问题只报planning，不绕过planning向用户停摆。最多同时3个有界subagent，无嵌套；明确文件责任，告知共享代码库不得回滚他人。subagent无独立Git提交、remote-write、GPU/Slurm、terminal或跨Gate权限。父executor整合和调度，唯一heartbeat、终态自唤醒exact executor；唯一GATE_C_FINAL_AUDIT或真正BLOCK须向planning可见送达，失败输出TERMINAL_DELIVERY_UNCONFIRMED。

## 起点与科学范围

Git根为当前外层项目下loopscope-tflt；分支loopscope，起点cc623e5e2cec22a56c6fa211574c9cd8de9e3b94，保护祖先4f59bd93eca4da3cbf458a93508f91c5b23912bc。初始dirty仅planning本次B验收/C handoff/control/plan治理文件，保留并可代提交。

B已PASS。本Gate完成MMLU全test14042、57subjects、plain5shot的68独立配置：旧9复用，新59合计828478新题级记录；86展示行中的18别名不重复运行/检验。三cell、fixed_t0/lag1/current_t含t0、九档lambda，无Matched-norm；模型/数据revision、dtype/cache、alpha、batch1、SVD、prefix/p、continuation全部按contract_v2及当前配置冻结。C不运行ARC正式数据，也不读取ARC target gold/outcome。

## 顺序和交付

1. 最小工程增量：支持FORMAL_TEST显式canonical index子集/分片及整cell合并闭合。先完整冻结59cell和每cell全14042身份；子集仅执行调度单位，不改变科学总体。每记录保留原index/identity，分片无重叠，最终并集恰为全pool，合并恢复canonical顺序。验证缺片、漏题、重复身份拒绝；现有整cell接口兼容。允许Phase10 verifier/manifest增加明确shard执行元数据，不改变科学字段含义，不降低完整cell验收条件。禁止将部分cell标成完整或混用不同配方/关键实现。历史9复用沿B已经核验的原分数和来源，不重做全套B、不读历史analysis。
2. 定向CPU tests、diff --check、受影响CLI help/bash -n。新已提交不可变源码部署后，在debug跑同launcher/env/runtime/producer/shard-verifier端到端预检，每job<30分钟。用无标签validation小集覆盖两模型及K2/K3，验证分片拼接与同输入未分片分数一致、完整性校验正确；延用B已经通过的策略验证，不重复全矩阵。只有此关键路径PASS才允许正式提交。关键路径修复后重新debug。
3. 正式A800可保留首批：batch1、packing1，在7个非重复cell×policy的lambda0.1各执行64题。每模型按完整test的无标签context长度排序（并列canonical index），取rank round(j*(N-1)/63), j=0..63，覆盖完整长度分布及最长样本，再按canonical顺序评分；执行前记录精确index。它们是真实冻结候选的正式分片，合格分数必须纳入最终cell，后续只补余下index。禁止读取gold、accuracy或据分数改参数。记录实际A800型号、loading、显存、时间和每题长度；按7类及全体无标签长度分布估算59cell剩余成本，明确估计不确定性。
4. 初始阶段最多16 GPUh（包含本Gatedebug及正式首批），总Gate硬上限240 GPUh，最多同时2 GPU。首批成功后，若已消耗加剩余保守预测的1.2倍不超过240 GPUh，可由同executor继续全量，无需再次等planning；先记录估计、单job切片/time和预算账。否则向planning报告已完成证据及最小资源补充，不擅自越限，继续安全CPU准备。可提交操作准入消息，Gate终态只发一次。默认保持packing1；本Gate不为追求利用率另开packing探索。分片大小依据实测长尾耗时确定，每job最多24h且满足现场合法限制。提交集合须让已消耗+所有在运行/待运行job的GPU×time上界不越硬预算；采用有界批次，不一次堆积全部长job。无标签计时估算可逐步校正，但不得改变科学集合。
5. 完成59新cell逐题raw评分和旧9引用后，先无标签全量闭合68×14042=954856记录/唯一identity、配置、候选finite分数、正确别名、来源、作业终态。正式canary不能重复计数。写MMLU_FULL_PANEL_CLOSED；此条件满足后本handoff授权一次MMLU gold接入和固定分析，无需新用户确认。之前禁止partial accuracy、gold/correctness、旧结果analysis。
6. 按contract_v2做68配置完整表、63×2扫描Holm family126、策略比较45项family，exact双侧McNemar、10000次subject内paired bootstrap seed20261002、nominal95%CI。7个非重复cell×policy各取MMLU microacc最高lambda，并列取较小值，写固定transfer-selection（身份/强度/规则/来源），供D使用。MMLU是开发选参，不宣称无偏泛化；不作方向特异性结论。合法负结果照常交付，不因不显著重跑/选新参。此Gate结束，D须新executor。

## 路径、调度与权限

本地仅Phase10模块/脚本/launcher/tests/docs及Gate C证据文件可写；治理control/contract/handoff只读（planning初始dirty可原样代提交）。可正常commit/push origin/loopscope，禁止force/rebase/reset或操作main。Phase8/9和protected wrapper/strategies/cache/config/eval_runner只读。外层资产/报告/phase10可放C结果表/说明，机器证据在新remote run，不镜像人类报告进Git。

HPC alias hpc2-hkustgz；先读hpc2-hkustgz-ssh skill，strict host key、BatchMode、ConnectTimeout、ClearAllForwardings=yes、UpdateHostKeys=no。旧source /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope 保持只读phase9-gate-c-20260922/a2c9e58a32163e1ed1527d3ab57798bc7378f486，既有 .venv-loopscope-cu121-20260711/bin/python、模型/cache不安装升级或下载。Slurm现场路径/权限直接核对（B为/opt/slurm/bin）。

可写workspace /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope 下 inputs/phase10-gate-c-<timestamp>、staging/phase10-gate-c-<timestamp>/source-<commit>、runs/phase10-gate-c-<timestamp>及对应logs。以已提交Git archive部署完整源码，immutable source不原地修。原MMLU canonical pool inputs/phase8-gate-d-20260905T142900Z-a1/test_pool.json，以及B的validation pool与历史reuse证据允许只读；精确原prompt来源沿当前adapter/配置读取并记录。gold只在步骤5全量闭合后按既有原revision映射接入。旧inputs/runs不覆盖、不移动、不删除；不新增摘要值。

用户最新调度硬规则：所有非正式任务（debug/preflight/smoke/probe/非正式canary）仅debug，每job00:29:00上限。所有正式任务使用普通用户合法最高优先级A800分区/QoS；B现场为emergency_gpu，提交前重新核对组/账户/分区/QoS资格与优先级，记录实际选择。debug不适配时报告准入问题，不能改投正式分区。正式canary是冻结59cell中的可保留数据，不能换名运行工程检查。预算240 GPUh含失败/重试、最多2 GPU，首批16 GPUh边界如上。

授权有进展的低风险工程修复和新attempt retry，保留失败证据，不重复已闭合分片。可取消自己确认错误/失败/重复待运行的C作业，不操作他人/Gate jobs，不为队列速度反复取消重投。硬件/调度失败可同配方新attempt续缺失分片；若部分输出不能可信闭合则保留原attempt、最小范围重跑并确定唯一canonical版本。科学/标签隔离/破坏性问题或预算扩张报planning。

## 监控、验收和送达

每次实际sbatch确认job ID后，若需跨turn等待，创建/更新executor唯一full heartbeat每60分钟，含PENDING，不人工轮询/长sleep。debug用10分钟且稳定RUNNING约60秒后才建立；短任务已结束无需建。单一monitor只读观察绑定job集合，无变化静默；terminal先停monitor，然后向exact executor发送AUTOMATION_TERMINAL_RESUME并保留可见送达；无支持则向planning发AUTOMATION_RELAY_REQUIRED。恢复后由executor闭合分片/继续预算内批次/诊断重试，monitor不代调度或判Gate。用户授权涵盖这些必要协作消息。

must-pass：新分片关键路径debug成功；合法A800全59新cell+9历史无标签闭合；仅闭合后MMLU一次分析；统计family/别名正确；7个lambda*固定且无ARC outcome；资源/源码/模型/数据来源明确，所有C jobs/monitor收尾。planning仅查三个决定性入口（source/预检，完整panel，analysis/selection），阶段E再总审。通常最多一次合并审计退修，非材料缺口不阻塞。

完成向planning 01a0fae2-12df-74e0-a874-190518ef0501 local可见发送唯一GATE_C_FINAL_AUDIT，请求PASS/PASS_WITH_FIXES/BLOCK，列授权和最终commit/dirty、改动、精确测试/exit、debug/formal jobs/root、数量闭合、预算、7迁移参数及未做事项。真正无法推进才BLOCK。不得自判PASS、进入D或创建D线程。交付确认失败必须输出TERMINAL_DELIVERY_UNCONFIRMED完整包。
