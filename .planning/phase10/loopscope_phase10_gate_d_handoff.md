# Phase10 Gate D：ARC-Transfer-Sweep

```text
COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
PLANNING_THREAD=01a0fae2-12df-74e0-a874-190518ef0501
EXECUTOR_THREAD=01a111f8-7ba3-7862-b0d6-78c0a5a2f767
EXECUTOR_HOST=local
EXECUTOR_TITLE=execute-LoopScope-ARC-Transfer-Sweep-第10阶段-Gate D
MODEL=gpt-6.1-sol
REASONING=high
MAX_CONCURRENT_SUBAGENTS=3
SCHEDULED_MONITOR=DISABLED_BY_USER
```

必须实际阅读 /Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md 及references/protocol.md，再依次AGENTS、PROJECT_MEMORY、planning README、live control、contract_v2、A数据binding/验收、B预检证据、C验收和本handoff。用户明确暂不定时监控，覆盖skill默认heartbeat；不要恢复旧monitor或新建替代。普通操作权限/问题只报planning；工具发送被审批策略阻止时保存完整TERMINAL_DELIVERY_UNCONFIRMED，不绕过审批。父executor整合/Git/远端/调度/唯一终态；最多同时3个有界subagent，无嵌套、无独立Git/remote-write/GPU/Slurm/terminal权限，明确文件所有权与共享代码库不回滚他人。

## 目标与冻结合同

C已PASS。只执行D：ARC-Challenge fixed revision210d026faf9955653af8916fad021475a3f00453全test1172，25shot native lm_eval0.4.11完整答案文本loglikelihood，primary acc_norm按原choice字符数归一，secondary acc描述。保持A已接受的train演示采样seed20261002/身份/顺序/原生边界、max_length32768及所有候选，不改题、不截断答案。三cell及fixed_t0/lag1/current_t、九档lambda、Native2/Loop3，共68独立配置79696条记录；86展示行含18 K2别名，不重复执行/检验。current_t含t0，同轮raw D_t拟合立即干预p；其余dtype/cache/alpha/batch1/SVD等原样。无Matched-norm。

C迁移清单在workspace/runs/phase10-gate-c-20261002T081604Z/mmlu-analysis-attempt1/transfer_lambdas.json；直接固定复制到D输入并核对7行：q17K2 current_t .2、fixed .9；q4K2 current_t .9、fixed .6；q4K3 current_t .9、fixed .1、lag1 .9。不得依据ARC改变或增加筛选门槛；MMLU无显著Loop优势不取消既定ARC实验。

## 执行顺序和资源

1. 检查当前代码/输入/计划，生成68完整ARC配置清单和1172 canonical身份，固定transfer清单。优先使用现有Phase10 producer/verifier/analysis，仅在正常执行必要时做最小Phase10增量。C分片变更后ARC尚无同源码端到端预检：在debug以A无标签validation小集跑两模型、K2/K3、三策略代表性完整文本评分→写分数→验证；采用同正式launcher/runtime/producer路径，job最多29min。可复用B的评分等价/方向检查，不重复全部B测试。若采用分片，针对ARC验证同小集分片合并与未分片分数一致。关键路径修复后重做最小debug。
2. debug PASS后正式A800先跑可保留首批：七个迁移参数对应的Online完整cell（每cell1172）加两模型Native，全部属于68冻结面板。最多2GPU、packing1、batch1；每首批job time最多4h，以有界队列总上界控制预算。首批累计含debug上限40GPUh。记录实际两模型/各策略耗时显存，按全68剩余任务估计成本。已消耗+1.2×剩余保守预测<=100GPUh时，同executor可继续其余59cell；否则向planning发最小预算补充，禁止越限或缩科学面板。总D硬预算100GPUh含失败/重试/debug，max2GPU。每job不超过8h且符合现场限制；任意提交批次已消耗+全部待运行/运行job GPU×time上界<=100GPUh。若首次完整cell无法在4h内完成，保留证据按授权低风险调度/分片恢复，不反复重跑成功数据。
3. 所有非正式任务仅debug。正式任务用普通用户合法最高优先级A800分区/QoS，提交前现场核实；此前emergency_gpu是最高合法选项，不盲继承。不要改batch/dtype/SVD算法换速度。先查既有D作业/输出再提交，避免重复。
4. 全68×1172=79696无标签原分数完整闭合，canonical身份/候选数量/有限值/同配方及来源、18展示别名、所有jobs终态与资源核实后写ARC_FULL_PANEL_CLOSED。此前禁止target gold、correctness、partial accuracy或按结果调参。闭合条件满足后本handoff授权一次读取固定revision ARC test答案、按id/question/choice身份严格join，唯一固定分析，无需再次请求用户确认。
5. 分析：acc_norm主，acc附描述。预选迁移7×[Loop,Native]=14一个Holm主family；全扫描63×2=126另一个探索Holm family；每lambda5个独立策略比较共45一个次family。exact双侧McNemar alpha.05、题级paired bootstrap10000 seed20261002 nominal95%CI。全网格含负结果，清楚区分MMLU预选迁移和ARC本地最优；无方向特异性结论。写完整68独立/86展示表、185 contrasts（14+126+45）、选参引用和资源。完成D终态，不进入E。

## 路径和工程权限

本地Git根为外层项目下loopscope-tflt，branch loopscope，起点ecaa1367eace0a1abb3268095057f4b22b616f36，保护祖先4f59bd93eca4da3cbf458a93508f91c5b23912bc。初始dirty仅planning C acceptance/control/D handoff，可原样代提交。只写Phase10必要模块/scripts/tests/docs和D evidence/resource；control/contract/handoff只读。protected runtime/wrapper/strategies/cache/config/eval_runner与Phase8/9只读。正常commit/push origin/loopscope，禁止force/rebase/reset/main修改。人类输出外层资产/报告/phase10/gate-d-arc-<date>，不镜像到Git。

HPC alias hpc2-hkustgz，实际读hpc2-hkustgz-ssh skill。严格HostKeyChecking、BatchMode/ConnectTimeout、ClearAllForwardings=yes、UpdateHostKeys=no；hostkey问题使用专用skill，不自动接受。VPN断线不是job失败，不擅改SSH/hosts/VPN/Clash或取消重投。

workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope。可写专属inputs/phase10-gate-d-<timestamp>、staging/phase10-gate-d-<timestamp>/source-<commit>、runs/phase10-gate-d-<timestamp>及logs，新attempt保留旧证据。已提交git archive完整部署，immutable source不原地修，修复新commit/source。旧source /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope保持只读，原.venv-loopscope-cu121-20260711/bin/python、shared模型/tokenizer/cache不安装升级下载。ARC输入只读workspace/inputs/phase10-gate-a-20261002T062100Z/prepared-attempt1/ARC-test-pool.json与validation池，原raw parquet target列仅步骤4授权后读取。C transfer只读；不重跑MMLU或重新选参。不新增摘要值。

CPU验证按实际改动做定向tests/CLI help/bash-n/diff--check，无代码改动不重跑全suite。低风险工程/调度失败在相同科学和预算内保留attempt后修复/重试，只续缺失，不覆写成功数据。父executor可取消自己确认错误/重复的D job，不操作别人/C作业；不能为排队盲重投。标签隔离、科学/预算/保护路径变更报planning。

## 等待与交付

用户暂不定时监控：不创建/恢复automation，不人工轮询或长sleep维持在线。提交后一次确认真实job ID/状态，保存精确剩余动作和预算，向用户说明等待点；在后续用户请求/明确消息恢复后先核实已有作业和记录再继续。长等待不等于Gate失败。此用户选择覆盖旧默认自动接续，不能宣称无人值守自动推进。

must-pass：同关键路径debug；合法A800完整68cell原分数闭合；ARC标签仅闭合后；预选7参数不变且14/126/45 families正确；全jobs/预算/source/数据/输出归属明确。planning轻量验收三个决定性入口，E一次阶段总审。内部有进展低风险修复不限次数，通常一次合并审计退修。

唯一GATE_D_FINAL_AUDIT或真正BLOCK向planning 01a0fae2-12df-74e0-a874-190518ef0501 local投递，列源码/最终Git状态/改动/测试exit/jobs/root/closure/analysis/transfer/预算/未做事项，请求PASS/PASS_WITH_FIXES/BLOCK，不自判PASS。若工具返回需要审批而policy never，保存原错误与完整包到 .planning/phase10/loopscope_phase10_gate_d_final_audit.md 和外层报告TERMINAL_DELIVERY_UNCONFIRMED.md，并在final明确链接，不能以写文件冒充送达或绕过审批。E须新独立线程。
