# GATE_B_HANDOFF：真实干预接入与双模型 debug

**必须按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行。实际读取 SKILL.md 和 references/protocol.md，不仅依赖摘要。**

Planning: 01a06fe9-c7bb-7d72-a906-234f301de317
Executor: 01a0704b-5798-7680-80a2-f1145f35f9ba
Title: execute-LoopScope-Intervention-Debug-第8阶段-Gate B
Admission: A 已 PASS，用户已确认剩余配方，[科学合同v1](loopscope_phase8_contract_v1.md)已冻结。
首次收到 GATE_B_HANDOFF_READY 后，重读 AGENTS.md→PROJECT_MEMORY.md→control→contract→本handoff，核对精确绑定再变更。

## 1. 目标与非目标

在保持原 Euler/评分/cache 路径的前提下，接入实际应用 residual 的 opt-in soft damping，并在 HPC debug 对两个模型、四窗口和 K2/4 验证。产物必须足以让 C 用同一实现采集512题方向、让 D 复用评分入口。不要实现通用 steering 平台或额外实验。

不运行正式 C_fit512/test14042、不读目标 gold/accuracy、不调lambda/K/window、不做逐prompt在线方向、不自行进入C。不重审已PASS的A。

## 2. 起点与本地修改权限

Repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
Branch=loopscope；已验收实现基线3fe34c46e3ac9774907349bdbf1702ab2180919f；本轮启动HEAD=2e56e05c46e8d74f32f4d47305b1e116cf0629a6。允许 READY 前追加仅 Phase8 科学/规划/配置同步的 planning commit；直接读diff核对，不回退。祖先4f59bd93eca4da3cbf458a93508f91c5b23912bc必须存在。初始工作树应clean。

获准最小 protected diff（依据A接入方案）：
- src/tflt/config.py：新增默认None的opt-in residual transform callback/context；明确只支持当前Euler/block路径。
- src/tflt/strategies.py：在opt-in分支原dtype residual计算后、原h更新前调用transform，提供t；None和zero-strength保留原计算路径，K1原分支保持。

允许 Phase8 专属新增或必要修订：src/tflt/loopscope/phase8_*.py、scripts/loopscope/*phase8*、tests/*phase8*、configs/loopscope/phase8_*.json、docs/loopscope_phase8.md 的命令/工件说明、.planning/phase8/loopscope_phase8_gate_b_evidence.md。
科学 contract/model_windows JSON 的已冻结字段不得修改；可以在新增 runtime JSON 中绑定环境、launcher、已冻结参数，不自行变更含义。planning-owned control/contract/plan/AGENTS 不可写。

wrapper.py/cache.py/eval_runner.py/其他历史源文件仍只读。评分接入优先 Phase8 专属 HFLM adapter/subclass/context manager，若确实无法正确传递位置，再向planning提交最小复现和拟议diff申请最小 protected-path 补充。禁止全局改site-packages或绕过标准choice-scoring。

允许自己的单一目的commit、普通push到origin/loopscope及推送前置规划提交；禁止force/rebase/main合并或回滚他人dirty。不要Git提交数据/weights/residual/basis/完整logs。

## 3. 冻结科学与最小实现

严格按contract_v1，尤其alpha=1、K={2,4}、rank1/lambda0.5、4B BF16/cachefirst、1.7B FP16/cachelast、batch1/plain5shot。

实际native residual在FP32中投影/减法，目标位置cast回原dtype，其他tokens/t0不变。向量按model/window/K区分；不跨K共用前缀。方向在执行前加载并固定；C生产512题basis前，B只有明确标注SMOKE_ONLY的微型basis。

实现有界answer-position collector：每题每个t只保存一个raw-native residual（转FP32），不得因四个choice请求/缓存stash把同一题重复计入D。真实position来自evaluator context切分和padding/truncation；若continuation不止一token，要保持正确评分与context位置。不得为简化而删题或截断成另一种prompt。

使用原版HFLM原始四choice scores验证adapter，无需gold。同model/window/K同prompt下，关闭transform/zero-strength与原版scores/输出在同runtime下应相同；若底层非确定性，先测原版自身重复误差，记录最小可解释容差，不能用宽松容差掩盖差异。使用小型tensor构造确保parallel/orthogonal/t0/其他tokens语义确实满足，而不要求随机真实题一定发生top1翻转。

basis拟合/加载最小闭环：按contract CPUfloat64 reducedSVD、FP32单位方向，显式source identities与K标注。B只用下面小型debug pool拟合，不替代C正式方向；保存/重载后方向及transform输出核对。

## 4. HPC路径、同步、环境权限

先使用 hpc2-hkustgz-ssh skill；alias=hpc2-hkustgz，ClearAllForwardings=yes。允许项目范围有界read-only repo/env/cache/scheduler/job检查。
Remote repo=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
Workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope
Existing runtime=<repo>/.venv-loopscope-cu121-20260711/bin/python
Read caches=/hpc2hdd/home/xhuang225/shared/hf_home；/hpc2hdd/home/xhuang225/shared/datasets

允许核对clean/branch/ancestry后专用remote clone的fetch及fast-forward-only同步到你的已push commit；不能reset覆盖dirty。A现场remote是08da773，须核实现在状态。不得在原固定复现checkout运行/写入/安装。

允许写 Workspace/staging/phase8-gate-b-<UTC>-<attempt>/、Workspace/runs/phase8-gate-b-<UTC>-<attempt>/、Workspace/inputs/phase8-gate-b-<UTC>/；所有重试新路径。不得修改历史输入/运行结果。

优先现有venv/cache，B允许临时环境变量 PYTHONDONTWRITEBYTECODE=1 等无科学影响设置。缺包/权重/数据时，先有界定位并报告具体缺项；不得自动升级环境或下载另一版本。planning可按实际需要补最小下载/versioned-venv权限，executor不自行改变依赖锁。不读取凭据，不绕过host-key校验。

## 5. Debug pool、资源与作业权限

CPU可读取dataset schema、subject/doc_index、validation question/choices长度和tokenization元数据，按contract生成共享512 identity清单（不forward全512、不读取目标gold）。dev五个演示答案是prompt必要输入，允许按标准renderer读取。

Debug总pool最多8个validation身份，两模型共享：C_fit按canonical顺序前4题用于SMOKE_ONLY拟合；另4题选择validation中按两模型tokenizer得到的最大prompt长度排名（按每题两模型长度max排序，平手subject/doc_index），排除前4题。这样明确覆盖上尾长度，不据结果选样本。只持久化这8题必要prompt/身份，不保存剩余target labels。test的schema/数量来源可查，B不得读取test题目或运行test。

在这8题内跑两个模型的四window/K组合；4拟合题无干预采样t1，4验证题比较native、原版Loop、transformNone/zero、Spectral，并核查collector/timestep/位置、score对齐及save/load。native每模型共享，重复运行仅限工程等价测试。debug得分只用于路径等价/有限性，禁止计算或用gold评定准确率优劣。

所有GPU/smoke必须 partition=debug，每job --time=00:29:00，最多1 GPU/8 CPU/128G内存；GPU至少满足该模型实测显存，允许debug合法可用类型，不能转formal分区。最多2个本Gate GPU jobs同时运行。先估计单job预计<30分钟；必要时按模型/功能拆小job，保持8题和冻结科学，不靠改dtype/batch简化。每次preflight记录commit/launcher/env/model/runtime/producer路径、实际job/partition/elapsed、峰值显存/吞吐/OOM。B证明功能路径，不直接授权formal packing；未来更大上尾需求必须重新测量。

允许executor自行提交上述debug jobs、只取消本Gate自己创建且确定无效或重复的jobs、在同资源上限内对工程/launcher/env invocation/serialization/resource故障使用新attempt修复重试。保留成功部分与失败证据；不重跑已有效数据仅为整理路径。依赖安装/额外资源/科学合同失败须向planning申请补充。不能取消用户或其他Gate的jobs。无formal partition/full/probe512/test权限。

若debug无法准入冻结任务，完成其余本地/CPU工作后报告具体admission blocker，不绕过debug。模型依赖正常路径故障先诊断，不因一个job exit1立刻发Gate BLOCK。

## 6. 验证与Agility budget

最小增量=两个protected文件的opt-in扩展＋Phase8 collector/basis/评分adapter＋一个debug launcher与结果verifier，不扩通用CLI或schema框架。

从repo root执行：
```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p '*phase8*.py'
git diff --check
```
原Euler/config受影响的现有精确测试模块按实际文件名补跑一次并记录命令，不运行无关full suite。纯Python数学与fake tensor测试可本地完成；torch/transformers真实测试在批准HPC环境进行。launcher固定解释器/工作目录/shell，不依赖compute节点Git或隐式Modules。

debug launcher至少支持--help/--dry-run，运行后提供同一run-root verifier入口及精确命令记录。verifier只检查计数/身份/配置、finite、方向单位长度、干预位置/步次与关闭路径一致，不另建完整审计平台。

Earliest real evidence=本地关键测试通过后立即提交一个模型的最小debug路径；再完成另一模型及上尾/配置覆盖。某个real path失败只修对应根因。Stop rule=两模型端到端与必要数值/身份checks通过就交付，不加对抗矩阵、泛化重构或重复full tests。

## 7. 三层责任与验收

Executor verification：实现、上述目标tests、CPU/真实debug、失败attempt保留与必要修复。
Planning acceptance只检查：1)真实更新/默认路径及位置；2)两个模型所有冻结window/K的debug证据与评分一致；3)basis/identity/env/资源证据可支持C。
Phase-end audit留到D后跨Gate最终科学结果，不在B重复A或预审未来全部输出。

正常路径材料性阻塞：目标token错位、干预未实际应用、None路径改分、混入stash/多choice重复行、数值非有限、误用K/dtype/cache、无有效debug准入。排版/未用schema/非运行边界不阻塞。

本Gate内部低风险修复无限但必须有材料性进展且不扩科学/权限；规划返修最多一次合并修复，原根因或直接回归可第二次。终态不是“拿到正acc”，是实现和debug可信。

## 8. 长作业与终态送达

Executor仅持有一个smoke heartbeat，10分钟。debug真实job已RUNNING约1分钟且无启动错误才创建；更早结束不创建。监控提示须绑定Gate B、executor 01a0704b-5798-7680-80a2-f1145f35f9ba、planning 01a06fe9-c7bb-7d72-a906-234f301de317、host/jobIDs/runroot与本handoff。正常不变状态安静，一次有界只读检查。

Attempt终态先暂停monitor，再用send_message_to_thread唤醒精确executor并保留确认：AUTOMATION_TERMINAL_RESUME含job状态/证据/下一步验证或许可内修复。无法唤醒则发AUTOMATION_RELAY_REQUIRED给planning。不要让monitor自己继续实现或仅输出“可以恢复”。planning不轮询。

完成后发送一次 GATE_B_FINAL_AUDIT，或穷尽授权内有进展的安全路径后发送真正BLOCK。包含executor/start-final branch/commit/dirty、files、exact commands/tests/exit、HPC job/partition/runroot/env、逐模型debug结果/偏差/未做事项，请求planning PASS/PASS_WITH_FIXES/BLOCK。你不能自发PASS，也不能进入C。

明确授权你用send_message_to_thread向planning 01a06fe9-c7bb-7d72-a906-234f301de317发送项目内终态/所需权限包。优先精简状态＋仓库证据文件路径，详细命令放 .planning/phase8/loopscope_phase8_gate_b_evidence.md。必须保留工具成功确认；不可用/被拒则输出TERMINAL_DELIVERY_UNCONFIRMED及准确原因，不能谎称已送达。无需上传模型或数据到外部系统。
