# GATE_C_HANDOFF：正式无标签校准与8个固定方向

**必须按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行；实际读取skill及references/protocol.md。**

Planning=01a06fe9-c7bb-7d72-a906-234f301de317
Executor=01a070f0-158e-72f2-9d06-5ba47c02e2bd
Title=execute-LoopScope-Residual-Calibration-第8阶段-Gate C
Admission=Gate B PASS，正式合同v1已冻结。收到GATE_C_HANDOFF_READY后才开始写入/SSH/执行。

## 1. 目标与最小增量

在已确定的512个validation身份上运行8个无干预Euler配置，保留有界pre-answer residual轨迹，用每配置t=1的512×d矩阵拟合独立rank1方向并冻结，为D提供可直接加载的8个basis。给出无标签谱/跨步诊断，不能以主方向“看起来不好”预判accuracy。

复用B已验收的runtime、HFLM adapter、safe renderer与数据加载路径。最小新增是512 pool导出、正式collector runner/launcher、完整性和basis verifier。不要复制/重写模型评分、训练模型、实现online directions、额外对照或通用实验平台。B的8个SMOKE_ONLY basis不可代替正式方向。

## 2. 基线、权限与可写路径

本地repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope；启动实现HEAD=ef54d2cdafa92483ed65575d0c5788b7980d60f3，B实际producer=c7e4e1f5e0b38874ee4e7eb1ac19dd5c7b6e59ba。允许READY前新增planning的B acceptance/C control/handoff/导航文档提交；核对diff，不reset。固定祖先4f59bd93eca4da3cbf458a93508f91c5b23912bc。

可写：
- src/tflt/loopscope/phase8_calibration*.py；phase8_pool.py仅增加正式512导出入口/最小共享renderer复用，B debug行为保持；phase8_runtime.py仅正式collector/序列化所需最小修补且不改投影/SVD科学。
- scripts/loopscope/*phase8_calibration*、tests/*phase8_calibration*及受上述改动影响的既有Phase8 pool/runtime测试。
- configs/loopscope/phase8_calibration*.json：只绑定执行布局/资源，不改已冻结model_windows/contract。
- docs/loopscope_phase8.md的C命令/工件说明；.planning/phase8/loopscope_phase8_gate_c_evidence.md。

config.py/strategies.py/wrapper.py/cache.py/eval_runner.py与其他阶段实现只读，C不应再改实际干预算法。若真实normal-path缺陷需要protected diff，给planning最小复现/拟议变更申请，不自行扩大。

用户已授权第八阶段源码提交；允许本人普通commit/push到既有git@github.com:singularity119/Training-Free-Looped-Transformers-Reproduction.git的loopscope，包括前置planning提交。禁止force/rebase/main合并、历史覆盖及任何模型/数据/basis/residual/完整logs入Git。平台实际权限以你自身直接收到的developer为准；若审批拒绝不可绕过。

## 3. 冻结科学与输入复用

读取 .planning/phase8/loopscope_phase8_contract_v1.md：Qwen3-4B-Base exact906bfd4...、BF16、windows12:15/13:16、cachefirst；Qwen3-1.7B-Base exactea980cb...、FP16、windows12:15/6:9、cachelast。K2/4，alpha1，h1/K，block、batch1、decodebypass、plainMMLU5shot/dev示例、HFLM0.4.11评分路径。

复用B已冻结identity文件：
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase8-gate-b-20260905T072208Z-retry6/calibration_identities.json
同目录debug_pool.json只作为renderer/source/上尾长度与B信息参照，不改变状态、不覆盖。当前dataset=cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe。正式pool必须与B的512身份、顺序、seed20260905逐项相同，按contract原算法重建检查，不新增抽样或替换题。

允许safe读取validation问题/choices/身份、dev五示例及答案，按B相同standardrenderer生成512 prompts；目标gold、test题目和所有既有test/accuracy结果不得读取。CPU构建中允许暂时遍历validation1531的safe字段，但只持久化512目标prompt及必要来源元数据。若cache缺少exactrevision先定位，禁止静默改版本/下载别的数据。

每model/window/K以direction=None、原Euler计算采集t0..K-1，只保留答案前位置的native out-x转CPU FP32；四candidate或stash不得重复入矩阵。正式每cell512身份，K2有1024条/t0..1、K4有2048条/t0..3，共12288条d维residual。共4096个配置-样本轨迹，8个t1矩阵各512×d。不得保存完整token hidden/residual，不需要保存choice scores或答案。

每cell独立CPUfloat64 reducedSVD、无中心化/行归一化，规范方向符号，运行方向FP32归一化；同K拟合同K应用。保存FORMAL_CALIBRATION作用域、模型/revision/window/K/cache/dtype/alpha/dataset/身份顺序、源producer/run路径和algorithm字段。已有schema可沿用，用scope区别SMOKE_ONLY；不增加摘要绑定。

## 4. 无标签诊断与完整性交付

每cell保存：
- 全512身份、每t的有界residual矩阵、位置和t计数，供诊断及复核；t1可沿用residual_t1.pt格式。
- basis.json（方向和现有谱诊断）；必要的singular values可作为小型向量保存。
- 每题每t的a_it=v^T delta_it、residual norm，以及按t汇总投影能量占比sum(a²)/sum(||delta||²)。用同一t1拟合v诊断后续t，K之间不共享v。
- repeated steps的累计投影h*sum_(t>=1)a_it；K4额外给出相邻重复步投影同号比例及coherence=abs(sum a)/sum abs a。分母零显式标undefined并报告数量，不能伪造为支持/反对。K2只有一次重复，coherence为1也不是持续累积证据。

这些诊断在拟合集上计算，属于in-sample描述；t1头部能量高与该方向由t1拟合有关，不作为独立泛化或有害性证据。没有预设正号/阈值准入，方向有效且完整就可以进入D。不要增加未批准的C_mech256或其他model/task/sweep。

一次fresh-process verifier检查8/8配置、每配置512同一身份顺序、各t形状/finite、正确model/window/K及dtype、basis单位长度/来源、无SMOKE_ONLY混入；用原D直接矩阵向量运算核对方向Rayleigh quotient和top singular value/eigen residual，允许FP32保存的合理舍入，不要求再做8次完整SVD。记录阈值与误差，正常模式误差<=1e-4相对量级作为初始工程容差，异常先查精度/来源，不据此调science。失败不覆盖成功cell。

## 5. HPC与资源授权

先读hpc2-hkustgz-ssh skill。host=hpc2-hkustgz，严格host-key、BatchMode/ConnectTimeout/ClearAllForwardings。VPN源地址会变，不硬编码旧IP；仅当需要时从当前ifconfig找有效10.x接口并对同一alias作每命令source binding，禁止擅改hosts/网络/SSHtrust。

Remote source=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
Runtime=<source>/.venv-loopscope-cu121-20260711/bin/python
Workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope
Read cache=/hpc2hdd/home/xhuang225/shared/hf_home；/hpc2hdd/home/xhuang225/shared/datasets
Read B roots=上述inputs retry6及runs/phase8-gate-b-20260905T072208Z-q17-a1、-q4-a1。

允许专用clone clean状态下fetch/fast-forward到已push实际实现commit，不接触固定复现checkout。只写workspace/{inputs,staging,runs,artifacts}/phase8-gate-c-<UTC>-<attempt>/前缀，所有新尝试新路径；不删除或修改B/历史工件。允许既有环境CPU/import/config/renderer检查；不升级包/下载weights，遇实际缺项请求planning最小补充。

先debug：新的正式producer/launcher同commit/环境/模型/runtime/序列化/verifier需两模型最小端到端preflight。debug partition每job29min、单GPU；选同一512内固定少数题并含其上尾长度验证正常路径，scope=PREFLIGHT_ONLY，不能当正式basis。B的功能证据复用，不整套重跑B；针对新增512读写/collector/export/fit/verifier的路径即可。实现或launcher改后相应debug重过；文档变化不取消既有有效job。

正式校准允许普通用户合法账户下最高可用优先级partition/QOS，由现场association/partition证据选择；不冻结GPU型号，不使用管理员特权。上限：同时总共2GPU，单job最多2小时、每GPU最多24CPU/128GiB主存；全Gate新增GPU尝试预算16 GPU-hours（含debug/失败计入），越界前报告。按测量选择每GPU1–3个独立cell进程的固定packing/worker backfill，最多同时6个模型进程；不跨cell共享模型/tensor，不改变batch1/dtype。

B在A40测得1.7B reserved5.90GB、4B10.88GB，validation上尾2954token，只作为起始估算。C要在新producer上测实际上尾和并发峰值/吞吐；不凭B单进程结果直接宣布3进程可用。保留至少20%显存或实际测得足够瞬时headroom，选改善/不损害time-to-result的packing。若debug无法复现formal GPU/资源形态，先最小有效formal canary计入本Gate512采集，验证后扩展；不得用未测模型/GPU组合外推。合格canary记录算正式工作，不为目录整齐重跑。

允许exactexecutor本人提交上述debug和正式calibration jobs、只取消自己本Gate确定无效/重复作业、以新路径重试工程/launcher/序列化/资源无效或缺失cell。保留成功/失败/取消分别记录；OOM降低packing但不改batch/dtype。不得取消他人job，不改科学/增样本，成本预算到达或无安全进展就向planning报告。MaxJobsPerAccount排队不自动授权换成未验证账户；允许选择明确属于本用户的合法association并记录，不绕限制。

## 6. Agility budget与验证责任

最小实现：512pool导出、无干预calibration runner、一个自包含launcher和独立verifier，沿用B tensor路径。采集/校准不是新干预算法。可将8cell作为独立可保留单元，避免失败一cell重跑整个模型。

从repo root执行：
```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p '*phase8_calibration*.py'
git diff --check
```
若修改pool/runtime只补原对应目标模块测试一次；既有config/strategies/全suite不因进入C自动重跑。launcher --help/--dry-run、bash语法、环境CPU检查与真实debug记录确切命令/退出码。

Earliest real evidence=新增入口的目标测试通过后即两模型最小debug；通过后运行512校准，不继续添加框架或对抗检查。完整性是C结果本身，所以最终8×512成员必须完整检查；不增加收据树。重复调用/排序/source混淆是正常路径材料性风险，直接检查即可。

Executor负责实施、debug/采集/fit和一次fresh verifier。Planning只检查1)8cell/512身份与basis正式作用域；2)决定性数值/来源与无标签边界；3)terminal作业及D可加载性。更广泛最终科学claims在D后一次phase audit。Gate内安全工程修复不限但有预算且须有进展；planning返修一次合并，原根因/直接回归才第二次。谱集中度不高不是工程BLOCK。

## 7. 长作业与终态连续执行

唯一executor-owned heartbeat：debug smoke10min、正式校准probe30min。debug真实job如未知PENDING等待，可从排队期启用同一个10min monitor（沿用B已证明的共享账户排队观察例外）；短job可直接完成而不创建monitor。正式非debug校准在verified真实jobID返回时即建/更新唯一30min monitor，覆盖PENDING和RUNNING。正常状态不变保持安静，planning不轮询。

prompt固定exact Gate C/executor/planning/host/jobset/runroot/下个已授权动作。attempt终态暂停monitor，再用send_message_to_thread给本executor发AUTOMATION_TERMINAL_RESUME并保留成功确认；无法触发则发AUTOMATION_RELAY_REQUIRED给planning。失败先在原许可内诊断修复，不自动Gate BLOCK。

吸取B实际交付延迟：收到terminal-resume的executor回合必须继续做verifier、保存证据、提交push与向planning送达，不能仅宣布“我将验证”就结束。若还有下一job，执行并正确接续同一monitor；只有在真实外部等待或Gate完整/真正BLOCK时才结束回合。

最终写 .planning/phase8/loopscope_phase8_gate_c_evidence.md，主动send_message_to_thread至planning 01a06fe9-c7bb-7d72-a906-234f301de317，发一次GATE_C_FINAL_AUDIT或真正BLOCK：exact身份、start/final/source commits/dirty、files、commands/tests/exit、HPC jobs/roots/资源/有效attempt、512/8完整性与basis路径、无标签诊断、错误偏差、未做test/outcome/D，请求PASS/PASS_WITH_FIXES/BLOCK。你不能自行PASS。工具确认是送达依据，本地final只作副本；失败输出TERMINAL_DELIVERY_UNCONFIRMED及原因。此前已解决BLOCK恢复后的新终态允许引用旧事件，不重复发送同一包。
