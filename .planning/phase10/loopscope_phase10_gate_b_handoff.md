# Phase10 Gate B：End-to-End-Preflight

按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，必须实际读取skill与references/protocol.md，以及AGENTS/PROJECT_MEMORY/control/contract_v2/current_t amendment/A验收/data binding/本handoff。

## 身份与准入

Planning=01a0fae2-12df-74e0-a874-190518ef0501，local。Executor=01a0fb63-097d-7c92-9bd6-1f1b8ddafe9f，local，已精确绑定。标题execute-LoopScope-End-to-End-Preflight-第10阶段-Gate B，gpt-6.1-sol/high。最多同时3个有界subagent，明确文件所有权，告知共享代码库不得回滚别人；无嵌套、无独立调度或跨Gate，父executor整合/Git/作业/交付。

A已PASS且ARC binding已接受。本地Git根 /Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt，分支loopscope，起点a7ca3626d639e1b271a1bb1b851a72e1299e359e。初始dirty仅planning验收/control/B handoff，保留并可治理提交。检查root/branch/HEAD/status/保护祖先。

## 科学与目标

contract_v2原样：三cell、fixed_t0/lag1/current_t、九档lambda、无Matched-norm，每数据集68独立/86展示。MMLU原5shot；ARC原生25shot完整文本评分主acc_norm附acc。B只工程验证和资源测量，不选参数、做accuracy或正式全test模型评分。

工作顺序：
1. 部署可追溯提交源码与既有环境，CPU/import/help校验；核验A的输入来源与长度绑定，不重建全部输入。只读核验MMLU9cell历史逐题raw score、身份、配置及完整性，不接入gold/correctness/历史analysis，不以分数设计方法。
2. 两模型的真实GPU tensor检查：lambda0恢复Loop，旧fixed0.5与Phase9路径一致，K2 fixed_t0/lag1在0.1/0.5/0.9等价；current_t含t0从原始残差拟合且只改p；三个策略逐题和逐候选状态释放正确。相同模型dtype/runtime输入应一致，先报最大绝对/相对误差；不得临时放宽阈值掩盖路径差异。SVD方向符号以投影比较；近重根若改变结果，记录并交planning，不删样本。
3. 小规模真实评分闭环：MMLU与ARC各从无标签validation取固定最多16题（覆盖首题、最长context/continuation、ARC实际3/4/5选项与数字标签）；按原排序选，输入不足某类别时只读test无标签结构构造合成边界单测，不为此跑test outcome。至少覆盖两模型、K2/K3、三策略，以及强度0.1/0.5/0.9代表组合；不要求全笛卡尔积。完整request→模型→transform→原始分数落盘→preflight verifier通过。ARC Native/Loop与未改HFLM在相同requests上完整文本分数等价；MMLU旧评分同样针对关键路径核查。
4. 测两模型/两数据集代表性长序列峰值、SVD耗时、吞吐、并发加载。batch1科学不变。先packing1，在有显存余量证据后可最多packing2实测；一旦OOM保留失败、降低packing，不能将加载峰值遗漏。正式test无标签长度若超过validation，允许按相同格式构造达到该形状的合成prefix资源样本，明确仅资源证据，不当正式可保留数据。不同model/task形状不凭比例外推。
5. 输出C/D正式任务预算与时间估计、建议packing、最高合法分区证据和可复用预检对象；B完成后停止，C另开线程。

## 权限与路径

本地可修改Phase10 src/modules、scripts/launchers、config的事实状态、tests、docs/loopscope_phase10.md，新增gate_b_evidence/data/resource附件；planning control/contract/handoff只读。原Phase8/9及protected wrapper/strategies/cache/config/eval_runner只读。若protected变更必要，提交最小复现/拟议diff由planning裁决。

允许正常commit/push origin/loopscope，禁止force/rebase/reset/stash他人变更/main写入。先提交后远端部署，不把dirty代码称为immutable commit。

HPC alias hpc2-hkustgz，先读hpc2-hkustgz-ssh skill，strict host key/ClearAllForwardings/UpdateHostKeys=no。旧source /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope 保持只读phase9-gate-c-20260922，不切分支或强行同步。授权从本地已提交loopscope通过git archive导出完整源码到workspace /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/staging/phase10-gate-b-<timestamp>/source-<commit>/；记录导出commit和直接文件/关键import来源，上传后该source不再原地修改。修复用新commit新source；不添加摘要值。

使用只读原环境 /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python，原shared模型/tokenizer/cache；不安装升级依赖或下载模型。可读相关历史inputs/runs的无标签配置/raw score及source，禁gold/correctness/analysis。ARC输入为workspace/inputs/phase10-gate-a-20261002T062100Z/prepared-attempt1；只读raw parquet默认不加载target标签。

可写仅专属workspace的inputs/phase10-gate-b-<timestamp>、staging/phase10-gate-b-<timestamp>、runs/phase10-gate-b-<timestamp>及相应logs。使用新attempt目录，禁止覆盖/删除/移动旧runs。

## GPU/Slurm预算和恢复

B总分配预算24 GPU-hours、同时最多2 GPU（含所有B子作业与packing），每个工程job walltime最多00:29:00。所有job为preflight/smoke，不授权正式FORMAL_TEST。按用户最新规则全部使用debug分区，提交前核对资格与适配GPU；正式A800最高合法优先级仅属于后续正式Gate，不能在B使用。不更改batch/dtype/数值算法换速度。无可合法使用资源时报告admission blocker。

仅executor可sbatch/srun管理本Gate作业。授权预算内工程修复后的新路径retry、取消自己已确认失败/重复待运行的B作业；不得操作其他Gate/他人jobs，不能因排队盲反复重投。统计全部失败和重试分配GPUh。预算不足先交已得证据及最小追加需求，不越限。job失败是attempt事件，低风险根因继续诊断修复，科学/信息/安全边界才Gate BLOCK。

每次关键launcher/runtime/hook/环境变化重做最小端到端；先一child通过再并行展开。正式C要复用B预检必须关键输入不变，否则C先重新预检。B预检结果不可混入正式test。

## 最小验证与交付

CPU：git diff --check；PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase10*.py'仅B修改影响时执行；关键CLI --help；bash -n两launcher。以现有run_phase10_accuracy.py --scope PREFLIGHT_ONLY、--engineering-check zero-strength/k2-policy与原phase10_preflight.sbatch实现代表性运行；新增必要真实tensor/兼容验证脚本归Phase10。runbook记精确命令、exit code、样本身份与job/root；不要虚构通过命令。

must-pass：实际GPU回调时序/边界和等价；两任务完整分数producer/verifier闭环；旧9cell原分数复用验证或确切不可复用缺口；资源峰值/吞吐足以给出安全C/D配置；无target gold/outcome读取。历史复用失败不自行补跑正式test，交planning决定。

Agility budget：只做上述代表性检查，不跑512诊断或重复全套历史suite；没有正常路径材料风险就停止加固并交B。executor工程验证；planning检查实际预检终态、关键等价/数据闭合、资源预算三项；阶段总审留E。低风险内部修复有进展即继续；规划退修通常一次合并，第二次限同根因/引入回归。

## 监控与终态

一个executor-owned heartbeat，当前smoke每10分钟，无变化静默。debug job在稳定RUNNING约60秒且无启动错误后建立唯一监控；更早结束无需新建，同一monitor跟踪自己的B job集合。每次一次有界只读检查，终态先停monitor再向exact executor发AUTOMATION_TERMINAL_RESUME，有可见送达响应；routing/fallback仅executor，不能monitor代替executor向planning判终态。短job结束前无须新建monitor；已有则关闭。监控不授予新权限。

用户当前逐Gate协作授权包含必要的monitor→executor、executor→planning、planning修复消息。完成后向planning 01a0fae2-12df-74e0-a874-190518ef0501 local发送一次GATE_B_FINAL_AUDIT或真正BLOCK，保留送达响应；失败输出TERMINAL_DELIVERY_UNCONFIRMED。列源码、测试、jobs/root、raw reuse、资源、失败和未做事项，请求PASS/PASS_WITH_FIXES/BLOCK；不得自判PASS或进入C。
