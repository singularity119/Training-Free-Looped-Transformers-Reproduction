# Gate A：第九阶段增量模块实现

按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，实际读取skill及references/protocol.md。涉HPC先读hpc2-hkustgz-ssh。

Planning/终态接收者：01a07ff5-79b1-79f0-809a-4d0971475686。
Executor：01a0c70e-2092-72b3-911c-6544822d585c。
Title：Gate A-execute-LoopScope-Online-Modules-第9阶段。模型gpt-5.6-luna，reasoning=max。

## 准入、路径与责任

用户已授权整个Phase9串行推进，科学合同contract_v1冻结。A目标仅完成最小可进入真实debug的增量代码，不执行B诊断或C正式评分。

本地唯一代码根为 /Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt，branch精确loopscope；源基点b281ebba85ae8febdcccedd0d201cc5185721552。实际启动HEAD是planning文档提交后的后代，在证据记录该HEAD；已知planning文档变更由planning完成提交，不覆盖他人改动。每次mutation检查path/branch/HEAD/dirty/保护祖先，禁main、reset/rebase/force。

科学以loopscope_phase9_contract_v1.md为准：两模型五窗K2/3/4、MMLU5shot、alpha1、rank1、lambda0.5、t0有效prompt SVD、t>=1答案位置干预。Matched-norm按自身轨迹定义。旧Phase8代码/basis/工件不可覆盖。

## 最小实现与允许写入

- 新增src/tflt/loopscope/phase9_spectral_core.py、phase9_runtime.py、phase9_adapter.py及必要phase9_accuracy/pool/verification旁路组件；优先复用旧评分和identity工具，不建立新通用框架。
- 新增configs/loopscope/phase9_*、scripts/loopscope/*phase9*、tests/test_phase9_*、docs/loopscope_phase9.md。交付B可用的debug/diagnostic入口和77配置manifest语义；C正式执行入口可在C补齐，不要求本门实现完整报告平台。
- 仅可写.planning/phase9/loopscope_phase9_gate_a_evidence.md记录结果；control/科学合同/handoff由planning拥有，不自行修改。
- wrapper.py、strategies.py、cache.py、config.py及所有phase8_*保持不变；现有residual_transform已足够优先复用。若实际必需改protected文件，提交最小原因/diff给planning，不擅改。
- 普通单目的本地提交与push origin loopscope获准，目标git@github.com:singularity119/Training-Free-Looped-Transformers-Reproduction.git；不提交模型、数据、basis、日志、凭据。先审查现有ahead提交仅项目文本，若发现外部审批拒绝保留理由报告，不绕过。

## HPC与信息边界

允许strict/key-only/ClearAllForwardings=yes通过hpc2-hkustgz只读检查：专用clone /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope；环境及 /hpc2hdd/home/xhuang225/shared 下项目缓存；workspace /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope 的sanitized validation512元数据、旧basis schema/配置/清单路径。不得读test正文、gold、旧逐题correctness或结果来调整方法。

可在现有已批准环境执行小型CPU合成张量验证/import，不加载模型，不提交Slurm/GPU，不下载或升级依赖，不写旧工件。必要源码同步仅可在先检查远端dirty/jobs后fast-forward专用clone，若新文档和代码同步能影响活动任务须报告；不得改只读复现checkout。连接故障不妨碍本地A工作，保留现状不自动取消/重提历史job。

## 决定性验证

1. 纯数学已知rank1矩阵的方向/投影、平行分量减半/正交不变、Matched-norm范数；t0完全不改、仅p修改、零强度走原路径、换请求释放状态。
2. fake HFLM测试真实context/continuation边界、padding/特殊token排除、左截断、多token候选不会进入估计；同prefix候选与重排不造成跨题串用。
3. 配置15个model/window/K与77逻辑cell唯一完整；旧Phase8原接口针对性回归。torch在运行时延迟导入，本地无torch可跑非张量测试；真实tensor验证安排B，不把skip当执行成功。

确定性命令：git diff --check；PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase9_*.py'（必须非零用例）；PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_loopscope_phase8_adapter.py'；PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_loopscope_phase8_spectral_core.py'；新增脚本CLI --help和sbatch/bash脚本bash -n。记录实际命令和exit；因实现文件名不同可新增相应定向命令，不能删科学必需验证。无credible broad regression不跑全历史suite。

## Agility budget、验收与修复

最小增量为Online/Matched-norm callback+prefix mask+可被B调用的runner；不要先重构Phase8。满足关键测试后交付A，B是最早真实模型实验。一次Gate evidence足够，不创建摘要值或重复receipt。允许同Gate低风险工程修复直到目标实现；planning通常一次集中返修，仅同根因/回归可第二次。科学与信息边界不变。

planning验收仅查看新代码实际关键路径、目标测试、旧实现无覆盖三个事实；不要求executor自我独立审计。阶段总审计留D。

允许最多3个并行bounded子agent做本Gate具体工程工作，不嵌套；你负责集成、Git、所有远程写入与终态。明确各自文件ownership且不回退他人编辑。子agent没有调度/科学变更/终态权限。

## 终态交付

完成后发送唯一GATE_A_FINAL_AUDIT给上述planning，包含起止branch/commit/dirty、文件、命令exit、测试数、HPC检查/未检查、接口说明、已完成和偏差、申请PASS/PASS_WITH_FIXES/BLOCK。必须用send_message_to_thread主动触发planning并保留成功送达；不可仅本地final。无法送达则输出TERMINAL_DELIVERY_UNCONFIRMED粘贴包。只有确无当前授权内安全进展才BLOCK；不得自PASS、不进入B。A无GPUjob无需heartbeat。
