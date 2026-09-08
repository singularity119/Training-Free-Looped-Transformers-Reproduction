# Gate F：t=0 共享方向与 t=1 配对比较

## SUPERSEDING AUTHORIZATION - current effective scope
User explicitly authorizes E and F to run concurrently; no E PASS or GATE_F_RUN_RELEASE is required. FULL_EXECUTION_AUTHORIZED now overrides every preparation-only, no-commit, no-remote-write, and wait-for-E restriction below. All specified implementation, Git, HPC, data, budget and analysis permissions are effective immediately. E retains its own executor and authority. F coordinates shared Git and remote-source changes with E before mutation, preserves E immutable producer and active jobs, and never overwrites another thread's dirty files. If E launchers use mutable source, coordinate a stable E source copy or wait only for the affected process to finish before sync; do not cancel E or force it onto F code.
F runs its 16 new cells independently now. Only final paired closure awaits valid E K3 comparator scores; no duplicate E runs or polling. Planning delivers accepted E evidence by event. Separate monitors and roots; each Gate at most 2 GPUs (combined 4), E 24 GPUh and F 48 GPUh. This explicit user exception applies only to E/F.


## 身份、状态与入场
按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行；实际读取 skill 和 references/protocol.md。
Planning：01a08024-3556-75c1-8209-64e89345b940。
Executor：01a08053-81a4-70b1-a306-a721189018a2。
Title：execute-LoopScope-T0-Shared-Direction-第8阶段-Gate F。
分支 loopscope；本地 loopscope-tflt 专用 clone；准备基点 3d1cd8012ff4daafe5540c9cdb6beb140ee29dc8。写前检查 branch/status/HEAD/保护基点祖先，不回退其他线程编辑。

用户2026-09-08明确要求立即创建新线程，故特例提前绑定 F 进行独立准备；不把此条推广为以后并行 Gate。当前仅 PREPARATION_AUTHORIZED：可读本地/HPC 相关证据，新增 F 专属脚本、配置、测试、evidence 并做本地 CPU 检查。E 仍独占共享 phase8 runtime 与远程执行；F 不改任何已有代码、不提交、不 push、不远程写入、不运行模型、不创建作业/monitor。完成准备后给 planning 发一次 F_PREPARATION_READY（非 Gate 终态），等待 planning 在 E PASS 后发送 GATE_F_RUN_RELEASE；不得自解锁。共享代码/Git/HPC 正式权限以下为冻结待放行范围，不是立即生效权限。

权威：AGENTS.md → PROJECT_MEMORY.md → .planning/README.md → control → 本 handoff；旧 contract_v1 提供未修改配方。原 A–D 和 E 结果保持独立。下一 Gate/Phase9 未授权。

## 科学问题和完整配置
比较首次窗口残差拟合出的固定方向，是否比原 t=1 拟合更适合后续循环的定向衰减。不是逐 prompt 在线估计，不宣称主方向有害或存在可靠收益。

| 模型 | inclusive 层窗口及边界 | dtype/cache | K |
|---|---|---|---|
| Qwen/Qwen3-4B-Base | 12:15 (B12→B16), 13:16 (B13→B17) | BF16/first | 2,3,4 |
| Qwen/Qwen3-1.7B-Base | 12:15 (B12→B16), 6:9 (B6→B10) | FP16/last | 2,3,4 |

模型 revision、数据 revision、tokenizer、lm_eval0.4.11、MMLU plain5shot/dev first_n5、validation512身份顺序/seed20260905、完整test14042/57 subjects 与 contract_v1 相同。damped_euler/block，alpha=1，h=1/K，总时长1，beta0，batch1，decode bypass，rank1，lambda0.5；无训练、无其他窗口/参数搜索。

t0 是首次经过窗口的 delta0=G(X0)-X0，尚未乘 h。四个模型/窗口分别从同一512样本 pre-answer 位置获得512×d矩阵；不中心化、不行归一化，native差值转FP32，再CPUfloat64 reduced SVD，top1 signpivotpositive，最终FP32unit。只拟合4份正式 t0 basis，逐份跨 K2/3/4 加载同一方向；元数据明确 fit_t=0、source_k、applies_to_k=[2,3,4]，不能伪装成t1或复制改k制造12份拟合。

优先从 Gate C 已保存、无干预的 t0 residuals 直接拟合，固定取各模型/窗口 K2 来源；只读核对512身份、位置、dtype和完整性。用小规模同 prompt K2/3/4 检查 t0 在真实代码路径的独立性。若历史残差不可用可新采同512四份t0，不要按K重复拟合；出现实质K依赖先报告原因，不盲目共享。t1新对照仍按其K拟合。

运行干预时机不变：t0 不衰减，只在 t>=1 的 pre-answer residual 做 delta-.5*v*(v^T delta)，FP32计算后转回原dtype；其他token不变。拟合时刻变化不等于干预时刻变化。

12配置每组3arm：普通Loop、Spectral-fit-t1、Spectral-fit-t0，共36逻辑cell。复用D中8Loop+8t1与E验收后4B K3的2Loop+2t1（共20cell）；新增12t0 testcell，加1.7B K3两窗口的2Loop+2t1（另拟合2份t1 basis），共16新testcell=224672条评分。无需新Native。复用以逐样本scores/identity和配方为准，不从四舍五入acc拼接；低成本等价检查若暴露影响分数的真实漂移，先向planning给最小修复/补跑范围，不静默复用。

## 结果与信息边界
新16cell全部唯一完整、四选项分数finite、配置闭合前不计算新增partialacc/flip，不读新增目标gold。全闭合后一次统一分析。已知D结果的探索性追加，不能称全新未触碰test或改写原主次检验。
主比较 t0−t1 共12项，Holm单一12项探索family；辅助 t0−Loop 共12项，另一个12项family。t1−Loop仅作描述对照，原D/E family不重写。各项报告correct/N、microacc、pp差、错→对/对→错、subjectmacro、配对subject bootstrap10000次seed20260905的nominal95%CI、exact McNemar及family Holm。明确nominalCI非同时区间，无跨模型通用性或选择最佳配置确证性结论。负/不显著结果有效，不能因此调参重跑。

## 放行后的实施与权限
只允许最小修改 src/tflt/loopscope/phase8*、scripts/loopscope/*phase8*、新 F configs和针对性tests；默认原t1行为保持，t0用显式opt-in元数据及加载校验。wrapper/strategies/cache/config继续只读，必要变更报planning。已有历史contracts/manifests/basis/runs不可改。
允许普通单目的commit/push origin loopscope、HPC专用clone快进；不rebase/force/main；执行前确认E不再写入。严格SSH hpc2-hkustgz，按hpc2 skill；source=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope；venv=.venv-loopscope-cu121-20260711；W=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope。读取shared hf_home/datasets和旧C/D/E相关证据，不更换环境或revision。新工件 W/{inputs,runs,artifacts,staging}/phase8-gate-f-<timestamp>-<purpose>-a<N>，write-once，只补无效/缺失工作。模型/数据/完整日志不入Git；本地报告仅外层资产/报告/phase8/gate_f/，不改Notion。

正式A800普通用户合法最高优先级，不能用管理员权限。单job<=6h，同时<=2GPU，总预算48GPUh含debug/retry，1GPU/8CPU/128G为起始；packing1–3独立进程按实际每模型上尾峰值/吞吐确定，留至少15%余量，不改batch/dtype。OOM保留attempt并降低packing。只可取消本F明确失败/超预算/有OOM风险作业，不动E或其他作业，不在连接故障时重提未知状态job。
所有debug/smoke在debug partition，<30分钟；同不可变producer/launcher/env/runtime/model路径先通过端到端debug才正式。两个模型都覆盖t0拟合/保存/跨K加载/后续干预/标准评分/写出/verifier，另覆盖1.7B K3 t1对照。微样本basis仅PREFLIGHT。资源形状不代表正式时先保留正式前512 canary，合格后补余量；代码关键路径变化重做对应debug。

## Agility budget及验收
最小增量是t0 fit/load opt-in、12+4cell panel及复用统计入口，不构建通用框架。针对性检查：t0跨K同方向加载、t0不干预而t>=1正确衰减、t1默认路径不变、36逻辑cell和配对family。确定性入口：git diff --check；PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase8_gate_f*.py'（不得0tests冒充通过）；新launcher bash -n；实际producer/verifier命令写一次Gate evidence。关键检查通过即进入debug/正式，不重复旧全套测试。
executor负责工程验证/运行/报告；planning只验收3件事：4份t0方向确实跨K共享及干预时机；36cell来源闭合和统计；终态作业/资源及报告不超证据。阶段末复用D/E验收做一次简洁综合结案。

## 修复与终态
低风险工程修复在本Gate冻结科学/48GPUh内次数不限，保留失败attempt；科学、protected、预算或信息边界需planning。一次集中验收退回修复，第二次仅同材料性根因或修复引入回归。最多3个并发 bounded 子agent，不嵌套、不调度、不跨Gate；不回退他人代码，executor独占Git/Slurm/终态职责。
放行后真实formal job ID返回即建唯一F executor-owned monitor含PENDING：probe30min/full60min；debug稳定RUNNING约1min后才smoke10min。切类型更新同一个monitor；终态先暂停，再send_message_to_thread向01a08053-81a4-70b1-a306-a721189018a2发AUTOMATION_TERMINAL_RESUME，触发验证/修复/继续。失败未必Gate BLOCK。无法确认唤醒给planning01a08024-3556-75c1-8209-64e89345b940发AUTOMATION_RELAY_REQUIRED。准备等待E不创建monitor，也不轮询E。
完成后向planning发唯一 GATE_F_FINAL_AUDIT 或真实 BLOCK：executor、起点/最终Git/dirty/push、变更、实际命令exit、jobs/roots、4t0+2t1 basis、16新增/20复用cell、统计/资源/偏差/报告，申请PASS/PASS_WITH_FIXES/BLOCK，保留送达成功证据；失败明确TERMINAL_DELIVERY_UNCONFIRMED。不能self-PASS或进入下一Gate。
