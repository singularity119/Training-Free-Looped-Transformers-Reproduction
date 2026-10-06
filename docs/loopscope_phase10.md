# LoopScope Phase 10 执行入口

科学范围与权限分别以 `.planning/phase10/loopscope_phase10_contract_v2.md`、control、exact Gate handoff 及 current_t amendment 为准。本文记录工程入口，不授予模型、作业或 outcome 权限。

## 面板与运行语义

每数据集 68 独立配置、86 展示行。两个 K2 cell 的 lag1 九档各复用 fixed_t0 结果，18 个别名不重复计数。三个 Loop 与两个 Native 共用基线；63 个独立 Online 配置采用 lambda 0.1–0.9。current_t 的三个 cell 九档共27配置均独立，K2不能复用旧策略结果。MMLU 预计复用 9、新跑 59；ARC 新跑 68。历史复用在 A 仅绑定配置/路径/schema，逐题分数与完整配方由 B 核验后决定是否可用。

`strength` 是移除方向分量的比例；Euler alpha 仍为 1，步长 1/K。仅修改最后有效 context 位置 p。S 仅包含实际保留 prefix 的有效非特殊 token；多 token continuation 不进入 S。fixed_t0 和 lag1 的 t0 不干预：前者固定每题v0，后者使用上轮未施加该轮干预前的残差方向，并在每次候选完整 forward 开始重启轨迹。current_t 每轮（含t0）先从自身尚未干预的 D_t[S,:] 拟合v_t，立即在p衰减，再做原Euler更新；不额外调用窗口，也不在已衰减残差上重新拟合。

MMLU 沿用 Phase9 的 plain 5-shot、dev first_n5、14042 test/57 subjects 及固定 revision。ARC 使用 lm_eval 0.4.11 原生 Question/Answer 模板、train 25-shot、完整 choice 文本及原生字符长度归一化 acc_norm，acc 作为描述项。候选保留 prefix 不一致时停止并交 planning 裁决，不自动删题、降 shot 或缩短答案。

## Gate D 单 allocation 入口

2026-10-07 用户要求 ARC 尽量在一个正式 Slurm job 内完成，精确权限见
`.planning/phase10/loopscope_phase10_gate_d_single_job_amendment.md`。
`run_phase10_arc_bundle.py` 在同一 allocation 内为每张 GPU 建立一个 worker，每个 cell
调用新的独立 `phase10_accuracy.sbatch` 子进程，并用原 verifier 闭合；不跨 cell
保留模型或 tensor 状态。正式先跑七个迁移 Online 与两个 Native，首批完成后用实测
cell 耗时预测余下59个配置，满足预算和剩余 walltime 条件就在同 job 内继续。

`phase10_arc_bundle.sbatch` 位置参数为：

```text
REPO PYTHON MANIFEST POOL PLAN RUN_ROOT COMMIT SCOPE GPU_COUNT PRIOR_GPUH WALLTIME_HOURS PREFLIGHT_INDICES_OR_DASH
```

新 runner 的 debug 预检使用 debug 合法数量的 GPU worker、已接受 validation
indices `[0,32,35,85,89,210]`、九配置54条完整文本评分及每 cell 验证。只有新 runner
的 debug 全部通过后才提交正式；旧单配置 launcher 的预检不替代新调度路径。

成本按整个 allocation 的 GPU 数乘以从 Slurm StartTime 起的墙钟时间计量，含空闲；
首批累计40GPUh、总100GPUh，先前 debug/失败分配也计入。余下配置按同模型、window、
K、policy 的首批完整 cell 耗时估算，Loop 用相同模型/window/K 已测 Online 最大耗时。
GPU 数量以当前 D 补充的现场合法上限为准，每 GPU packing1、batch1、同 source/runtime/数值配方。
本入口枚举本节点 CUDA_VISIBLE_DEVICES，不支持跨节点 GPU；多节点需要另行验证的分布式 launcher。
正式 worker 数由 CPU 定向测试覆盖，debug 与正式卡数差异必须记录。预算准入失败保存
已成功 cells 后停止，不丢配置或读取部分 outcome。正式合法时限与预算容许时优先
申请能覆盖全部68配置的长 allocation；原分批9/59及每job8h限制已被补充取代。

所有 target gold/accuracy 仍在68配置共同闭合和作业终态核实后才可读取。
用户暂不定时监控继续有效，真实等待点保存接续状态。

## Gate A 定向检查

在 Git 根运行：

```bash
git diff --check
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase10*.py'
PYTHONPATH=src python3 scripts/loopscope/build_phase10_manifest.py --help
PYTHONPATH=src python3 scripts/loopscope/run_phase10_accuracy.py --help
PYTHONPATH=src python3 scripts/loopscope/verify_phase10_scores.py --help
PYTHONPATH=src python3 scripts/loopscope/analyze_phase10.py --help
```

测试使用 fake scorer/tensor 或合成 outcome，只是工程证据。真实模型上的 lambda0、lambda0.5、K2 两策略等价、原生 full-continuation 分数等价与显存/吞吐由独立 Gate B 执行。A 不提交 Slurm，不创建 heartbeat。

## 数据输入与评分工件

ARC CPU 输入 builder 为 `scripts/loopscope/build_phase10_arc_inputs.py`，仅在已授权的新 inputs/staging 根运行。test/validation 在载入时剥离目标标签；train 可用答案渲染示例。每题记录 canonical identity、25 个示例 identity、完整候选文本、两模型 tokenization 和 retained-prefix 一致性。精确 revision、种子、长度与全量检查结果见 `.planning/phase10/loopscope_phase10_gate_a_data_binding.md`。

评分入口 `run_phase10_accuracy.py` 接受面板、cell、输入、scope 与新 output 根。formal 只允许全 test；preflight 可取固定工程样本。每条评分持久化所有原始候选 loglikelihood 和长度，不持久化 target gold；输出以 exclusive-create 写入。完整面板闭合先运行 `verify_phase10_scores.py`，后续 Gate 才可解封 gold 并运行分析。

`phase10_preflight.sbatch` 与 `phase10_accuracy.sbatch` 提供同一 producer 的 launcher；partition/QoS/GPU/time 等资源必须由 B/后续 exact handoff 提供。按2026-10-02用户最新规则，非正式任务统一debug；正式任务使用普通用户合法最高优先级A800。preflight仍<30min，异卡显存/packing不外推。A 不执行模型或提交 launcher。`--engineering-check zero-strength` 支持无干预等价；`--engineering-check k2-policy` 将canonical fixed_t0-K2用lag1路径运行以验证别名。两者仅限PREFLIGHT_ONLY、独立新根、明确engineering身份，不进入formal闭合。launcher通过`PHASE10_ENGINEERING_CHECK`传递。current_t的K2独立检验，不能通过k2-policy别名复用。

Gate B从本地已提交loopscope以git archive部署完整源码到新专属staging/source-commit；旧远端分支只读，不切换或同步。修复用新commit新source。Gate A操作补充仅允许专属staging CPU overlay；数据附件记录实际import来源。

## 分析与迁移

`analyze_phase10.py` 在获准读取完整 outcome 的 Gate 使用。MMLU 以 micro acc 为七个非重复 cell×policy 选 lambda，精确并列选较小值，ARC outcome 前固定。ARC 主迁移 14 比较作为一个 Holm family；每数据集探索扫描 126 比较作为一个 family；策略45比较作为另一个family（每lambda两个K2 current_t−fixed_t0/lag1共享基线，K3三策略两两三比较）。exact 双侧 McNemar、alpha0.05，paired bootstrap10000/seed20261002，MMLU subject 内、ARC题级，nominal95%CI。

分析入口读取`--closure`，通过同一verifier重新闭合raw score roots并按canonical identity对齐，再打开`--gold`的`{identities,gold}`。MMLU含`--include-reuse`；ARC先读取和验证已冻结的`--transfer-lambdas`。目标标签文件由后续有outcome权限的Gate准备，A不创建或读取。

完整扫描与历史选窗属于探索；无 Matched-norm，不作方向特异性机制结论。Online−Native 与 Online−Loop 分别回答总方案和额外干预收益。current_t同时改变干预起点和方向时序，其对比是组合策略消融，不能单独归因于t0或当前方向。

## Gate B 实际预检对象（2026-10-02）

Gate B工程证据见 `.planning/phase10/loopscope_phase10_gate_b_evidence.md` / `loopscope_phase10_gate_b_resources.json`，待planning决定。最终实际GPU源码为 `1cee7a5fc0731361d3d2b4471234426b4d7f10aa`，以完整git archive部署在专属staging；旧远端分支保持只读。32个debug/A40作业全部完成，0.236389 GPUh，最多同时2GPU，无GPU失败/OOM。

CPU输入入口为 `prepare_phase10_preflight.py`：原安全MMLU renderer生成validation1531，Phase10 loader明确识别新validation schema且formal拒绝；ARC直接引用A已接纳pool。选择MMLU `[0,368]`、ARC `[0,32,35,85,89,210]`。`check_phase10_reuse.py` 已核验9历史cell×14042 raw记录，无outcome计算。

`phase10_engineering.sbatch SOURCE PYTHON CHECK_SCRIPT [args...]` 在debug调用 `check_phase10_gpu.py` 或 `measure_phase10_resources.py`。GPU检查在相同requests上对原生HFLM比较完整分数，并以真实模型CUDA residual和actual current_t trajectory验证数值/时序；只允许validation。producer仍用 `phase10_preflight.sbatch`，通过 `PHASE10_PREFLIGHT_INDICES` 固定输入，工程K2 alias通过 `PHASE10_ENGINEERING_CHECK=k2-policy`，分数由 `verify_phase10_scores.py --scope PREFLIGHT_ONLY` 闭合。全部精确submit参数、job ID、env/command_args和source import入口保留在统一run根 `runs/phase10-gate-b-20261002T070214Z`。

资源脚本在validation模板补普通token达到MMLU context3096/continuation1、ARC context1242/continuation46，只作资源证据。先packing1，再带 `--packing1-summary` 同形状实测独立进程packing2，包含加载峰值；原生logits cache的实际forward与候选评分数分别记录。A40 packing2安全但四组聚合吞吐略降，建议该卡packing1；A800配置由后续正式Gate实测，不外推。本预检不混入正式test；关键source/launcher/runtime改动后重做debug。
