# LoopScope Qwen3-1.7B 第一阶段全局计划与控制

最后更新：2026-07-11

## 1. 文件职责

本文件是 LoopScope 第一阶段唯一的可变全局计划和 Gate 控制面。它回答：项目当前处于哪一 Gate、哪个 commit 已通过审计、哪个线程被授权执行、下一步需要什么授权。

信息源分工：

- `../loopscope-tflt/AGENTS.md`：长期稳定的实验边界、工程规则和 Gate 验收标准。
- 本文件：当前计划、Gate 状态、线程分配、审计决策和下一步授权条件。
- `../loopscope-tflt/docs/loopscope_phase1.md`：具体命令、工件和运行说明。
- 旧工作区 `.planning/`、历史 handoff 和线程聊天：只读历史档案，不再作为当前状态的权威来源。
- Codex 原生 plan/goal：当前线程内的临时步骤。
- Codex memory：辅助回忆，不作为 commit、实验结果或 Gate 决策的证据。

更新规则：只在 Gate 决策、线程分配、已审计 commit、实验配方或授权边界变化时更新。禁止记录逐次工具调用流水账。

## 2. 全局目标与冻结范围

目标：在不训练、不改模型参数的前提下，验证内部信号能否预测 Qwen3-1.7B 在 MMLU 5-shot 上不同宽度为 4 的 loop window 的真实收益。

第一阶段冻结配方：

```text
model=Qwen/Qwen3-1.7B-Base
model_revision=ea980cb0a6c2ae4b936e82123acc929f1cec04c1
task=mmlu
num_fewshot=5
dtype=float16
k=2
iteration_mode=block
strategy=damped_euler
alpha=1.0
beta=0.0
cache_strategy=last
decode_mode=bypass
window_width=4
```

第一阶段不扩展到其他模型、K/alpha/beta/cache/decode 消融、训练型 probe、逐输入动态 window 或使用测试标签选窗。

## 3. 当前权威状态

```text
former planning/audit thread=019f4c7b-e5eb-77f2-b007-59d004896550; historical read-only source after handoff
planning/audit thread=019f5035-590c-74e0-bebd-015ccead906d
planning handoff=user-authorized successor effective 2026-07-11; all future planning, audit decisions and post-Gate-E governance belong to successor thread
immutable run planning_thread_id=019f4c7b-e5eb-77f2-b007-59d004896550; preserve in existing manifest/approvals; do not rewrite during successor handoff
current gate=post-Gate-E workspace governance PASS; project idle pending a new user/planning decision
last completed gate=post-Gate-E workspace governance PASS at loopscope@2c3acb34c07362bf82c0f7a0af59e0d435ab44cd; Phase 1 experiment PASS remains closed
post-Gate-E workspace governance executor=019f5132-cd06-7281-be3b-a01a06f7a757; responsibility closed after planning audit PASS; historical evidence only
post-Gate-E workspace governance status=PASS; successor planning thread independently verified two single-purpose commits fb17d5afe1b5310e7ec857fbc93c6cc7bb7bd2cb and 2c3acb34c07362bf82c0f7a0af59e0d435ab44cd, clean local/origin/HPC2 Git, protected reproduction main@3dae7586294789892b22cadb7e69592f294882c2, local and remote 143 tests, seven empty 0755 project workspace leaves, unchanged unique legacy symlink, compatible unchanged versioned venv/lock, runtime uv-module removal, and unchanged Phase 1 canonical manifest/freeze/analysis hashes
next admission condition=user and planning/audit thread must define and explicitly authorize a new scientific or engineering Gate in a new executor; no experiment, migration, cleanup, or follow-on run is currently authorized
audited branch=loopscope
local B0 HEAD=99250a49763e3b8620b0f4b731d77c73aa9c9e07
origin/loopscope=e9d5a30c5cd9fae5e497432d8e4fc64f801656c8
local ahead of origin=0
local original reproduction=clean main@4f59bd93eca4da3cbf458a93508f91c5b23912bc
Gate B execution thread=019f4d58-42a4-73b2-9b5d-6c64ac68f43b
B0 baseline=3e2dcfa46dd825c6ad0f8d4b638fb034cbe45875
B0 status=PASS at 99250a49763e3b8620b0f4b731d77c73aa9c9e07; 118 tests PASS
Gate B status=PASS; closed; local/origin/HPC2 dedicated clone all at 99250a49763e3b8620b0f4b731d77c73aa9c9e07
Gate B venv=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711
Gate B lock_sha256=f9f8c780f602cd9ae641be56cd5206779a253fc6ace948fce03d8eaf9654a9d2
Gate B validation=118 tests PASS; imports/versions/compile/shell/help PASS
remote original reproduction=clean main@3dae7586294789892b22cadb7e69592f294882c2 before and after
Gate C execution thread=019f4d7c-ab2d-7c61-a365-41f974a6c2cb
Gate C authorization=authorized by user for full Gate C execution under the frozen phase-one contract
Gate C first attempt=BLOCK before immutable inputs/run root/Slurm; lm-eval TaskConfig adapter defect confirmed
Gate C repair-1=accepted at acf385cd19203cb520c038195ad1c07f7df52447; TaskConfig adapter fixed; 121 tests PASS
Gate C second blocker=per-subject task datasets lack auxiliary_train; exact dataset topology exposes validation/dev/test per subject and auxiliary_train only under aggregate all
Gate C repair-2=accepted at 161ebefb06533ba5182aa95cd725348ca4734acd; target_split=validation frozen; 125 tests PASS
Gate C third blocker=global basename rglob finds 11 lm-eval task-family YAMLs; actual TaskManager.task_index provides one authoritative yaml_path for each loaded task
Gate C repair-3=accepted at 37e34c10c9cac5a93ff773fce960346790a5b33b; TaskManager registry YAML provenance bound; 128 tests PASS
Gate C autonomous repair-1=accepted at ea8764eb3e5d5576efb3d4f3d8e2d6a75007d2fb; duplicate dev rows receive distinct source occurrence IDs; 129 tests PASS
Gate C autonomous repair-2=accepted at 5d5899eacaedab8e79275b95277c3b2fb5d91669; JSONL LF record boundary fixed; 130 tests PASS
Gate C fourth blocker=submit wrapper invokes prepare helper with ambient python3 and no repository src PYTHONPATH; deterministic ModuleNotFoundError before attempt/receipt/sbatch
Gate C repair-4=accepted at 24bd49cf1d7cb2b10984f6e9f814be037024e2fc; repository-local helper imports fixed; 132 tests PASS
Gate C first submitted array=job 9963296 submitted exactly once; task 0 FAILED effective rank zero matrix; tasks 1 and 2 COMPLETED; immutable and never retry
Gate C metric decision=freeze gram_spectrum_shannon_effective_rank v2: exact zero centered spectrum maps to effective_rank=0.0 with explicit zero_centered_spectrum=true and centered_spectrum_mass=0.0; every positive spectrum retains the squared-singular-value entropy formula
Gate C metric repair=accepted at e9d5a30c5cd9fae5e497432d8e4fc64f801656c8; probe schema v3; 142 tests PASS
Gate C successful array=job 9963382; all three tasks COMPLETED 0:0; strict artifacts and 3-observation revision closure independently verified
Gate C audited run=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-20260711-053022
Gate C run_manifest_sha256=e497983d6b47ba2ff8e296a5375838737ad666b5f7dc95b1f366a8bd21111750
Gate C status=PASS; closed; execution thread 019f4d7c-ab2d-7c61-a365-41f974a6c2cb must stop and never enter Gate D
Gate D execution thread=019f4f46-1af8-7100-a6ab-54e9dc9556fa
Gate D authorization=authorized for the existing audited run only; gate-d-limit baseline plus 10 frozen windows at limit=5; no scientific ranking and no Gate E
Gate D first submission=sbatch rejected before job creation with QOSMaxSubmitJobPerUserLimit; immutable attempt/receipt retained; no retry/job/claim/output
Gate D QOS audit=debug MaxSubmitJobsPU=8 and MaxJobsPU=10; user queue empty; original 11-element array can never pass admission unchanged
Gate D replacement=jobs 9964000 indices 0-5 and 9964017 indices 6-10; all 11 tasks COMPLETED 0:0; sample identity aligned; 12-observation revision closure match
Gate D status=PASS; closed; execution thread 019f4f46-1af8-7100-a6ab-54e9dc9556fa must stop and never enter Gate E
Gate E former execution thread=019f4f94-378a-7672-acc1-ad5087fe85b4; unresponsive; all write/SSH/Slurm/timer/retry/analysis authority revoked by user
Gate E current execution thread=019f5031-90dc-7eb0-8659-a53ec7761079
Gate E current thread title=execute-LoopScope 1.7B 第一阶段-Gate E 修复收尾
Gate E authorization=one completely offline write-once repair for missing indices9/10/12 only, exclusive results promotion, then full revision closure and one paired analysis; existing 11 results must not be rerun
Gate E retry throttle authorization=job 9964801 may receive a live array throttle adjustment by the current executor only, with an explicit upper bound of %8; this does not authorize a second submission, retry, requeue, resubmit, task remap, or scientific/config/artifact change
Gate E final audit status=PASS; successor planning thread independently verified sender, clean local/origin/HPC2 dedicated Git, protected original reproduction Git, empty Slurm queue, historical jobs, unique offline retry 9964801/02/03 COMPLETED/0:0, all supplemental self-hashes, 14 outputs each with 57 tasks and 14042 samples, ordered identity SHA256 872c9f6bd7f40e3c6fa40b13cc862033379ac0a9cbc439a1593653945f3d521d, three exclusive source-identical promotions, 18-observation revision closure at ea980cb0a6c2ae4b936e82123acc929f1cec04c1, and one full 13-window paired analysis; audit harness returned AUDIT_OK
Gate E scheduler audit=manifest GPU runners request debug/24h but debug MaxTime=30m; user final choice is unified A40 probe/full on partition/QOS emergency_gpua40 with PriorityTier/Factor=300 and MaxTime=7d
Gate E submitted probe=A800 job 9964116; user explicitly authorizes one cancellation only if every element remains unstarted/PENDING with no claim/output; otherwise BLOCK without replacement
Gate E scheduler override=after safe cancellation, one A40 replacement probe array 0-1%2; full array 0-13%2; partition/qos/gres/time only; task/science/cpus/memory/output unchanged; score remains original debug CPU job
Gate E completed=A800 probe 9964116 safely cancelled unstarted; A40 probe 9964377 PASS; score 9964401 PASS; score freeze canonical SHA256 4b1ac9b081e86eb774cb233fb67203349ae882b08ea45f01c22901112118998a PASS
Gate E full blocker=job 9964413 task0 baseline FAILED before science on gpu3-13 because uv modulefile missing; baseline claim exists, output absent; tasks1/2 initially RUNNING on gpu3-6; indices3..13 initially PENDING
Gate E containment=preserve running tasks1/2; cancel only still-unstarted indices3..13; any cancellation race BLOCK
Gate E repair authorization=conditional on tasks1/2 PASS and clean containment: one baseline infrastructure replacement on gpu3-6 with exact manifest argv and separate write-once retry claim, followed serially by one original-runner array 3-13%2 on gpu3-6; no repo/manifest/freeze mutation and no further retry
Gate E throttle override=user explicitly authorized changing live remaining array 9964609 from %2 to %4; scheduler-only, scientific task mapping and argv unchanged; this is not an unauthorized deviation
Gate E full current=baseline repair 9964590 PASS; original indices1/2 PASS; remaining 9964609 indices3,4,5,6,7,8,11,13 PASS; indices9,10,12 FAILED HTTP 429 from hf-mirror dataset tree; 11/14 outputs complete; queue empty; full revision closure and paired analysis not run
remaining-gates workflow=event-driven; one execution thread per complete Gate; no planning-thread polling
```

Gate A 已于 2026-07-11 由规划/审计线程只读复审并判定 `PASS`。Gate A 执行线程 `019f4c7d-e579-7891-8753-f0b5c0a25128` 已结束职责，不得继续执行 Gate B。

旧规划/审计线程已依次对 B0、Gate B、Gate C 和 Gate D 完成独立只读复审并判定 `PASS`。用户随后将所有后续计划、Gate E 最终审计和 post-Gate-E workspace 治理移交给 successor 规划/审计线程 `019f5035-590c-74e0-bebd-015ccead906d`。当前只授权 Gate E 新执行线程 `019f5031-90dc-7eb0-8659-a53ec7761079` 完成既定的 HTTP 429 三结果离线修复、revision closure 与 paired analysis；旧规划线程及其他历史执行线程均不得继续操作。

## 4. Gate 总表

| Gate | 目标 | 执行线程 | 状态 | 进入下一 Gate 的条件 |
|---|---|---|---|---|
| A | 本地实现、测试、revision/freeze 闭环 | `019f4c7d-e579-7891-8753-f0b5c0a25128` | `PASS` at `3e2dcfa` | 已满足；该线程停止 |
| B0 | 本地治理迁移与审批契约加固 | `019f4d58-42a4-73b2-9b5d-6c64ac68f43b` | `PASS` at `99250a4` | 已完成并关闭 |
| B | Git 同步、HPC2 专用 clone、固定环境、CPU/import/unit/help smoke | `019f4d58-42a4-73b2-9b5d-6c64ac68f43b` | `PASS` at `99250a4` | 已完成并关闭；该线程停止 |
| C | 1.7B GPU probe 与 loop-effect audit | `019f4d7c-ab2d-7c61-a365-41f974a6c2cb` | `PASS at e9d5a30; job 9963382` | 已满足；job 9963296 永久不得重试；Gate C 线程停止 |
| D | baseline 与全部候选 window 的 `--limit 5` 工程 smoke | `019f4f46-1af8-7100-a6ab-54e9dc9556fa` | `PASS; jobs 9964000 + 9964017` | 已满足；11 jobs/identity/revision closure 通过；limit accuracy 未用于选窗；Gate D 线程停止 |
| E | full probe、冻结评分与正式候选网格 | `019f5031-90dc-7eb0-8659-a53ec7761079` | `PASS; Phase 1 complete` | 已满足；Gate E executor 职责关闭，不得进入 workspace governance |

剩余 Gate 采用固定事件驱动协议：每个 Gate 使用独立的 `gpt-5.6-sol / high` 执行线程；Gate 内连续执行并可谨慎使用 subagent；结束时只向规划线程发送一次结构化最终审计请求。规划线程平时停止等待、不轮询；`PASS` 后制定并分发下一 Gate，`PASS_WITH_FIXES/BLOCK` 时把修复计划发回原执行线程。只有规划/审计线程可给出 `PASS / PASS_WITH_FIXES / BLOCK`。

### Gate 执行线程有限自主修复协议

为减少工程适配问题在执行线程与规划线程之间反复往返，Gate 执行采用“事件驱动 + 有界自主修复”的混合模式：

1. Gate 执行线程可自行诊断并修复低风险工程问题，包括单一根因的运行时 adapter、兼容性、路径解析或 provenance 绑定缺陷；前提是修复不改变冻结的科学实验契约、受保护代码、审批语义或外部授权边界。
2. 每次 Gate 授权默认包含最多两轮低风险自主修复。每轮必须先保留可复现失败证据或 red test；一个 commit 只处理一个根因；随后完成全量本地测试、适用的远端 CPU 回归，并在全新 write-once staging 中重跑原失败命令。
3. 已在该 Gate 获授权的普通 fast-forward push、fetch 与 `merge --ff-only` 可作为修复闭环的一部分。禁止覆盖或复用证据路径、重写 Git 历史、删除重建异常远端状态，亦禁止自动重试已经提交的 Slurm/GPU job。
4. 下列情况必须立即停止并通知规划/审计线程，不得自行选择方案：可能改变 model、dataset/split、task、样本数、seed、window 集合、评分/选择标准或其他实验语义；修改 manifest/schema/approval 契约或受保护文件；扩大网络、模型、数据或外部系统权限；需要破坏性操作；存在多个科学上合理的修复方向。
5. 当 Gate 完成、触及上述升级边界，或两轮自主修复额度耗尽时，执行线程只发送一次结构化终态消息。规划线程不轮询，收到消息后才进行一次新的只读审计。
6. 执行计划与 handoff 使用中文。执行线程可保守使用 subagent，最多同时四个且无需用满；主执行线程始终独占共享工作树写入整合、Git、SSH/HPC2、Slurm、实验状态和最终证据责任。
7. 任何导致执行线程当前无法继续的条件，都必须在暂停、结束 turn 或退出前主动向规划/审计线程发送一次结构化 `BLOCK`；不得静默中断，也不得依赖用户人工转述。强制报告范围包括工具/usage limit、审批或 reviewer 拒绝、SSH/HPC2/Slurm 不可用、host-key/权限失败、意外既有文件、Git dirty/mismatch、scheduler/QOS 拒绝、job 失败、工件缺失、多个合理修复方向或科学/授权边界。该消息属于唤醒规划线程的必要终态事件，不属于禁止的小里程碑。若 `send_message_to_thread` 本身不可用，必须在执行线程自己的 final response 中输出同样的 `decision requested: GATE_X_FINAL_AUDIT` 与 `result: BLOCK`，并明确已完成状态和未执行动作；不得无声停止。
8. GPU/Slurm 长任务等待采用执行线程自有的定时器/heartbeat，不保持活跃 turn 持续轮询。每次定时唤醒只做一次有界只读检查；仍为 PENDING/RUNNING 且无异常时使用 `DONT_NOTIFY` 并结束本次唤醒，终态成功时自动推进下一阶段，终态失败时停用监控并主动回传结构化 `BLOCK`。执行线程等待期间显示 `idle` 是正常状态，但必须能够证明定时器已真实启用；规划/审计线程不参与定时轮询。

判例：修复 lm-eval `TaskConfig` 取值或把已加载 task 绑定到 `TaskManager` 的权威 YAML 属于可自主处理的 adapter/provenance 工程问题；把 `auxiliary_train` 改为逐 subject 的 `validation` 会改变数据 split 契约，必须由规划/审计线程批准。

## 5. Gate A 已接受实现

审计接受的本地提交范围为 `origin/loopscope@836de84..3e2dcfa`，共 10 个提交。关键闭环包括：

- LoopScope schema、指标、候选 grid、layer/window probe、评分与 paired analysis。
- N+1 boundary-state window 语义和平方奇异值谱 effective rank。
- `lm-eval==0.4.11` MMLU renderer provenance 与 probe-pool 防篡改。
- model/tokenizer/manifest 精确 revision closure；tokenizer commit 只从 authoritative resolved artifact 提取。
- Gate E full-probe freeze 在 attempt/`sbatch` 前验证。
- 根目录 `AGENTS.md` 是唯一执行规则文件。

Gate A 最终独立验证：targeted 23 tests 与 full 110 tests 通过；shell/CLI/diff 检查通过；专用 clone 与原复现 checkout 均 clean。

## 6. Gate B 已完成范围

Gate B 由固定执行线程 `019f4d58-42a4-73b2-9b5d-6c64ac68f43b` 完成，授权范围为：

1. 仅以普通 fast-forward 将 `loopscope@99250a4` 推送/同步到允许的远端目标。
2. 在 HPC2 创建或核对专用 LoopScope clone，不改固定复现 checkout。
3. 使用锁定依赖建立独立 versioned venv。
4. 只运行 CPU/import/unit/compile/CLI help 和最小 preflight；不运行 GPU probe、MMLU eval 或 Slurm 实验。
5. 回传 branch/commit/dirty、依赖版本、命令、exit code、远端路径和保护证明，然后停止等待复审。

任何 Git/SSH/依赖/版本/测试条件失败均立即 `BLOCK`；不得改变路径、复用环境或进入 Gate C。

## 7. Gate 决策记录

| 日期 | Gate | 决策 | 审计对象 | 说明 |
|---|---|---|---|---|
| 2026-07-11 | A | `PASS` | `loopscope@3e2dcfa` | 110 tests；R1-R4、tokenizer resolver 和 pre-submit freeze 闭环通过；未执行远端工作 |
| 2026-07-11 | B0 | `PASS` | `loopscope@99250a4` | 两个指定提交；本地 118 tests 与审批契约正负测试通过；规划线程独立复审通过 |
| 2026-07-11 | B | `PASS` | local/origin/HPC2 `loopscope@99250a4` | 新建独立 venv；远端 118 tests、版本/import/compile/shell/help 全部通过；原复现受保护；未进入 Gate C |
| 2026-07-11 | C | `BLOCK` | `loopscope@99250a4` renderer preflight | lm-eval 0.4.11 `TaskConfig` 的 Mapping `.get()` 返回 `None` 而属性已填充；当前 `_config_value` 错误优先 Mapping，未创建输入工件、run root 或 Slurm job |
| 2026-07-11 | C repair-1 | `accepted; Gate C remains BLOCK` | `loopscope@acf385c` | TaskConfig adapter 最小修复与 121 tests 通过；真实 exporter 随后证明 per-subject dataset 不含 `auxiliary_train` |
| 2026-07-11 | C repair-2 | `accepted; Gate C remains BLOCK` | `loopscope@161ebef` | validation/dev/test 契约冻结与 125 tests 通过；真实 exporter 随后证明 basename YAML 搜索无法唯一绑定实际 task family |
| 2026-07-11 | C repair-3 | `accepted; Gate C continued` | `loopscope@37e34c1` | TaskManager 公共 registry 唯一绑定实际 `mmlu/default` YAML；真实 renderer dry-run 继续暴露重复 dev 行 provenance ID 问题 |
| 2026-07-11 | C autonomous repair-1 | `accepted; Gate C continued` | `loopscope@ea8764e` | 按出现次序消费重复 dev 行 source index；内容与 split 契约不变；129 tests 通过 |
| 2026-07-11 | C autonomous repair-2 | `accepted; Gate C remains BLOCK` | `loopscope@5d5899e` | JSONL 严格按 LF 分隔，避免 U+0085 被 `splitlines()` 误切；130 tests 通过；随后 submit dry-run 暴露 ambient Python import 缺陷 |
| 2026-07-11 | C repair-4 audit | `BLOCK; narrow repair authorized` | `loopscope@5d5899e` | `prepare_qwen17_phase1.py` 在无 `PYTHONPATH` 的 ambient Python 下确定性缺少 `tflt`；失败位于 attempt/receipt/`sbatch` 前，不涉及科学契约或实验重试 |
| 2026-07-11 | C repair-4 | `accepted; Gate C submitted once` | `loopscope@24bd49c` | submit helper 使用 repository-local `src`；132 tests 通过；新 run dry-run 通过并仅提交一次 array job `9963296` |
| 2026-07-11 | C job 9963296 | `BLOCK; metric contract decision required` | task 0 failed; tasks 1/2 completed | centered all-identical answer-position matrix产生严格零谱；旧 v1 对零矩阵报错；未重试、未进入 Gate D |
| 2026-07-11 | C effective-rank v2 decision | `repair authorized` | new commit/run required | exact zero centered spectrum 显式扩展为 `effective_rank=0.0`，同时记录 zero flag 与 spectrum mass；正谱公式不变；辅助指标不得改变主 ranking |
| 2026-07-11 | C effective-rank v2 repair | `accepted` | `loopscope@e9d5a30` | estimator v2、probe schema v3 与 zero-spectrum diagnostics 闭环；本地/远端 142 tests 通过；主 ranking 未变 |
| 2026-07-11 | C final | `PASS` | run `20260711-053022`, job `9963382` | 三 task 均 COMPLETED/0:0；B_0=0/true/0，其余正谱；anchor loop effective、calls=2、restore=true；sentinels 全 valid；revision closure match；无 Gate D 行为 |
| 2026-07-11 | D first admission | `BLOCK; no job created` | existing run, `gate-d-limit` | `debug` runner `0-10%2` 被 `QOSMaxSubmitJobPerUserLimit` 拒绝；attempt/receipt 唯一且保留；无 task/claim/output、无 Gate E |
| 2026-07-11 | D QOS audit | `split replacement authorized` | debug QOS | 规划线程只读确认 user queue 为空而 `MaxSubmitJobsPU=8`；11 元素 array 结构性超限；保持同一 task scripts，顺序提交 `0-5%2` 与 `6-10%2`，每批一次且不得重试 |
| 2026-07-11 | D final | `PASS` | jobs `9964000`, `9964017` | 11 tasks 全 COMPLETED/0:0；57 subtasks × 5 samples 对齐；identity一致；12 observations revision match；无 accuracy 排名、无 Gate E |
| 2026-07-11 | E scheduler pre-audit | `long-A40 override superseded before submission` | existing immutable run | 最初识别 i64m1tga40u；用户随后明确允许更高级 GPU 与账户最高调度层级；若旧 authorization 已落盘则只读保留并由新 authorization 显式 supersede |
| 2026-07-11 | E scheduler A800 authorization | `superseded after probe submission` | existing immutable run | 只读核验未发现 A100；曾授权 emergency_gpu/A800，probe job `9964116` 已提交；随后用户要求 probe/full 统一 A40，并明确允许取消该 job |
| 2026-07-11 | E scheduler final authorization | `unified highest-tier A40 authorized` | existing immutable run | 仅当 `9964116` 所有元素仍 PENDING、从未运行且无 claim/output 时，允许一次取消并一次 A40 replacement；probe/full 使用 emergency_gpua40、gpu:a40:1、24h；并发严格保持冻结的 `%2`；若发生启动竞态则 BLOCK |
| 2026-07-11 | E probe/score/freeze | `PASS` | jobs `9964377`, `9964401` | A800 probe `9964116` 安全取消且从未运行；A40 full probes、4-observation revision closure、label-free score 与 write-once freeze 均通过 |
| 2026-07-11 | E full first array | `BLOCK; containment ordered` | job `9964413` | task0 在 gpu3-13 科学命令前因缺少 `uv` modulefile FAILED/1:0；task1/2 在 gpu3-6 正常运行；仅取消仍未启动的 3..13，保留全部证据 |
| 2026-07-11 | E full uv-node repair | `one bounded replacement authorized` | existing immutable run | 条件为 task1/2 PASS 且 3..13 安全取消；先在 gpu3-6 补唯一 baseline，再串行提交原 runner `3-13%2` 到 gpu3-6；原 claim/attempt/receipt 永久保留，不修改科学 argv/manifest/freeze，任一步失败不得再试 |
| 2026-07-11 | E remaining throttle | `user-authorized scheduler adjustment` | job `9964609` | 用户明确授权将 live array throttle 从 `%2` 调整为 `%4`；不得把该授权定性为违规或虚构；task mapping、单任务资源和科学 argv 未变 |
| 2026-07-11 | E remaining terminal | `BLOCK; 11/14 outputs complete` | jobs `9964413`, `9964590`, `9964609` | baseline 与 indices 1..8、11、13 成功；indices 9/10/12 在 dataset initialization 访问 hf-mirror 时 HTTP 429 FAILED/1:0；队列已空，revision closure/paired analysis 未执行 |
| 2026-07-11 | E executor replacement | `new executor assigned` | `019f5031-90dc-7eb0-8659-a53ec7761079` | 旧线程 `019f4f94-378a-7672-acc1-ad5087fe85b4` 因持续无响应被撤销全部执行权限；新线程只负责完全离线补齐三项缺失结果、revision closure 与 paired analysis |
| 2026-07-11 | planning/audit handoff | `successor assigned` | `019f5035-590c-74e0-bebd-015ccead906d` | 用户将后续计划、最终 Gate E 审计与 workspace 治理从旧线程 `019f4c7b-e5eb-77f2-b007-59d004896550` 移交；existing run 的旧 planning_thread_id 绑定保持不可变 |
| 2026-07-11 | E retry live throttle | `authorized up to %8` | job `9964801` | 当前执行线程可仅对既有 retry array 做 live throttle 调整；不得新提交、重试、requeue、resubmit、改 task mapping、科学参数或工件契约；该 array 仅有 3 个 task，实际并发仍受 task 数限制 |
| 2026-07-11 | E final audit attempt | `BLOCK; planning audit infrastructure` | executor packet + live Git/Slurm/artifacts | sender、clean commits、空队列、历史 jobs 与 `9964801/02/03 COMPLETED/0:0`、唯一 retry/限流/promotion 工件链均已只读核对一致；14 份 results 的全量断言脚本在远端执行前因 Codex usage limit 被拒绝，未产生远端副作用；不要求 executor 修复，恢复工具后继续同一次审计，当前不得宣告 Gate E/Phase 1 PASS |
| 2026-07-11 | E final audit resumed | `PASS; Phase 1 complete` | run `20260711-053022`; jobs `9964377`, `9964401`, `9964413`, `9964590`, `9964609`, `9964801` | successor 规划线程独立重读 live Git/Slurm/工件；全量只读 harness `AUDIT_OK`：14/14 outputs、57 tasks/14042 samples、ordered identity、exclusive promotion、18-observation revision closure、13-window full paired analysis 与 supplemental 自哈希全部通过；旧 executor 关闭，workspace governance 需新线程 |
| 2026-07-11 | post-E governance admission | `authorized; executor assigned` | `019f5132-cd06-7281-be3b-a01a06f7a757` | Codex thread handlers 恢复后已在 LoopScope local project 创建新的独立执行线程并发送完整 handoff；旧 Gate E executor 不复用；规划线程不轮询 |
| 2026-07-11 | post-E governance final | `PASS; executor closed` | `loopscope@fb17d5a..2c3acb3`; executor `019f5132-cd06-7281-be3b-a01a06f7a757` | 规划线程独立复审两项单一目的提交、local/origin/HPC2 clean sync、本地/远端 143 tests、7 个空目录、legacy symlink、venv/lock、runtime uv guard 与 Phase 1 canonical hashes；原复现和历史 workspace 未改；未授权下一轮实验 |

## 8. 当前执行安排

B0、Gate B 与 Gate C 均已由规划/审计线程判定 `PASS` 并关闭。Gate C 的最终已审计基线为 clean `loopscope@e9d5a30`，成功 run 为 `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-20260711-053022`，array job `9963382` 的三项任务全部 `COMPLETED/0:0`；layer probe、anchor audit、sentinel probe 与三方 revision closure 均由规划线程重新读取验证。旧失败 job `9963296` 及其 run 永久保留且不得重试。Gate C 执行线程职责结束，不得继续操作。

Gate D 已由规划/审计线程判定 `PASS` 并关闭。QOS split jobs `9964000` 和 `9964017` 的 11 个 tasks 均 `COMPLETED/0:0`，索引 0..10 无重复/缺口；规划线程重新解析 11 组 command/results/revision 工件，确认 57 subtasks、每项 5 samples、ordered identity 完全一致，12-observation revision closure match，且没有读取/排序 limit accuracy。原 11-array admission failure 与 supplemental authorization/attempt/receipt 全部保留。Gate D 执行线程职责结束，不得继续操作。

Gate E 的原执行线程 `019f4f94-378a-7672-acc1-ad5087fe85b4` 已因持续无响应被用户撤销全部执行权限，只保留为历史证据。A800 probe `9964116` 已按用户特批在未运行、无工件状态安全取消；A40 replacement probe `9964377`、score job `9964401`、4-observation revision closure 与 write-once score freeze 均已通过。首次 full array `9964413` 的 task0 baseline 在 gpu3-13 科学命令启动前因缺少 `uv` modulefile 失败；task1/2 在 gpu3-6 成功。一次性 baseline repair `9964590` 已成功。remaining array `9964609` 最初以 `3-13%2` 提交，随后按用户明确授权将 live throttle 调整为 `%4`；该调整合法且不改变科学 argv。最终 indices 3、4、5、6、7、8、11、13 成功，indices 9、10、12 在 dataset initialization 访问 `hf-mirror.com` 时收到 HTTP 429 并 FAILED/1:0。当前队列为空，14 个 full outputs 中 11 个完整，缺少 window 20:23、22:25、14:17；不得运行不完整的 full revision closure 或 paired analysis。

当前唯一 Gate E 执行线程为 `019f5031-90dc-7eb0-8659-a53ec7761079`，标题 `execute-LoopScope 1.7B 第一阶段-Gate E 修复收尾`，配置 `gpt-5.6-sol / high`。新线程不得重做 11 个成功结果；只获授权在全新 retry staging 中以 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1` 精确补跑 indices 9/10/12，保留原 partial dirs 与失败证据。三项 retry 全部成功并与已有样本 identity/revision 对齐后，只允许通过 write-once promotion authorization 将缺失的 `results.json` 以 exclusive create 补入 canonical dirs；不得覆盖原 command_args/env/model_revision，且 supplemental provenance 必须明确实际 retry producer。若现有 validator 无法在不伪造或覆盖 provenance 的情况下接受该方案，必须 `BLOCK`。14/14 完整后才可执行 standard + supplemental revision closure 和一次 paired analysis；任何 repair 失败不得再试。该线程的终态 `GATE_E_FINAL_AUDIT` 或 `BLOCK` 必须发送给 successor 规划/审计线程 `019f5035-590c-74e0-bebd-015ccead906d`；existing run 内旧 `planning_thread_id` 仍保持不可变。

## 9. Gate E 最终审计后的 workspace 治理

用户已于 2026-07-11 确认：只有 Gate E 完整结束并由 successor 规划/审计线程 `019f5035-590c-74e0-bebd-015ccead906d` 给出最终 `PASS` 后，才由该 successor 线程负责 workspace 文件结构与相关默认路径调整。Gate E 执行线程和旧规划线程均不得提前进入该治理工作。

目标结构为项目级兄弟目录：

```text
/hpc2hdd/home/xhuang225/workspaces/
├── training_free_looped_transformers_reproduction/
│   ├── runs/
│   ├── logs/
│   └── artifacts/
└── training_free_looped_transformers_loopscope/
    ├── inputs/
    ├── runs/
    ├── staging/
    └── artifacts/
```

约束：

1. 当前 Phase 1 canonical run `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-20260711-053022` 及其全部历史 siblings 永久保留原位；不得移动、改名、复制后替换或通过新 symlink 冒充原路径，因为 manifest、runner、freeze 与分析均绑定绝对路径。
2. 新的项目级根目录只作为后续新 run/input/artifact 的默认位置；不追溯改写已有 provenance。
3. `/hpc2hdd/home/xhuang225/shared/{hf_home,datasets,uv}` 继续作为跨项目共享 cache，不复制到项目 workspace。
4. `runs.legacy-project-symlink-20260701-031953` 在任何调整前必须先只读核验类型、目标、所有者和引用；不得自动删除、覆盖或改链。
5. repo 默认路径、文档、测试和 `AGENTS.md` 必须同步更新并完成本地/远端回归；路径治理与移除运行时 `uv` module 依赖分别使用单一目的提交。
6. Gate E 最终审计前，本节仅为已授权的后续任务，不构成当前远端文件写入或迁移许可。
7. Gate E 最终审计 `PASS` 后，LoopScope 后续新产生的证据/工件、staging 与 runs 必须落在 `training_free_looped_transformers_loopscope` 独立 workspace，不得再与原复现项目的对应输出目录混用；LoopScope 专属 input 如需新建也放入该独立 workspace，共享且已有的输入/缓存不为目录整齐而复制。
8. Python venv、wheel/cache 与兼容依赖不要求随 workspace 重复建设：在 lock、Python/CUDA/torch ABI、包版本和只读复用边界一致时优先复用既有环境（包括当前 LoopScope 固定 venv，或经审计兼容的原复现环境）；不得原地升级、修改或污染原复现 venv。若不兼容，另建 versioned venv，并记录选择依据与 provenance。
