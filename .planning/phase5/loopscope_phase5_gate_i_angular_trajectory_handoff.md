# GATE_I_HANDOFF

```text
Project/phase=LoopScope Phase 5 post-terminal diagnostic extension
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019fa464-6734-7ec0-a57a-e07a33223331
Authorized executor title=execute-LoopScope-Angular-Trajectory-第5阶段-Gate I
Gate=I
Objective=fresh native-no-loop adjacent raw-residual angular-distance acquisition for exact 1.7B and 4B validation-1531 cells, aggregate/Kneedle/verifier/dual-model figure
Decision requested=GATE_I_FINAL_AUDIT or BLOCK
Terminal recipient=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
```

## 1. Admission

执行前必须重读：

1. `loopscope-tflt/AGENTS.md`
2. `.planning/loopscope_phase5_control.md`
3. `.planning/loopscope_phase5_gate_i_angular_trajectory_contract.md`
4. 本 handoff
5. `configs/loopscope/phase3_card.json`
6. `configs/loopscope/phase5_card.json`
7. Phase 3/5 acquisition code与 Phase 5 figure contract
8. `hpc2-hkustgz-ssh` skill

只有以下状态同时成立才 admission：

```text
repo=/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
branch=loopscope
HEAD=97de998504a9bad9867688f307a8bfc6fffdfb81
protected_base=4f59bd93eca4da3cbf458a93508f91c5b23912bc is ancestor
dirty=only the two pre-existing untracked files under loopscope-tflt/报告/
origin/loopscope=f33a37a4e5c0cfc461184a3471277897c77cf83b
control ACTIVE_GATE=I
control AUTHORIZED_EXECUTOR=019fa464-6734-7ec0-a57a-e07a33223331
run root does not already exist
```

两份既有 untracked report 必须保留，禁止移动、覆盖、提交或清理：

```text
loopscope-tflt/报告/LoopScope_第五阶段相对双相反转双模型离线重选窗.md
SHA256=1376222e3abeb12dcd72bbb6fa23f3d90c1677b1d930d1744b61fb056ee6e113
loopscope-tflt/报告/LoopScope_第五阶段多宽度Top4全量Outcome.md
SHA256=e6827ef0386b5f74a4d8bbf49507c0501cfc318176f67c884c3f1a236bda47c3
```

任一 binding、branch、base、dirty 或 run-root 条件不符，先 `BLOCK`，不得修正历史状态。

## 2. Frozen science

完整科学合同以
`.planning/loopscope_phase5_gate_i_angular_trajectory_contract.md` 为唯一数学权威。
关键点：

- exact Qwen3-1.7B-Base@`ea980...` float16，B0...B28；
- exact Qwen3-4B-Base@`906bfd...` bfloat16，B0...B36；
- exact MMLU validation 1,531 / 57，5-shot dev renderer；
- final non-padding pre-answer token；
- native no-loop、one forward/identity、`use_cache=false`、`output_hidden_states=true`；
- raw residual before FinalNorm；
- primary metric=`acos(clamp(cos(raw B_l,raw B_(l+1))))/pi`；
- no gold/test/outcome/accuracy；
- Kneedle 只是 descriptive，合法 `NO_STABLE_KNEE` 不阻塞 Gate。

特别禁止直接复用现有 `decoder_boundary_states(outputs.hidden_states,L)` 作为 raw
`B_0...B_L`：其 `B_L` 是 FinalNorm 后状态。本 Gate 必须按合同捕获 FinalNorm 输入并闭合。

## 3. Allowed implementation and local outputs

允许新增/修改仅限：

```text
configs/loopscope/phase5_angular_trajectory_gate_i_card.json
src/tflt/loopscope/phase5_angular_trajectory.py
scripts/loopscope/run_qwen_phase5_gate_i.py
tests/test_loopscope_phase5_angular_trajectory.py
docs/loopscope_phase5.md  # 仅在命令骨架确有必要时的小幅追加；非必需
报告/figures/phase5_angular_trajectory/  # 新目录，仅最终镜像
报告/LoopScope_第五阶段GateI相邻层角距离诊断.md  # 新文件
```

优先把 producer/analyzer/verifier/plotter 放入一个窄 module + 一个 launcher，不建设通用 hook
framework，不修改 CLI。允许一个单目的本地 commit 绑定 implementation provenance；不得
push、fetch、pull、rebase、merge、force-push 或公开发布。生成报告和 figures 不进入
implementation commit。

默认保护且不得修改：

```text
AGENTS.md
.planning except executor must not modify control/handoff/contract
src/tflt/wrapper.py
src/tflt/strategies.py
src/tflt/cache.py
src/tflt/config.py
src/tflt/eval_runner.py
src/tflt/loopscope/phase3_acquisition.py
src/tflt/loopscope/phase5_acquisition.py
all Phase 1-5 cards/selectors/outcomes/reports/run artifacts
all existing figures
```

## 4. HPC2/GPU/Slurm authority

Host/root：

```text
host=hpc2-hkustgz
source_checkout=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
runtime_venv=<source_checkout>/.venv-loopscope-cu121-20260711
run_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-i-angular-trajectory-20260727T161957Z
pool_root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase3-p3b-20260715T223304Z
```

允许：

- project-scoped read-only SSH/provenance/cache/environment checks；
- 将 exact authorized implementation files hash-closed stage 到 run root；
- 两模型各 4 identities 的 smoke；
- 两模型 exact validation-1531 formal native no-loop acquisition；
- 每 job 1 GPU；A800 或 A40；只为本 Gate；
- Slurm submission、只读监控、terminal seal；
- login CPU merge/aggregate/bootstrap/Kneedle/plot/verifier；
- 若 `kneed==0.8.5` 缺失，在 run-root-local `deps/` 从国内 PyPI mirror 做一次 pinned
  安装并记录 hash；禁止升级共享 venv；
- smoke/formal 长等待使用 executor-owned 低频 heartbeat。terminal heartbeat 必须
  pause/delete 后向 exact executor 发送 `AUTOMATION_TERMINAL_RESUME`；失败则向 planning
  发送 `AUTOMATION_RELAY_REQUIRED`。

调度约束：

- freeze exact model membership、shard mapping、argv、code/card hashes 后才 submit；
- 允许在 job 仍 PENDING、attempt/forward/record 全为零时做一次 scheduler-only partition/GPU
  migration，保留 science/membership/root 且不得双重提交；
- 任一正式 task 已发生 forward 后，不允许代码修复、覆盖、append、retry、requeue 或
  salvage；partial formal failure 必须 `BLOCK`；
- failed smoke 只可在一个 fresh attempt sibling 重试一次，且不得进入 formal 前隐瞒原因。

禁止：

- login node 跑模型 forward；
- 除 exact two models × validation-1531 × primary metric 外的 GPU/Slurm；
- 加载 MMLU test、gold/label/correctness；
- loop insertion、loop/outcome、accuracy；
- 额外模型、数据、split、position、metric 或 population；
- 覆盖任何旧 root 或使用旧 partial records。

## 5. Ordered execution path

### I-0：admission 与最小实现

1. 核对 local Git、control binding、cards/hashes、remote runtime/model caches/pool。
2. 实现 narrow card/module/launcher/tests。
3. targeted engineering verification：

```bash
PYTHONPATH=src python3 -m unittest tests.test_loopscope_phase5_angular_trajectory -v
python3 -m py_compile src/tflt/loopscope/phase5_angular_trajectory.py scripts/loopscope/run_qwen_phase5_gate_i.py tests/test_loopscope_phase5_angular_trajectory.py
PYTHONPATH=src python3 scripts/loopscope/run_qwen_phase5_gate_i.py --help
PYTHONPATH=src python3 scripts/loopscope/run_qwen_phase5_gate_i.py dry-run --card configs/loopscope/phase5_angular_trajectory_gate_i_card.json
git diff --check
```

最小 synthetic tests 必须覆盖：

- identical/orthogonal/antipodal vectors -> angle 0/0.5/1；
- clamp closure、zero norm/non-finite failure；
- raw final-boundary prehook count/FinalNorm closure；
- 28/36 transition length；
- forbidden persistence rejection；
- aggregate SD/quantile/bootstrap deterministic recompute；
- Kneedle forward/reverse mapping、non-unique/shape-invalid/unstable -> `NO_STABLE_KNEE`；
- figure source-data membership。

不要跑 full suite 或重审旧 Gates。

### I-1：two-model smoke

每模型独立冻结 4 exact validation identities，运行一次 4-identity smoke。每个必须证明：

```text
record_count=4
forward_calls=4
unique identities=4
subjects>=2
boundary_count=29 or 37
transition_count=28 or 36
answer_position=last non-padding token
final_norm_prehook_calls_per_forward=1
final_norm(raw_B_L) closes outputs.hidden_states[-1]
loop_insertions=0
all angles finite in [0,1]
hidden/logits/probabilities/forbidden fields absent
```

两模型 smoke 都 PASS 才能 formal。

### I-2：two-model formal validation-1531

分别 freeze model-specific shard manifest 并 submit。允许并行运行两个模型，但 artifacts、
job/membership/receipts 必须完全分离。formal seal 只验证 scheduler、identity、counts、hashes
和 sanitized schema，不提前运行 aggregate/Kneedle 或画图。

每模型必须：

```text
1531 records
1531 successful forwards
1531 unique identities
57 subjects
exact equality to canonical pool membership/order after merge
no missing/duplicate/extra/partial
```

### I-3：one aggregate/analyzer/verifier pass

两模型 formal seal 均闭合后：

1. 每模型 merge 一次；
2. 每模型 aggregate/bootstrap 一次；
3. 两模型 Kneedle descriptive analysis 一次；
4. 生成一张 composite figure 与 source-data；
5. fresh-process independent verifier 一次；
6. 写中文报告和 Gate manifest；
7. 只读复制最终 figure/report 到新本地 sibling。

达到 must-pass 后停止，不追加 hardening。

## 6. Agility and repair budget

```text
smallest increment=one sidecar module + one launcher + one card + targeted tests
earliest real experiment=two 4-identity smokes immediately after targeted tests
decisive checks=raw FinalNorm-prehook closure; exact 1531/57 identity closure; finite 28/36 angles; independent aggregate/Kneedle/figure verification
stop rule=verifier PASS and required outputs complete -> terminal packet; no full suite or old-Gate audit
executor-owned pre-formal low-risk repair loops=2
planning-audit-returned repair cycles=1
```

任何可能改变 model/data/split/position/raw-vs-normalized metric/population/Kneedle parameters
的修复不属于 low-risk，必须 `BLOCK`。

## 7. Required terminal packet

只允许向 planning thread `019f8604-4717-7be2-8bf8-9d4a26a3d7f7` 可见发送一次：

```text
GATE_I_FINAL_AUDIT
```

或：

```text
BLOCK
```

terminal packet 至少包含：

- executor thread ID/title；
- authorized base、final commit/diff/dirty、未 push 证明；
- exact local/remote commands、tests/exit codes；
- model/tokenizer revisions、snapshot/config/weight hashes；
- dataset/renderer/pool/identity hashes；
- exact run root、Slurm job IDs/tasks/states/resources；
- smoke/formal record/forward/subject/boundary/transition closure；
- metric numeric closure与 forbidden persistence scan；
- 两模型 transition summaries、bootstrap digests；
- Kneedle per-direction status/frequency/reason 和 model-level status；
- SVG/PDF/PNG/source-data/report paths 与 SHA256；
- verifier result/receipt/manifest；
- repair loops/deviations；
- 明确未读取 test/gold/outcome，未跑 loop/accuracy，未改 selector/history；
- decision requested=`PASS / PASS_WITH_FIXES / BLOCK`。

executor 不得自判 PASS，不得进入 Gate J。若跨线程 delivery 无可见确认，输出
`TERMINAL_DELIVERY_UNCONFIRMED` paste-ready packet，不得声称已通知 planning。
