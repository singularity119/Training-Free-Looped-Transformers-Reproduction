# GATE_I_POST_FORMAL_PLOT_REPAIR_HANDOFF

```text
Project/phase=LoopScope Phase 5 post-terminal diagnostic extension
Gate=I
Exception type=one post-formal presentation-only repair
Planning decision=PASS_WITH_FIXES
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019fa464-6734-7ec0-a57a-e07a33223331
Authorized executor title=execute-LoopScope-Angular-Trajectory-第5阶段-Gate I
Current implementation commit=c8858d5e7d448752a23722ec6fbbd530312bd26e
Formal jobs=10079877;10079878
Canonical run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-i-angular-trajectory-20260727T161957Z
Terminal recipient=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
```

## 1. Audited defect and materiality

两模型 formal acquisition 与 seals 已 PASS，且尚未运行 aggregate、bootstrap、Kneedle、
figure、verifier 或报告。planning 独立复核 frozen plotter 与 gnuplot 5.2.8 状态：

```text
plot_body is executed sequentially for SVG, PDF and PNG
Panel B executes: set xrange [0:1]
unset multiplot / a new set multiplot does not reset xrange
next format Panel A has no autoscale/raw-range reset
```

因此 SVG 的 Panel A 首次可正确 auto-range，而 PDF/PNG 的 Panel A 会继承 `[0,1]`，
违反三格式同图与 raw transition-index 合同。这是 presentation state leakage，不改变
已封存的模型 forward、identity、angular-distance scalars、formal seals、统计定义或
Kneedle 参数。

## 2. Exact authorized change

只允许修改
`src/tflt/loopscope/phase5_angular_trajectory.py` 的 figure-rendering `plot_body`：

1. 在每次 Panel A 开始、任何 raw-index arrow 和 plot 命令之前执行
   `set autoscale x`；
2. 同处执行 `unset arrow`，随后由已有 frozen `raw_arrows` 命令重新建立 raw-index
   Kneedle arrows；
3. Panel B 继续使用 frozen `set xrange [0:1]`、先 `unset arrow` 再建立
   normalized-depth arrows。

允许在现有 targeted test 中增加最小 regression，证明三次 render block 的 Panel A
均在 raw arrows/plot 前恢复 autoscale，并证明 gnuplot 执行
`set xrange [0:1] -> set autoscale x` 后不再保持 fixed `[0,1]`。

禁止修改 card、producer、records、aggregate/bootstrap/Kneedle 数学、source-data
schema、颜色/字体/尺寸、模型/data/split/metric/population 或其他文件。不得借本修复
调整 knee 结果、曲线、CI、图注或科学结论。

## 3. Provenance and execution

- 允许 amend 当前单目的 implementation commit，并 hash-close exact authorized files；
- 不得 push/fetch/pull/rebase/merge；
- 不得覆盖原 staged implementation；将修复后的 module/test 放入 fresh
  analysis-stage sibling，并记录旧/新 implementation hash；
- formal records、merged records、seals 和 jobs `10079877/10079878` 全部只读；
- 禁止任何 smoke/formal/model forward/GPU/Slurm retry、rerun、requeue 或 salvage。

只运行 targeted plot regression、原 targeted Gate I tests、py_compile、dry-run 与
`git diff --check`。通过后，使用已封存的 exact formal inputs运行原计划中唯一一次
aggregate/bootstrap/Kneedle/figure analysis，再运行唯一一次 fresh-process verifier。
由于此前 analysis count=0、figure count=0，本次仍是 canonical first analysis，不是
outcome-informed rerun。

## 4. Acceptance and stop rule

must-pass：

- executable gnuplot state test证明 Panel A autoscale 已恢复；
- generated script 的 SVG/PDF/PNG 三个 Panel A 均具有 raw transition-index range；
- 三种格式均非空并表达同一两面板 source data、CI、fit 与 knee annotations；
- aggregate/Kneedle/figure/verifier 所有原 Gate I must-pass 成立；
- formal artifacts hashes 不变；
- information barrier 保持。

任一 targeted check、analysis、render 或 verifier 失败，立即 `BLOCK`。本例外仅一次，
不得再次修代码、重跑 analysis、覆盖 artifacts 或扩展 Gate。

最终仍只能向 planning thread 发送一次 `GATE_I_FINAL_AUDIT` 或 `BLOCK`。不授权
Gate J、test/gold/outcome/accuracy、loop、selector 修改或 public push。
