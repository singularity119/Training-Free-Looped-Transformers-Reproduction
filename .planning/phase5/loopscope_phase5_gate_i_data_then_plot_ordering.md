# GATE_I_DATA_THEN_PLOT_ORDERING

```text
Project/phase=LoopScope Phase 5 post-terminal diagnostic extension
Gate=I
User instruction=scientific data first; figure rendering afterward as a separate step
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019fa464-6734-7ec0-a57a-e07a33223331
Terminal recipient=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
```

## Binding ordering correction

用户明确要求：绘图放在得到实验数据之后单独进行，不与实验数据分析绑定为一个不可分割
步骤。Gate I 后续顺序冻结为：

1. **I-3A scientific data analysis**：只读消费已封存的两模型 formal scalar records，
   各执行一次 aggregate、subject-stratified bootstrap 与 descriptive Kneedle，写出
   transition tables、analysis JSON、Kneedle summary 和 figure source-data CSV；
2. **I-3A data verification**：fresh process 独立复算并闭合上述机器可读科学数据；
3. **I-3B presentation**：只读取 I-3A 已闭合的 source-data/summary，单独生成
   SVG/PDF/PNG；
4. **I-3B figure verification/report**：检查三格式结构一致性并完成中文报告与 Gate
   manifest。

若现有 launcher 把 analysis 与 plotting 写成同一函数，允许做最小 orchestration
拆分或增加明确的 data-only / plot-only subcommand；必须复用原 frozen 计算函数、schema、
seed、draw、Kneedle 参数与 source-data，禁止重写数学或重复运行 I-3A。

## Failure semantics

- I-3A 科学数据一旦写出并通过 data verifier，必须保持 immutable；
- I-3B 失败不得删除、覆盖、回滚或重新计算 I-3A；
- plotter 不得读取 raw per-identity records，只读取已闭合的 aggregate/source-data；
- plotter 或格式问题不得改变 transition estimates、CI、Kneedle status 或报告中的数值；
- Gate I 最终交付仍包含用户已要求的 figure；若 I-3B 再失败，terminal `BLOCK` 必须明确
  报告 I-3A 已完成的数据状态，等待后续纯展示修复，不得重跑实验或统计。

原 post-formal plot repair 仍只允许 Panel A autoscale/arrow state reset。本 ordering
correction 不授权新的科学修复、GPU/Slurm/model forward、formal rerun、outcome access、
Gate J 或额外模型/数据/指标。
