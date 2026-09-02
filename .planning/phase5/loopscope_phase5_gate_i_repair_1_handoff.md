# GATE_I_REPAIR_1_HANDOFF

```text
Project/phase=LoopScope Phase 5 post-terminal diagnostic extension
Gate=I
Repair cycle=1 of 1 planning-audit-returned repair
Planning decision=PASS_WITH_FIXES
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019fa464-6734-7ec0-a57a-e07a33223331
Authorized executor title=execute-LoopScope-Angular-Trajectory-第5阶段-Gate I
Original implementation commit=5166ecbf80f86a44c259d30cf77be98a9f4b101a
Original smoke jobs=10079768;10079769
Original run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase5-gate-i-angular-trajectory-20260727T161957Z
Terminal recipient=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
```

## 1. Audited material defect

两模型 smoke 均已完成 4 次 forward，并写出各 4 条 sanitized records；记录侧已闭合
boundary/transition count、raw FinalNorm prehook、zero loop、finite angle 与 closed-world
schema。失败发生在 records 写完之后、acquisition receipt 构造时：

```text
src/tflt/loopscope/phase5_angular_trajectory.py:789
NameError: name 'torch' is not defined
```

planning 复核确认 `_load_runtime(model)` 返回 tuple 的 `runtime[0]` 是本次实际加载的
`torch` 模块，而 receipt 代码错误地引用未绑定的全局 `torch`。两份 stderr 均为同一
NameError。该问题不改变科学合同、forward、hidden capture、angular-distance 公式、
membership 或 sanitized record。

## 2. Exact authorized repair

只允许在 acquisition receipt 所在函数内，把本次 runtime tuple 的第 0 项窄幅绑定为
局部 `torch`（例如紧邻 `runtime = _load_runtime(model)` 后执行
`torch = runtime[0]`），使以下现有 receipt 字段读取该 exact runtime：

```text
torch.__version__
torch.cuda.get_device_name(torch.cuda.current_device())
```

允许在现有 targeted test 文件中增加一个直接覆盖 receipt runtime serialization 的
最小回归测试。禁止修改：

- angular-distance、cosine/clamp、raw residual/FinalNorm prehook 的任何公式或实现；
- card、模型、revision、dtype、renderer、identity、membership、position、population；
- scheduler recipe、GPU type、formal shard mapping；
- Kneedle、aggregate、bootstrap、figure、verifier 科学合同；
- 其他源码、旧 records、旧 attempts 或 protected artifacts。

修复后允许 amend 原单目的 implementation commit；不得 push/fetch/pull/rebase/merge。
重新 hash-close repaired module/test/launcher/card，并保留旧 implementation/stage hashes
作为 invalid smoke provenance。

## 3. Verification and fresh smoke

只运行原 handoff 的 targeted tests、py_compile、launcher dry-run 和 `git diff --check`；
不得跑 full suite 或扩展 hardening。测试必须证明 receipt 使用 `runtime[0]`，且不依赖
模块级 `torch`。

旧 attempt-0001 artifacts 和 jobs `10079768/10079769` 必须保持只读、不得 append、
覆盖、salvage 或 promotion。使用同一 canonical run root 下两个全新的
`attempt-0002` sibling，按原 exact 4 identities 对两模型各重新运行一次 smoke。
这是 Gate I 唯一允许的 smoke retry。

两份 fresh smoke receipt 均 PASS，且原 handoff I-1 的所有 must-pass closure 成立后，
才可进入原 handoff I-2 formal。若任一 fresh smoke 再失败，立即 `BLOCK`；不得第三次
smoke、再次修代码、retry/requeue 或换科学合同。

## 4. Continuing authority and stop conditions

fresh smoke PASS 后，原 Gate I handoff 的 formal validation-1531、aggregate/Kneedle、
figure、verifier、报告、information barrier、write-once 与 terminal delivery 条款恢复。
本 repair 不授权 Gate J、test/gold/outcome/accuracy、loop、selector 修改、额外
模型/数据/split/metric/population、旧 artifact 覆盖或 public push。

最终仍只能向 planning thread 发送一次 `GATE_I_FINAL_AUDIT` 或 `BLOCK`。
