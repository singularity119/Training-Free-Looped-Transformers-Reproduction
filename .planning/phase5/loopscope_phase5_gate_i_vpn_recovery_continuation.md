# GATE_I_VPN_RECOVERY_CONTINUATION

```text
Project/phase=LoopScope Phase 5 post-terminal diagnostic extension
Gate=I
Planning/audit thread=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
Authorized executor thread=019fa464-6734-7ec0-a57a-e07a33223331
Authorized executor title=execute-LoopScope-Angular-Trajectory-第5阶段-Gate I
Supersedes=I-0 network-admission BLOCK only
Original handoff=.planning/loopscope_phase5_gate_i_angular_trajectory_handoff.md
Terminal recipient=019f8604-4717-7be2-8bf8-9d4a26a3d7f7
```

## Planning decision

`PASS_WITH_FIXES`：原 `BLOCK` 的唯一原因是本机未连接 HKUST-GZ VPN，导致 HPC2
private DNS、SSH 与 write-once run-root absence 无法核验。用户已手动连接学校 VPN，
planning 于 2026-07-28 完成以下只读复核：

```text
hpc2login.hpc.hkust-gz.edu.cn=10.120.18.63
ssh host alias=hpc2-hkustgz
observed login node=mgmt-4
remote checkout status=loopscope...origin/loopscope
canonical run root=ABSENT
```

因此 Gate I admission 的网络部分现已闭合。恢复同一 executor；不得创建替代 executor，
不得修改原科学合同、模型、数据、split、metric、样本总体、run root 或授权边界。

## Continuation authority

执行线程必须重新读取 live control、原 contract、原 handoff 与本 continuation，然后从
原 handoff 的 I-0 admission/最小实现开始继续。原 handoff 的全部允许、禁止、repair
budget、write-once、GPU/Slurm、information barrier 与 terminal delivery 条款保持不变。

本 continuation 仅修复外部网络 admission：

- 允许按原 handoff 使用 HPC2、GPU、Slurm 完成 exact two-model validation-1531
  adjacent-angular-distance Gate I；
- 不允许借 VPN 恢复扩大远程路径、模型、数据、任务、指标或 outcome 权限；
- 若 canonical run root 在 executor 重新核验时已存在、Git/control binding 漂移，或
  private DNS/SSH 再次失败，必须再次 `BLOCK`；
- 最终仍只发送一次 `GATE_I_FINAL_AUDIT` 或 `BLOCK` 给 planning thread。

## Actions not authorized

不授权 Gate J、test/gold/label/correctness/outcome/accuracy、loop run、selector 修改、
额外模型或数据、旧 root 覆盖、formal partial retry/salvage、public push 或 destructive
Git/filesystem 操作。
