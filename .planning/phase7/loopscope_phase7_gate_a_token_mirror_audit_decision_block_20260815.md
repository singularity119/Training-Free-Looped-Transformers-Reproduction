# Gate A Token-Backed Mirror Audit Decision — BLOCK

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=A
AUDITED_EXECUTOR=01a00159-6c55-72b3-bbf8-5ac798868258
EVENT=GATE_A_TOKEN_MIRROR_FINAL_AUDIT
DECISION=BLOCK
BLOCKER=BLOCK_GEMMA_ACCESS_AWAITING_REPO_REVIEW
NEXT_GATE_STATUS=LOCKED
```

## 决定性证据

- Sender 与 Supplement 5 授权的 exact Gate A executor 一致。
- Admission sidecar 可解析，Qwen 为 PASS，Llama 为 PASS（revision
  `13afe5124825b4f3751f836b40dafda64c1ed062`、`L=28`、26 candidates），Gemma unresolved。
- Llama 与 Gemma 使用同一 token-backed mirror 路径；Llama 成功，而 Gemma 返回明确的 gated 403
  `awaiting review`。因此 token 文件、`HF_TOKEN_PATH` 与 mirror endpoint 已正常工作。
- Gemma cache 没有 snapshot 文件，live revision、`L`、路径、choice surfaces 和 candidate domain 仍未闭合。
- Live Git 无 staged membership；七个 protected dirty paths 保持原状。未发生权重、forward、GPU/Slurm、
  MMLU 数据、评分、绘图或 Gate B 工作。

## 材料性与决定

Gate A 的三模型 admission 是 Gate B smoke 的必要输入。Gemma 缺失会直接改变正式 membership 和后续
计算路径，不能作为非材料偏差接受。该状态不是 executor 可修复的工程问题，也不是 planning 的 ordinary
operational-permission 代决权可以覆盖的外部仓库审批。

Decision：`BLOCK`。Gate A executor 权限撤销；Gate B–D 保持锁定。

## 唯一恢复条件

用户在已登录的 Hugging Face 账号中打开 `google/gemma-2-2b`，确认 access 已从 pending/awaiting
review 变为 accepted，并且模型文件可直接读取。仅点击申请、同意许可或看到“请求已提交”不满足恢复
条件。确认 accepted 后，planning 可恢复同一个 Gate A executor，仅重试 Gemma 的 token-backed mirror
metadata/tokenizer admission；不得重复 Qwen/Llama 或进入 Gate B。
