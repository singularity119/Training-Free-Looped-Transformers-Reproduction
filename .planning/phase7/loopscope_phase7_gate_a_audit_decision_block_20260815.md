# Gate A Audit Decision — BLOCK_MODEL_ACCESS

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=A
DECISION=BLOCK
AUDITED_EXECUTOR=01a00159-6c55-72b3-bbf8-5ac798868258
AUDITED_EXECUTOR_TITLE=execute-LoopScope-三模型Provenance与轨迹探测实现-第7阶段-Gate A
EXECUTOR_AUTHORITY=REVOKED_PENDING_USER_DECISION
NEXT_GATE_STATUS=LOCKED
BLOCKER=BLOCK_MODEL_ACCESS_LLAMA32_3B_AND_GEMMA2_2B
```

## 决定性证据

- `configs/loopscope/phase7_model_admission.json` 显示 Qwen/Qwen2.5-3B 已从 HF mirror 完成
  metadata/tokenizer admission：真实 `L=36`、四个 exact choice surfaces 均 single-token 且互异、
  path closure 与 42 个 candidate windows 闭合。
- `meta-llama/Llama-3.2-3B` 与 `google/gemma-2-2b` 在 existing shared cache 中缺失；HF mirror
  对两者返回 gated 403，official fallback 在 repo metadata 请求时连接重置；没有 snapshot 文件、
  权重、替代模型或其他镜像 membership 被接受。
- 官方 Hugging Face 模型页当前分别要求 Llama 3.2 访问者同意共享联系信息并接受其许可，Gemma 2
  访问者登录并同意 Google 使用许可。该事实解释了 mirror 403，并把剩余问题定位为账号/许可
  权限，而非 launcher 或下载命令缺陷。
- Planning 独立复跑 Phase 7 focused suite：17 tests 通过。抽查确认 causal-LM final hidden/logits
  closure、Gemma final-logit softcap、显式 CUDA model/input binding 与 cache-dir 透传已经落入正常
  producer/launcher 路径。
- Git staged membership 为空，七个 protected dirty paths 仍保留；没有 Gate B、模型 forward、
  CUDA/Slurm、dataset 或 outcome 动作。

## 材料性与已接受部分

三个代码缺陷的返修与 Qwen admission 接受，不要求重复返修。Gate A 仍不能 `PASS`，因为 frozen
membership 要求三个 exact repos 都闭合真实 revision、`L`、paths、choice surfaces 与 candidate
domain；缺少其中两个会直接改变 Gate B 的模型矩阵和可执行性。

## 用户决定与恢复条件

首选恢复路径由账号持有人完成，不在本 planning thread 代办：

1. 在 Hugging Face 上分别申请/接受 `meta-llama/Llama-3.2-3B` 与 `google/gemma-2-2b` 的访问条件；
2. 让 HPC2 上现有的非交互 Hugging Face authentication context 对两个 exact repos 可读；不要在
   Codex 消息、日志或 artifact 中粘贴 token；
3. 向 planning 明确确认“两个 repo 的许可与 HPC2 访问已完成”。Planning 随后重新授权同一
   Gate A executor，只下载 metadata/tokenizer、补齐两模型 admission 并发送新的 terminal audit。

若用户不愿接受上述许可，则必须由用户明确选择是否修改 Phase 7 frozen model matrix；这是科学
范围变更，planning 不会自行删除或替换模型。

在用户决定前：Gate A executor 保持停止；Gate B–D 与 planning closeout 全部锁定；禁止新的
下载重试、凭证操作、模型替换、权重/forward、GPU/Slurm 或数据访问。
