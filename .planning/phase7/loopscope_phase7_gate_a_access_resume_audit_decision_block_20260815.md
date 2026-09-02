# Gate A Access-Resume Audit Decision — BLOCK

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=A
AUDITED_EXECUTOR=01a00159-6c55-72b3-bbf8-5ac798868258
EVENT=GATE_A_ACCESS_RESUME_FINAL_AUDIT
DECISION=BLOCK
BLOCKER=BLOCK_HPC_OFFICIAL_CONNECTIVITY_AND_PHASE7_TOKEN_NOT_YET_PERSISTED
NEXT_GATE_STATUS=LOCKED
```

## 决定性证据

- Sender 与 live control 中恢复授权的 Gate A executor 一致。
- Llama/Gemma 的 existing cache 均为空；镜像 metadata/tokenizer 请求均返回 gated 403。
- official Hub 正常请求与 supplement 允许的唯一一次 fresh retry 均在 TLS/repo-info 阶段被连接重置。
- 因此两个模型的 resolved revision、真实层数、路径闭合、choice-surface 闭合和 candidate domain 仍未建立。
- 本地 Phase 7 working set、branch/HEAD、staged membership 和七个 protected dirty paths 均未发生变化。
- 未下载权重，未执行 forward、CUDA/GPU、Slurm、MMLU 数据读取、评分或 Gate B 工作。

## 材料性与权限边界

该缺口直接阻止三模型 Gate A admission，因而不能进入真实 GPU smoke。普通网络路由与 bounded
retry 可由 planning 的既有 operational-permission 代决权处理，但 token 的读取、输入和持久化属于
用户凭证操作，不能由 planning 或 executor 代办，也不能通过聊天传输。

用户已在浏览器端创建新的 read token。恢复 Gate A 前，用户需要在 HPC2 的交互式终端亲自执行
`hf auth login` 并粘贴 token；planning/executor 只接受“登录完成”的确认，不读取 token 内容。

后续诊断确认 `hf auth login` 会在持久化前调用 official `whoami`，并因同一 TLS reset 失败；这不是
token 无效证据。远端库支持 `HF_TOKEN_PATH`，因此用户可改为把新 token 交互式写入独立的
`~/.cache/huggingface/phase7_read_token`（mode `600`），不覆盖默认 token。Planning 只接受文件非空
与 mode 的确认，随后另行授权 executor 通过该路径访问既有镜像路由。

## 当前状态

- Gate A：`BLOCK`，executor 活动权限撤销。
- Gate B–D：`LOCKED`。
- 允许的下一步：用户交互式写入 Phase 7 专用 token 文件；随后 planning 编写并投递新的 Gate A
  access/connectivity supplement。
- 仍禁止：模型权重、forward、GPU/Slurm、MMLU 数据、正式 acquisition、V3 scoring、绘图和 Gate B。
