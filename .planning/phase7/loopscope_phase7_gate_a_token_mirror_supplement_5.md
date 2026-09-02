# Gate A Token-Backed Mirror Supplement 5

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=A
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a00159-6c55-72b3-bbf8-5ac798868258
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-三模型Provenance与轨迹探测实现-第7阶段-Gate A
RESUME_REASON=USER_CREATED_PHASE7_READ_TOKEN_FILE_MODE_600
SCIENCE_CHANGE=NONE
NEXT_GATE_AUTHORITY=NONE
TERMINAL_EVENT=GATE_A_TOKEN_MIRROR_FINAL_AUDIT_OR_BLOCK
```

## 1. 恢复事实与凭证边界

用户已在 HPC2 交互式 shell 中亲自创建只读凭证文件：

```text
/hpc2hdd/home/xhuang225/.cache/huggingface/phase7_read_token
```

Planning 只读确认文件非空、owner=`xhuang225`、mode=`600`，没有读取或输出内容。Executor 只被
授权让 `huggingface_hub` 通过 `HF_TOKEN_PATH` 标准机制消费该文件；不得以 `cat`、`head`、`sed`、
Python print、shell expansion、环境 dump、debug HTTP 或任何其他方式读取、显示、复制、移动、修改、
删除或重新持久化 token。不得运行 `hf auth login/logout/switch/list`，也不得报告账号名或 token 长度。

## 2. 唯一授权工作

Executor 必须完整重读 `research-gate-orchestrator/SKILL.md`、`references/protocol.md`、live control、
本 supplement 与既有 Gate A handoff/supplements，然后只执行：

1. 不重复 Qwen admission。对 `meta-llama/Llama-3.2-3B` 和 `google/gemma-2-2b`，在单命令环境中设置：

   ```text
   HF_TOKEN_PATH=/hpc2hdd/home/xhuang225/.cache/huggingface/phase7_read_token
   HF_ENDPOINT=https://hf-mirror.com
   ```

2. 使用此前冻结的 metadata/tokenizer `snapshot_download` allow-list 与 shared cache
   `/hpc2hdd/home/xhuang225/shared/hf_home/hub`。只允许 `config.json`、tokenizer 配置/词表/merges/
   sentencepiece 与 chat-template 轻量文件；显式禁止任何 model weight、checkpoint、index 或 shard。
3. Mirror 成功后，立即切换为 offline `AutoConfig`/`AutoTokenizer` 与必要的
   `accelerate.init_empty_weights` meta construction，闭合每个模型的 resolved model/tokenizer revision、
   真实 `L`、decoder/embedding/FinalNorm/lm-head path、四个 exact choice surfaces 的 single-token/distinct
   boolean，以及公式生成的 candidate domain；不得输出 token IDs。
4. 仅在 admission 事实变化时更新 `configs/loopscope/phase7_model_admission.json` 与必要的
   `docs/loopscope_phase7.md`。运行 JSON parse、focused admission/dry-run、`git diff --check`、staged
   membership 与 protected dirty-state 检查；代码未变时不重复完整 tests。

## 3. Stop conditions 与 bounded repair

- 两模型均闭合：立即停止并发送 terminal packet，不进入 Gate B。
- Mirror 返回 401/403：允许一次不涉及 token 内容的参数/环境路径检查；确认 `HF_TOKEN_PATH` 被库识别、
  exact repo 与 endpoint 正确后仍失败，报告 `BLOCK_MIRROR_GATED_TOKEN_NOT_ACCEPTED`。不得打印 token、
  改权限、重做许可、切换账号、尝试 alternate repo 或自行修 official connectivity。
- 一次普通网络 timeout/reset：同一路由允许一次 fresh retry；仍失败则报告精确 connectivity blocker。
- 普通 shell quoting 或 allow-list 构造错误可在不发出请求或不改变科学合同的前提下立即修正。

## 4. 仍然禁止

任何权重/checkpoint/index/shard、模型 forward、CUDA/GPU/Slurm/job、MMLU row/dev/gold/outcome/test、
loop、generation、formal scoring、绘图、Gate B executor/work、official Hub fallback、proxy/environment install、
credential/profile/license 操作、remote project checkout 写入、protected path 变更、commit 或 push。

## 5. Agility budget

- 最小增量：两次 token-backed mirror metadata acquisition，加离线 admission closure。
- 最小检查：两模型 exact files/revision、offline path/choice/candidate closure、sidecar parse 与 protected state。
- Stop rule：一旦三模型 admission 完整，立即发 terminal event；不追加网络框架、代理或额外 hardening。

## 6. Terminal packet

向 planning thread 精确发送一次：

```text
GATE_A_TOKEN_MIRROR_FINAL_AUDIT
Project/phase and executor:
Supplement:
Per-model mirror command and exit status, excluding all credential content:
Llama/Gemma resolved revisions, L, path/choice/candidate closure:
Updated admission state and focused validation:
Weight/forward/CUDA/Slurm/data/credential actions not taken:
Final Git/protected state:
Remaining blocker:
Decision requested: PASS / BLOCK
```

若无法确认投递，输出 `TERMINAL_DELIVERY_UNCONFIRMED` paste-ready packet；不得自行进入 Gate B。
