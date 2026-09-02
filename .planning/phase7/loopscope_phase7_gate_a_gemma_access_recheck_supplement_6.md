# Gate A Gemma Access Recheck Supplement 6

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=A
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a00159-6c55-72b3-bbf8-5ac798868258
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-三模型Provenance与轨迹探测实现-第7阶段-Gate A
RESUME_REASON=USER_REQUESTED_EXECUTOR_DIRECT_GEMMA_ACCESS_RECHECK
SCIENCE_CHANGE=NONE
NEXT_GATE_AUTHORITY=NONE
TERMINAL_EVENT=GATE_A_GEMMA_RECHECK_FINAL_AUDIT_OR_BLOCK
```

## 1. 唯一授权动作

Executor 必须先完整重读 `research-gate-orchestrator/SKILL.md`、`references/protocol.md`、live control、
本 supplement 与 Supplement 5。随后只执行一次 `google/gemma-2-2b` token-backed mirror metadata/
tokenizer access attempt：

```text
HF_TOKEN_PATH=/hpc2hdd/home/xhuang225/.cache/huggingface/phase7_read_token
HF_ENDPOINT=https://hf-mirror.com
cache_dir=/hpc2hdd/home/xhuang225/shared/hf_home/hub
revision=main
```

沿用 Supplement 5 的 exact allow-list，仅允许 config/tokenizer 轻量文件。不得重复 Qwen 或 Llama，
不得 official fallback、proxy、install、账号/许可动作或第二次 access retry。

## 2. 成功路径

若 mirror 成功，立即 offline `AutoConfig`/`AutoTokenizer` 加必要的 empty-weight meta construction，闭合
Gemma resolved model/tokenizer revision、真实 `L`、decoder/embedding/FinalNorm/lm-head path、四个 exact
choice surfaces 的 single-token/distinct boolean 与 candidate domain；不得输出 token IDs。只更新 admission
sidecar 与必要 runbook，做 JSON parse、focused dry-run、`git diff --check`、staged/protected-state 检查，
然后停止并投递 terminal event。

## 3. 失败路径与 stop rule

- 仍返回 401/403、pending 或 awaiting review：立即报告 `BLOCK_GEMMA_ACCESS_AWAITING_REPO_REVIEW`；不做
  参数重查、重试、alternate repo、official fallback、账号切换或许可动作。
- 普通 timeout/reset：报告精确 connectivity blocker；本 supplement 不授权重试。
- Shell quoting 在发出请求前失败，可修正同一命令一次；请求一旦发出即计为唯一 access attempt。

## 4. 仍然禁止

读取、打印、复制、修改或删除 token；任何 `hf auth` 命令；权重/checkpoint/index/shard；模型 forward；
CUDA/GPU/Slurm/job；MMLU row/dev/gold/outcome/test；loop/generation/scoring/plotting；Gate B executor/work；
protected path 变更、commit 或 push。

## 5. Terminal packet

向 planning thread 精确投递一次：

```text
GATE_A_GEMMA_RECHECK_FINAL_AUDIT
Project/phase and executor:
Supplement:
Single Gemma mirror attempt and exit status, excluding credential content:
Gemma admission closure or exact blocker:
Focused validation and updated files:
Forbidden actions not taken:
Final Git/protected state:
Decision requested: PASS / BLOCK
```

若无法确认投递，输出 `TERMINAL_DELIVERY_UNCONFIRMED` paste-ready packet；不得自行进入 Gate B。
