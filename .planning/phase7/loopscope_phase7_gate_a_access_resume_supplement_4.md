# Gate A Access Resume Supplement 4

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=A
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a00159-6c55-72b3-bbf8-5ac798868258
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-三模型Provenance与轨迹探测实现-第7阶段-Gate A
RESUME_REASON=USER_COMPLETED_LLAMA_AND_GEMMA_HF_ACCESS
SCIENCE_CHANGE=NONE
NEXT_GATE_AUTHORITY=NONE
TERMINAL_EVENT=GATE_A_ACCESS_RESUME_FINAL_AUDIT_OR_BLOCK
```

## 1. 恢复事实

用户已确认 `google/gemma-2-2b` 授权完成，并在同一已登录 Hugging Face 浏览器会话直接打开
`meta-llama/Llama-3.2-3B/blob/main/config.json` 看到 JSON 内容。账号持有人许可边界已由用户本人
完成；executor 不得重复申请、接受许可或操作浏览器账号。

## 2. 唯一授权工作

Executor 必须先完整重读 `research-gate-orchestrator/SKILL.md`、`references/protocol.md`、live
control、本 supplement 与此前 Gate A handoff/supplements。随后只执行：

1. 使用 HPC2 当前已经存在的非交互 Hugging Face authentication context，重试
   `meta-llama/Llama-3.2-3B` 与 `google/gemma-2-2b` 的 config/tokenizer metadata acquisition；
   不重复 Qwen admission。
2. 路由仍为 existing shared cache → 单命令 `HF_ENDPOINT=https://hf-mirror.com` → official
   `huggingface.co` fallback；allow-pattern 与 supplement 2/3 相同，只允许 AutoConfig/
   AutoTokenizer 所需轻量文件。
3. 不读取、打印、复制或修改 token，不执行 `hf auth login`、credential/profile/license 操作，
   不在 terminal packet 报告账号名；直接以两个 exact repo 文件能否读取作为 access 结果。
4. 成功后用 offline `AutoConfig`/`AutoTokenizer` 与必要的
   `accelerate.init_empty_weights` meta construction 闭合各自 resolved revision、真实 `L`、
   decoder/embedding/FinalNorm/lm-head path、四个 choice surfaces boolean 与公式生成 candidate
   domain；不加载权重、不做 forward、不输出 token IDs。
5. 仅更新 `configs/loopscope/phase7_model_admission.json` 和必要的 `docs/loopscope_phase7.md`；
   三模型全部闭合后状态改为 admitted。运行 JSON parse、focused admission command、
   `git diff --check` 与 protected/staged membership 检查即可；代码未变时不重复完整 tests。

## 3. Stop conditions

- 成功闭合两个剩余模型即停止并发送 terminal packet，不进入 Gate B。
- 若任一 repo 仍返回 401/403，报告 `BLOCK_HPC_AUTH_CONTEXT_NOT_ENTITLED`；不得登录、索取 token、
  重复许可、切换账号、换 repo 或下载权重。
- 若只是一次 connection reset/timeout，可以在同一路由内作一次 fresh network retry；仍失败则
  精确报告 connectivity blocker，不作环境或代理改造。

## 4. 仍然禁止

任何模型权重/checkpoint/index、模型 forward、CUDA/GPU/Slurm/job、MMLU row/dev/gold/outcome、
test split、loop、generation、formal scoring、绘图、Gate B executor、remote project checkout 写入、
环境安装升级、credential/token/profile/license 操作与 protected path 变更。

## 5. Terminal packet

向 planning thread 精确发送一次：

```text
GATE_A_ACCESS_RESUME_FINAL_AUDIT
Project/phase and executor:
Supplement:
Per-model cache/mirror/official route and exit status:
Llama/Gemma resolved revisions, L, path/choice/candidate closure:
Updated admission state and focused validation:
Weight/forward/CUDA/Slurm/data/credential actions not taken:
Final Git/protected state:
Remaining blocker:
Decision requested: PASS / BLOCK
```

无法确认投递时输出 `TERMINAL_DELIVERY_UNCONFIRMED` paste-ready packet；不得自行进入 Gate B。
