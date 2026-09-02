# LoopScope Phase 7 Gate A Audit Decision — PASS

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=A
EXECUTOR_THREAD=01a00159-6c55-72b3-bbf8-5ac798868258
EXECUTOR_TITLE=execute-LoopScope-三模型Provenance与轨迹探测实现-第7阶段-Gate A
DECISION=PASS
EXECUTOR_AUTHORITY=REVOKED_AFTER_TERMINAL
NEXT_GATE=B
NEXT_GATE_STATUS=AUTHORIZED_IN_SEPARATE_EXECUTOR
AUDIT_DATE=2026-08-15
```

## 决定

Gate A `PASS`。Exact executor 的终包与 planning 的最小独立验收一致：三模型 provenance、
架构路径、choice surface 和由真实深度生成的 candidate domain 均已闭合；Phase 7 focused
discovery tests 为 17/17；Git staged membership 为空，七个受保护的前序 dirty paths 未被
覆盖或暂存。

| Model | Resolved revision | L | Candidate windows | Admission |
| --- | --- | ---: | ---: | --- |
| `Qwen/Qwen2.5-3B` | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | 36 | 42 | PASS |
| `meta-llama/Llama-3.2-3B` | `13afe5124825b4f3751f836b40dafda64c1ed062` | 28 | 26 | PASS |
| `google/gemma-2-2b` | `c5ebcd40d208330abc697524c919956e692655cf` | 26 | 26 | PASS |

三模型的 decoder blocks、embedding、FinalNorm 与 lm head 路径均闭合；`" A"`、`" B"`、
`" C"`、`" D"` 在各 tokenizer 上均为单 token 且彼此不同。Gemma 的单次用户要求复查已
通过 token-backed `hf-mirror.com` 获取 metadata/tokenizer，并完成离线 empty-model admission；
未读取凭证内容，未下载权重，也未执行 forward、CUDA、Slurm 或 MMLU 数据访问。

## Planning 独立验收

Planning 直接解析 `configs/loopscope/phase7_model_admission.json`，确认状态为
`PASS_THREE_MODEL_ADMISSION`，exact 三模型 membership、`L`、candidate count、path closure 与
choice closure 全部一致。随后核对 branch `loopscope`、HEAD
`37241f6fb52aa6ee89a5b4e24ac607f7312e77ac`、required ancestry、空暂存区和 `git diff --check`；
结果均通过。

终包中的一次非 canonical `unittest` module-list 调用只产生 sibling import loader error；按
handoff 约定的 discovery 命令随后 17/17 通过，未触发代码修复。该偏差不影响 Gate 决定。

## 下一步

Gate A executor 权限终止。Gate B 由独立 executor
`01a001c1-3ff9-7370-abbb-b32128897a08` 承担，只运行三模型四样本 CUDA trajectory smoke。
Gate A 代码目前是通过测试但未提交的 Phase 7 专用工作集；Gate B 将先在不触碰 protected
dirty paths 的前提下完成 focused Phase 7 integration commit/push 与 HPC2 fast-forward，再以
该 immutable commit 执行 `debug` preflight。Gate C、Gate D 与 planning closeout 仍锁定。
