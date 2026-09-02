# Gate A Repair and Operational Permission Supplement 2

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=A
PLANNING_AUDIT_THREAD=01a0013e-71c3-7c90-a547-4059b462dc7e
AUTHORIZED_EXECUTOR_THREAD=01a00159-6c55-72b3-bbf8-5ac798868258
AUTHORIZED_EXECUTOR_TITLE=execute-LoopScope-三模型Provenance与轨迹探测实现-第7阶段-Gate A
AUDIT_DECISION=PASS_WITH_FIXES
AUDIT_RETURNED_REPAIR_CYCLE=1
SUPPLEMENTS=.planning/phase7/loopscope_phase7_gate_a_handoff.md
SCIENCE_CHANGE=NONE
NEXT_GATE_AUTHORITY=NONE
POST_REPAIR_TERMINAL=GATE_A_REPAIR_FINAL_AUDIT_OR_BLOCK
```

## 1. 决定与材料性

Planning 对首次 terminal packet 做了 skill 要求的轻量验收，独立确认：

- exact executor、branch/base 与 protected dirty membership 一致；Phase 7 新文件未暂存，
  `git diff --check` 与 12 个 targeted tests 通过；
- 三个 frozen target repo 的 snapshot 在两个已审计 HPC2 HF cache 根均缺失，因此 Gate A
  must-pass 的 resolved revision、真实 `L`、live choice-surface closure 与正式 candidate table
  仍未满足；
- `run_phase7_base_trajectory.py` 实际加载 `AutoModelForCausalLM`，而
  `phase7_producer.py` 从输出读取 `last_hidden_state`；标准 causal-LM 输出正常路径通常不提供该
  字段；
- launcher 尚未显式把模型和 tensor inputs 放到 CUDA，且声明的 `--cache-dir` 没有传入
  tokenizer/model load。这会让 Gate B 的 true-CUDA、audited-cache smoke 无法可信执行。

以上均直接影响 Gate B admission，但能由同一 executor 在 frozen science 下作一次捆绑、低风险
返修；因此决定为 `PASS_WITH_FIXES`，不创建 Gate B executor。

## 2. 最小代码返修

只允许修改原 Gate A 已授权的 Phase 7 路径，并完成以下三项：

1. **Causal-LM final closure**：把 temporary native FinalNorm forward-hook 捕获的 normalized
   output 作为最终 hidden，不再要求 causal-LM output 暴露 `last_hidden_state`。用同一 native
   forward 的 probe-position logits 与 `lm_head(native_final_hidden)` 做最小数值闭合；不得对
   raw `B_L` 再调用 FinalNorm，不得增加第二次 model forward。
2. **True-CUDA launcher contract**：runtime mode 必须显式要求 CUDA 可用，把模型与所有 tensor
   inputs 放到同一 CUDA device，并在 forward 前 fail closed；dry-run/help 保持无 GPU 依赖。
   不在本 Gate 实际运行 CUDA 或模型 forward。
3. **Audited cache binding**：当 `--cache-dir` 给定时，必须将其传入 tokenizer 与 model 的
   `from_pretrained`；继续保留 exact repo/revision、`local_files_only=true`、
   `trust_remote_code=false`。加入只覆盖上述正常路径的 focused tests，不扩展为通用 device
   framework，不跑与本缺陷无关的 full suite。

## 3. Metadata-only operational permission

用户已把 ordinary operational-permission 代决权委托给 planning。现向同一 Gate A executor
授予下述最小充分权限：

```text
host=hpc2-hkustgz / account=xhuang225
writable_cache_root=/hpc2hdd/home/xhuang225/shared/hf_home
network_destination=official Hugging Face Hub needed for the three exact frozen repos
exact_repos=Qwen/Qwen2.5-3B; meta-llama/Llama-3.2-3B; google/gemma-2-2b
allowed_download_class=config and tokenizer metadata/assets only
allowed_runtime=AutoConfig/AutoTokenizer plus optional accelerate.init_empty_weights model construction
forbidden_download_class=all model weights/checkpoints and all datasets
```

允许 executor：

- 对三个 exact repo 解析 live immutable Hub revision，并把 `config.json`、tokenizer config/vocab/
  model/JSON、special/added-token 等 `AutoConfig`/`AutoTokenizer` 实际需要的轻量文件写入上述
  cache root；可以使用受控 allow-pattern 或逐文件下载，但不得下载任何 safetensors、bin、pt、
  ckpt 或其 shard/index；
- 只使用当前 HPC2 环境中已经存在且非交互可用的 Hub authentication context；不得读取、打印、
  复制、修改或新增 token/credential，不得代表用户接受 license 或申请 gated access；
- 运行 `trust_remote_code=false`、不加载权重、不做 forward 的 `AutoConfig`/`AutoTokenizer`
  admission；必要时可在 `accelerate.init_empty_weights` 下从 config 构造 meta model，以闭合
  decoder container/blocks、embedding、FinalNorm、lm-head 路径，不能实例化真实参数存储；
- 在内存中检查 exact surfaces `" A"`, `" B"`, `" C"`, `" D"` 各为一个且彼此不同的 token；
  不输出或持久化 token/input IDs；
- 将三个 repo 的 resolved model/tokenizer revision、真实 `L`、path closure、choice boolean
  closure、cache/access state 和公式生成的 candidate domain 写回
  `configs/loopscope/phase7_model_admission.json`。只有三模型全部闭合才能把 blocker 状态改为
  admitted。

若任一 repo 返回真实 401/403/gated/license 错误，或 metadata-only 获取仍要求新增 credential、
license acceptance 或权重下载，立即停止该模型，不绕过、不换镜像、不替换模型，并在 post-repair
terminal 中精确报告剩余 `BLOCK_MODEL_ACCESS`；planning 无权代决 credential/license。

## 4. 仍然禁止

- 任何模型权重下载、加载或 forward；CUDA/GPU/Slurm/job/run-root；MMLU dataset、validation
  row、dev demos、gold/outcome/test；正式 trajectory/V3；loop/generation/绘图。
- 远端 project checkout 写入、Git pull/switch/install/upgrade、环境变更；除上述 exact cache
  metadata/tokenizer 文件外的远端 mutation。
- 修改科学合同、模型 membership、prompt、probe、choice surfaces、candidate rule、V3 参数、
  schema 的科学语义或 protected paths。
- 进入 Gate B、创建 Gate B executor、commit/push protected dirty state。

## 5. 决定性复验与 stop rule

返修后只需：

1. 重跑 `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase7_*.py'`，新增 tests
   必须覆盖 causal-LM 无 `last_hidden_state`、CUDA/model-input device binding 与 cache-dir 透传；
2. 重跑 Phase 7 compile/help/dry-run 与 `git diff --check`；无需 full repository suite；
3. 给出三模型 readable admission/candidate table，确认没有持久化 token IDs、没有权重文件被
   获取、没有模型 forward；
4. 确认七个 protected dirty paths 仍未被修改、暂存或提交。

以上闭合即停止 Gate A，不增加 hardening。向 planning thread 精确发送一次返修终态：

```text
GATE_A_REPAIR_FINAL_AUDIT
Project/phase:
Gate/executor:
Repair supplement:
Material fixes and files changed:
Metadata-only acquisition commands and exit codes:
Three-model live admission and candidate-domain result:
Focused tests/commands and exit codes:
Weight/forward/CUDA/Slurm/dataset actions not taken:
Final branch/commit/dirty and protected-state confirmation:
Remaining deviations or blocker:
Decision requested: PASS / BLOCK
```

若无法确认 cross-thread 投递，则输出 `TERMINAL_DELIVERY_UNCONFIRMED` 的 paste-ready packet；
不得自行进入 Gate B。
