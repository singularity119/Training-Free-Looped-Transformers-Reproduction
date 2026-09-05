# GATE_A_HANDOFF：配方现场核对与谱操作最小核心

Project/phase: LoopScope / 第8阶段
Planning recipient: 01a06fe9-c7bb-7d72-a906-234f301de317
Authorized executor: 01a07030-6413-7660-b712-c18d9e93d26e
Executor title: execute-LoopScope-Recipe-and-Spectral-Core-第8阶段-Gate A

## 1. 目标与范围

用户已授权阶段连续推进，但本任务仅 Gate A：在不运行真实模型、不看目标 outcome 的前提下，核对四个模型窗口配置的可实现性，完成后续实际干预所需的最小纯 Python 数学参考及接入方案。现有前期总体计划已被 v0.2 替代，不运行旧 17 配置方案。

先读 repo AGENTS.md → PROJECT_MEMORY.md → 当前 control → 总体计划 → 本 handoff。若未收到 GATE_A_HANDOFF_READY，保持等待。科学待定项不阻塞本 Gate 的独立工作，也不能由 executor 自行选定。

## 2. 起点与可写路径

Local repo: /Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
Branch: loopscope
Implementation starting HEAD: b53ff067992e41b66a8df01794dfb106c61b29d0；允许规划任务在本次 handoff 下发前追加仅 Phase 8 规划/入口文档的提交，须读取 diff 验证。不得重置到旧 HEAD。
Protected ancestor: 4f59bd93eca4da3cbf458a93508f91c5b23912bc
Expected dirty: 以 READY 消息给出的规划提交为准；已有规划文档属于规划任务，不覆盖。

本 Gate 仅允许新增/修改：
- src/tflt/loopscope/phase8_spectral_core.py：不依赖 torch 的 rank-1 投影、soft damping、同范数 uniform 数学参考；不做通用框架。
- tests/test_loopscope_phase8_spectral_core.py：有限决定性数学测试与四窗口映射。
- configs/loopscope/phase8_model_windows.json：只记录用户已确认模型/window/cache 和明确 PROPOSED/PENDING 项，不能伪称 runtime 合同已冻结。
- .planning/phase8/loopscope_phase8_gate_a_evidence.md：一个精简 Gate 证据总结，含接入点/最小 proposed diff、实际版本与路径。

不得写 control、总体计划、AGENTS.md、其他阶段文件。默认 protected `config.py/strategies.py/wrapper.py/cache.py/eval_runner.py` 均只读，本 Gate 不实现 hook。仅允许本 Gate 文件的单一目的 Git commit、普通 fast-forward push 到 origin/loopscope；可以推送前置规划提交，禁止 force/rebase/merge/main 变更。不得覆盖他人 dirty；遇到影响当前 provenance 的新改动向规划报告。

## 3. 已确认与待定科学配置

Qwen/Qwen3-4B-Base：inclusive 12:15(B12→B16)、13:16(B13→B17)，cache first。
Qwen/Qwen3-1.7B-Base：inclusive 12:15(B12→B16)、6:9(B6→B10)，cache last。
MMLU plain 5-shot，普通循环为 damped_euler（euler alias），h=alpha/K。K 包含首次 t=0。

待用户回复：test14042 或 validation1019 评测、K=2/4 范围、原提案 rank1/lambda0.5 确认。默认方法草案为 calibration uncentered t=1 residual top-1，pre-answer token、t>=1 投影软衰减；不得在本 Gate 拟合方向、读 target gold、选 alpha/lambda/window 或运行模型。

可提出 dtype/runtime/scoring 建议，但由 planning 在 B 前冻结。核对现有 FP16 1.7B 与 BF16 4B 的精确历史运行路径，不假定跨模型 dtype 相同。明确 evaluator 的 prefix token 位置、choice loglikelihood、padding、cache body vs stash 调用，以及原始 residual 与 FP32测量差分的区别。

## 4. HPC 基础权限

先应用 hpc2-hkustgz-ssh skill。允许使用 hpc2-hkustgz、ClearAllForwardings=yes 作有界只读 SSH；允许在已存在环境运行无模型 forward 的 CPU/import/config/tokenizer 元数据检查。远程 shell、缓存元数据读取不等于运行数据集评测。

只读根：
- /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
- /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/{inputs,runs,staging,artifacts}：仅来源/环境/身份清单元数据，不读历史或新逐题结果、labels/accuracy。
- /hpc2hdd/home/xhuang225/shared/hf_home、/hpc2hdd/home/xhuang225/shared/datasets：model config/tokenizer/revision/cache 文件元数据；不将目标 gold 或结果输出到工具。

候选 runtime：专用 clone/.venv-loopscope-cu121-20260711/bin/python，必须现场验证。
历史模型 revision 线索：4B=906bfd4b4dc7f14ee4320094d8b41684abff8539，1.7B=ea980cb0a6c2ae4b936e82123acc929f1cec04c1；核实缓存存在和 config，不自动改成最新。

禁止远程写入、同步 clone、安装/下载、运行模型、CUDA、Slurm submit/cancel/retry，以及任何旧 artifact 变更。允许读取当前用户项目相关 squeue/sinfo/account 的资源元数据，为下一 Gate 提出 debug 资源建议。连接/环境不足时先完成所有本地工作，并向规划报告最小实际缺项；不得绕过 host-key 信任校验。

## 5. 顺序、测试与 Agility budget

1. 核对本地分支/根/祖先/dirty，并只读 HPC 的 repo/env/model config 和 MMLU task 配置来源。
2. 实现最小数学函数，写目标测试；非单位 v/非有限输入应显式失败，不掩盖正常路径错误。
3. 给出最小实际 adapter 接入方案：现有 audit_collector 的返回值被忽略，所以观测不能当干预；说明最少需要改哪个 protected 文件以及关闭时原算法保留方式。只写方案，不改文件。
4. 完成关键验证，提交/推送自己的四个文件，发回证据。

从 repo root 运行：
```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_loopscope_phase8_spectral_core.py'
git diff --check
git status --short --branch
```

测试覆盖 lambda=0 恒等、lambda=1 删除方向分量、lambda=0.5 投影能量/范数恒等式、正交分量保持、v 与 -v 相同效果、同范数 uniform 确实保留同一向量谱衰减后的范数（零 residual 明确为零），以及四个 inclusive/boundary 映射。此处都是科学计算所需的边界，不扩展成穷举框架。不要重复 full suite；CPU tests 通过且接入路径明确就提交。最早真实模型实验在 Gate B 的 debug partition，本 Gate 不运行。

Agility budget：一个数学模块、一份模型窗口配置、一份针对性测试、一份证据总结；不新增 CLI/schema registry/receipt tree，不把模型评测框架重写为 Gate A 工作。任何额外验证须能改变 B 的 admission，否则列为可选并停止。

## 6. 验收与修复

Executor verification：目标测试、只读现场事实与最小接入设计。
Planning acceptance：1) 四窗口/模型 config 一致；2) 投影/同范数关键测试；3) 可实现的最小 opt-in 干预和原评分路径方案。运行合同待定不视为本 Gate 工程失败，但锁住 B。
Phase-end audit：留待最终配对结果、完整身份和 claims 的一次综合检查。

Gate 内低风险本地工程修复次数不限，须持续有进展且不扩路径/科学/远程权限；规划返修最多一次合并材料性修复，原根因或直接回归才可第二次。纯格式/非运行路径问题不阻塞。遇到缺少具体 operational 权限发最小 BLOCK 由规划补充，不自行越权。

## 7. 终态交付

只在 Gate 完成或确实无法继续时发一个 GATE_A_FINAL_AUDIT 或 BLOCK，包含 exact executor、起点与终点 branch/commit/dirty、四文件变更、命令/exit codes、HPC 路径/实际版本、发现、未运行的事项、请求规划 PASS/PASS_WITH_FIXES/BLOCK。你不能宣告自己的 PASS。

必须用 send_message_to_thread 主动发送给规划任务 01a06fe9-c7bb-7d72-a906-234f301de317，保留工具确认；本地 final 不能替代送达。工具不可用时输出 TERMINAL_DELIVERY_UNCONFIRMED 可粘贴包。无长作业，不创建 heartbeat。发终态后不进入 Gate B。
