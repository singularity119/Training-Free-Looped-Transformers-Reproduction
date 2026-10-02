# Phase 10 Gate A：Lambda-and-ARC-Adapters

按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行。执行者必须实际读取该文件及 references/protocol.md，再读 AGENTS.md、PROJECT_MEMORY.md、当前control、本handoff与科学合同。不得仅依赖继承上下文。

## 身份、准入和基点

- Planning：01a0fae2-12df-74e0-a874-190518ef0501，host local。
- Executor：01a0fb34-5c36-7210-b17d-744410853d1f，host local；已精确绑定，仅本Gate权限。
- 标题：execute-LoopScope-Lambda-and-ARC-Adapters-第10阶段-Gate A。
- 配置：gpt-6.1-sol / high。用户明确允许最多同时3个subagent；仅限本Gate具体子任务，禁止嵌套委派、独立调度作业、跨Gate或代替executor发终态。必须划定各自文件所有权，并告知共享代码库、不得回滚他人修改。父executor整合、提交与交付。
- 用户2026-10-02已授权planning逐Gate派发推进Phase10。A准入成立；B–E仍锁定。
- Git根：/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt。
- 分支loopscope；基点ee9feb4dbee1caca53a2fc23f02527c2bbe05106。初始dirty仅planning生成的 .planning/README.md 与 .planning/phase10/，必须保留，允许随A单独治理提交纳入Git。写入前检查root/branch/HEAD/status和保护祖先4f59bd93eca4da3cbf458a93508f91c5b23912bc。

## 目标与最小实现

实现第十阶段可执行配置、面板、评分适配和分析路径，使Gate B可立即进行真实预检。冻结科学见contract_v1及其引用的plan §§2–4。

1. 41个独立cell/数据集；59行展示包含18个K2等价别名。三组模型窗口K、两方向策略、lambda0.1至0.9，只有Online、Loop、Native。明确拒绝Matched-norm进入Phase10面板。
2. 复用Phase9Runtime及Phase9GateERuntime或只读辅助函数，新增phase10旁路入口，不放宽历史phase9校验、不覆盖旧runtime。lambda、direction_policy和dataset须实际传至producer并写入可读cell身份。
3. MMLU严格保持原评分、模板与revision。ARC接入lm_eval0.4.11标准任务、25-shot、完整choice文本、主acc_norm附acc，支持实际选项数和数字标签；真实pre-answer与有效prefix屏蔽必须保留，多token continuation不进入SVD。
4. 用官方metadata及HPC缓存只读确定ARC数据revision、tokenizer max_length等事实，固定seed20261002；示例抽取采用该版本原生sampler，记录身份。可在HPC专用inputs/phase10-gate-a-<timestamp>/创建无标签输入和示例清单，train示例答案可读，正式test目标gold不可导出/查看/用于配置，validation只作无标签工程样本。读取数据载入器时尽早剥离target标签；不运行模型。缺失数据只报告确切依赖，A不下载大数据或模型。
5. 检查ARC全test无标签token长度/候选截断是否一致；若标准25-shot出现候选特异prefix，给出最小实例、候选解决办法及影响，交planning决定；不得删题、降shot、缩短答案或自行改评分。
6. 新增统一闭合验证、历史9cell复用mapping（A只核配置/路径/schema，不读逐题score/gold）、分析和迁移lambda选择程序。分析仅用合成数据测试，禁止现在分析真实outcome。
7. 提供B可用的debug/preflight与formal launcher/dry-run，但A禁止提交或运行模型。日志/新产物按write-once根规划。

## 可写路径与操作权限

本地新增/修改仅：src/tflt/loopscope/phase10*.py，scripts/loopscope/*phase10*，configs/loopscope/phase10*，tests/test_phase10*，docs/loopscope_phase10.md，.planning/phase10/loopscope_phase10_gate_a_evidence.md 及事实性 data binding 附件。planning所有control、contract、handoff、plan不得由executor自行改权；事实附件交planning接纳。允许新增同命名小辅助文件，不做通用重构。

只读：Phase8/9源码配置、旧报告工件及protected wrapper.py/strategies.py/cache.py/config.py/eval_runner.py。必须改protected路径时先送最小复现及拟议diff，不得自行实施。

Git：允许专用clone正常fetch、单目的commit、非强制push至origin/loopscope；不得rebase/reset/stash他人修改或改main。远端分叉先保留并回报，不自主merge科学差异。允许将本轮已有planning文件独立提交，再提交实现。

HPC：先实际读取 hpc2-hkustgz-ssh skill；仅hpc2-hkustgz alias，严格host key且ClearAllForwardings=yes。可读项目专用source /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope、workspace /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope、共享/hpc2hdd/home/xhuang225/shared下相关环境/cache/数据。允许专用source保持loopscope的fast-forward-only同步（clean且符合分支后），现有环境中targeted CPU/import/help/tokenization检查；只可写专用workspace的inputs/phase10-gate-a-<timestamp>与staging/phase10-gate-a-<timestamp>。禁止修改旧run/cache/固定复现clone、安装升级环境、下载模型、大规模数据下载、GPU/Slurm提交/重试/取消、模型forward、真实accuracy分析。SSH暂不可达时先完成本地独立工作，再交具体缺口，不以连接失败伪称实验失败。

## 定向验证与交付

从Git根运行并保留exit code：
- git diff --check
- PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase10*.py'
- PYTHONPATH=src python3 scripts/loopscope/build_phase10_manifest.py --help
- PYTHONPATH=src python3 scripts/loopscope/run_phase10_accuracy.py --help
- PYTHONPATH=src python3 scripts/loopscope/verify_phase10_scores.py --help
- PYTHONPATH=src python3 scripts/loopscope/analyze_phase10.py --help

入口名称按上述实现；如需要最小合理变更，在runbook列出实际等价命令。纯Python测试无需torch，真实张量/模型等价留B。必要时补受影响的旧adapter测试，不重复全套suite。

必须通过：面板41独立/59展示及准确reuse mapping；lambda及policy传递；K2时间表等价；t0不干预及p位置边界；ARC候选完整文本与标签映射、acc_norm遵循原生版本；lambda0与0.5兼容设计；合成数据分析的8项迁移family/72项扫描family/9项K3方向family与固定选参tie-break。代码级fake验证不能冒充GPU验证，未验证项明确交B。

证据：一个gate_a_evidence.md，列branch/commits/dirty、改动文件、精确命令与exit、最小验证结果、remote访问/事实绑定、未完成风险及B建议。无需摘要值或多层receipt。Gate B资源仍未授权。

## Agility budget / 验收 / 修复

最小增量是phase10旁路和ARC适配，不改通用框架、不重审旧PASS、不补512诊断。上述关键检查足够后交付，不扩展假想异常矩阵。最早真实模型预检属于Gate B。

executor负责工程验证；planning仅检查面板与科学参数、评分边界、定向测试与数据binding三个关键方面；跨Gate审计留阶段末。低风险同权限工程修复不限制次数，有进展就继续；科学、数据、权限改变回planning。普通审计最多一次合并退修，第二次限同根因或引入回归。

## 终态交付

仅完成或真正阻塞时，向planning线程01a0fae2-12df-74e0-a874-190518ef0501（local）发送GATE_A_FINAL_AUDIT或BLOCK，列真实证据，请求PASS/PASS_WITH_FIXES/BLOCK；不得自判PASS。用户已授权此Gate终态、修复交付与planning往返消息。通过send_message_to_thread保留成功响应，若失败输出TERMINAL_DELIVERY_UNCONFIRMED可粘贴副本。只发一次正式终态，修复后用明确superseding包。不得进入B、建下一Gate或启动heartbeat（A无作业）。
