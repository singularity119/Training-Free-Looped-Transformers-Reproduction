# Gate B：按用户要求以流程验证收口

2026-09-22 用户明确要求：debug 512 诊断因运行时限仅部分完成，只要流程跑通即可尽快进入全量，并尽量利用 A800 显存。本补充覆盖原 B 必须完成 9×512 诊断的 admission；科学 contract_v2 与 47 配置全量面板不变。

执行任务仍为 01a0c72c-5462-71c3-8a7f-a2c4862680c1。按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，实际读取 skill 与 references/protocol.md；监控路由以 monitor_routing_amendment 为准，仅向执行器自身汇报。

立即停止补齐 validation512 的计划，不新提交这类补齐作业。核对 12832136/37 的终态、已完成产物和超时原因，保留全部原始证据；未完成或半写记录不得作为有效诊断样本。无需因纯时限中断重跑已验证流程。

B 最低收口证据：既有九配置真实 debug 全部有效；零强度等价、mask/顺序独立、t0 SVD 与后续答案位置处理正确；部分诊断能证明真实 producer 到落盘/读取的正常路径，无未解决数值或科学错误。汇总各 cell 实际有效样本数、缺失范围和超时状态，不将部分样本描述为全512分布结论。资源数据有则提供，无则明确留给 C 的代表性资源 canary，不为收集完整诊断拖延 B。

请尽快整理并发送 GATE_B_FINAL_AUDIT 给 planning，申请按修订标准验收。B 不直接进入 C、不读取 test/gold；规划验收后立即创建独立 C 执行任务。

C 资源准备方向：优先 A800，实际记录型号/总显存。冻结 batch=1、dtype、评分缓存、模型和方法数值行为，通过独立进程 packing 或有限 worker pool 填充显存；按各模型及长 prefix 的实测峰值和吞吐决定并发，不把 packing 称为 batch。避免 OOM 是边界，保留实际瞬态余量；不机械追求100%显存，也不沿用未实测的 packing 上限。由 C 在正式分区先跑可保留的最小有效 canary，必要的 launcher/runtime 变化先做短 debug；同卡代表性资源测量后扩展。正式资源预算和并发上限由 C handoff 绑定，此段不授权 B 提交 C 作业。
