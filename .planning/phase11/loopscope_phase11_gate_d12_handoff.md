# D12 — 原版 Qwen3-4B nonthinking GPQA

COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
EXECUTOR_THREAD=01a15ddb-5ea0-7112-8468-62b463c996b5
PLANNING_THREAD=01a15d9c-ee8b-7502-97b2-a40f732e00f5
HOST=local
STATE=AUTHORIZED_IMPLEMENTATION

planning直接读取用户注释请求（userMessage 01a15fe1-eb00-7c53-8a21-f77af9a9b023）及澄清回复（01a15fe3-386d-72f2-9c28-d7c3132580a5）：“greedy，整段作为final”。标注本次灵境指task4953520，原版Qwen3-4B。所有科学选择已明确，允许准备完成后直接提交，无需重复确认。用户在既有executor明确指定本实验，沿用该线程为例外，D12独立新Gate，不重开D9B。

模型oss://sampool-bucket/deltalake/other_sample/users/270228/huggingface/Qwen/Qwen3-4B/，原生模板enable_thinking=False，greedy/do_sample=False，max_new_tokens32768；整个生成文本作为final，用原S2语法，合法答案冲突及无match计错，448完整分母。保持原GPQA Main448身份、顺序、shuffle seed20261008及generation seed20261018+canonical_index。Native/current_t K2 λ.5/fixed_t0 K3 λ.9，各448共1344；window15:18 inclusive、alpha1 beta0 h1/K、block damped_euler/cachefirst/decodefull及原rank1 FP32方向语义保持。完整预测封存后才读gold，两项配对McNemar/Holm及10000 bootstrap seed20261018保持。相对4953520同时改变thinking、采样到greedy、评分区域，不能声称thinking单因素因果。

资源H20-8x-640G总16卡、quota h20_agi_large_16，双节点各8独立单卡worker，batch1/packing1。当前checkout /Users/singularity/Desktop/tflt_digai/tflt；只增加隔离D12 recipe/worker/launcher/tests及必要接入，保护其他dirty和既有科学核心。新资产统一users/557030/tflt/phase11/d12/fresh-root；模型只读，实际revision UNVERIFIED_OSS_COPY，不将历史HF revision写为当前验证结果。禁止HPC，不复活已取消的独立来源比较或独立容量准入；保留真实启动/设备/内存观测，未验证容量如实记录。

沿D11交接的低风险工程修复范围，定向核对模板关闭thinking、greedy、whole-final S2、32768、H20设备识别、双节点1344分片并集及启动链路，然后提交。工程失败允许同科学fresh恢复缺失/无效项，保留失败回执，成功样本不覆盖。D10/D11及各自监控保持独立，不改不取消。D12真实受理后exact job绑定自身唯一full60min monitor，健康无变化静默，终态暂停并唤醒executor，按research-gate-orchestrator向planning投递GATE_D12_FINAL_AUDIT。规划1–3决定性证据验收；当前仅授权实现，不代表受理、运行或完成。
