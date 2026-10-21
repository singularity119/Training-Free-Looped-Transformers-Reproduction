## 2026-10-22 最新资源覆盖：8卡恢复

planning直接核实用户回复01a15ff1-15ee-7103-8bf2-a6837574cf1e：“搞错了，超额度了，改为8卡，其他不变”。D12后续使用H20-8x-640G总8卡、单节点8独立worker，quota仍h20_agi_large_16；覆盖此前16卡/双节点配置。其余原版Qwen3-4B、enable_thinking=False、greedy、32768、whole-final S2、三组448共1344及seed/loop完全保持。失败原因为user-reported quota，未由平台日志独立证实；不再等待旧启动日志，不需重复许可。保留4954247/4392120/21150146 FAILED及0launch/0records证据。定向验证单节点bootstrap/build和8GPU分片1344并集后fresh提交，旧monitor d12-qwen3-thinking在新真实job后更新同一个exact绑定，当前PAUSED；D10/D11不变。

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

## 真实受理记录

- task4954247 / modelTrainingJob4392120，name expr_gq_q4_nt_r_20261022031602，backend tj20261022031741-560a3053；2026-10-22 03:17:41 China初受理NEW，尚不代表CUDA启动或完成。
- runtime source 25f8337a336bfc65dfd8b85ef4b88992fc12f275；OSS root `oss://sampool-bucket/deltalake/other_sample/users/557030/tflt/phase11/d12/20261021T191602Z/`，九文件代码/配置/计划及真实receipt逐文件回读一致。
- 独立monitor `d12-qwen3-thinking`（可见名D12 Qwen3关闭thinking全量）ACTIVE/full60min/exact executor；D10 d9b和D11 d11-instruct2507继续独立。machine receipt `.planning/phase11/gate_d12_submission_20261022.json`。
- 10项定向验证通过；模型实际revision UNVERIFIED_OSS_COPY，runtime smoke尚未观察，长容量NOT_MEASURED。用户确认greedy/wholefinal，保留原32768与三组各448，非thinking单因素因果实验。

## 用户8卡资源修订及fresh恢复（2026-10-22）

用户明确回复：“搞错了，超额度了，改为8卡，其他不变”。该最新指令覆盖前述16卡/双节点资源要求，当前为H20-8x-640G总8卡、单节点、quota h20_agi_large_16。旧task4954247/model4392120/fission21150146平台FAILED，OSS无launch/完成记录；quota原因来自用户说明，未冒称平台失败日志独立证实。原receipt与失败证据保留。

- fresh task4954265/model4392138，name expr_gq_q4_nt_r_20261022033048，backend tj20261022033145-0ec4ada6；2026-10-22 03:31:45 China初受理NEW，03:32:49状态API再次HTTP200确认NEW，尚无启动证明。
- source cc2025c27dbfe32ecbf4f821fcea02848e6009d6，root `oss://sampool-bucket/deltalake/other_sample/users/557030/tflt/phase11/d12/repair8-20261021T193048Z/`。
- 11项定向验证通过；科学字段逐项与失败attempt直接比较一致，8worker×3组×56题=1344完整并集；九文件及真实receipt回读一致。模型/greedy/32768/wholefinalS2/seed/loop不变。runtime smoke尚未观察，长容量NOT_MEASURED。
- 同一唯一monitor d12-qwen3-thinking先PAUSED后更新绑定4954265/4392138并ACTIVE/full60min；不新增monitor，不改D10/D11。新machine receipt `.planning/phase11/gate_d12_submission_8gpu_20261022.json`，旧machine receipt保留历史。
