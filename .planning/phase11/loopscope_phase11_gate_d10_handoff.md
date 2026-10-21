# D10 — Thinking-2507 GPQA greedy8192

COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
EXECUTOR_THREAD=01a15ddb-5ea0-7112-8468-62b463c996b5
PLANNING_THREAD=01a15d9c-ee8b-7502-97b2-a40f732e00f5
HOST=local

按已读取research-gate-orchestrator skill/protocol执行，用户在此既有独立executor直接指定新实验，本次按其选择沿用线程，D9B保持PASS_CLOSED。本handoff同步已核实用户授权，不要求重复问提交许可。D10作为独立配方/运行/评分，不重写D9结果。

模型oss://sampool-bucket/deltalake/other_sample/users/270228/huggingface/Qwen/Qwen3-4B-Thinking-2507/，实际revision以指定OSS副本未绑定HF如实记录；读取小metadata、tokenizer及实际版本，不恢复已取消逐字节来源比对。原GPQA Main448、原CSV/单rng20261008选项顺序、0-shot题面沿用，使用新模型自身模板并确认thinking呈现。三组Native/current_t K2 λ0.5/fixed_t0 K3 λ0.9，window15:18、alpha1/beta0、block damped_euler、cachefirst/decodefull及原方向拟合不变。用户明确D4 greedy/do_sample=False、max_new_tokens8192和整个输出文本S2（包括生成thinking文本）；不得仍套D8 first-think-close后final-only，不从合法答案冲突中择优。448分母、三组1344，完整raw/预测封存后gold，原两项McNemar/Holm和配对bootstrap作为当前预先指定比较；新模型+长度+解码+评分区域同时变化，不声称单因素因果。

资源H20-8x-640G总8卡/quota h20_agi_large，与D9的16卡额度区别；先定向验证新模板/greedy/整段S2/上下文/8worker路由，再按用户明确请求提交。使用现有灵境运行包机制，真实启动/错误/显存可观测；不把本地测试或旧被豁免预检称新CUDA通过。代码repo /Users/singularity/Desktop/tflt_digai/tflt，branch loopscope，起点a6d305d；保护现有dirty与旧D8/D9/数值核心，仅新增隔离thinking2507模块/launcher/测试和最小必要接入；fresh资产全部在oss://sampool-bucket/deltalake/other_sample/users/557030/tflt/下独立D10 root。源GPQA已发布只读复用，不反复获取；不连接HPC。仅stage自有文件和获准planning记录，不提交模型/题集/raw/bundle。

允许真实提交、同科学低风险工程修复和fresh缺失项恢复；源与旧任务只读、不取消其他任务、不覆写成功记录。不确定的科学选择向planning报告，由planning向用户集中确认，不能自行调参。Agility budget限新checkpoint/greedy/whole-output parser与部署必要最小改动；定向测试通过即提交，不重复已PASS审计。真实job后唯一full60min monitor，健康静默，失败attempt恢复原executor；终态先暂停，再给本planning GATE_D10_FINAL_AUDIT并确认投递；不能投递给TERMINAL_DELIVERY_UNCONFIRMED。规划验收只查模型/输入边界、完整性、冻结评分决定性证据。负结果可PASS，不自动追加新实验。
