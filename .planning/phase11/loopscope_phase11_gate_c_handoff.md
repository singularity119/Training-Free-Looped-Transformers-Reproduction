# Gate C — A800 Formal Canary

2026-10-09用户最新操作修订：本阶段允许最多同时8张A800，覆盖下文max2GPU限制；受现场更低普通用户配额约束。每job1GPU、同卡packing<=2、C累计12GPUh/216记录及信息边界保持。不得为扩并发重复成功记录或盲取消运行任务。当前接续优先闭合已有任务并交付终态。

2026-10-08。Executor01a11b40-07d3-7342-9d01-f809b480f49e/local；planning01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local。模型gpt-6.1-sol/high。按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，读skill/protocol、AGENTS、control、contract_v1及GPQA source binding。B PASS，source基线4d4821810532a919140c35a7035e18f39339cf73，runtime已验ffb1e3824e85484e346b1090b7b5f6f572f6b4f3；科学不变。

## 目标与Agility budget

最小改动为正式launcher/调度清单/资源记录，不改核心runtime。同commit/launcher关键路径debug synthetic预检通过后，在A800获取可计入最终面板的216条正式记录，闭合此固定canary集合，报告真实生成长度、吞吐、峰值/设备内存和成本建议。只做决定D准入的工作，不运行全量，不提前解封gold。原D报告Gate顺延E，D为后续独立全量executor。

## Canary membership

每任务固定4个identity，所有18臂共享，共3*4*18=216。仅依据A frozen input的token长度排序（长度相同按canonical index），选排序下标floor((N-1)*q)，q=0.25,0.5,0.75,1.0；生成用prompt长度、ARC用该题最大actual candidate input长度。在任何正式forward之前输出仅identity/index/长度的固定membership并写一次保存。这是正式完整面板的资源分批，不是新测试子集或科学筛选。禁止依据预测、gold或输出选样。

优先使用现有--shard i/N对一个canonical index执行，不新增删题/重编号路径；每臂4个单例shard可合并调度，仍保留正式完整pool/manifest。每任务正式总N仍1172/12032/448；后续D按identity排除已完成canary，不能混淆4题闭合与完整任务闭合。canary raw保持完整原始输出但executor/planning只看运行元数据，不读生成正文、predicted choice、gold/correctness/accuracy或模型对比结果。

## 路径与Git

嵌套loopscope-tflt/loopscope，每次写入/提交/远程执行检查真实root、HEAD/dirty、固定基点祖先。本地限Phase11 launcher/调度/资源与必要最小engineer修复、targeted tests、docs、C证据。protected wrapper/strategies/cache/config只读，修改需向planning给材料性最小复现和diff。允许单一目的commit和普通push；规划包可同提交，禁止无关stage/覆盖他人修改。

HPC2 host hpc2-hkustgz/user xhuang225，按SSH skill；旧source和环境/cache不变。复用B pinned Python与已接受A inputs，只读。部署真实Git loopscope源码至专属workspace/staging/phase11-gate-c-*，同commit新路径；新写入限该workspace/{staging,inputs,runs,logs}/phase11-gate-c-*。无本地模型/数据下载。可复用原B只读Git对象传输方法，无伪造HEAD。source检查与实际commit一致。

## 资源与分区

C累计预算12 allocation GPUh（debug/正式/失败均计入），最多并发2GPU，每job1GPU/<=8CPU。非正式debug/smoke统一debug、每job<30min、合适非A800 GPU，遵守项目2026-10-02规则；正式可保留canary使用现场普通用户合法最高优先级A800 partition及最高合法QoS，核对资格/优先级，不能按名字猜。正式每job最多2h，未完成继续新合法任务，提交预留总预算不得超12。仅自己的C失败/无进展任务可取消，新路径工程retry保留失败；不能操作其他作业。不继承B12GPUh余额或Phase10无上限授权。

先packing1跑每任务最长样本的K3current_t lambda0.9（正式记录，成功后保留），直接测设备显存占用与进程峰值并依据实际A800容量及2048KV需求留至少20%显存余量。满足容量条件才允许packing2独立模型进程（batch1、<=4CPU/worker、无tensor共享、唯一输出）。不得把两进程allocator峰值相加说成设备实测。若packing2无明确吞吐收益则回packing1；比较不同题的单点不能宣称确定加速，只能作排程估计。超过2packing需后续planning新授权。最长题可能提前EOS，资源准入仍用已知全2048KV形状和实际显存估计保留余量，不把提前EOS当最坏验证。不得在正式分区额外跑不留存synthetic debug。

若12GPUh不能完成固定216，不提前超支，保留已完成并带实际成本回planning；普通工程修复在同budget内不限轮次。不可缩短2048上限、减少shot、换SVD/精度/模板或优化为不同科学方法。对数值不变的性能优化先提供材料性瓶颈及等价验证方案，不悄悄更改。

## 执行与验收

1. 核对现场scheduler、固定canary membership、写最小正式launcher。改launcher后同commit在debug运行三任务synthetic producer与attempt verifier，<30min且相同加载/输入/输出路径，无需重跑整套B checker。formal模式/分片边界CPU针对性检查，synthetic不冒充formal。
2. 只有上述通过才提交最高合法A800真正canary。按上述packing顺序，不重复成功的identity/cell。现有formal shard数据必须保留供D复用；记录source/config/model/data/seed不变。
3. canary membership完整闭合216，允许专用canary verifier以冻结集合验证，不改全任务closure准入或提前gold读取。仅汇总题耗时、prefill/解码耗时、长度、截断率、memory、OOM及成本；提取结果/失败率等答案相关内容留完整任务解封。报告54配置成本范围与prompt长度分层局限。给出D建议预算/并发/packing/分片、当前已完成复用索引和源commit兼容性；不把长度偏置的4题估计说成总体无偏预测。

真实预检/失败/调度/内存证据为must-pass，无需全套历史测试。最多3子agent、明确文件责任，只有executor做调度/远端写入/Git整合与终态。规划轻量核对运行状态、canary完整性、资源三项；材料性修复通常一轮。

## 监控与终态

正式job一经取得可验证ID（含PENDING）即建立唯一executor-owned full heartbeat每60min；状态不变静默，有多个C jobs由同一monitor观察。短debug稳定RUNNING约1分钟后可smoke10min，同一heartbeat后续改full，禁止重叠。terminal暂停/删除monitor并唤醒exact executor，若一批完成还需C下一批则executor按当前预算续跑；job失败不是Gate失败。可跨线程terminal resume给本executor、relay/最终给planning，保留工具确认。不由planning轮询。

完成唯一GATE_C_FINAL_AUDIT或真正BLOCK发planning，包含commit/dirty、命令exit、job/partition/QoS、budget、216闭合与资源、失败/偏差、monitor暂停确认、未解封/未扩展声明。未确认送达则TERMINAL_DELIVERY_UNCONFIRMED。不得自行PASS、不得进入D或无上限full submission。
