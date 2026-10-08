# Gate B — AUTHORIZED

2026-10-08。Executor01a11af4-0dbc-7fd0-92ec-bb66169ea433/local，标题Gate B-execute-LoopScope-Debug-Smoke-第11阶段，gpt-6.1-sol/high。Planning01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local。Gate A PASS，base c60c091aa9a09d15b9cee55e2b19bedf0950d23c。用户授权本阶段独立Gate、跨线程派发/修复/终态与连续推进。

按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，实际读取skill/protocol、AGENTS、control、contract_v1与gpqa_source_binding。科学合同不变，当前只授权B。

## 目标与Agility budget

先运行已存在check_phase11_gpu.py真实合成CUDA检查，再三任务synthetic producer→attempt verifier闭合，再最小代表性序列资源测量。修复仅可信正常路径缺陷。通过后立即交付，不构建通用框架，不重审A或历史Gate，不跑正式test forward或读取outcome。长生成路径不以3token smoke假装2048token显存验证。

## 身份与路径

写入前核对嵌套loopscope-tflt真实根、loopscope分支、HEAD/dirty及固定基点祖先。本地写入限Phase11源/测试/scripts/config工程元数据、docs/loopscope_phase11.md、B证据；科学字段不改。允许targeted测试后单一目的commit与普通push origin/loopscope，规划本轮文件可一并提交，不stage其他改动。protected wrapper/strategies/cache/config默认只读，必要时最小复现+拟议diff交planning。

HPC2 host=hpc2-hkustgz，user=xhuang225，按SSH skill禁forward、密钥改变停。旧source /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope保持只读，不切分支、不更新旧环境。允许以本地git bundle或独立clone部署完整Git源码至专属workspace/staging/phase11-gate-b-*，真实branch loopscope/HEAD必须匹配提交；不伪造.git或绕开producer源码检查。修复后新commit、新source路径。Python复用旧source/.venv-loopscope-cu121-20260711/bin/python，cache只读。远端新写入仅专属workspace/{inputs,staging,runs,logs}/phase11-gate-b-*，A所有inputs只读。输出不覆盖；本地无模型/数据下载。

## GPU与修复权限

所有B任务仅debug，每job<=00:29:00、1GPU、<=8CPU，最多同时2GPU，累计allocation GPUh上限12（失败尝试计入）。现场核对debug普通用户资格、GPU类型/内存和QoS；优先可用A40或其他合适非A800卡，避免A800例外歧义。hostmem按现场限制和需要申请，不用特权资源。不得把B预检改名正式canary转投正式分区。可取消自己确认为B的失败/无进展作业并释放资源，低风险修复后新路径重试；不得取消其他阶段/用户作业。科学异常不能重试掩盖，提交不足或预算不足带实测证据回planning。

输入先用A remote_ready_paths中的synthetic pools；资源合成prompt可根据A已绑定长度2860/2819和ARC1242/46补普通token，严禁用真实test forward。仅工程资源路径允许强制固定2048生成步测试最坏KV增长，明确PREFLIGHT_ONLY，不能替代正式greedy EOS行为。先测packing1；仅在峰值与安全余量支持且预算允许时测独立进程packing2，batch1、无模型共享、输出隔离。异GPU测量不直接作为A800 packing依据；若debug无法容纳代表性资源形状，如实报告限制给C规划。

## 实施与决定性验证

1. 只读检查环境/调度，部署完整已提交源码。按现有preflight launcher运行check_phase11_gpu.py，必要时拆分小检查，<30min。验证Native接口、Loop/lambda0、K2fixed/lag1同请求等价，K3三策略实际方向bank及decode无SVD、K次body/单stash、所有层cache增长。lambda0.1/0.9至少补一项参数传递/真实应用抽查，非新增面板。
2. 三任务各最小synthetic producer实际输出并运行verify_phase11_scores.py --scope PREFLIGHT_ONLY --attempt。MMLU/GPQA至少prefill+两decode步；ARC完整候选likelihood与原生HFLM一致。实际提交和验证参数保留，不仅--dry-run。
3. 最少代表性生成与ARC资源实测：模型加载峰值、prefill/解码峰值、分阶段时间、tokens/s、题吞吐、GPU类型/内存、packing及OOM。包含达到冻结生成上限的KV形状；不能因synthetic早停低估。记录实测支持的C时间/成本范围和不确定性。真正固定2048步若29分钟不够，保持最小路径并回报，不擅改科学。

允许B新增最小资源脚本/launcher与针对性测试，修改关键路径后重做受影响debug。可最多3并发subagent，明确文件所有权、不覆盖他人；调度/远端写入/Git整合/终态仅executor。低风险工程迭代不限次数但受12GPUh与权限约束。实验形状/数据/模型/策略不得改。无目标gold读取、accuracy计算、完整正式panel或下一Gate。

## 监控与终态

用户已授权skill监控协议。短debug只有真实RUNNING约1分钟且无启动错误才创建一条executor-owned smoke heartbeat每10分钟，未变状态静默；更大资源probe可将同一heartbeat改30分钟，绝不重叠。快速结束无需监控。任务终态暂停/删除monitor并触发exact executor继续诊断/闭合，失败是attempt事件，不直接Gate BLOCK。monitor仅观察、无独立提交/重试权限；terminal resume发送本executor，relay发送planning，均保留工具确认。不得让planning轮询。

最后发送唯一GATE_B_FINAL_AUDIT或真正BLOCK至planning，附source commit/branch/dirty、精准命令/exit、job/partition/time、fresh run路径、tests及上述三项证据、资源建议和明确未做项；不自判PASS。消息未确认则输出TERMINAL_DELIVERY_UNCONFIRMED。规划只查1至3项材料性证据，普通audit-returned repair最多一轮（同根因例外按skill）。验收前C锁定。工程命令入口参考docs/loopscope_phase11.md，不把未执行计划写成成功证据。
