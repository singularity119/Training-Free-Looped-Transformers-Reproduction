# Gate A handoff — AUTHORIZED

2026-10-08正式绑定补充，优先于下文历史未派发措辞：executor=01a11ac8-ca15-7573-8ff6-c1dc0892faab，host=local，gpt-6-luna/max；planning=01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f，host=local，已工具核对。用户已明确授权独立Gate创建、跨线程派发/回传/修复及逐Gate推进。当前只授权A。

允许针对性验证后单一目的commit并正常push origin/loopscope，包含Phase11规划包；只stage本Gate文件，不force或改写历史。此条取代下文不自动提交推送限制。终态必须通过send_message_to_thread送至上述planning并保留工具成功确认；交付失败输出TERMINAL_DELIVERY_UNCONFIRMED，不声称已交付。当前无GPU长任务，不创建监控。

Agility budget：只做Phase11旁路适配及所列关键测试，通过即交付B预检，不构建通用框架。最多3个有独立文件责任的并发subagent，不覆盖他人修改；子agent无远端写入/调度/Git/终态权限。HPC host=hpc2-hkustgz，user=xhuang225，source=/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope只读；新CPU写入仅/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-a-*及staging/phase11-gate-a-*，cache按AGENTS只读。缺操作权限向planning给最小材料性证据，不自行扩范围。

拟定标题：Gate A-execute-LoopScope-Generation-Adapters-第11阶段。Executor尚未绑定，本文本身不授予执行权限。创建时使用用户配置的默认模型，除非用户明确指定模型/推理档位。

按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，实际读取skill与references/protocol.md。政策AGENTS.md、Phase11 control和contract_v1共同约束；旧阶段control无授权作用。

## 目标及最小增量

在loopscope专用clone/branch上新增Phase11 panel、输入适配、prefill方向银行与incremental decode回调、生成/ARC评分入口和闭合/统计接口。优先复用既有wrapper residual_transform，Phase9/10历史数值模块只读。关键风险是跨decode步错误重置方向、当前位置索引、cache写入与输入标签泄漏，不建设通用框架。

首次写入前检查git status --short --branch、rev-parse HEAD/show-toplevel和固定基点祖先。起点81f873b38942474a012e2648db3e81d3085abdd6；Phase11规划文件允许未提交，其他变更保留不覆盖。若branch已推进，核对diff与合同相关路径后向规划回报，不reset。

## 允许范围（绑定后）

本地src/tflt/loopscope/phase11*、configs/loopscope/phase11*、scripts/loopscope/*phase11*、tests/test_phase11*、docs/loopscope_phase11.md与本Gate证据。不得修改wrapper.py/strategies.py/cache.py/config.py；确有需要，先提交最小原因与拟议diff给规划。不得修改Phase10控制面。

允许本地CPU/fake-tensor针对性测试；本地无模型/数据下载与forward。按HPC2专用skill允许项目相关只读SSH/环境/cache元数据核对，允许专属workspace新inputs/phase11-gate-a-*、staging/phase11-gate-a-*的CPU输入构建与tokenization；远端源clone、历史环境和旧runs只读。模型权重不下载；缺数据访问先报告，常规已授权可访问的小型数据只进入新远端input根。完整CoT/raw数据不进Git。无GPU/Slurm、无正式生成、无目标outcome、无下一Gate权限。不自动提交推送，交付可审阅diff与证据。

## 具体工作

1. 绑定模型/tokenizer与三数据revision、实际split/题数、示例identity、选项顺序及完整token长度。GPQA构建器可接触正确答案字段用于选项构造和密封映射，评分器/模型输入不能获得gold角色。展示仅合成模板，不输出真实GPQA题。
2. 固定18独立臂和K2别名；生成任务真实prefill一次拟合所需v_t，后续单token解码复用；ARC完整候选评分保留prompt-only方向/干预位置，不误将完整候选forward当生成decode。
3. 实现strict Final answer提取、EOS/长度终止分类、完整分母与失败记录、fresh输出路径和任务内identity闭合。实现冻结统计，不读取真实gold执行分析。
4. 产出B可直接运行的最小debug命令与工程合成prompt，验证至少两个解码步的K次调用和方向冻结；写明待真实GPU验证项，不能以fake测试宣称模型等价。

## 验证与交付

git diff --check；PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase11*.py'；新入口--help/--dry-run。覆盖54配置、K2别名、跨题reset、decode无拟合、逐t方向选择、lambda0无改动、非目标位置不变、合法/重复/无效答案与truncation计分、标签不入prompt、完整性缺项拒绝。只做可信正常路径关键检查，不扩大全套历史审计。

交付gate_a_evidence.md、data_binding.json和可审阅diff，包含实际命令/exit、路径/versions、差异、未验证项。规划核对面板/输入绑定和方向时序三项决定性证据后验收。允许同范围低风险工程修复；科学变更或受保护路径变更回报规划。审计返回修复通常最多一轮，材料性问题才返修。

终态唯一GATE_A_FINAL_AUDIT或真正BLOCK，包含executor身份、branch/revision/dirty、文件/命令/证据、未做事项、请求规划决定；接收线程须派发时工具核对并写入control。只有用户明确授权跨线程消息后才工具发送；否则最终输出TERMINAL_DELIVERY_UNCONFIRMED粘贴包，不声称已交付。当前未派发，不创建监控。
