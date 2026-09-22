# Gate B：真实debug与无标签诊断

按照 [$research-gate-orchestrator](/Users/huangxutao/.codex/skills/research-gate-orchestrator/SKILL.md) 的协作约定执行，实际读取skill及references/protocol.md；HPC操作先读hpc2-hkustgz-ssh。

Planning/终态接收者=01a07ff5-79b1-79f0-809a-4d0971475686。
Executor=01a0c72c-5462-71c3-8a7f-a2c4862680c1。
Title=Gate B-execute-LoopScope-Debug-and-Diagnostics-第9阶段；gpt-5.6-luna/max。

## Admission、身份与科学

A已由planning验收PASS，代码8f785198c18889f78c2cce7e3f26395bc7dd5cfb；实际启动HEAD包含后续planning文档提交，记录现场HEAD。唯一专用本地clone为项目外层下loopscope-tflt，branch loopscope；保护祖先4f59bd93eca4da3cbf458a93508f91c5b23912bc。每次写入核对path/branch/HEAD/dirty，不回退他人编辑，不进入原复现checkout。

科学完全按contract_v2：4B13:16/15:18、1.7B12:15，K2/3/4，alpha1，batch1，dtype/cache/model/data固定。两个取消窗口不运行。Online-t0与Matched-norm不改为近似SVD，不扩大lambda/rank/token子集。B只用原固定sanitized validation512及合成张量，不读test正文、test scores/gold、validation目标gold/correctness/accuracy，不进入C。

## 工作包与可写范围

1. 首次HPC只读检查专用clone状态/已有jobs/环境/缓存/校准pool，确认不影响历史任务。原Phase8基线、basis、结果永远只读，不接管原jobs/monitor。
2. 在src/tflt/loopscope/phase9_*、scripts/loopscope/*phase9*、tests/test_phase9_*、docs/loopscope_phase9.md新增/补齐真实模型debug、无干预诊断collector、pool读取、输出与verifier；允许configs/loopscope/phase9_*加入执行字段，不改冻结科学。control/contract/handoff仅planning可写；executor仅写gate_b_evidence。
3. 现有run_phase9_debug.py只有manifest/合成tensor入口，需要补齐原HFLM full-continuation评分和实际LoopConfig callback连接。复用phase8稳定评分/加载/pool工具，不重新实现评分，不改任何phase8_*、wrapper/strategies/cache/config。若确需protected修改，向planning提交最小理由/diff。
4. 修正并测量真实GPU SVD耗时：显式同步或CUDA events包围被计时代码，避免异步launch时间当SVD时间。临时全token矩阵只在当前prompt保留，诊断结束释放，不能持久化512×S×d。bounded诊断记录允许保留标量及必要单题参考。
5. 原MMLU validation512身份按Phase8既有sanitized pool直接复用。该池含dev示例答案是正常prompt；禁止读取目标题label。若只找到含gold源文件，先由无标签字段导出器构建sanitized池，不打印/保存gold，记录路径，不能自己换样本。

## 真实debug最早路径

先定向本地测试/CLI/bash-n，通过后debug分区单次上限29分钟、1GPU，每次代表最小subset，两模型与三窗口/K2/3/4通过有界作业覆盖。先少量真实prompt（每模型固定前2题加从sanitized pool按token长度选最长1题），不按答案筛选；必要时分job，不能降低科学配方以塞进时限。

决定性检查：
- 真实native residual、t0不改、t>=1仅p变化、正确调用K次（stash不计）；原Loop与关闭干预/零strength原选项分数等价，不能只看tensor finite。
- exact prefix mask不含continuation/padding/special；同题各候选独立重算t0参考方向与缓存路径投影一致、候选顺序改变不污染；跨题方向释放。若p不在S，不能改变科学而强行把special token加入S，报告具体触发证据。
- BF16/FP16真实投影；CPU FP64精确SVD参考的正常谱间隔样本相对投影误差<=1e-3（零参考范数使用明确绝对误差），近重根报告谱间隔和误差、不按几何筛题；若材料差异持续，返回具体科学问题给planning，不换算法。
- 两新arm实际有干预，Matched-norm按自身轨迹；K2同状态范数对照，K3/K4不声称跨臂逐步等范数。
- GPU/CUDA/加载/hook/写出/verifier链路全部完成且无非有限。写清0strength仍估计的额外成本，不用它代表Loop速度。可在保持科学不变下修复真实路径问题并新路径重试。

## 正式无标签诊断probe

同commit/launcher/env/runtime/producer的debug preflight通过后，授权在普通用户合法最高优先级资源运行512诊断：三窗口×K2/3/4各512条无干预Loop轨迹，共4608配置—样本；收集E0、A_t、C0t、谱间隔、norm比、token数和耗时。v_t仅诊断，不用于主干预。分母零明确null/原因。禁止由这些诊断筛除窗口或改变方法。

对两新arm性能只需相同代表性长度集（固定每模型前2题和最长题，以及debug暴露的真实上尾样本），不要求512×两臂全部评分。输出原生选项分数可用于等价检查，但不得接gold或报告acc。

资源：B同时最多2GPU，总累计24GPUh含debug/失败/重试；单formal job<=6小时，普通用户合法partition/QoS，不锁GPU型号。初始每GPU8CPU/128GiB host memory以内，实际节点上限更低则合理收缩。正式packing1–3依据当前卡型/长序列实测，至少15%显存余量，不改batch/dtype或跨进程共享模型张量。若SVD临时分解占用大，降低packing而非换算法。每次新launcher修复先一个child/canary再扩。新卡型/资源形状需实测，保留可用canary。

只允许本B失败/超预算/明确OOM风险jobs的有界取消；不因PENDING随意取消重提，不动历史/其它任务。预算不可满足时向planning申请补充，保留运行证据。

## 远端路径、Git与修复

SSH alias hpc2-hkustgz，strict/BatchMode/ClearAllForwardings=yes；未知key走专用恢复，不自动接受。远端代码 /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope；workspace /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope 的inputs/runs/staging/artifacts下新增phase9-gate-b-<timestamp>-aN。HF/dataset缓存沿用/hpc2hdd/home/xhuang225/shared及已固定环境，不升级。

允许普通本地提交/push origin loopscope至现有GitHub仓库，先查文本范围；远端仅clean且未影响活动job时fast-forward同步专用clone或使用项目内新源码快照。不得force/rebase/reset。允许已批准环境中CPU/import和必要dependency缓存检查；如缺依赖先报告确切需求，不能无声升级锁定栈。低风险工程/launcher/serialization/resource修复可持续，保留失败、新路径仅补无效缺失工作；不改科学/信息边界。

最多3个并行bounded子agent，不嵌套、不调度/远端写入；你负责所有Git/GPU/Slurm/集成/终态。分配文件ownership且不回退他人编辑。

## 验证、Agility budget与验收

命令最低：git diff --check；PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase9_*.py'（非零用例）；新增launcher bash -n、runner/verifier --help；实际debug/diagnostic命令及exit记录一份evidence。只有受影响旧接口回归才运行对应旧测试，不全历史重跑。

最小增量是真实HFLM callback+无干预collector+一份诊断verifier，先跑少量真实题，再512，不建设通用审计平台。主路径/资源证据足够即完成B，不增加防御性矩阵。planning只验收真实debug语义、9×512诊断闭合、资源/成本与C可运行预算三项。输出C建议总GPUh、卡型候选、packing/吞吐、前向/SVD分项成本；不凭GPU型号名称估算。

## 监控与终态

唯一executor-owned heartbeat：debug稳定RUNNING约1分钟且无启动错误后smoke10min；formal probe真实jobID返回即从PENDING建probe30min，类型切换更新同一个。健康无变化安静。终态先暂停再向exact executor发AUTOMATION_TERMINAL_RESUME，确认送达；失败是attempt事件，先诊断原范围修复；自动触发不可用则向planning发AUTOMATION_RELAY_REQUIRED。不把空final当恢复。

完成暂停monitor，唯一GATE_B_FINAL_AUDIT主动发planning并确认送达，含起止revision/dirty、代码/命令exit、job/root、preflight、512身份计数、无gold证明、数值诊断/成本/资源/OOM、偏差和申请PASS/PASS_WITH_FIXES/BLOCK。无新C/test/gold权限，绝不自PASS。未确认送达输出TERMINAL_DELIVERY_UNCONFIRMED。
