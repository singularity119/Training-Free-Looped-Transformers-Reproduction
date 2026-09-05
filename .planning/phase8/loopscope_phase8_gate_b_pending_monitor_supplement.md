# Gate B operational supplement：debug排队期监控

Planning: 01a06fe9-c7bb-7d72-a906-234f301de317
Executor: 01a0704b-5798-7680-80a2-f1145f35f9ba
Date: 2026-09-05

## 需要与授权

executor报告真实debug job12645346为PENDING/MaxJobsPerAccount、StartTimeUnknown；仓库证据记录其submission、commit c7e4e1f、1.7B四window/K cell和资源。规划有界SSH复核因本地source-address错误未连通，不据此认定job失败或改变其状态。排队时间未知时，原“RUNNING稳定1分钟才建monitor”会令queued job缺少后续唤醒；本补充为这一实际调度情形授予最小观察例外。

允许绑定executor立即创建唯一10分钟smoke heartbeat，覆盖job12645346的PENDING期；创建前由executor用其已验证连接方式核对job仍存在及当前状态。原job进入RUNNING后沿用同一monitor，不再新建。未来本Gate原handoff已授权debug job若同样进入未知排队期，可以更新此唯一monitor的job集合，不创建重叠monitor，不增加调度权限。

冻结当前路由：host hpc2-hkustgz；job12645346；run root=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase8-gate-b-20260905T072208Z-q17-a1；logs在同workspace/staging/phase8-gate-b-20260905T072208Z-gpu1/。monitor必须绑定exact executor和planning IDs，引用原handoff与本补充。

## 观察与恢复

每次只作一次有界read-only状态/必要日志检查。PENDING/RUNNING且无新材料性问题时保持安静，不催促用户、不写逐次control记录。队列原因变化本身不允许改account/partition/QOS/资源或取消他人job。

成功/失败/取消/超时等attempt终态：先暂停monitor，再用send_message_to_thread向exactexecutor发送AUTOMATION_TERMINAL_RESUME，附job/状态/路径/下一步验证或原许可内根因修复，并保留工具确认。终态job不是自动Gate BLOCK。若触发不可用，发送AUTOMATION_RELAY_REQUIRED给planning，不能仅输出“可以恢复”。

SSH/DNS暂时故障不等于job终态；不得伪造Slurm状态。出现需要人工恢复的持续访问问题可报告访问证据并请求executor处理，仍不让monitor接管实施/调度。

本补充只改变监控启动时机；没有新科学、账户、GPU、重试、取消或下一Gate权限。debug资源仍29min/1GPU/8CPU/128G每job、最多2job并行。保持已提交job及write-once路径，不为同步这份纯文档补充而取消/重跑。遵循research-gate-orchestrator其余全部协议。
