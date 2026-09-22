# Gate B：用户指定512诊断转debug

遵照research-gate-orchestrator协作约定。Planning=01a07ff5-79b1-79f0-809a-4d0971475686；Executor=01a0c72c-5462-71c3-8a7f-a2c4862680c1。

用户在执行任务要求两个512无标签诊断改投debug，executor已提交12832136/12832137（model0/model1），原12831872/12831873仍保留。planning直接squeue核对四个job全部PENDING且elapsed0。

授权同一executor在动作前重查状态：若原12831872/12831873仍PENDING，取消这两个重复正式作业，记为用户分区迁移取消、保留路径/提交记录，不记科学失败；保留debug12832136/12832137。若任何旧job已开始运行，先报告进度及有效产物再由planning决定，不依据过期状态取消。不得动其它job。当前保持同一970ef52科学/环境/runner及完整9×512身份，不改sample数。

用户请求的30分钟debug诊断属于本次运行资源选择，先前不足30分钟preflight已通过；不能把用户估计当作完整512必能完成的实测结论。若超时，保留有效产物，诊断是否支持恢复/分片；不自动重跑完成样本或改科学，不未经批准返回formal。

唯一heartbeat只观察实际保留的debug12832136/12832137，10分钟。本补充明确授予这两个job从PENDING起观察的最小例外；删除已取消旧job观察对象，不并行维护第二监控。终态先暂停并主动唤醒exact executor，失败先原范围诊断；健康无变化安静。24GPUh、最多2GPU及无test/gold/C边界保持。
