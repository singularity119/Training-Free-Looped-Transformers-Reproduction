# Gate D 全量预算提案 — 尚未授权提交

2026-10-09，C已PASS，216记录可复用，remaining245520。用户max8A800已冻结；不代表承诺持续分配8卡。当前需决定全量重大计算预算，不要求重复确认科学配置。

建议按完整54配置推进，最多8个单A800 job、每卡packing2独立batch1进程，job<=2h；第一轮最多16 allocation GPUh，包含成功canary复用、原index不重编号且无重复的正式小批，按实测更新成本。全量建议硬上限4800 allocation GPUh（含实际运行及当前作业预留、失败重试），每次只释放后续有限批次；接近上限不提交超预算作业。若用户偏好先短批，D第一轮总授权可限制128GPUh，达到该额度前回报吞吐/长度和剩余成本；仍保留全部成功正式记录、不改最终科学目标。

预算依据：C每臂4输入长度点外推2648.22GPUh；8卡理想连续13.79天。此为偏置小样本代理，不含排队/模型启动/尾部/重试，不是预测或保证。4800为规划缓冲上限，不是承诺全部花完；128用于先取得更有代表性的全量实际分布。费用币种/单价未绑定，不能转换财务成本。

执行准备范围：最小非重叠index批处理使同模型进程摊销加载；冻结runtime/science不变，任何生产路径变化做针对性CPU与<30min debug同commit预检，保存兼容证据才能与216复用。若改变数值行为需planning重新准入，不能为了性能改变SVD、batch、2048、shot或模板。每任务18臂全量identity完整且raw闭合后才允许对应gold/统计解封；不根据部分outcome修改网格。

拟定新独立执行聊天Gate D-execute-LoopScope-Full-Acquisition-第11阶段，gpt-6.1-sol/high。科学/paths按现有contract和HPC专属根；最高合法A800/QoS、现场配额核对；正式ID取得立即唯一full60min monitor，terminal自唤醒后executor负责完整闭合或授权范围内修复，不能停在observer角色。E仍为阶段综合审计。预算决定后写exact handoff并绑定新executor，本提案不授予GPU/Slurm动作。
