# Gate A 科学及执行范围修订：current_t

2026-10-02。用户明确新增当轮估计、当轮衰减，规划正式授权同一Gate A executor 01a0fb34-5c36-7210-b17d-744410853d1f在原路径及工程权限内加入current_t，按contract_v2执行。此修订优先于旧handoff的41-cell、59展示、两策略、所有t0不干预、禁止当轮方向等文字。

新增正常路径验收：t0及每个t先从未干预D_t估v_t再改p，其他位置不变；方向来自自身轨迹且candidate continuation排除；不得复用旧policy的t0-return捷径。lambda0仍回Loop，current_t与旧策略K2必须独立；验证方向来源时刻与应用时刻，不把fake验证冒充真实GPU证据。

更新配置、runtime、manifest、analysis、tests和runbook为68独立/86展示、MMLU9复用59新/ARC68新、7个迁移lambda及14/126/45统计families。原Phase9和现有旧策略语义保持。现有3个subagent可由父executor重新分工整合，不新增独立Gate、不嵌套委派。

原Gate A可写路径/Git/CPU权限、ARC小文件及overlay补充保持，仍无模型forward/GPU/Slurm/真实outcome权限。新分支真实张量和代表性资源验证留B，资源预算届时重新估计。工程完成后同一executor发送GATE_A_FINAL_AUDIT，不自行进入B。
