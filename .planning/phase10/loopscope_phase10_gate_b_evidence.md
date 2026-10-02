# Phase10 Gate B 工程验证交付

2026-10-02。Executor=01a0fb63-097d-7c92-9bd6-1f1b8ddafe9f；planning=01a0fae2-12df-74e0-a874-190518ef0501。结果为AUDIT_REQUESTED，由planning作Gate决定；没有进入C。

起点a7ca3626d639e1b271a1bb1b851a72e1299e359e、loopscope。先提交planning治理内容f3ec305/6bed0bd，验证实现3d0cf5e/f30895f，最终实际GPU源码为 **1cee7a5fc0731361d3d2b4471234426b4d7f10aa**。该源码以git archive完整部署到专属staging后没有原地修改。

## 证据入口

统一远端workspace=/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope：

- source：staging/phase10-gate-b-20261002T070214Z/source-1cee7a5fc0731361d3d2b4471234426b4d7f10aa。
- inputs：inputs/phase10-gate-b-20261002T070214Z/prepared-attempt1；ARC继续直接引用A的inputs/phase10-gate-a-20261002T062100Z/prepared-attempt1/ARC-validation-pool.json。
- run：runs/phase10-gate-b-20261002T070214Z；核心GATE_B_ENGINEERING_AGGREGATE.json、historical_reuse_verified-attempt1.json、mmlu-eight-producers-closure.json、arc-eight-producers-closure.json；32个job的可读命令在job_submissions.jsonl及本资源附件列出的job IDs/初始两job记录。
- 本地资源附件：loopscope_phase10_gate_b_resources.json。源import来源已记录source-1cee7a5fc0731361d3d2b4471234426b4d7f10aa-cpu.json。

远端旧source保持clean phase9-gate-c-20260922/a2c9e58a32163e1ed1527d3ab57798bc7378f486；原环境Python、torch2.3.1+cu121、transformers4.51.3、lm_eval0.4.11没有安装/升级/修改。模型与tokenizer固定revision，dtype、batch1、cache、Euler、CUDA FP32 reduced gesvd均保持。

## 验证结果

CPU：git diff --check、三个launcher bash -n、关键CLI help/import均exit0；31项原Phase10定向测试在validation/资源计量修改后通过，两个新增检查模块加入后38项通过。没有重复历史suite。

MMLU原loader只能接受test14042，因此在Phase10新增明确validation1531输入入口（FORMAL_TEST拒绝该输入），不改Phase8 renderer/validator。CPU复用原安全列投影及标准5-shot，固定canonical选择[0,368]；ARC沿A原pool选[0,32,35,85,89,210]，覆盖首题/最长context和continuation、3/4/5选项及数字标签。

六个真实模型×任务×K检查均GPU_CHECK_COMPLETE。Native/Loop adapter对未改HFLM完整分数、零强度恢复、Phase9 fixed0.5、K2 fixed/lag1在0.1/0.5/0.9的真实CUDA残差结果/投影全部exact，最大绝对/相对误差0。current_t实际模型轨迹每轮从raw D_t拟合，包含t0，独立重新拟合oracle相等，仅p改变；候选/题/配置状态边界通过。最小观测top1相对奇异值gap=0.373494，没有观测到改变结果的近重根。

每任务8个代表性producer组合覆盖两模型、K2/K3、三策略及0.1/0.5/0.9；MMLU16条、ARC48条完整原分数由同verifier闭合。初始ARC Native6条也独立闭合。B所有新模型目标均validation，资源输入为合成模板扩展；没有正式test模型评分、gold/correctness/accuracy/历史analysis读取。

历史9cell同配方raw复用核验全部通过：9×14042=126378条，canonical identity/order、逐候选四个finite raw scores、配置、revision/dtype/cache、runtime版本、已闭合source与shards均匹配。仅输出counts/provenance，没有计算outcome。

32个GPU作业全部debug/A40、COMPLETED/0:0；最长65秒，总分配 **0.236389 GPUh**，实测最多同时2GPU，OOM及GPU失败attempt均0。所有job已终态且队列无本Gate作业。全程executor保持活动；短job在检查时已终态，未建立跨turn heartbeat，没有残留monitor或待唤醒executor。

## 显存与吞吐

MMLU合成context3096/continuation1；ARC合成context1242/continuation46（组合最长值的保守包络）。保持原validation模板/Answer末端，补普通文本token；不是正式可保留数据。两模型均实际测量，同卡独立进程packing2，不共享model/tensor。CUDA峰值与0.5s nvidia采样均含model loading；采样峰值是下界，精确torch峰值见附件。原生logits cache下候选评分数与实际forward分开记录。

| 任务 | 模型 | packing | whole-GPU采样峰值 GiB | 聚合候选scores/s |
|---|---|---:|---:|---:|
| arc_challenge | 1.7B K2 | 1 | 4.920 | 0.7724 |
| arc_challenge | 1.7B K2 | 2 | 9.831 | 0.7597 |
| arc_challenge | 4B K3 | 1 | 9.365 | 0.4643 |
| arc_challenge | 4B K3 | 2 | 18.722 | 0.4570 |
| mmlu | 1.7B K2 | 1 | 5.922 | 0.5309 |
| mmlu | 1.7B K2 | 2 | 11.835 | 0.5111 |
| mmlu | 4B K3 | 1 | 10.504 | 0.2178 |
| mmlu | 4B K3 | 2 | 20.999 | 0.2057 |

packing2四组均安全，但聚合吞吐比packing1低约1.6%–5.6%；A40建议packing1。SVD主要占耗时，双进程各自CUDA SVD时间约翻倍；不能因显存余量自动选择packing2。A800卡型/容量/吞吐未在B测量，不外推其packing。

## C/D资源与预算建议（不授予后续执行权）

当前普通用户xhuang225的A800最高合法分区为emergency_gpu：AllowGroups/Accounts/Qos=ALL、RootOnly=NO、PriorityTier/JobFactor=300，合法assoc含emergency_gpu QoS，QoS priority0（同级）。正式提交时应重读现场资格。

B极值validation样本的逐题timing线性压力投影：MMLU59新Online约805.07 A40 GPUh；ARC63 Online约44.02 A40 GPUh，另5个无SVD baseline不在该数内（按最慢现测Online保守加约6.57 GPUh）。这些不是总体运行时间估计，也不证明A800耗时：MMLU正式无标签context median524/p901602/p992670/max3096，而B特意包含最长2954。不能直接申请805h并称为预期成本；正式A800预算必须以首个可保留正式canary的时间重新定标。

建议C/D先A800 packing1、batch1、单cell或规划明确授权的可保留正式shard，测实际GPU loading/长尾峰值及吞吐后再决定packing2/并行。现有FORMAL_TEST入口仅支持全cell，尚无正式小shard接口；若选择shard canary，C handoff需明确允许完整cell分片且最终保持全14042/1172闭合，工程改动后重做debug预检。若不分片，首个完整cell作canary：A40压力样本现测MMLU最慢Online cell约24.5h、最大shape合成路径约36h，可据此讨论48h上限，但不能视作A800保证。ARC最慢现测全cell压力投影约1.31h；保守覆盖最长文本和5候选可讨论首cell4h上限。此前debug卡资源仅功能/同卡证据，A800资源准入留正式Gate。

可复用预检对象为源码1cee7a5、现有phase10_accuracy/preflight launcher、原venv/模型/cache、两adapter/producer/verifier；source/launcher/runtime/hook/serialization等关键对象变更时重新debug；B原分数不混入正式test。

## 偏差与交付边界

第一次CPU部署bootstrap在显式source /etc/profile时exit1，没有提交job；改用已核验login shell/绝对Slurm路径完成。一次无写入的多余CLI试调用被argparse拒绝，随后执行真实verifier命令通过。

批量第9次submission返回1且未获得job ID，捕获stderr未保留；现场debug QoS MaxSubmitPU=8，缩小到4个的有界提交批次后成功，不宣称已证实原返回1的精确根因。已获job ID全部保留且没有重复提交对应任务；无GPU失败/覆写/删除。此项为提交工具层限制，不影响最终producer证据。

三个有界子任务分别拥有GPU检查脚本、输入/历史reuse脚本、资源脚本；父executor审阅整合/Git/部署/调度/闭合/终态。无嵌套或独立调度。原Phase8/9及wrapper/strategies/cache/config/eval_runner未改；planning的AGENTS/plan/control/handoff调度治理编辑仅代为提交，未自行更改科学/授权。请求planning验收PASS或返回材料性修复；不自行判PASS、不进入C。
