# D11 — Instruct-2507 GPQA A10016

COLLABORATION_PROTOCOL=MANDATORY_RESEARCH_GATE_ORCHESTRATOR
EXECUTOR_THREAD=01a15ddb-5ea0-7112-8468-62b463c996b5
PLANNING_THREAD=01a15d9c-ee8b-7502-97b2-a40f732e00f5
HOST=local

按已读取research-gate-orchestrator及protocol执行。planning已直接核实用户在本executor新指令，明确指定该聊天继续新Instruct实验，沿用线程为本次用户选择例外；D11独立于D10和关闭的D9。无需重复请求提交许可，不因文档同步阻断已授权实现。

模型oss://sampool-bucket/deltalake/other_sample/users/270228/huggingface/Qwen/Qwen3-4B-Instruct-2507/，小metadata/原生模板核对，actual revision按用户指定OSS副本如实记录未绑定HF。不重启用户已取消的逐字节来源比对。原GPQA Main448原序、原选项shuffle20261008、D4 0-shot CoT题面和整段输出S2；greedy/do_sample=False、max_new_tokens8192。Native/current_t K2 λ.5/fixed_t0 K3 λ.9三组1344，window15:18 alpha1 beta0 block damped_euler/cachefirst/decodefull/原rank1方向语义不变。Instruct原生模板不强制开启thinking；不要把Thinking模板硬套进来。整段输出中合法答案冲突按原S2计错，无match计错、全部448分母；原完整后gold屏障、两项McNemar/Holm及配对bootstrap。固定输入/已看过题集配置属于探索性复现，不声称跨模型平台单因素因果。

资源A100-8x-640G总16卡，quota=a100_agi_large_32。核对当前CLI/平台合法资源与真实提交字段，保持8+8双节点/16独立单卡worker、batch1及保守packing，不混入D10 H20任务。起点repo /Users/singularity/Desktop/tflt_digai/tflt/loopscope，HEAD ef60e48；保留dirtyplanning，仅新增隔离Instruct recipe/worker/launcher/tests和必要最小接入，旧D8/D9/D10/数值核心只读。源数据只读复用tflt OSS已发布448，所有新run/bundle/log/receipts放users/557030/tflt/phase11/d11/fresh-root。不连接HPC、不改外部部署repo或ACL。

Agility budget：只验证新模板/greedy/S2/设备识别/双节点分片1344并集和提交路径，复用此前正常启动逻辑；定向测试后真实提交，保留启动健康与整卡显存日志，不称未做预检或容量已PASS。独立运行root，成功样本不覆盖；同科学低风险工程错误可fresh恢复缺失/无效项，不停其他任务。科学不确定向planning报告集中确认，不能自行择优改参数。

用户同时保留D10任务，D11不能取消/重用其root/score或替换掉其监控路由。一个Gate一个有效monitor，D11真实job后建自身full60min，D10唯一monitor继续独立；二者可同executor但按exact Gate/job分开，健康无变化静默。终态monitor暂停并唤醒本executor，分别GATE_D11_FINAL_AUDIT/GATE_D10_FINAL_AUDIT向planning投递确认。无双重监控同一job。规划只读1–3决定性证据验收；负结果可PASS，未经用户不增加后续实验。

## 真实受理记录

- task4954217 / modelTrainingJob4392090，name expr_gq_i4_g1_r_20261022025735，backend tj20261022025759-a35f5e85；2026-10-22 02:57:59受理NEW，尚不代表CUDA启动或完成。
- runtime source 3bf5d241aa6732c13714bfce41537db2e9dbbb26；OSS root `oss://sampool-bucket/deltalake/other_sample/users/557030/tflt/phase11/d11/20261021T185735Z/`，代码包九文件及真实receipt逐文件回读一致。
- 单独monitor `d11-instruct2507` ACTIVE/full60min/exact executor；D10 d9b继续独立。machine receipt `.planning/phase11/gate_d11_submission_20261022.json`。
- 9项定向检查通过；runtime smoke尚未观察，8192长序列容量NOT_MEASURED。首请求create前benchmark缺失明确1008拒绝，证据独立保留；补平台现有benchmark字段后当前fresh受理，科学配置未变。
