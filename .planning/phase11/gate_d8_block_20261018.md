# BLOCK — Phase 11 Gate D-8 提交准入

EXECUTOR_THREAD=01a14eeb-0aae-7733-a86e-62b55c8dde54
EXECUTOR_HOST=local
PLANNING_THREAD=01a14da8-5e0b-7550-9874-5bd47416bc6f
PLANNING_HOST=local
RUNTIME_SOURCE_COMMIT=9f7ca50fef085529b15d2cc4f46774bae19845a3
RESULT=BLOCK_REQUESTED

## 材料阻断

2026-10-18T12:37:01Z，真实 D8 debug 资源请求的 `sbatch --test-only` 返回 exit 1：
`AssocGrpSubmitJobsLimit` / `Job violates accounting/QOS policy`。
2026-10-18T12:43:42Z 再读 association，`hkuhpc-ai/root/xhuang225` 的 `GrpSubmit=0`；用户队列为空。
这不是排队等待，当前账户不能新增本 Gate 的 debug/正式作业。未更换账户、QoS 或分区绕过。

## 已完成

- 从 branch `loopscope`、base `4ccd8c18d674cf8e33b6e7ffee8e035d8881cdb4` 开始，提交并推送 `da751a3` 实现与 `9f7ca50` 合成 fixture 修复。新 D8 producer、独立 thinking-final-span S2、断点记录、batch/launcher/debug verifier、闭合和统计实现已落盘。
- 既有生成入口只新增默认关闭的 sampling/special-token opt-in；wrapper/strategies/cache/config 数值核心未修改。
- 官方模型/tokenizer 固定 `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`。config/tokenizer 已完整缓存；36 层、hidden2560、原生容量40960、无 RoPE scaling；边界 token 为151667/151668。权重仍不完整，尚无模型加载或真实 CUDA 运行证明。
- 新原生 thinking 模板的正式448题输入已完成；全部原题、identity、canonical order、选项及0-shot绑定不变。prompt 为89–2819 tokens，最长与32768预算相加后尚余5373 tokens。单user、add_generation_prompt=True、enable_thinking=True。
- 当前源码远端 D8 定向检查11项通过（9项输入/采样/RNG/span、2项统计）；既有生成入口6项通过。CLI、compile、shell syntax、diff check通过。
- 远端旧合成fixture仅A/B两个选项导致一次CPU输入准备失败；已保存失败证据，新增独立四选项工程fixture并回归通过。正式题集/配方未变，未运行模型、未读gold。

## 保存状态与决定性证据

Remote artifact root:
`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/artifacts/phase11-gate-d8-20261018T121451Z`

决定性文件：

1. `debug-submit-test-only.json`：exact资源请求、退出码、调度拒绝与association。
2. `prepared-block-summary.json` / `input-verification-r2.json`：448题身份/题目/选项及容量；CPU checks、源码与保留状态。
3. `download-tunnel-lifecycle.json`：仅停止自身下载PID3021221；临时58018端口已无listener，配置/tokenizer/部分下载保留。

本地聚合副本：`.planning/phase11/gate_d8_preparation_block.json`。
远端源码：`staging/phase11-gate-d8-20261018T121451Z/source-r2`。
恢复计划：同 staging 下 `debug-plan-r2.json`。
正式输入：`inputs/phase11-gate-d8-20261018T121451Z/formal-r2/thinking-pool.json`。
固定解释器：`/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python`。
源码通过 committed/clean runtime export 固定，HEAD/ref 文件仅作 export provenance metadata；未改共享远端 checkout。

## 完成数量、成本与保护状态

- 真实 job IDs：无；成功 raw：0/1344，完成配置0/3；实际/失败 GPUh均为0。
- 没有真实job，故没有创建D8 heartbeat。下载专用SSH连接已退出，临时listener已核对关闭。
- D7未接管，未启停/迁移其monitor，未提交、取消或重跑其作业。
- 未读目标gold、逐题正确性或accuracy；未运行部分评分、未改S2、未调参。
- 原题、prepared token、raw/gold/逐题预测留HPC；仅取回聚合状态。历史source/inputs/raw/gold/score未改。
- 他人 `AGENTS.md` 默认执行模型dirty保留，不纳入本Gate提交。

## 恢复所需

需要外部恢复当前普通用户的合法 submit admission，再由 exact planning 恢复同一 D8 executor。
之后完成同一固定官方模型的缺失权重下载，先用source-r2与debug-plan-r2跑同路径debug；通过后推进原三组1344条、全量raw/预测闭合、既定gold和两项Holm分析。仅真实job存在后创建本Gate唯一heartbeat。
不请求科学配方变化，不请求绕过scheduler，不进入下一Gate。

Decision requested: planning确认提交准入BLOCK及恢复条件。
