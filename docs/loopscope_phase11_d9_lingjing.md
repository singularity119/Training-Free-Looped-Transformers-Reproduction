# D-9 灵境代码准备与恢复入口

当前仅完成 D-9-A 代码准备。用户 2026-10-21 随后通知 `h20_agi_large` 有8卡可用，
当前提交候选已改为总8卡、单节点；该通知不是本 Gate 的计算提交权限。
不得轮询额度、提交 CPU/GPU 任务或建立额度监控。本文命令是后续独立 executor 的
恢复 runbook，不授予执行权限。动态范围见 `.planning/phase11/loopscope_phase11_control.md`
和 `loopscope_phase11_gate_d9a_handoff.md`，原 D8 关闭、D7 不变。

## 已准备内容

- `configs/loopscope/phase11_d9_lingjing.json`：8 H20 候选配置、原模型/数据身份、fresh OSS 根。
- `prepare_phase11_d9_inputs.py`：云端匿名读取原作者固定归档的 Main CSV，复用原
  `prepare_gpqa_inputs` 的一个 `Random(20261008)` 连续选项排列，再调用原 D8
  `prepare_pool` 重渲染单 user、0-shot、thinking prompt。`--synthetic` 使用旧 D8 工程 fixture，
  不联网、不建 gold。正式构建器单独生成 sealed gold，推理只读无角色的 question/choices/prompt。
- `phase11_d9_lingjing.py`、`phase11_d9_lingjing_worker.py`：真实平台 rank 0 启动
  本节点 8 个独立单卡进程，packing=1；每 worker 顺序处理三个 cells。24 个独立 raw roots，
  每臂全部原 index 0..447 恰好一次，共 1344。逐题 seed 仍为 `20261018+canonical_index`。
  使用原 `acquire_row`、`answer_span`、`validate_bundle` 和 `RecordStore`，不修改 wrapper、
  cache、策略或 SVD 数值实现，不使用 vLLM/TP、不伪造 Slurm 环境。
- `phase11_d9_bootstrap.py`：短绝对挂载入口，解包 code bundle，保留镜像的
  torch/CUDA/flash/triton 包；每节点 leader 安装 frozen 高层依赖并保留安装日志。
  rank 0 在获准云端构建输入。保留旧16卡方案的rank 8路由支持，但当前配置/导出计划仅申请8卡。
  启动器逐秒记录所分配卡的整卡显存。
- `check_phase11_d9_gpu.py`：复用原 D8 thinking/final-S2/decode 路径检查，允许合法 no_match，
  但工程预检仍需真实 S2 正路径。短 synthetic 自然 EOS **不构成 32768 上尾容量证明**。
- `analyze_phase11_d9.py`：本地挂载 tokenizer 复验全部 raw、恢复原序并封存预测；
  第二次独立调用才读 gold，复用原 D8 exact McNemar、两项 Holm、10000 次 seed20261018 bootstrap。
- `build_phase11_d9_bundle.py`：只导出已提交代码；不上传、不调用 quakecmd，不包含
  `.planning`、测试题、模型、gold、历史结果或凭据。

## 来源与平台证据的边界

GPQA 使用 `idavidrein/gpqa@56686c06f5e19865c153de0fdb11be3890014df7` 的
`dataset.zip` / `dataset/gpqa_main.csv`，与原 source binding 完全同源。
2026-10-21 本机仅 HEAD 得到 200、application/zip、2348038 bytes，没有下载归档。
云端实际读取仍必须通过 Main=448、五字段完整与原构建器检查。
`alignment.json` 明确标记 `SOURCE_CONTRACT_REUSED` 及 `hpc_artifact_readback_performed=false`，
不声称已在此设备对旧 HPC pool 做逐题回读。已放弃的镜像路线曾返回 403 GatedRepo，
不会向镜像发送 token 或以另一副本顶替。

OSS 候选为 `oss://sampool-bucket/deltalake/other_sample/users/270228/huggingface/Qwen/Qwen3-4B/`。
已只读核对 36 层、hidden2560、native40960、BF16、EOS `[151645,151643]`、thinking template、
三片 safetensors/398 tensor entries/8044936192 bytes。目录及对象 metadata 没有可读 acquisition revision；
因此**尚未证明该副本是原 `1cfa9a7208912126459214e8b04321603b3df60c`**。
启动前需要该副本的原下载 repo/revision 记录或获准的原固定 revision 资产直接比对证据，
不能把同名、同 config 或大小当作 revision 证明。

暂停前已核对账号 557030 的成功任务 4952206 / 合约21143688：16 卡 H20-8x-960G，
额度组 h20_agi_large、zhangbei、绝对挂载 Python 入口 + tar bundle；页面返回的镜像为
`hub.docker.alibaba-inc.com/cro-aispace/quake:torch-2.10-cuda128-cudnn9-rocky9-train-v2.0-20260820`。
候选提交沿用已受理镜像别名和 `git@gitlab.alibaba-inc.com:Yufeng_Intern/Big_Model_Intern.git hxt`，
不写该仓库、不改 remote。挂载 `/jindofs_temp:jindo:jindo-lake3:/other_sample`。
这些是部署方式证据，不证明本代码已在 CUDA 上运行，也不证明当前有空余额度。
96G/141G 均列于规格页；候选选 96G，总8卡=单节点8卡，单卡batch1/packing1。

## 已收到8卡通知后的准入项

1. Planning 新建并 exact 绑定 D-9-B；先闭合上述模型/tokenizer revision，再填真实
   demand_id 与独立 fresh 尝试路径。quota_project 按用户指定填 h20_agi_large；可用8卡来自用户
   通知，不冒充重新读取平台实时库存。候选中的 null 是未决项，禁止直接提交。
2. 明确灵境的非正式 debug 资源映射。目前 CLI help/已成功任务只证明常规训练入口，
   没有证明存在等价 HPC `debug` 分区；`waiting_for_error` 不是 debug 调度分区。
   Planning 必须确认合法的短任务映射/例外后，才能执行预检，不能凭本 runbook 自动绕过。
3. 最小预检候选为 1 H20、上限1740秒，原 synthetic 三 cells 顺序执行，同 bundle、镜像、
   tokenizer、CUDA/loop/缓存/写盘/验证器路径。预计 5–20 分钟只是按旧 D8 75秒预检作的
   保守准备估计，未在本镜像实测；预算上限约0.4833 GPUh。需包含启动、安装及输入构建总耗时。
4. 正式前另取得代表性 prompt 与32768上尾的整卡/进程显存证据，约20%余量，packing固定1。
   当前 bundle **不强制生成32768 token，不把短 synthetic 预检外推为容量准入**。
   容量测量的具体工作负载与保留正式 canary 方案由 D-9-B handoff 冻结；未取得证据时
   admission 的 `preflight_passed`/`capacity` 保持未通过，formal worker 拒绝启动。
   不继承旧 D8 packing2 的1.09%余量，也不为资源更改正式batch/长度/采样。

## 本地代码打包

只在 loopscope 专用 clone 中执行（无云端副作用）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/loopscope/build_phase11_d9_bundle.py \
  --output-dir .planning/phase11/gate_d9a_code_bundle_20261021
```

输出 code-only `runtime.tar.gz`、`start.py`、真实 source commit 的 `bundle-info.json`、
1卡 preflight 与8卡 formal 的 plan/registry、`admission-template.json` 和
`submission-candidate.json`。需求ID、额度、模型来源证据、debug映射、显存证据均需真实填入。
导出器拒绝打包尚未提交或 dirty 的运行源码，planning/control 的独立改动不混进 bundle。

## 后续云端恢复命令模板

以下均在新独立 D-9-B handoff 到位及相关准入闭合后使用，**D-9-A 不执行**。
将新 code bundle、start.py、plan、admission 放到候选 fresh OSS 根；模型保持指定根只读。
把提交回执原文保存到 `receipt-<stage>.json` 并发布到共享挂载，bootstrap最多等待120秒，
从真实 `modelTrainingJobId` 读取身份，而非在提交前猜 ID。提交结果不明先精确查任务，不盲重提。

```bash
# 云端预检实际通过后，验证三 cells 的 raw/lifecycle/S2；不会读 gold。
PYTHONPATH=src python scripts/loopscope/check_phase11_d9_gpu.py \
  --plan /CLOUD_ROOT/plan-preflight.json --model-dir /ADMITTED_MODEL \
  --output /CLOUD_ROOT/preflight-verified.json

# 全部1344 raw完成：先封存全量预测，--registry 和 --gold 不可同用。
PYTHONPATH=src python scripts/loopscope/analyze_phase11_d9.py \
  --registry /CLOUD_ROOT/registry-formal.json --model-dir /ADMITTED_MODEL \
  --output-dir /CLOUD_ROOT/closure-fresh

# 完整预测封存之后，独立调用冻结统计分析。
PYTHONPATH=src python scripts/loopscope/analyze_phase11_d9.py \
  --closure /CLOUD_ROOT/closure-fresh/closure.json --model-dir /ADMITTED_MODEL \
  --gold /CLOUD_ROOT/inputs/formal/sealed/gpqa_main-gold.jsonl \
  --output-dir /CLOUD_ROOT/analysis-fresh
```

真实提交时使用 quake-train skill 的 `quake_train.py` 先做本地命令预览；补齐真实参数
并由获授权 D-9-B executor 执行一次提交。bootstrap/launcher 的每次重试使用新 launcher-output，
raw root 内 RecordStore 只恢复缺失/半条记录，验证成功记录不重算、不覆盖。
worker/node失败保留 failure.json、安装日志、worker日志及整卡显存CSV；正式全量未闭合不读gold。
冷启动依赖/框架/数值环境与旧HPC不同，真实路径验证和实际版本另存env，不能把差异单独归因于硬件。

后续真实作业存在后，D-9-B 才可建立唯一 heartbeat：smoke10min/full60min，终态暂停并
唤醒 exact executor，按协议回传 planning。当前没有作业、没有 monitor，也没有新增 GPUh。
