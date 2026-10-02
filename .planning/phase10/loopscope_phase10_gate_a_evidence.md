# Phase10 Gate A 工程交付证据

日期2026-10-02，executor `01a0fb34-5c36-7210-b17d-744410853d1f`，planning `01a0fae2-12df-74e0-a874-190518ef0501`（local）。状态：工程完成、请求planning验收；不自判PASS，不进入B。

## Provenance与范围

实际Git根为专用 `loopscope-tflt`，分支始终`loopscope`。授权起点 `ee9feb4dbee1caca53a2fc23f02527c2bbe05106`，保护基点 `4f59bd93eca4da3cbf458a93508f91c5b23912bc` 为祖先，检查exit0。初始dirty仅planning文件，保留后独立提交：

- `1940961422b3df298108056d4d34a5dd168552ba`：原Gate A规划绑定。
- `aff38d9c38eddf711c370cb374a24294d5d72a79`：用户current_t合同v2、amendment与ARC操作补充。
- `ccf5293941e18c397eb868ed4268a78dc5761ae4`：完整Phase10旁路、配置、tests与runbook（21个新文件）。

本证据及data binding另作证据提交，终态包记录最终HEAD/push/dirty状态。无reset/rebase/stash/force-push。`git fetch origin`后原分支关系ahead1/behind0；其后仅本executor正常追加提交。

改动入口：`src/tflt/loopscope/phase10_{panel,analysis,adapter,data,runtime,accuracy}.py`，`configs/loopscope/phase10_panel.json`，六份`tests/test_phase10*.py`，七个`scripts/loopscope/*phase10*`入口/launcher，`docs/loopscope_phase10.md`。规划变更来自planning的正式授权文件；executor只新增事实/证据附件。

## 最小实现与定向结果

每数据集68独立/86展示/18个旧K2策略别名；63Online+3Loop+2Native。MMLU9预期复用59新（828478条新题级记录）、ARC68新（79696），合计908174。历史9cell只核配置、来源路径与schema；没有读取真实逐题score/gold，validated_reuse_cell_count仍为0。

fixed_t0与lag1复用不变的Phase9数值callbacks，t0不干预；lag1在每个候选forward开始重启轨迹。current_t使用原数值SVD helpers，每轮先从自身未干预的prompt残差拟合，含t0立即只改p；fit_t=used_t=t，不额外调用模型/window。K2 current_t独立。lambda0保持原对象，lambda0.5旧fixed路径兼容。真实tensor/model等价留B。

ARC完整文本continuation、3/4/5选项、数字/字母标签及原生字符长度acc_norm可用。S/p从真实保留context界定，continuation不进入SVD。同题候选保留prefix不一致显式停止，不改变shot/题数/答案。完整候选分数写入新根，不含目标gold；verifier按canonical identity闭合并直接为analysis提供对齐输入。analysis再次读取正常score roots验证闭合后才打开identity-bound gold；ARC先验证MMLU选参文件。

合成分析支持七个迁移lambda（并列选小值），Holm families为ARC14主迁移、每数据集126扫描、45策略比较（K2别名不重复）；exact双侧McNemar及bootstrap10000/seed20261002，MMLU subject内、ARC题级。无Matched-norm；current_t是起点和方向时序的组合消融。

父executor最终运行以下精确命令，exit均为0：

```bash
git diff --check
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_phase10*.py'
PYTHONPATH=src python3 scripts/loopscope/build_phase10_manifest.py --help
PYTHONPATH=src python3 scripts/loopscope/run_phase10_accuracy.py --help
PYTHONPATH=src python3 scripts/loopscope/verify_phase10_scores.py --help
PYTHONPATH=src python3 scripts/loopscope/analyze_phase10.py --help
bash -n scripts/loopscope/phase10_accuracy.sbatch scripts/loopscope/phase10_preflight.sbatch
```

最终unit output：`Ran 31 tests in 0.829s / OK`。覆盖面板、reuse mapping、全部统计family、tie-break、lambda/policy、三个策略时序、p-only/continuation mask、候选完整性与闭合，以及无site-packages的help/dry-run。ARC子任务还跑了不变的Phase9 adapter4项，exit0；未重复全suite。

## 实际HPC CPU/data证据

严格SSH选项统一为`-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o ClearAllForwardings=yes hpc2-hkustgz`；明确的临时代理tunnel采用已有SSH配置的reverse forward，下载后主动关闭。login mgmt-4；只读source保持clean `phase9-gate-c-20260922` / `a2c9e58a32163e1ed1527d3ab57798bc7378f486`，没有切分支/同步。版本及ARC事实详见同目录`loopscope_phase10_gate_a_data_binding.md`。

全部新inputs/staging在`/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/`。固定revision `210d026faf9955653af8916fad021475a3f00453`三parquet：

```bash
curl -x http://127.0.0.1:57997 --fail --location --connect-timeout 20 --max-time 55 \
  https://huggingface.co/datasets/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/train-00000-of-00001.parquet \
  -o /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase10-gate-a-20261002T062100Z/raw/train-00000-of-00001.parquet
```

test/validation用同参数各自split名称执行，均exit0，实际大小203808/189909/55743，总449460。直连attempt1退出35（TLS reset），失败根`inputs/phase10-gate-a-20261002T061700Z`保留；代理成功重试不覆盖失败路径。没有共享cache下载或安装。

实际builder命令（下述绝对路径省略统一workspace前缀以便阅读，运行时均为绝对路径）：

```text
PYTHON=.venv-loopscope-cu121-20260711/bin/python（只读source下）
STAGE=staging/phase10-gate-a-20261002T061700Z
INPUT=inputs/phase10-gate-a-20261002T062100Z
build_phase10_arc_inputs.py --model-config STAGE/configs/loopscope/phase10_panel.json
 --dataset-revision 210d026faf9955653af8916fad021475a3f00453
 --cache-dir INPUT/cache --max-length 32768 --data-dir INPUT/raw
 --source-manifest INPUT/source_urls.json --output-dir INPUT/prepared-attempt1
```

通过`runpy.run_path`运行上述argv：先`sys.path.insert(0, SOURCE/src)`，再`import tflt.loopscope; tflt.loopscope.__path__.insert(0, STAGE/src/tflt/loopscope)`。设置PYTHONDONTWRITEBYTECODE=1、HF_HOME为既有shared/hf_home、TOKENIZERS_PARALLELISM=false、OMP_NUM_THREADS=1；不调用模型。每个target parquet读取时仅pyarrow列投影id/question/choices；train可含示例answerKey。命令exit0、target_gold_loaded=false、test1172/validation299。两模型test最大context1242/continuation46/input1243，validation1207/34/1207；全部截断候选0、prefix不一致0。逐题25个唯一train示例identity已持久化。实际producer `load_pool`对两split输入接受exit0；test有4三选项、3五选项、22数字标签题。

最终implementation commit另上传小型源码到`staging/phase10-gate-a-20261002T063600Z`，包含当前Phase10文件及所需只读Phase9 numerical helpers快照。真实import解析均指向该staging的phase10_panel/runtime/accuracy及phase9_gate_e_runtime；旧source提供其他只读包。CPU check通过`phase10_panel.build_panel/validate_panel`检查两个数据集，建立`arc_validation_preflight_manifest.json`，再实际调用`run_phase10_accuracy.main`如下等价argv：

```text
--dataset arc_challenge --manifest STAGE/arc_validation_preflight_manifest.json
--pool INPUT/prepared-attempt1/ARC-validation-pool.json
--cell-id q4-w15-18-k3-online-{fixed-t0,lag1,current-t}-lambda0.5
--run-root STAGE/not-executed-<cell> --commit ccf5293941e18c397eb868ed4268a78dc5761ae4
--scope PREFLIGHT_ONLY --max-length 32768 --dry-run
```

三策略分别运行，另运行current_t-K2 `--engineering-check zero-strength`、fixed_t0-K2 `--engineering-check k2-policy`，共五次实际CPU dry-run，各选择6个无标签工程输入。最后返回`PHASE10_CPU_DRY_RUN_COMPLETE; model_forward=0; gpu_jobs=0; target_gold_loaded=false`，exit0。原始source未被新模块遮蔽为错误旧实现。dry-run不创建model score根。

## 子任务与修复

三个有界subagent分别拥有panel/analysis、ARC adapter/data、runtime/producer/verifier的互斥文件；无嵌套委派、Git整合、remote-write、模型或作业权限。父executor审阅并整合、提交、远端CPU/data执行和终态交付。

本Gate内低风险迭代包括ARC每候选lag1重启、原生fewshot默认train兼容、最早pyarrow列投影、真实builder/producer输入schema衔接、old reuse schema和preflight-only engineering身份。用户中途明确新增current_t后暂停依赖矩阵的写入，收到正式v2后完成更新。没有审计退修循环，没有根据真实outcome选参数。

## B准入建议与实证边界

- Planning需接受此ARC binding及A实现，再另建B独立executor；A不授予资源。
- 远端旧分支保持只读；B需明确专用source分支/commit部署权限，不把A staging CPU snapshot当成已完成B deployment。
- B执行两模型、K2/K3、三策略、lambda0.1/0.5/0.9的真实代表性预检；lambda0与旧0.5兼容、旧K2两路径等价、原生完整continuation评分/因果prefix实证，历史9cell逐题分数和配方核验仍待完成。
- 新预算按59个MMLU新cell、68个ARC新cell重估；current_t含t0额外SVD，不能套用旧两策略预算。测真实长prefix和并发加载峰值、无OOM、吞吐，采用本阶段A800最高合法优先级例外且preflight<30min。

没有GPU/Slurm、model forward、真实accuracy分析、target gold读取、Shared重拟合、Matched-norm或下一Gate动作。Phase8/9源码/config/results及protected wrapper/strategies/cache/config/eval_runner未改（基点到交付改动路径均为上述Phase10新增和planning文件）。唯一终态路由为planning线程；可见send_message_to_thread响应保留在本executor聊天。
