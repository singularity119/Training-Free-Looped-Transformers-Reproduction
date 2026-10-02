# Gate A ARC 数据与长度事实附件

此文件为 executor 的事实建议，须由 planning 接受才准入 Gate B；不修改合同或授权。

2026-10-02 严格 SSH 到 `hpc2-hkustgz` 成功，观测 login `mgmt-4`。专用 source 为 clean `phase9-gate-c-20260922`、commit `a2c9e58a32163e1ed1527d3ab57798bc7378f486`；未切分支或同步。ARC 在已检查的 shared HF hub/datasets 缓存及 shared 中不存在。

HPC 解释器：`/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711/bin/python`；lm_eval0.4.11、transformers4.51.3、datasets5.0.0、torch2.3.1+cu121。只导入库、载入 tokenizer 与 AutoConfig，无模型实例或 forward。

官方 [dataset metadata](https://huggingface.co/api/datasets/allenai/ai2_arc) 实时查得 revision `210d026faf9955653af8916fad021475a3f00453`（lastModified2023-12-21）。该 revision 的 ARC-Challenge 文件：test203808、train189909、validation55743 bytes，总计449460。拟只把三文件下载到新的专属 inputs/raw，不写共享 cache。

原生 [arc_challenge.yaml](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/v0.4.11/lm_eval/tasks/arc/arc_challenge.yaml) 引入 [arc_easy.yaml](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/v0.4.11/lm_eval/tasks/arc/arc_easy.yaml)：train示例池，test正式目标、validation工程目标；`Question: {{question}}\nAnswer:`，choice文本完整continuation，标签通过 `choices.label.index(answerKey)` 映射。主 acc_norm 按 Python 原候选文本字符数归一化；acc描述。

两模型固定 revision 同 Phase9。cached tokenizer 标称 `model_max_length=131072`，AutoConfig `max_position_embeddings=32768`。现场 `HFLM.max_length` 源码先取模型配置的 `n_positions/max_position_embeddings/n_ctx`，再取 tokenizer，所以拟冻结实际评分 `max_length=32768`，不用标称131072。少量数值长度覆盖不改变任何 shot 或评分规则。

fewshot seed20261002，使用安装的0.4.11原生 ContextSampler；每个split重置种子，在 canonical target 顺序下推进一次 RNG，并逐题保存25个 train示例identity。Native/Loop/Online与两模型使用同一已保存 prompt。train答案可用于原生渲染；test/validation加载时仅投影id/question/choices，目标gold不进入配置、持久化输入或长度核验。

## 实际全量核验（完成，等待 planning 接纳）

操作补充1获准后，固定三文件实际下载到 `/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase10-gate-a-20261002T062100Z/raw/`，`wc -c` 为203808/189909/55743，合计449460。直连失败尝试保留在 `inputs/phase10-gate-a-20261002T061700Z/`；成功下载使用既有SSH代理配置的临时reverse tunnel，结束后已Ctrl-C关闭（SSH退出255是主动关闭），没有持续监控或作业。

实际CPU builder返回exit0、target_gold_loaded=false、test1172、validation299。输出目录：`inputs/phase10-gate-a-20261002T062100Z/prepared-attempt1/`（均在上述workspace）：

- `arc_input_facts.json`：revision、版本、Arrow读取列、tokenizer、sampler和长度证据。
- `ARC-test.jsonl` / `ARC-test-pool.json` 与 `ARC-validation.jsonl` / `ARC-validation-pool.json`：无标签完整输入。
- `ARC-test-demo-identities.json` / `ARC-validation-demo-identities.json`：每题25个train示例身份；构造时逐题检查25个示例。
- 原始来源URL记录：同inputs根的 `source_urls.json`。

两模型的结果相同：

| split | N | 最大context tokens | 最长continuation tokens | 最大实际input tokens | 截断候选 | prefix不一致题 |
|---|---:|---:|---:|---:|---:|---:|
| test | 1172 | 1242 | 46 | 1243 | 0 | 0 |
| validation | 299 | 1207 | 34 | 1207 | 0 | 0 |

无需删题、降shot或修订统一前缀。此证据只来自tokenizer/renderer，不证明实际GPU评分、tensor等价或accuracy。

实际producer的`load_pool`接受两份原生builder输入（exit0）；逐题检查25个示例身份且不重复。test选择数分布：1165题四选项、4题三选项、3题五选项，22题包含数字标签；validation295/3/1分别为四/三/五选项，4题包含数字标签。没有强制四选项或字母标签。

实际import解析：phase10_data.py与phase10_adapter.py来自 `staging/phase10-gate-a-20261002T061700Z/src/tflt/loopscope/`；phase9_spectral_core.py来自只读专用source。用显式 `tflt.loopscope.__path__.insert(0, staging_path)` 避免旧模块遮蔽；入口在同staging的scripts/loopscope。local committed base为1940961422b3df298108056d4d34a5dd168552ba，执行增量是本Gate未提交的adapter/data文件快照，最终实现commit由Gate A evidence绑定。未复制环境或模型，未修改远端旧分支。
