# HPC 历史 GPQA 最终选择重评分

规则原文见 [gpqa_final_choice_v1_protocol.md](gpqa_final_choice_v1_protocol.md)，
实现为 `src/tflt/loopscope/gpqa_final_choice.py`。本工具是已有 D13 v1 的移植，
不修改生成模型或历史结果。复杂语义由无 GT 审阅者判定；正则只提供候选，
不能把候选文件直接当成已完成的审阅。

## 可直接复制到另一设备的 prompt

```text
请在这台设备的 LoopScope 专用仓库执行已有 HPC GPQA 输出的离线重评分。
先 git fetch origin，在保留本地改动的前提下更新 loopscope 分支；不要重置、
强推或改 main。阅读 AGENTS.md、docs/gpqa_final_choice_v1_protocol.md、
docs/gpqa_final_choice_hpc_runbook.md 和 src/tflt/loopscope/gpqa_final_choice.py。
本次用户授权是整理分析已有输出：可读取现有 HPC 结果，并在新目录写 CPU 分析
工件；不重新推理、训练，不提交 GPU/Slurm 作业，不创建定时监控，不改历史结果
或旧 Gate 裁决。此前本地 D13 的“不连 HPC”是当时设备范围，不应误当本次的禁止；
只使用本设备已有的合法 HPC 接入方式，连接缺失时报告实际阻塞，不修改 VPN。

先盘点所有可取的历史 GPQA 原始输出及 producer/configuration，区分生成配置、
重复评分视图和不同 seed。不要把 S0/S1/S2 三张表当三次独立推理，不把 seed 或
重复 identity 合并成更大分母。每组记录模型、实际可核实 revision、Thinking/
chat template、采样或 greedy、温度/top_p/top_k/min_p、token 上限、seed、窗口、
K、lambda、方向来源、Euler 参数、cache/decode、dtype、batch、原 run/job/path；
缺失项明确 UNVERIFIED。完整 GPQA Main 是448题；缺失、重复、额外 identity 必须
列出，未闭合面板不算完整增益。D3等不平衡 pilot 单列，不补成448。

在读逐题 GT、旧预测、旧 correctness 前，冻结本次可完整比较的配置和 contrast
family。每个同实验、同 seed 的非 Native 对 Native 主比较纳入预先固定的 family；
匹配普通 Loop 比较另列探索 family。不要见分数后缩小 family，也不要覆盖已完成
D9/D10/D11/D12的旧 family。HPC新增结果作为新批次；若合并历史比较，另报补充
family。按下述格式把已有输出归一化，再运行 prepare。配置/join-map/旧结果留在
private；审阅者只看随机匿名 id/output/options，不得看运行名、模型、配置、
canonical index、GT或旧分数。prepare只产生候选和REVIEW_PENDING模板，必须完成
实际审阅后才能seal；有既往分数暴露必须披露，不能伪称从未见过数据。

按D13-final-choice-v1.0提取明确最终承诺：接受自然语言、大小写、Markdown、
boxed字母、独立字母和无需解题即可唯一对应的选项文本；排除试探、引用、排除项
和演算字母。明确更正可覆盖先前承诺，未解决冲突、标号和内容矛盾记歧义；没有
最终承诺记NO_COMMITMENT。不得替模型解题，不从boxed数值直接猜字母。复杂文本
检查最终承诺、前后撤回与矛盾，必要时扩展上下文；如仅聚焦审阅须如实披露，
不能称所有推理token逐字通读。截断数字前缀等边界不能因GT事后挑读法：没有
单义已完成承诺就保守记歧义/未表态；合理替代读法在GT前记alternative_labels及
原因，主判定封存后不得修改。历史D11 C.1/C.3存在不一致，不能照抄为先例。

每题保存selected_label/null、类型、版本、原文证据字符区间、理由、审阅方法和
审阅者。所有题完成后seal，再读GT评分。歧义/未表态仍在448分母中；条件准确率
如计算须另列且不能替代主指标。输出Native/各Loop或干预配置的正确数、准确率、
对Native增益(pp)、W→R/R→W、配对bootstrap95%CI(20000次，seed20261022)、
exact McNemar与预冻结family的Holm。先封存的替代读法另报敏感性分析，不替换主分。

历史S2必须按原producer真正使用的文本区域和parser重放，不把record.predicted_label
的S0当S2，也不普遍把final_span.final_text当S2输入。已知D9用首个</think>后区域，
D10/D11/D12用whole-output；尤其D10 final_span.final_text只是诊断片段。HPC更早
实验逐个查producer和原parser版本：原来没有S2的可以另算同版本S2，但标注为新算，
不能冒称历史S2。无法闭合旧S2来源就单列未验证，先交付新分，不能猜。

提供新旧逐题差异和人工归因：纯格式、自然语言最终承诺、明确更正/冲突、
thinking/final区域、选项文本映射。不要把所有得分恢复都叫格式影响，更不能叫
推理提升。计算相对增益变化=(方法新旧正确差-Native新旧正确差)/448*100。
最后说明扣除评分口径影响后剩下哪些点估计、CI、校正后统计证据及边界敏感性；
同题事后探索不能写成新题泛化或因果推理收益。没有可靠正证据不等于证明效果为零。

报告和运行证据放新资产/结果目录，代码仓库不提交原始输出、GT或大工件。交付
盘点清单、每组完整配置、主结果表、盲提取/封存/首次GT访问回执、逐题差异、
歧义与边界清单、实际命令、统计和未完成范围。遇到普通适配问题在此范围修复，
不要扩展框架，不因低风险动作反复请求确认，不自行启动下一实验。
```

## 最小输入和命令

工具不加载模型，无第三方依赖。项目根目录运行，`ROOT` 指向仓库外的新分析目录。

输入 `source.jsonl` 每行**仅**含以下五个字段，不包含GT或旧预测：

```json
{"cell":"runA/native","identity":"original-id","canonical_index":0,"output":"model output","options":{"A":"text A","B":"text B","C":"text C","D":"text D"}}
```

`manifest.json` 在GT前写好；每cell需有448条并在同group内identity/index/options完全对齐。
configuration填实际完整配置；下面只展示结构。不同模型/seed不能用同group作配对。

```json
{
  "rule_version":"D13-final-choice-v1.0", "expected_n":448,
  "cells":[
    {"cell":"runA/native","group":"runA","configuration":{"model":"actual model","arm":"Native","seed":"actual seed"}},
    {"cell":"runA/c2","group":"runA","configuration":{"model":"actual model","arm":"C2","seed":"same seed"}}
  ],
  "contrasts":[{"id":"runA/c2-v-native","reference":"runA/native","treatment":"runA/c2","family":"hpc-primary"}]
}
```

```sh
PYTHONPATH=src python3 -m tflt.loopscope.gpqa_final_choice prepare --source /path/source.jsonl --manifest /path/manifest.json --root "$ROOT"
```

把 `review/decision-template.jsonl` 另存为 `decisions.jsonl` 并完成每题判断。
只把 `blind/all.jsonl` 和候选交给审阅者；候选也要核查冲突。决策行示例：

```json
{"id":"R000001","rule_version":"D13-final-choice-v1.0","selected_label":"B","type":"CLEAR_NATURAL_COMMITMENT","evidence_start":0,"evidence_end":11,"evidence":"I choose B.","reason":"明确最终选择且无未解决冲突","method":"manual_focused_context","reviewer":"blind-reviewer-1","alternative_labels":[]}
```

null类型用 `NO_COMMITMENT`、`AMBIGUOUS_CONFLICT`、`AMBIGUOUS_LABEL_TEXT`、
`AMBIGUOUS_OPTION_MAPPING` 或 `OPTION_MAPPING_MISSING`。证据区间为Python字符索引，
不是UTF-8字节或token。无结论可引用末段证据；空输出允许空区间。
`disclosure.json` 如实填：

```json
{"gold_accessed_in_new_extraction":false,"historical_exposure":"实际已见的历史材料/汇总，或无","manual_scope":"实际阅读范围、候选核查方式和未覆盖边界"}
```

```sh
PYTHONPATH=src python3 -m tflt.loopscope.gpqa_final_choice seal --root "$ROOT" --decisions /path/decisions.jsonl --disclosure /path/disclosure.json
PYTHONPATH=src python3 -m tflt.loopscope.gpqa_final_choice score --root "$ROOT" --gold /path/gold.jsonl --legacy /path/legacy.jsonl
```

GT每行 `cell/identity/gold_letter`，须覆盖完整面板。旧S2每行
`cell/identity/predicted_label/provenance/parser_region`；label允许null，provenance填
原producer/parser版本和结果路径，parser_region填真正输入区域。没有已验证旧S2时
省略`--legacy`。数据归一化和S2区域重放由执行者按实际历史schema完成，不能通用猜测。

输出含 `scoring/results.json`、逐题评分、歧义清单、可选新旧差异。
归因、文字报告、预封存替代读法的敏感性分析由执行者基于这些工件完成，脚本不自动
替人工判断。封存是本地只读快照和过程约束，不是密码学证明，也不能证明操作者此前
从未接触GT。评分工程失败后保留失败目录，用 `score --output /fresh/attempt2` 重试，
保留 `private/first-gold-access.json`，不得重新盲判或覆盖封存结果。

最小验证：`PYTHONPATH=src python3 -m unittest discover -s tests -p test_gpqa_final_choice.py`。
