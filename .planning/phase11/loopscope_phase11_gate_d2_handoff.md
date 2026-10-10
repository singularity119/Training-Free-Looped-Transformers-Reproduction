# Gate D-2 — GPQA post-hoc extraction sensitivity

2026-10-10。用户最新“继续”批准上轮提出的事后提取分析。独立executor01a124db-33fa-7322-9317-12e53d3d3861/local，gpt-6.1-sol/high；planning01a11ab9-23cc-76d2-afd7-b5f9d9dc5e3f/local。按照research-gate-orchestrator协作约定执行，读skill/protocol与AGENTS。D已PASS撤权；本Gate独立新增分析，不重开采集。

## 冻结分析规则（读取新增准确率之前）

唯一新增parser S1：按splitlines逐行strip后，re.fullmatch以下表达式，re.IGNORECASE：
`final[ \t]+answer[ \t]*[:：][ \t]*(?:\([ \t]*([A-D])[ \t]*\)|([A-D]))[ \t]*[.!;,．！。]?`

接受整行Final answer前缀后的括号或裸A-D，大小写、水平空白、中英冒号及最多一个列出的尾标点。两个捕获组取非空值并大写。全输出恰好一条匹配行才有效；零条或多条（即使字母相同）均失败。禁止Markdown移除、散文猜测、最后一项兜底、boxed提取或新增parser扫描。以实际写入handoff的regex为准；先检查其转义正确并做合成fixture测试。

S0为原严格提取/评分，仅读取展示不得覆盖。对全部18×448既有输出应用S1，同一规则不按cell或gold调整。先生成全量S1预测及状态并保存，再加载已绑定GPQA gold评分。失败计错，主分母每cell448。length-limit也仅用同一唯一整行规则；无追加生成/修复。报告每cell全分母准确率、相对Native与Loop差值、提取成功率、截断率；native-EOS/length-limit组的总数/成功数/正确数作描述，条件准确率若提供须写分母和选择偏差，不能替代全样本主指标。

沿用原30/12/2比较族、paired bootstrap10000 seed20261008与Holm；全部明确标注事后探索，不能因Holm显著便称预注册确认。S1规则由已见格式诊断提出，有后验选择局限。无额外lambda/模型/样本筛选。比较S0/S1提取状态及有效共同样本预测一致性；不得改变S0历史工件。

## 执行范围与交付

本地nested loopscope-tflt/loopscope，基点c081f979cf3e68312a5181d6be1ef25855e5ca70。保留规划未提交的D acceptance/control。新增独立CPU脚本/针对性测试/.planning/phase11机器证据；中文报告在外层资产/报告/phase11。原runtime、producer、strict parser、D结果只读。允许单一目的commit及正常push，仅stage本Gate文件和此次规划文件。

SSH hpc2-hkustgz；项目workspaces/training_free_looped_transformers_loopscope范围，原D raw closure列明的C/D8064输出只读；只读既有冻结GPQA gold进行本次评分。新CPU结果写唯一phase11-gate-d2-*目录，原工件不覆盖。使用原pinned Python环境，无安装、模型加载、GPU、Slurm提交/取消、MMLU数据/gold或原固定复现checkout修改。原监控保持PAUSED，不创建新monitor。

验收仅三项：8064身份/分母完备且S0未改；S1精确规则与歧义处理测试通过；18cell表/冻结统计族/事后局限齐全。最小合成测试含裸/括号、大小写、水平空白、尾标点、多行重复/冲突、Markdown/散文拒绝。科学规则问题回planning，不能自行放宽。完成发送GATE_D2_FINAL_AUDIT至planning（既有用户授权的派发/终态/修复协议），含证据路径、版本、测试、限制，不自行PASS。若发送不可用给可转交包。MMLU继续暂停。
