# Gate A 普通Git合并补充授权

按照research-gate-orchestrator协作约定执行。Planning=01a07ff5-79b1-79f0-809a-4d0971475686；Executor=01a0c70e-2092-72b3-911c-6544822d585c。科学合同v2与scope supplement保持。

Planning直接核对：本地9588e69工程提交已完成；origin/loopscope唯一远端独有ae40ce6d87dcf08e31d18a349e2ccd73745756a9，仅增加Phase8 Gate E evidence及docs/loopscope_phase8.md历史报告文字。非fast-forward为版本分叉，非科学变更。

授权同一A executor在loopscope分支把该exact远端提交普通merge进当前HEAD（包含本补充的planning提交），保留双方提交历史，然后普通push origin loopscope。允许仅为该合并解决两份历史文档的文本冲突：保留本地较新E/F/G结果和远端E原始记录，不覆盖或删除有效结果、不回退到旧状态。不得选择整文件ours/theirs丢掉另一方更新，不改Phase8运行代码或工件，不rebase/reset/force。若远端再次新增其它提交，先核对其范围再报告，不自动扩大本授权。

合并后检查git diff --check及合并涉及文档，工程代码未改变不需重跑已通过模型/完整suite；若有实际代码冲突则停止合并解决并向planning报告。此授权不启动B/模型/GPU。完成继续原Gate A终态交付，记录合并commit、push结果和最终dirty状态。
