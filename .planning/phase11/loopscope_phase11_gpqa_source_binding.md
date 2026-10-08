# Phase11 GPQA official source binding — ACCEPTED

2026-10-08，规划接纳Gate A候选来源，不等于Gate A整体PASS。此文件为contract_v1的GPQA来源补充，取代其中待绑定HF acquisition repo/revision/physical split，其他科学配方不变。

- dataset entity: GPQA-Main，完整448题。
- source_kind: author_github_archive。
- github_repo: idavidrein/gpqa。
- github_commit: 56686c06f5e19865c153de0fdb11be3890014df7。
- archive: dataset.zip；member: dataset/gpqa_main.csv。
- logical_split: main_csv；无虚构train/dev物理split。
- canonical_order: 固定commit归档成员CSV行顺序，index从0开始。
- identity: github:idavidrein/gpqa@56686c06f5e19865c153de0fdb11be3890014df7:dataset/gpqa_main.csv:<index>。
- 使用Question、Correct Answer、Incorrect Answer 1/2/3字段；不使用Pre-Revision或Extra Revised字段，不把Explanation、验证者正确率、反馈等列送入prompt或用于选题。正确答案角色仅隔离构建器接触，按冻结seed20261008打乱并封存gold映射。

规划已通过只读SSH直接读取候选metadata并独立CSV DictReader计数：448行，所需字段存在，CSV3214603bytes。候选文件为staging/phase11-gate-a-20261008T094044Z/gpqa_official_source_candidate.json；实际输入在同stage的official-gpqa-inspection/extracted/gpqa_main.csv。作者官方README明确将归档与HF列为替代分发。未核实与HF候选revision83022cefff930aea54f654c0b282e74b9eeda5c6的逐题内容/顺序等同，不作此声明、不将GitHub commit伪装为HF revision。接纳依据为作者原始发布与实际Main结构，不是声称与不可访问副本等价。

允许原Gate A executor据此完成projection、tokenization、示例/选项配置和sealed映射，并更新本Gate机器配置及data binding。既有inspection文件保持原样，新接纳状态另存证据；不新增blob/content digest记录，不改写历史检查文件。无GPU、正式生成、outcome或下一Gate权限。报告必须保留实际source_kind和上述等同性边界。
