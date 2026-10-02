# Gate A 操作补充 1：固定 ARC 小文件与 CPU overlay

2026-10-02。接收exact executor 01a0fb34-5c36-7210-b17d-744410853d1f的范围内依赖请求。规划认定这是完成已授权ARC数据binding所必需的小规模准备，不改变科学配方。此补充仅覆盖旧handoff中相关下载/运行路径限制，其他边界保持。

授权同一executor在hpc2-hkustgz下载 allenai/ai2_arc 固定revision `210d026faf9955653af8916fad021475a3f00453` 的ARC-Challenge train/test/validation三个parquet文件，官方报告大小合计449460 bytes。下载来源限官方Hugging Face该repo及其正常文件重定向；核对响应、可读metadata、split行数与路径即可，不新增摘要值证据。

目标限 /hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase10-gate-a-<timestamp>/raw/；新建路径，不覆盖旧文件或写共享cache。原始parquet下载不可避免携带标签列，不视为授权读取目标标签；test/validation的CPU读取必须用pyarrow列投影仅id/question/choices，禁止读取/导出/打印answerKey或用其计算结果。train仅为已冻结25-shot示例可读取answerKey。

可上传当前Gate增量代码到同workspace的staging/phase10-gate-a-<timestamp>/，用既有只读source/package配合明确记录的overlay路径进行CPU数据构建、tokenizer和边界检查。记录真实代码commit/未提交diff、路径、Python import解析来源、环境和命令；不得让旧模块误遮蔽新模块。允许上传专用运行所需的小型源文件，不拷贝模型或整个环境。

远端旧phase9-gate-c-20260922分支保持只读，不切换、不同步、不修改。若overlay不能运行，继续最小诊断并报告实际缺失依赖，不静默升级环境。只用已有tokenizer/cache；没有模型forward、GPU、Slurm、target outcome权限。

完成后把ARC实际revision、文件路径、split数、fewshot身份和无标签prefix检查写入A事实附件交planning接受。Gate B仍锁定。此为操作补充，不是A验收或数据binding已通过。
