# Gate A Download Route Supplement 3

```text
PROJECT_PHASE=LoopScope Phase 7
GATE=A
AUTHORIZED_EXECUTOR_THREAD=01a00159-6c55-72b3-bbf8-5ac798868258
SUPPLEMENTS=.planning/phase7/loopscope_phase7_gate_a_repair_and_permission_supplement_2.md
CHANGE_CLASS=OPERATIONAL_DOWNLOAD_ROUTE_ONLY
SCIENCE_CHANGE=NONE
NETWORK_PRIORITY=EXISTING_CACHE_THEN_HF_MIRROR_THEN_OFFICIAL_HUGGING_FACE
```

用户明确允许缺失模型文件优先从 HF 清华镜像下载。Gate A metadata-only acquisition 的路由更新为：

1. 先复用 `/hpc2hdd/home/xhuang225/shared/hf_home` 中已经存在且可读的 exact snapshot 文件；
2. 缺失的 config/tokenizer 轻量文件优先在单条命令作用域设置
   `HF_ENDPOINT=https://hf-mirror.com` 获取；不得把环境变量写入 shell profile 或改系统配置；
3. 镜像缺文件、revision 解析异常或无法服务 gated repo 时，可以回退到官方
   `https://huggingface.co`，仍只使用 supplement 2 已允许的 metadata/tokenizer 文件类；
4. 两条来源都失败时保留 exact 错误并按 supplement 2 的 `BLOCK_MODEL_ACCESS` 规则停止，不换
   repo、镜像模型或科学 membership。

其余边界完全不变：禁止权重/checkpoint、dataset、模型 forward、CUDA/GPU/Slurm；禁止新增或
暴露 credential、交互登录、license acceptance、安装 hfd/aria2 或改环境。只可使用已有且
非交互可用的认证上下文；不得在命令、日志、artifact 或 terminal packet 中显示 token。

在 post-repair terminal 中按模型记录实际来源为 `existing_cache`、`hf-mirror.com` 或
`huggingface.co`，以及是否发生 fallback；不记录 token ID 或任何 digest。
