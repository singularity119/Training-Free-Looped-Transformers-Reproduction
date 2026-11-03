# Phase 11 D4 GPQA Main 原始结果

仅 D4：Qwen/Qwen3-4B-Instruct-2507，18 个配置 × 448 = 8,064 条。无 D5/D6 结果。
包含完整 raw 文本、生成 token IDs、telemetry、command_args、env、历史 producer/分析源码、S0/S1/S2 原预测、分析/closure、GPQA prepared pool、输入 token IDs、选项排列、gold 侧车、原作者 CSV/zip 和相关日志。

历史 CoT S2：Native 225/448，普通 LoopK2 226/448，普通 LoopK3 215/448，C2（K2 current_t λ=0.5）244/448，F3（K3 fixed_t0 λ=0.9）244/448。其余配置见 coverage.json。
全部配置各 448 唯一 identity，无重复/缺失，原始数据与输入/选项/gold 已核验闭合。归档逐文件解压字节与源文件直接比较通过。没有重跑模型、换 parser 或新增 hash 清单。

在本目录执行（目标目录应是新的空目录）：

```bash
mkdir -p /Users/singularity/Desktop/tflt_digai/资产/数据/phase13/gpqa_d4_history
cat phase11_d4_gpqa_main.tar.gz.part-* | tar -xz -C /Users/singularity/Desktop/tflt_digai/资产/数据/phase13/gpqa_d4_history
```

解包后根目录为目标目录下的 phase11_d4_gpqa_main/。保留远端相对目录结构；manifest.json 中每个 files 项的 source 是原路径，local 是相对解包根的路径。原 JSON 内绝对路径不改写，按此映射读取。raw_roots 中的路径同样去除 remote_root 前缀即可定位。

原始 raw predicted_label 与冻结 S2 不同；复核 226/244 应使用 artifacts/phase11-gate-d4-final-20261013T141800Z/analysis/predictions.jsonl 内 S2 对原 gold，不能用原严格提取字段替代。实际上限为 8192；初始 input_facts 的 2048 不代表 D4 运行上限。cache=first，decode=full，window=15:18，BF16、greedy、native EOS；精确参数看原 command_args/env。

D4 旧 plan 中有 1,029 条不存在的计划路径，详见解包 manifest 的 missing_plan_references；最终 closure 的全部 raw 已齐全。运行包没有独立 generation_config.json，不补造。模型权重/HF 缓存/虚拟环境不包含在内。

本次按用户追加授权，仅将 D4 原始数据分卷提交到 Git。每卷不超过 40 MiB；所有卷均须下载后再解包。
