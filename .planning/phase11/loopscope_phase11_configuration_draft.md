# Phase 11 配置确认草案

已由2026-10-08的loopscope_phase11_contract_v1.md取代；下文保留配置讨论历史，不再作为当前授权入口。

2026-10-08。状态：部分科学配置已由用户确认，其余待确认；无执行 Gate、GPU、Slurm 或 outcome 授权。当前聊天为第十一阶段规划线程。本文不改变其他阶段 control。

## 已确认

- 目标：Qwen/Qwen3-4B-Instruct-2507，参考第十阶段配置组；精确 revision 后续核对。用户选中模型窗口方案，规划按认可该方案理解。
- 单窗口 inclusive 15:18，K=2/3。
- 任务：ARC-Challenge 25-shot、MMLU-Pro 5-shot CoT、GPQA-Main 0-shot。
- GPQA-Main：0-shot CoT 生成后提取答案，用户已明确选择；不采用直接候选似然评分。
- lambda：0.1、0.5、0.9，取代第十阶段九档。
- decode=full：prefill 及每个生成 token 的解码步都执行 Loop。
- 方向仅在各配置自身的 prefill 轨迹估计；decode 不重新拟合，复用其对应 prefill 方向。
- 拟合位置为实际保留 prompt 的全部非 padding、非 special token，包含示例和问题；FP32、未中心化、未行归一化的 exact reduced SVD top-1，沿用第十阶段数值语义。
- prefill 仅衰减最后有效输入位置；decode 仅衰减当前生成位置。
- t 指 Loop 内部迭代，每个生成 token 从 t=0 开始，不是生成 token 序号。
- fixed_t0：prefill t0 估 v0 不干预，后续使用 v0；decode 每个 token 的 t0 不干预，t>=1 使用该 prefill v0。
- lag1：prefill t0 估 v0 不干预，t>=1 使用自身上一轮干预前残差估出的 v_(t-1)；decode t0 不干预，t>=1 使用 prefill v_(t-1)。
- current_t：prefill 每轮从当轮干预前残差估 v_t 并立即干预，包含 t0；decode 每轮使用 prefill v_t 干预，包含 t0。
- 报告必须称为 prefill 方向时序、decode 冻结复用；不能称为 decode 实时重估。current_t 同时改变干预起点与方向时序，不能单独归因于任一因素。
- 不使用单个 decode token 的残差做 rank-1 SVD：其衰减退化为整体残差缩放。不新增历史生成 token 的滚动方向估计。

## 尚待用户确认的建议

- 模型 dtype 建议 BF16；revision 后续只读核对后绑定。
- Native、普通 Loop 对照；三策略、三档 lambda；不加入 Matched-norm。
- ARC 完整候选文本似然评分，主 acc_norm、附 acc；MMLU-Pro 与 GPQA 的生成模板、长度、答案提取及停止规则待冻结。
- batch=1、block damped Euler、alpha=1、每步 1/K、beta=0、cache=first；生成长度、模板、提取和停止规则尚未冻结。
- 固定全网格探索及多重比较方案尚未最终冻结，不将测试集当地最优称为迁移验证。

若上述单窗口 K2/K3 面板通过确认，且实际实现验证 K2 fixed_t0/lag1 等价，则每任务 1 Native + 2 Loop + 6 K2 干预 + 9 K3 干预 = 18 独立配置，三任务共 54。此数量为条件性计划，不代表已运行。

## 下一步

确认剩余科学选择后再建立正式 contract/control 和分 Gate 计划。当前不创建执行线程，不提交实验，不继承第十阶段的资源或跨线程授权。
