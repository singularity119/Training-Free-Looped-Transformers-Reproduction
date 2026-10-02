# Phase 10 科学合同 v1

2026-10-02，依据用户已确认的三组cell、双方向策略、九档lambda、取消Matched-norm、MMLU与ARC25-shot口径，以及当前逐Gate执行授权。

科学配方规范引用 loopscope_phase10_plan.md v2 的 §§2–4，作为本合同组成部分冻结：每数据集41独立配置，K2经验证等价后别名复用；MMLU预计32新+9复用，ARC41新；fixed_t0/lag1仅Online干预，Native/Loop对照；MMLU原14042/57subjects/5shot，ARC全test1172/25shot标准文本评分acc_norm主、acc描述。

继承Phase9模型/tokenizer和MMLU revision及dtype/cache/Euler/rank1/FP32 exact SVD语义。ARC具体revision、原生sampler seed20261002、示例identity、max_length在Gate A事实附件提出并由planning接受后，才准入B；不得根据outcome决定。

MMLU选参：4个非重复cell×策略各取Online micro acc最高lambda，并列较小lambda；在ARC outcome前固定。主ARC迁移8比较一组Holm；每数据集36×2=72扫描比较一组探索Holm；K3 lag1−fixed_t0九比较一组次family。exact双侧McNemar，alpha0.05，配对bootstrap10000次seed20261002，MMLU按subject内、ARC按题，nominal95%CI。

无Matched-norm，不作方向特异性机制结论。禁止自动扩窗/K4/增lambda、改shot/删题/变评分、重拟合Shared或训练。ARC候选截断不一致是需要planning裁决的科学口径问题，不可静默处理。

执行权限仅由control/exact handoff授予。本合同不授权任何executor跨Gate。
