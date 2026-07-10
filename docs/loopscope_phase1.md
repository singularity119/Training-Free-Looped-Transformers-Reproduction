# LoopScope 第一阶段执行说明（Qwen3-1.7B × MMLU）

## 范围与结论边界

本阶段只验证：冻结模型后，不同宽度为 4 的 loop window 所产生的内部信号，能否预测真实 MMLU 收益。它不是通用自动选窗器的完成版，也不扩展 K、步长、策略、缓存、模型或任务。

固定配方为：

```text
model=qwen3-1.7b-base
revision=ea980cb0a6c2ae4b936e82123acc929f1cec04c1
task=mmlu
num_fewshot=5
dtype=float16
k=2
iteration_mode=block
strategy=damped_euler
alpha=1.0
beta=0.0
cache_strategy=last
decode_mode=bypass
window_width=4
```

本地原复现项目与 HPC2 原复现项目都是只读参照：

```text
/Users/huangxutao/Desktop/Training-free looped transformer/Training-Free Looped Transformers Reproduction
/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_reproduction
```

LoopScope 只允许在新 clone 的 `loopscope` 分支开发和运行：

```text
本地：/Users/huangxutao/Desktop/Training-free looped transformer/LoopScope_Entropy-Aware Window Selection for Training-Free Looped Transformers/loopscope-tflt
HPC2：/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope
基点：4f59bd93eca4da3cbf458a93508f91c5b23912bc
```

不得在原复现 checkout 中 fetch、切分支、安装环境或运行 LoopScope。不得覆盖、移动、删除或复用已有 run、output、cache、模型及日志。

`configs/profiles/hpc2-hkustgz.toml` 的 `remote_src` 指向原始复现 checkout，只保留给复现流水线作历史配置；LoopScope **禁止** 使用该 profile 进行同步、安装、运行或提交。LoopScope 必须显式使用上面的专用 checkout。

## Window 边界与有效秩冻结语义

对于含 N 个 decoder layer 的模型，`probe-layers` 输出 N+1 个边界状态 `B_0..B_N`。`B_j` 表示已经执行前 j 个 decoder layer 后的状态，因此两端均包含的 window `[a,b]` 统一使用入口 `B_a` 与出口 `B_(b+1)`。entropy、KL-to-final 和有效秩必须使用同一对边界；`a=0` 直接使用 `B_0`，不允许负索引。工件使用 `boundary_index`、`after_layer` 和 `before_layer` 显式记录映射，不再用 layer output 下标代替 window 边界。

有效秩固定为 `gram_spectrum_shannon_effective_rank` v2，工件 schema 固定为 `loopscope.probe.v3`：先逐向量拒绝零输入并做单位归一化，再跨校准样本中心化 answer-position hidden matrix，取奇异值 `s_i` 并记录 `centered_spectrum_mass=sum_i(s_i^2)`。当且仅当该质量精确为 `0.0` 时，报告 `effective_rank=0.0`、`zero_centered_spectrum=true` 和 `centered_spectrum_mass=0.0`，表示跨样本变化的有效维数为零；禁止 epsilon 分母、近零阈值、`1.0`、null、NaN 或无显式 flag 的零值。质量为有限正值时，必须报告 `zero_centered_spectrum=false`，并严格使用 `p_i=s_i^2/sum_j(s_j^2)` 计算 `exp(-sum_i p_i log p_i)`。原始奇异值直接归一化的定义禁止用于第一阶段工件。有效秩始终仅作辅助诊断；`criterion_v0` 的 effective-rank gate 保持关闭，不得进入主排序或选窗。

## Gate 顺序

1. **Gate A（本地代码门）**：完成代码、单元测试、compile 和 CLI smoke 后必须停止，把 branch、commit、dirty、diff 与测试结果交给规划线程 `019f4c7b-e5eb-77f2-b007-59d004896550`。未得到明确 `PASS`，不得推送功能代码、SSH 或进入 HPC2。
2. **Gate B（远端环境门）**：Gate A PASS 后，才允许同步 Git、建立或核对 HPC2 专用 clone、创建独立 versioned venv，并完成 CPU/import/unit/compile/CLI help smoke。Gate B 不运行 GPU probe、MMLU eval 或 Slurm 实验；完成后停止审计。
3. **Gate C（GPU 仪器门）**：Gate B 明确 PASS 后，才运行四样本 probe、锚点 loop-effect audit 和三个 sentinel window probe。完成后再次停止审计。
4. **Gate D（工程 smoke 门）**：Gate C PASS 后，运行 baseline 和所有候选 window 的 `--limit 5`。`limit=5` 只验证工程链路、样本对齐和 schema，**不得按小样本准确率排序、选择最佳 window 或形成科学结论**。完成后停止审计。
5. **Gate E（完整信号与正式网格门）**：Gate D 明确 PASS 后仍须依次完成完整校准池 probe、revision 一致性、离线 score 与 score freeze；只有这些工件全部通过才允许提交 full MMLU window grid。

`prepare_qwen17_phase1.py` 会预先生成后续阶段的不可变命令，但生成不等于授权；`submit_qwen17_phase1.sh` 仍要求上一 Gate 的结构化 PASS 文件。

## Gate A 本地检查

Gate A 只在 LoopScope 新 clone 中执行：

```bash
git rev-parse --show-toplevel
git rev-parse --abbrev-ref HEAD
git status --short --branch
git diff --check
PYTHONPATH=src python3 -m unittest discover -s tests
python3 -m compileall -q src tests scripts/loopscope
PYTHONPATH=src python3 -m tflt.cli probe-layers --help
PYTHONPATH=src python3 -m tflt.cli probe-window --help
PYTHONPATH=src python3 -m tflt.cli make-window-grid --help
PYTHONPATH=src python3 -m tflt.cli score-windows --help
PYTHONPATH=src python3 -m tflt.cli analyze-window-grid --help
python3 scripts/loopscope/build_mmlu_probe_pool.py --help
PYTHONPATH=src python3 scripts/loopscope/export_mmlu_renderer.py --help
python3 scripts/loopscope/prepare_qwen17_phase1.py --help
bash -n scripts/loopscope/submit_qwen17_phase1.sh
```

完成以上检查后停在 Gate A，不运行模型、不创建 HPC2 run root。

## 构建无标签校准池

校准输入必须由正式评测所用的同一 lm-eval MMLU renderer 预渲染，严格冻结 `lm-eval==0.4.11`、target split=`validation`、5-shot、fewshot split=`dev`、`chat_template=false`、`multiturn=false`。结构化 `question/choices` 零样本自动渲染路径已禁用。

第一阶段只把 57 个 subject 的 `validation`（共 1531 条）作为非 test 校准候选；`dev`（共 285 条）只提供每个 subject 的 5-shot demonstrations；`test` 只允许在规则冻结后的正式收益评估中使用。`auxiliary_train` 只存在于聚合数据配置，不属于逐 subject lm-eval task split，因此禁止作为第一阶段 renderer target，也不得映射或静默分配到 subject。

renderer manifest v2 除自哈希外，必须记录实际安装包版本、被调用源码文件及逐文件 SHA、task YAML/config SHA、精确 dataset commit、raw/processed split fingerprint、完整 source projection SHA 和 render contract SHA。每条目标记录必须记录 `task_name`、目标 split/index/doc SHA、逐条 render SHA，以及 5 个有序唯一、同 subject 的 demo provenance（ID/index、dev split、doc/render/gold SHA）。目标 gold 不得进入投影或 probe pool 字段；5 个 demo answer 必须由真实 renderer 写入 prompt，并以 `uses_target_gold_labels=false`、`fewshot_answers_present=true` 区分两种语义。

先由 exporter 从精确数据集 revision 生成 projection；builder 随后必须重新加载同一 `lm-eval==0.4.11` 与数据集、逐条重渲染并逐字节比较。仅填写格式正确的 64 位哈希、手写 manifest 或手写 prompt 都会失败。生产 CLI 不提供 `trust`/`skip verify` 开关。

```bash
PYTHONPATH=src python scripts/loopscope/export_mmlu_renderer.py \
  --output-jsonl /path/to/new/lm_eval_mmlu_projection.jsonl \
  --manifest /path/to/new/lm_eval_mmlu_renderer_manifest.json \
  --dataset-revision <exact-lowercase-dataset-commit> \
  --target-split validation \
  --fewshot-split dev \
  --seed 20260710
```

若 exporter 无法定位实际 task YAML/source、无法取得 dataset fingerprint、实际 prompt 不以目标 `Answer:` 结束，或 5 个 dev demonstrations 不能固定，立即停止；不得用手写模板、零样本 prompt 或 MMLU test 替代。

先检查，再在一个从未使用过的目标路径写入：

```bash
python scripts/loopscope/build_mmlu_probe_pool.py \
  --input-jsonl /path/to/new/lm_eval_mmlu_projection.jsonl \
  --output-jsonl /path/to/new/probe_pool.jsonl \
  --manifest /path/to/new/probe_pool_manifest.json \
  --renderer-manifest /path/to/lm_eval_mmlu_renderer_manifest.json \
  --source 'cais/mmlu@<exact-revision>' \
  --split validation \
  --count 512 \
  --seed 20260710 \
  --dry-run

python scripts/loopscope/build_mmlu_probe_pool.py \
  --input-jsonl /path/to/new/lm_eval_mmlu_projection.jsonl \
  --output-jsonl /path/to/new/probe_pool.jsonl \
  --manifest /path/to/new/probe_pool_manifest.json \
  --renderer-manifest /path/to/lm_eval_mmlu_renderer_manifest.json \
  --source 'cais/mmlu@<exact-revision>' \
  --split validation \
  --count 512 \
  --seed 20260710
```

抽样按 subject 分层，seed 固定为 `20260710`。manifest 的 rendering subset hash 覆盖目标 provenance、renderer contract、全部固定 demo ID/hash 与逐条 prompt hash。任一目标文件已存在时脚本直接停止，不覆盖。

## 冻结候选窗口

实际 `num_hidden_layers` 经检查后，用 CLI 生成一次新的 manifest：

```bash
PYTHONPATH=src python -m tflt.cli make-window-grid \
  --num-hidden-layers 28 \
  --output /path/to/new/window_grid.json \
  --width 4 \
  --anchor 12:15 \
  --random-seed 20260710
```

这里的 `28` 只是 Qwen3-1.7B 当前预期值，远端仍须用实际模型 config 核对。Gate D 开始后不得修改该 manifest；如需修改，创建新的 timestamped run root/version。

## 准备新的 HPC2 run root

Gate A PASS、HPC2 新 clone 和独立环境通过后，在干净的 `loopscope` checkout 中执行：

```bash
python scripts/loopscope/prepare_qwen17_phase1.py prepare \
  --repo-root /hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope \
  --planning-thread-id 019f4c7b-e5eb-77f2-b007-59d004896550 \
  --venv '/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-YYYYMMDD[-vN]' \
  --window-grid /path/to/new/window_grid.json \
  --probe-pool-jsonl /path/to/new/probe_pool.jsonl \
  --probe-pool-manifest /path/to/new/probe_pool_manifest.json \
  --timestamp YYYYMMDD-HHMMSS \
  --dry-run
```

确认预览后去掉 `--dry-run`。目标必须形如：

```text
/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs/loopscope-qwen17-mmlu-phase1-<timestamp>
```

repo、run base 和 venv 路径是硬保护，不是可替换示例：repo 必须精确为上述 HPC2 LoopScope clone，run root 必须是固定 run base 的直接子目录且使用完整时间戳，venv 必须是 clone 内 `.venv-loopscope-cu121-YYYYMMDD[-vN]`。任何相邻路径都拒绝。run root 已存在时脚本立即停止。

Gate C 生成的 `probe-layers` 命令始终显式传入 `--input-manifest`；`probe-window` 同时显式传入 `--input-manifest` 和冻结的 `--window-grid`。缺少任一 provenance 工件时不允许运行。

## 审批文件与提交

Gate A PASS 文件示例：

```json
{
  "planning_thread_id": "019f4c7b-e5eb-77f2-b007-59d004896550",
  "gate": "A",
  "decision": "PASS",
  "approved_commit": "<Gate A commit>"
}
```

Gate C 或 D 的 PASS 文件还必须包含当前 `phase1_run_manifest.json` 的 `run_manifest_sha256`。`PASS_WITH_FIXES` 不等同于 PASS：先落实修正并重新审计。

提交前默认只预览：

```bash
bash scripts/loopscope/submit_qwen17_phase1.sh \
  --run-root <new-run-root> \
  --stage gate-c \
  --approval-file <gate-a-pass.json> \
  --dry-run
```

确认后将 `--dry-run` 换成 `--submit`。后续顺序固定为：

1. `gate-d-limit`：全部候选的 limit=5 工程 smoke；
2. Gate D PASS 后运行 `gate-e-probe`：完整校准池的一次 `probe-layers` 和覆盖全部 frozen candidate 的 `probe-window`；
3. 完整 probe revision 一致后运行 `gate-e-score`：用冻结 criterion 离线 `score-windows`；
4. 执行 `prepare_qwen17_phase1.py freeze-score`，冻结并核验 score report 自哈希/文件 SHA、criterion SHA、grid SHA 和 full-probe revision report SHA；
5. `gate-e-full` 提交脚本在创建 attempt 文件和调用 `sbatch` 前，必须调用 `validate-full-freeze` 重新计算两份 full probe 的文件/canonical hash、绝对路径、完整 pool IDs/hash、render contract、grid、criterion 和三方 revision 绑定；任一变化立即失败。
6. 只有上述 pre-submit 复核完整通过，才运行 `gate-e-full`。

可独立执行同一只读 helper 做预检：

```bash
python scripts/loopscope/prepare_qwen17_phase1.py validate-full-freeze \
  --run-root <new-run-root>
```

Gate C 的四样本/sentinel 不能替代完整校准池 probe。脚本只调用一次 `sbatch`，不自动重试；attempt、receipt、claim 或 output 已存在时拒绝重复提交。

## 模型与 tokenizer revision 的强制闭合

phase-one config 与 run manifest 固定精确 HF snapshot commit。`probe-layers`、`probe-window`、loop audit 及 smoke/full eval 的 model/tokenizer load 都显式传入该 commit；加载后必须从两个实际对象解析 resolved commit，并与 manifest commit 三方完全一致：

- probe：`probe_report.json` 的 `model.revision`、`tokenizer.revision` 与 `revision_closure`；
- loop audit：`audit_report.json` 的 `model.commit_hash`、`tokenizer.commit_hash` 与 `revision_closure`；
- lm-eval baseline/window：`model_revision.json` 的 `model_commit`、`tokenizer_commit` 与 `manifest_commit`。

每个阶段结束后运行：

```bash
python scripts/loopscope/prepare_qwen17_phase1.py verify-revisions \
  --run-root <new-run-root> \
  --stage gate-c
```

Gate D、完整 probe 和 full 分别使用 `gate-d-limit`、`gate-e-probe`、`gate-e-full`。revision report 必须通过 canonical 自哈希，并精确绑定当前 run manifest、stage、预期 jobs、artifact 路径与三方 observations；不能只看 `match=true`。缺工件、缺 resolved commit、manifest 不一致或 model/tokenizer 不一致都整体停止。

离线 score 完成后执行：

```bash
python scripts/loopscope/prepare_qwen17_phase1.py freeze-score \
  --run-root <new-run-root>
```

`gate-e-full` 提交时会重新计算 score report canonical hash/文件 SHA，并核对 criterion SHA、window-grid SHA、score job 路径和 full-probe revision report；任一字节或绑定变化都会停止。

## 每个 Gate 的回传内容

```text
Gate:
branch/commit/dirty:
files changed:
exact commands:
tests and exit codes:
remote host/job ids/run root:
artifact paths:
observed result:
deviations/errors:
requested decision: PASS / PASS_WITH_FIXES / BLOCK
```

Gate D 只回传流程、样本 ID、schema、job 状态与错误；不要展示按 limit accuracy 排列的 window 表，更不能据此选窗。
