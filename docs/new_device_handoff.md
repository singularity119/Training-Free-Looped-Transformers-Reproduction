# 在新设备继续 LoopScope

本文落实 2026-09-02 用户授权的项目上下文迁移。原 `AGENTS.md` 按用户要求保持原样，
没有用 `AGENTS.override.md` 或其他自动覆盖文件替换它。
当前操作是接续已有 `loopscope` 分支，不是从研究初始基点重新建立项目。

## 1. 获取已有分支

在新设备上选择一个新的专用目录，然后执行：

```bash
git clone --branch loopscope --single-branch \
  git@github.com:singularity119/Training-Free-Looped-Transformers-Reproduction.git \
  loopscope-tflt
cd loopscope-tflt
git status --short --branch
git remote get-url origin
git rev-parse --show-toplevel
git merge-base --is-ancestor 4f59bd93eca4da3cbf458a93508f91c5b23912bc HEAD
```

已有 checkout 应先查看并保护本地改动，再 `git fetch origin loopscope` 和
`git pull --ff-only origin loopscope`。若分支已分叉，应检查双方更改，不能 reset、force-push 或删除重建。
`main` 仍只作只读复现参照，不能把研究变更合并回去。

原 `AGENTS.md` 中的旧 Mac 绝对路径是历史设备路径；迁移后的本地根由用户在新设备选择，
以 `git rev-parse --show-toplevel` 为准。它对原复现 checkout、冻结实验和远端写入的保护继续适用。
旧“初始化时从基点创建分支”的规则对应首次建立研究项目；本次应保留已经存在的研究历史。

## 2. 建立旧 planning 路径的兼容链接

Git 中唯一真实 planning 目录是 `<repo>/.planning/`。原 `AGENTS.md` 和部分历史程序使用
`<repo>/../.planning/`，因此新设备先创建指向仓库内目录的兼容链接。
在仓库根执行以下脚本；若外层已有其他项目的 `.planning`，脚本会停止，保留现有目录：

```bash
python3 - <<'PY'
from pathlib import Path
import os
import subprocess

repo = Path(subprocess.check_output(
    ['git', 'rev-parse', '--show-toplevel'], text=True).strip()).resolve()
target = repo / '.planning'
legacy = repo.parent / '.planning'
if not target.is_dir():
    raise SystemExit('Missing tracked .planning directory; update the loopscope branch first.')
if legacy.is_symlink():
    if legacy.resolve() != target:
        raise SystemExit('Existing .planning link points elsewhere; inspect it before continuing.')
elif legacy.exists():
    raise SystemExit('Existing external .planning must be reviewed; do not overwrite it.')
else:
    legacy.symlink_to(os.path.relpath(target, legacy.parent), target_is_directory=True)
print('Planning context:', target)
print('Compatibility link:', legacy)
PY
```

这只是让旧地址访问同一份文档，不创建第二份可独立修改的 control。
若设备不支持符号链接，应先解决该兼容要求；不要复制两份 planning 并分别维护。
仓库内 `.planning` 根目录的 50 个旧短路径链接已经删除，文件只在 `phaseN/` 中保留。
这里需要的只是仓库外层的一个目录兼容链接；历史正文里的短路径按阶段号映射到规范路径。

## 3. 在 Codex 中建立新任务

打开 Git 根目录 `loopscope-tflt`，向新任务发送：

> 请按 README 的项目接续入口读取 AGENTS.md、PROJECT_MEMORY.md 和
> docs/new_device_handoff.md，然后读取记忆入口所指的当前 control。
> 这是我授权的新设备接续，请使用当前 checkout 的实际路径，保留既有 loopscope 分支历史。
> 先确认当前阶段、最近结论、未解决问题和下一步，再继续我指定的实验工作。

上传文档不会复制旧 Codex 聊天、账号记忆、task ID、automation、插件连接或 SSH 登录状态。
旧 thread ID 仅作 provenance；已关闭的 executor 和 heartbeat 不会因 clone 而重新激活。
不要全量读入一百多份历史文件；按“摘要 → 当前 control → 对应合同/证据”逐层检索。

## 4. 本地与 HPC 环境

- 新设备先准备 Git、Python 3、Codex，以及本机的 GitHub 访问权限。
- 若沿用研究 Gate 工作流，需要在新设备准备 `research-gate-orchestrator` 技能；
  SSH 工作流还使用 `hpc2-hkustgz-ssh`，遇 host-key 变化才使用 `hpc2-recover-host-key`。
  这些技能属于原设备 Codex 配置，不在本仓库；从原设备对应技能目录迁移，或从可信安装源安装并阅读。
- HPC2 的 VPN、SSH host alias、私钥、known_hosts 和账号访问需要在新设备配置；不要写进 Git。
- 只读 SSH 检查使用 `BatchMode=yes`、有界 `ConnectTimeout`、`ClearAllForwardings=yes`。
  host-key 不匹配时核对可信指纹，不能关闭 host-key checking。
- 本机只保存代码和文档；模型、数据、完整 eval 结果、缓存和 GPU 实验继续留在授权的 HPC2 路径。
- 先核对既有已审计远端环境，不能因为换设备就重建或升级运行中的 venv。
  环境规范见 [envs/hpc2](../envs/hpc2/README.md)，实验入口见对应阶段 runbook。

本机可以做无模型下载的导入/CLI 检查：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m tflt.cli --help
git diff --check
```

这不证明 GPU 数值路径或所有历史冻结 verifier 可在新代码上复跑；真实实验仍按新的授权范围验收。
历史阶段的 verifier 可能绑定当时的不可变卡或控制文件，不能为“通过检查”而改写它们。

## 5. 本次迁移保存和未包含的内容

- `.planning` 的 102 个原始 Markdown 文件移入 Git；曾一起迁入的 50 个根目录短路径软链接已清理。
  除索引 README 外，原始 control、contract、handoff、supplement、规则与审计原文保留。
- 新增 `PROJECT_MEMORY.md` 和本文，更新 README 与阶段 runbook 的导航/路径。
- 旧设备的外层 `.planning` 改为指向仓库内目录的链接。
- 原 `AGENTS.md` 内容保持迁移前本地版本；按用户追加授权，它已有的本地改动随此次提交保存。
- 删除短路径软链接时，一起提交了两个 Phase 4 文件已有的路径修正：
  `scripts/loopscope/run_qwen4_phase4_p4b.py`、`src/tflt/loopscope/phase4_outcome.py`。
  三处 control 引用改为 `.planning/phase4/loopscope_phase4_control.md`，历史冻结检查逻辑保持不变，
  避免删除旧链接后因找不到 control 而跳过检查。
- 五个历史阶段 runbook 中的导航更新随本次迁移保存，既有报告目录修正保留。
- 外层 `资产/` 中的中文报告、图片和 source-data 没有上传；可按需单独复制。
- 不包含模型、数据、运行目录、完整日志、venv、个人 Codex memory 或凭据。

本次迁移没有启动 GPU/Slurm 作业、改动方法公式、重新计算 outcome 或授予下一 Gate 权限。
