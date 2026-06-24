"""Remote command generation for Slurm and SSH backends."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tflt.config import LoopConfig


@dataclass(frozen=True)
class EvalSpec:
    model: str
    tasks: str
    limit: Optional[int]
    loop_config: Optional[LoopConfig]
    output_dir: str
    num_fewshot: Optional[int] = None
    batch_size: str = "auto"
    dtype: str = "bfloat16"


def build_lm_eval_command(spec: EvalSpec) -> List[str]:
    cmd = [
        "python",
        "-m",
        "tflt.eval_runner",
        "--model",
        spec.model,
        "--tasks",
        spec.tasks,
        "--batch-size",
        spec.batch_size,
        "--dtype",
        spec.dtype,
        "--output-dir",
        spec.output_dir,
    ]
    if spec.limit is not None:
        cmd.extend(["--limit", str(spec.limit)])
    if spec.num_fewshot is not None:
        cmd.extend(["--num-fewshot", str(spec.num_fewshot)])
    if spec.loop_config is not None:
        cfg = spec.loop_config
        cmd.extend(
            [
                "--loop",
                "--window",
                "%d:%d" % cfg.window,
                "--k",
                str(cfg.k),
                "--iteration-mode",
                cfg.iteration_mode,
                "--strategy",
                cfg.strategy,
                "--alpha",
                str(cfg.alpha),
                "--beta",
                str(cfg.beta),
                "--cache-strategy",
                cfg.cache_strategy,
                "--decode-mode",
                cfg.decode_mode,
            ]
        )
        if cfg.first_n is not None:
            cmd.extend(["--first-n", str(cfg.first_n)])
    return cmd


def shell_join(argv: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def make_run_dir(root: str = "runs") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(root) / ts


def render_slurm_script(
    command: List[str],
    job_name: str,
    result_root: str,
    partition: str = "gpu",
    gres: Optional[str] = None,
    gpus: int = 1,
    cpus: int = 8,
    mem: str = "64G",
    time_limit: str = "24:00:00",
    remote_src: Optional[str] = None,
    hf_endpoint: str = "https://hf-mirror.com",
    hf_home: Optional[str] = None,
    transformers_cache: Optional[str] = None,
    hf_datasets_cache: Optional[str] = None,
) -> str:
    body = shell_join(command)
    hf_home = hf_home or "{result_root}/hf_home".format(result_root=result_root)
    transformers_cache = transformers_cache or "$HF_HOME/transformers"
    hf_datasets_cache = hf_datasets_cache or "$HF_HOME/datasets"
    gres = gres or "gpu:{gpus}".format(gpus=gpus)
    remote_src = remote_src or "."
    return """#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={result_root}/control/%x-%j.out
#SBATCH --error={result_root}/control/%x-%j.err

set -euo pipefail

mkdir -p "{result_root}/control"
echo "$0" > "{result_root}/control/sbatch_script.txt"
printf '%s\n' {quoted_command} > "{result_root}/control/command.txt"

source /etc/profile >/dev/null 2>&1 || true
module load anaconda3 cuda/12.4 uv
cd "{remote_src}"
. "{remote_src}/.venv/bin/activate"

python -V | tee "{result_root}/control/python_version.txt"
nvidia-smi | tee "{result_root}/control/nvidia_smi_before.txt"

export HF_ENDPOINT="${{HF_ENDPOINT:-{hf_endpoint}}}"
export HF_HOME="${{HF_HOME:-{hf_home}}}"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-{transformers_cache}}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-{hf_datasets_cache}}}"
export RESULT_ROOT="{result_root}"

{body}

nvidia-smi | tee "{result_root}/control/nvidia_smi_after.txt"
""".format(
        job_name=job_name,
        partition=partition,
        gres=gres,
        gpus=gpus,
        cpus=cpus,
        mem=mem,
        time_limit=time_limit,
        result_root=result_root,
        remote_src=remote_src,
        hf_endpoint=hf_endpoint,
        hf_home=hf_home,
        transformers_cache=transformers_cache,
        hf_datasets_cache=hf_datasets_cache,
        body=body,
        quoted_command=shlex.quote(body),
    )


def render_ssh_script(command: List[str], result_root: str, session_name: str) -> str:
    body = shell_join(command)
    return """#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="{result_root}"
SESSION_NAME="{session_name}"
mkdir -p "$RESULT_ROOT/control"
python -V | tee "$RESULT_ROOT/control/python_version.txt"
nvidia-smi | tee "$RESULT_ROOT/control/nvidia_smi_before.txt"
printf '%s\n' {quoted_command} > "$RESULT_ROOT/control/command.txt"

tmux new-session -d -s "$SESSION_NAME" 'bash -lc {tmux_command}'
tmux display-message -p -t "$SESSION_NAME" '#{{session_name}} #{{session_created}}' \
  > "$RESULT_ROOT/control/tmux_session.txt"
""".format(
        result_root=result_root,
        session_name=session_name,
        quoted_command=shlex.quote(body),
        tmux_command=shlex.quote(body + " 2>&1 | tee \"$RESULT_ROOT/control/stdout.log\""),
    )


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        mode = path.stat().st_mode
        path.chmod(mode | 0o111)


def env_snapshot() -> Dict[str, str]:
    keys = [
        "HF_ENDPOINT",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _loop_metadata(config: LoopConfig) -> Dict[str, object]:
    return {
        "tflt_loop": {
            "model_alias": config.model_alias,
            "window": list(config.window),
            "k": config.k,
            "iteration_mode": config.iteration_mode,
            "strategy": config.strategy,
            "alpha": config.alpha,
            "beta": config.beta,
            "cache_strategy": config.cache_strategy,
            "decode_mode": config.decode_mode,
            "first_n": config.first_n,
        }
    }
