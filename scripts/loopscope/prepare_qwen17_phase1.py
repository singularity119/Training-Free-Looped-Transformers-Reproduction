#!/usr/bin/env python3
"""Prepare a write-once LoopScope Qwen3-1.7B phase-one run root.

This script only prepares manifests and Slurm task files.  It never submits a
job.  Submission is deliberately separated into ``submit_qwen17_phase1.sh`` so
that every remote stage remains behind an explicit planning-thread approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BASE_COMMIT = "4f59bd93eca4da3cbf458a93508f91c5b23912bc"
EXPECTED_ORIGIN = "git@github.com:singularity119/Training-Free-Looped-Transformers-Reproduction.git"
PLANNING_THREAD_ID = "019f4b5a-79ac-78c1-9196-c7fd733cf04d"
RUN_PREFIX = "loopscope-qwen17-mmlu-phase1-"
DEFAULT_RUN_BASE = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/runs"
)
DEFAULT_REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
DEFAULT_VENV_NAME = ".venv-loopscope-cu121-20260710"
RUN_SCHEMA_VERSION = "loopscope.phase1-run.v1"


class PreparationError(ValueError):
    """Raised before submission when a run would violate the frozen contract."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Create one new timestamped run root and immutable job manifests."
    )
    prepare.add_argument("--repo-root", default=str(DEFAULT_REMOTE_REPO))
    prepare.add_argument("--venv", default=None, help="Versioned venv path (default under repo).")
    prepare.add_argument("--config", default="configs/loopscope/qwen17_mmlu_phase1.json")
    prepare.add_argument("--criterion", default="configs/loopscope/criterion_v0.json")
    prepare.add_argument("--window-grid", required=True)
    prepare.add_argument("--probe-pool-jsonl", required=True)
    prepare.add_argument("--probe-pool-manifest", required=True)
    prepare.add_argument("--run-base", default=str(DEFAULT_RUN_BASE))
    prepare.add_argument(
        "--run-root", default=None, help="Exact new run root; overrides timestamp."
    )
    prepare.add_argument(
        "--timestamp",
        default=None,
        help="UTC YYYYMMDD-HHMMSS (default: current UTC; ignored with --run-root).",
    )
    prepare.add_argument("--partition", default="debug")
    prepare.add_argument("--gres", default="gpu:a40:1")
    prepare.add_argument("--cpus", type=int, default=8)
    prepare.add_argument("--mem", default="64G")
    prepare.add_argument("--debug-time", default="00:30:00")
    prepare.add_argument("--full-time", default="24:00:00")
    prepare.add_argument("--max-concurrent", type=int, default=2)
    prepare.add_argument("--dry-run", action="store_true")

    verify = subparsers.add_parser(
        "verify-revisions",
        help="Compare actual model revisions after a completed model-using stage.",
    )
    verify.add_argument("--run-root", required=True)
    verify.add_argument(
        "--stage",
        choices=["gate-c", "gate-d-limit", "gate-e-probe", "gate-e-full"],
        required=True,
    )
    freeze = subparsers.add_parser(
        "freeze-score",
        help="Validate and freeze the label-free score artifact before full MMLU.",
    )
    freeze.add_argument("--run-root", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        return prepare_run(args)
    if args.command == "verify-revisions":
        return verify_revisions(Path(args.run_root).expanduser().resolve(), args.stage)
    if args.command == "freeze-score":
        return freeze_score(Path(args.run_root).expanduser().resolve())
    raise AssertionError("unreachable command")


def prepare_run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser().resolve()
    venv = (
        Path(args.venv).expanduser().resolve()
        if args.venv
        else repo_root / DEFAULT_VENV_NAME
    )
    run_root = _requested_run_root(args)
    run_base = Path(args.run_base).expanduser().resolve()
    _validate_hpc2_paths(repo_root, run_base, run_root, venv)
    config_path = _resolve_repo_path(repo_root, args.config)
    criterion_path = _resolve_repo_path(repo_root, args.criterion)
    expected_config = repo_root / "configs" / "loopscope" / "qwen17_mmlu_phase1.json"
    expected_criterion = repo_root / "configs" / "loopscope" / "criterion_v0.json"
    if config_path != expected_config or criterion_path != expected_criterion:
        raise PreparationError("phase config and criterion must use the exact committed paths")
    grid_path = Path(args.window_grid).expanduser().resolve()
    pool_path = Path(args.probe_pool_jsonl).expanduser().resolve()
    pool_manifest_path = Path(args.probe_pool_manifest).expanduser().resolve()

    if run_root.exists():
        raise FileExistsError("refusing to reuse existing run root: %s" % run_root)
    if args.max_concurrent < 1 or args.max_concurrent > 2:
        raise PreparationError("phase one permits at most two concurrent GPU jobs")
    if args.cpus < 1:
        raise PreparationError("--cpus must be positive")

    git = _git_provenance(repo_root)
    config = _read_json(config_path)
    _validate_phase_config(config)
    criterion = _read_json(criterion_path)
    _validate_criterion(criterion)
    grid = _read_json(grid_path)
    _validate_grid(grid, config)
    pool_manifest = _read_json(pool_manifest_path)
    pool_bytes = _validate_pool(pool_path, pool_manifest)

    jobs = _build_jobs(
        run_root=run_root,
        repo_root=repo_root,
        venv=venv,
        config=config,
        grid=grid,
        criterion_path=run_root / "manifests" / "criterion_v0.json",
    )
    manifest: Dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "planning_thread_id": PLANNING_THREAD_ID,
        "run_root": str(run_root),
        "git": git,
        "venv": str(venv),
        "cache": {
            "HF_HOME": "/hpc2hdd/home/xhuang225/shared/hf_home",
            "TRANSFORMERS_CACHE": "/hpc2hdd/home/xhuang225/shared/hf_home/hub",
            "HF_DATASETS_CACHE": "/hpc2hdd/home/xhuang225/shared/datasets",
            "UV_CACHE_DIR": "/hpc2hdd/home/xhuang225/shared/uv",
        },
        "inputs": {
            "phase_config": {
                "source_path": str(config_path),
                "snapshot_path": str(run_root / "manifests" / "qwen17_mmlu_phase1.json"),
                "sha256": _file_sha256(config_path),
            },
            "window_grid": {
                "source_path": str(grid_path),
                "snapshot_path": str(run_root / "manifests" / "window_grid.json"),
                "manifest_sha256": grid["manifest_sha256"],
            },
            "criterion": {
                "source_path": str(criterion_path),
                "snapshot_path": str(run_root / "manifests" / "criterion_v0.json"),
                "file_sha256": _file_sha256(criterion_path),
                "criterion_sha256": manifest_sha256(criterion),
            },
            "probe_pool": {
                "source_path": str(pool_path),
                "snapshot_path": str(run_root / "manifests" / "probe_pool.jsonl"),
                "manifest_source_path": str(pool_manifest_path),
                "manifest_snapshot_path": str(
                    run_root / "manifests" / "probe_pool_manifest.json"
                ),
                "pool_sha256": hashlib.sha256(pool_bytes).hexdigest(),
                "manifest_sha256": pool_manifest["manifest_sha256"],
                "count": int(pool_manifest["count"]),
                "seed": int(pool_manifest["seed"]),
                "split": str(pool_manifest["split"]),
                "task_group": "mmlu",
                "num_fewshot": 5,
                "uses_target_gold_labels": False,
                "fewshot_answers_present": True,
                "render_contract_sha256": _nested_value(
                    pool_manifest, ["renderer", "render_contract_sha256"]
                ),
                "render_contract_subset_sha256": pool_manifest[
                    "render_contract_subset_sha256"
                ],
            },
        },
        "frozen_recipe": {
            "model": "qwen3-1.7b-base",
            "repo_id": "Qwen/Qwen3-1.7B-Base",
            "task": "mmlu",
            "num_fewshot": 5,
            "dtype": "float16",
            "k": 2,
            "iteration_mode": "block",
            "strategy": "damped_euler",
            "alpha": 1.0,
            "beta": 0.0,
            "cache_strategy": "last",
            "decode_mode": "bypass",
            "window_width": 4,
        },
        "revision_policy": {
            "cli_pins_revision": False,
            "rule": (
                "All actual probe/audit/eval model commit hashes must exactly match. "
                "A mismatch stops the stage and its results must not be merged."
            ),
            "verification_command": (
                "python scripts/loopscope/prepare_qwen17_phase1.py verify-revisions "
                "--run-root %s --stage <gate-c|gate-d-limit|gate-e-probe|gate-e-full>"
                % run_root
            ),
        },
        "stages": {
            "gate-c": {
                "requires_gate": "A",
                "purpose": "four-sample instrumentation and real loop-effect audit",
                "scientific_selection_allowed": False,
                "jobs": jobs["gate-c"],
            },
            "gate-d-limit": {
                "requires_gate": "C",
                "purpose": "all-candidate limit=5 engineering smoke",
                "limit": 5,
                "scientific_selection_allowed": False,
                "jobs": jobs["gate-d-limit"],
            },
            "gate-e-probe": {
                "requires_gate": "D",
                "purpose": "full-pool layer probe and all frozen candidate window probes",
                "scientific_selection_allowed": False,
                "jobs": jobs["gate-e-probe"],
            },
            "gate-e-score": {
                "requires_gate": "D",
                "requires_stage": "gate-e-probe",
                "purpose": "offline label-free scoring with frozen criterion",
                "scientific_selection_allowed": False,
                "jobs": jobs["gate-e-score"],
            },
            "gate-e-full": {
                "requires_gate": "D",
                "requires_stage": "gate-e-score",
                "purpose": "full frozen MMLU grid after score freeze and Gate D PASS",
                "scientific_selection_allowed": True,
                "jobs": jobs["gate-e-full"],
            },
        },
        "scheduler": {
            "partition": args.partition,
            "gres": args.gres,
            "cpus": args.cpus,
            "mem": args.mem,
            "debug_time": args.debug_time,
            "full_time": args.full_time,
            "max_concurrent": args.max_concurrent,
            "automatic_retry": False,
        },
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    run_root.mkdir(parents=True, exist_ok=False)
    directories = (
        "manifests",
        "control",
        "control/provenance",
        "control/submissions",
        "jobs",
        "slurm",
    )
    for relative in directories:
        (run_root / relative).mkdir(exist_ok=False)
    _write_new_bytes(run_root / "manifests" / "qwen17_mmlu_phase1.json", config_path.read_bytes())
    _write_new_bytes(run_root / "manifests" / "window_grid.json", grid_path.read_bytes())
    _write_new_bytes(run_root / "manifests" / "criterion_v0.json", criterion_path.read_bytes())
    _write_new_bytes(run_root / "manifests" / "probe_pool.jsonl", pool_bytes)
    _write_new_bytes(
        run_root / "manifests" / "probe_pool_manifest.json", pool_manifest_path.read_bytes()
    )
    _write_new_json(run_root / "control" / "phase1_run_manifest.json", manifest)

    for stage, stage_jobs in jobs.items():
        _write_stage_files(
            run_root=run_root,
            repo_root=repo_root,
            venv=venv,
            stage=stage,
            jobs=stage_jobs,
            git_commit=git["commit"],
            partition=args.partition,
            gres=args.gres,
            cpus=args.cpus,
            mem=args.mem,
            time_limit=(
                args.full_time if stage in ("gate-e-probe", "gate-e-full") else args.debug_time
            ),
            requires_gpu=stage != "gate-e-score",
            max_concurrent=args.max_concurrent,
        )

    print(str(run_root))
    print(str(run_root / "control" / "phase1_run_manifest.json"))
    print(manifest["manifest_sha256"])
    return 0


def _build_jobs(
    run_root: Path,
    repo_root: Path,
    venv: Path,
    config: Mapping[str, Any],
    grid: Mapping[str, Any],
    criterion_path: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    del repo_root, venv
    model = str(config["model"]["alias"])
    grid_windows = [str(item["window"]) for item in grid["windows"]]
    anchor = str(grid["anchors"]["required"])
    sentinel_windows = _sentinel_windows(grid_windows, anchor)
    pool = run_root / "manifests" / "probe_pool.jsonl"
    pool_manifest = run_root / "manifests" / "probe_pool_manifest.json"
    grid_snapshot = run_root / "manifests" / "window_grid.json"

    gate_c = [
        _job(
            "probe-layers-4",
            "gate-c",
            run_root / "gate-c" / "probe-layers-4",
            [
                "python", "-m", "tflt.cli", "probe-layers",
                "--model", model,
                "--input-jsonl", str(pool),
                "--input-manifest", str(pool_manifest),
                "--output-dir", str(run_root / "gate-c" / "probe-layers-4"),
                "--max-examples", "4",
                "--dtype", "float16",
                "--device", "cuda",
                "--choice-labels", "A,B,C,D",
                "--erank-max-vectors", "1024",
            ],
            "probe_report.json",
            ["model", "revision"],
        ),
        _job(
            "audit-anchor-12-15",
            "gate-c",
            run_root / "gate-c" / "audit-anchor-12-15",
            [
                "python", "-m", "tflt.cli", "audit-loop-effect",
                "--model", model,
                "--output-dir", str(run_root / "gate-c" / "audit-anchor-12-15"),
                "--window", anchor,
                "--k", "2",
                "--iteration-mode", "block",
                "--strategy", "damped_euler",
                "--alpha", "1.0",
                "--beta", "0.0",
                "--cache-strategy", "last",
                "--decode-mode", "bypass",
                "--dtype", "float16",
                "--device", "cuda",
            ],
            "audit_report.json",
            ["model", "commit_hash"],
        ),
        _job(
            "probe-sentinels",
            "gate-c",
            run_root / "gate-c" / "probe-sentinels",
            [
                "python", "-m", "tflt.cli", "probe-window",
                "--model", model,
                "--input-jsonl", str(pool),
                "--input-manifest", str(pool_manifest),
                "--window-grid", str(grid_snapshot),
                "--output-dir", str(run_root / "gate-c" / "probe-sentinels"),
                "--max-examples", "4",
                "--dtype", "float16",
                "--device", "cuda",
            ] + [value for window in sentinel_windows for value in ("--window", window)],
            "probe_report.json",
            ["model", "revision"],
        ),
    ]

    gate_d = [
        _eval_job("baseline-limit5", "gate-d-limit", run_root, None, limit=5)
    ]
    gate_d.extend(
        _eval_job(
            "window-%s-limit5" % window.replace(":", "-"),
            "gate-d-limit",
            run_root,
            window,
            limit=5,
        )
        for window in grid_windows
    )

    full_layer_output = run_root / "gate-e-probe" / "probe-layers-full"
    full_window_output = run_root / "gate-e-probe" / "probe-windows-full"
    gate_e_probe = [
        _job(
            "probe-layers-full",
            "gate-e-probe",
            full_layer_output,
            [
                "python", "-m", "tflt.cli", "probe-layers",
                "--model", model,
                "--input-jsonl", str(pool),
                "--input-manifest", str(pool_manifest),
                "--output-dir", str(full_layer_output),
                "--dtype", "float16",
                "--device", "cuda",
                "--choice-labels", "A,B,C,D",
                "--erank-max-vectors", "1024",
            ],
            "probe_report.json",
            ["model", "revision"],
        ),
        _job(
            "probe-windows-full",
            "gate-e-probe",
            full_window_output,
            [
                "python", "-m", "tflt.cli", "probe-window",
                "--model", model,
                "--input-jsonl", str(pool),
                "--input-manifest", str(pool_manifest),
                "--window-grid", str(grid_snapshot),
                "--output-dir", str(full_window_output),
                "--dtype", "float16",
                "--device", "cuda",
            ] + [value for window in grid_windows for value in ("--window", window)],
            "probe_report.json",
            ["model", "revision"],
        ),
    ]

    score_output = run_root / "gate-e-score" / "score-windows"
    gate_e_score = [
        _job(
            "score-windows",
            "gate-e-score",
            score_output,
            [
                "python", "-m", "tflt.cli", "score-windows",
                "--layer-probe", str(full_layer_output / "probe_report.json"),
                "--window-probe", str(full_window_output / "probe_report.json"),
                "--output-dir", str(score_output),
                "--criterion", str(criterion_path),
                "--boundary-stat", "mean",
                "--window-stat", "median",
            ],
        )
    ]

    comparison = [str(value) for value in grid["comparison_windows"]["random_in_band"]]
    full_windows = _deduplicate(grid_windows + comparison)
    gate_e = [_eval_job("baseline-full", "gate-e-full", run_root, None, limit=None)]
    gate_e.extend(
        _eval_job(
            "window-%s-full" % window.replace(":", "-"),
            "gate-e-full",
            run_root,
            window,
            limit=None,
        )
        for window in full_windows
    )
    return {
        "gate-c": gate_c,
        "gate-d-limit": gate_d,
        "gate-e-probe": gate_e_probe,
        "gate-e-score": gate_e_score,
        "gate-e-full": gate_e,
    }


def _job(
    job_id: str,
    stage: str,
    output_dir: Path,
    argv: List[str],
    revision_artifact: Optional[str] = None,
    revision_key: Optional[List[str]] = None,
) -> Dict[str, Any]:
    job = {
        "job_id": job_id,
        "stage": stage,
        "output_dir": str(output_dir),
        "claim_dir": str(output_dir) + ".claim",
        "argv": argv,
        "command": shlex.join(argv),
        "automatic_retry": False,
    }
    if revision_artifact is not None:
        job["revision_artifact"] = str(output_dir / revision_artifact)
        job["revision_key"] = list(revision_key or ())
    return job


def _eval_job(
    job_id: str,
    stage: str,
    run_root: Path,
    window: Optional[str],
    limit: Optional[int],
) -> Dict[str, Any]:
    output_dir = run_root / stage / job_id
    argv = [
        "python", "-m", "tflt.eval_runner",
        "--model", "qwen3-1.7b-base",
        "--tasks", "mmlu",
        "--output-dir", str(output_dir),
        "--num-fewshot", "5",
        "--batch-size", "auto",
        "--dtype", "float16",
    ]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    if window is not None:
        argv.extend(
            [
                "--loop",
                "--window", window,
                "--k", "2",
                "--iteration-mode", "block",
                "--strategy", "damped_euler",
                "--alpha", "1.0",
                "--beta", "0.0",
                "--cache-strategy", "last",
                "--decode-mode", "bypass",
            ]
        )
    return _job(
        job_id,
        stage,
        output_dir,
        argv,
        "model_revision.json",
        ["commit_hash"],
    )


def _write_stage_files(
    run_root: Path,
    repo_root: Path,
    venv: Path,
    stage: str,
    jobs: Sequence[Mapping[str, Any]],
    git_commit: str,
    partition: str,
    gres: str,
    cpus: int,
    mem: str,
    time_limit: str,
    max_concurrent: int,
    requires_gpu: bool,
) -> None:
    stage_dir = run_root / "jobs" / stage
    stage_dir.mkdir(exist_ok=False)
    task_paths = []
    for job in jobs:
        task_path = stage_dir / (str(job["job_id"]) + ".sh")
        _write_new_text(
            task_path,
            _task_script(repo_root, venv, git_commit, job),
            executable=True,
        )
        task_paths.append(str(task_path))
    _write_new_text(stage_dir / "task_files.txt", "\n".join(task_paths) + "\n")
    _write_new_json(stage_dir / "jobs.json", {"stage": stage, "jobs": list(jobs)})
    _write_new_text(
        stage_dir / "runner.sbatch",
        _array_runner(
            run_root,
            stage,
            len(jobs),
            partition,
            gres,
            cpus,
            mem,
            time_limit,
            max_concurrent,
            requires_gpu,
        ),
        executable=True,
    )


def _task_script(
    repo_root: Path,
    venv: Path,
    git_commit: str,
    job: Mapping[str, Any],
) -> str:
    output = shlex.quote(str(job["output_dir"]))
    claim = shlex.quote(str(job["claim_dir"]))
    repo = shlex.quote(str(repo_root))
    venv_quoted = shlex.quote(str(venv))
    command = shlex.join([str(value) for value in job["argv"]])
    return """#!/usr/bin/env bash
set -euo pipefail

repo_root={repo}
output_dir={output}
claim_dir={claim}
venv={venv}

[[ "$(git -C "$repo_root" rev-parse --show-toplevel)" == "$repo_root" ]]
[[ "$(git -C "$repo_root" rev-parse --abbrev-ref HEAD)" == "loopscope" ]]
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "{git_commit}" ]]
[[ -z "$(git -C "$repo_root" status --porcelain)" ]]
if [[ -e "$output_dir" ]]; then
  echo "Refusing to reuse existing output: $output_dir" >&2
  exit 73
fi
mkdir -p "$(dirname "$output_dir")"
if ! mkdir "$claim_dir"; then
  echo "Refusing duplicate execution; output claim already exists: $claim_dir" >&2
  exit 73
fi
if [[ -e "$output_dir" ]]; then
  echo "Output appeared after claim creation; stopping: $output_dir" >&2
  exit 73
fi
if [[ ! -x "$venv/bin/python" ]]; then
  echo "Versioned LoopScope venv is missing: $venv" >&2
  exit 72
fi

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  set +e
  source /etc/profile.d/modules.sh >/dev/null 2>&1
  set -e
fi
if command -v module >/dev/null 2>&1; then
  module load anaconda3 cuda/12.4 uv
fi
. "$venv/bin/activate"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${{PYTHONPATH:+:$PYTHONPATH}}"
export HF_ENDPOINT="${{HF_ENDPOINT:-https://hf-mirror.com}}"
export HF_HOME="${{HF_HOME:-/hpc2hdd/home/xhuang225/shared/hf_home}}"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-$HF_HOME/hub}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-/hpc2hdd/home/xhuang225/shared/datasets}}"
export HF_HUB_DISABLE_XET="${{HF_HUB_DISABLE_XET:-1}}"
export UV_CACHE_DIR="${{UV_CACHE_DIR:-/hpc2hdd/home/xhuang225/shared/uv}}"

exec {command}
""".format(
        repo=repo,
        output=output,
        claim=claim,
        venv=venv_quoted,
        git_commit=git_commit,
        command=command,
    )


def _array_runner(
    run_root: Path,
    stage: str,
    count: int,
    partition: str,
    gres: str,
    cpus: int,
    mem: str,
    time_limit: str,
    max_concurrent: int,
    requires_gpu: bool,
) -> str:
    task_file = run_root / "jobs" / stage / "task_files.txt"
    gres_line = "#SBATCH --gres=%s\n" % gres if requires_gpu else ""
    return """#!/usr/bin/env bash
#SBATCH --job-name=loopscope-{stage}
#SBATCH --partition={partition}
{gres_line}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --array=0-{last}%{max_concurrent}
#SBATCH --output={run_root}/slurm/%x-%A_%a.out
#SBATCH --error={run_root}/slurm/%x-%A_%a.err

set -euo pipefail
task_list={task_file}
line_number=$((SLURM_ARRAY_TASK_ID + 1))
task_script="$(sed -n "${{line_number}}p" "$task_list")"
if [[ -z "$task_script" || ! -f "$task_script" ]]; then
  echo "Missing task script for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 74
fi
exec bash "$task_script"
""".format(
        stage=stage,
        partition=partition,
        gres_line=gres_line,
        cpus=cpus,
        mem=mem,
        time_limit=time_limit,
        last=count - 1,
        max_concurrent=max_concurrent,
        run_root=run_root,
        task_file=shlex.quote(str(task_file)),
    )


def verify_revisions(run_root: Path, stage: str) -> int:
    manifest_path = run_root / "control" / "phase1_run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise PreparationError("unsupported phase-one run manifest")
    _verify_manifest_hash(manifest, "run manifest")
    if Path(str(manifest.get("run_root"))).resolve() != run_root:
        raise PreparationError("run manifest path does not match --run-root")

    expected_jobs = _revision_jobs_for_stage(manifest, stage)

    observations = []
    missing = []
    for job in expected_jobs:
        artifact = Path(str(job["revision_artifact"]))
        if not artifact.is_file():
            missing.append(str(artifact))
            continue
        payload = _read_json(artifact)
        revision = _nested_value(payload, [str(value) for value in job["revision_key"]])
        if not isinstance(revision, str) or not revision.strip():
            raise PreparationError("missing model revision in %s" % artifact)
        observations.append(
            {
                "job_id": job["job_id"],
                "stage": job["stage"],
                "artifact": str(artifact),
                "revision": revision,
            }
        )
    if missing:
        raise PreparationError(
            "stage is incomplete; revision artifacts are missing: %s" % ", ".join(missing)
        )

    revisions = sorted({item["revision"] for item in observations})
    report = {
        "schema_version": "loopscope.model-revision-check.v1",
        "run_manifest_sha256": manifest["manifest_sha256"],
        "stage": stage,
        "reference_policy": "all actual commit hashes must exactly match",
        "expected_jobs": [_revision_job_contract(job) for job in expected_jobs],
        "observations": observations,
        "unique_revisions": revisions,
        "match": len(revisions) == 1,
    }
    report["manifest_sha256"] = manifest_sha256(report)
    validate_revision_report_payload(report, manifest, stage)
    report_path = run_root / "control" / "provenance" / (stage + "-model-revisions.json")
    _write_new_json(report_path, report)
    print(str(report_path))
    if not report["match"]:
        print(
            "model revision mismatch; stop and do not merge this stage's results",
            file=sys.stderr,
        )
        return 2
    print(revisions[0])
    return 0


def _revision_jobs_for_stage(
    manifest: Mapping[str, Any], stage: str
) -> List[Mapping[str, Any]]:
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping) or stage not in stages:
        raise PreparationError("stage is absent from run manifest: %s" % stage)
    if stage not in ("gate-c", "gate-d-limit", "gate-e-probe", "gate-e-full"):
        raise PreparationError("stage has no model-revision contract: %s" % stage)
    jobs: List[Mapping[str, Any]] = []
    if stage != "gate-c":
        jobs.append(stages["gate-c"]["jobs"][0])
    if stage in ("gate-e-probe", "gate-e-full"):
        jobs.append(stages["gate-d-limit"]["jobs"][0])
    if stage == "gate-e-full":
        jobs.extend(stages["gate-e-probe"]["jobs"])
    jobs.extend(stages[stage]["jobs"])
    for job in jobs:
        if "revision_artifact" not in job or "revision_key" not in job:
            raise PreparationError("revision stage contains a non-model job")
    return jobs


def _revision_job_contract(job: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": str(job["job_id"]),
        "stage": str(job["stage"]),
        "artifact": str(job["revision_artifact"]),
        "revision_key": [str(value) for value in job["revision_key"]],
    }


def validate_revision_report_payload(
    report: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    stage: str,
) -> str:
    if report.get("schema_version") != "loopscope.model-revision-check.v1":
        raise PreparationError("unsupported model-revision report schema")
    _verify_manifest_hash(report, "model-revision report")
    if report.get("run_manifest_sha256") != run_manifest.get("manifest_sha256"):
        raise PreparationError("model-revision report is bound to another run manifest")
    if report.get("stage") != stage:
        raise PreparationError("model-revision report stage mismatch")
    expected = [_revision_job_contract(job) for job in _revision_jobs_for_stage(run_manifest, stage)]
    if report.get("expected_jobs") != expected:
        raise PreparationError("model-revision expected jobs differ from the run manifest")
    observations = report.get("observations")
    if not isinstance(observations, list) or len(observations) != len(expected):
        raise PreparationError("model-revision observation count mismatch")
    revisions = []
    for observation, contract in zip(observations, expected):
        observed_contract = {
            "job_id": str(observation.get("job_id", "")),
            "stage": str(observation.get("stage", "")),
            "artifact": str(observation.get("artifact", "")),
            "revision_key": contract["revision_key"],
        }
        if observed_contract != contract:
            raise PreparationError("model-revision observation job/artifact mismatch")
        revision = str(observation.get("revision", "")).strip()
        if not revision:
            raise PreparationError("model-revision observation is empty")
        revisions.append(revision)
    unique = sorted(set(revisions))
    if report.get("unique_revisions") != unique:
        raise PreparationError("model-revision unique set is inconsistent")
    if report.get("match") is not (len(unique) == 1) or report.get("match") is not True:
        raise PreparationError("model revisions do not match")
    return unique[0]


def freeze_score(run_root: Path) -> int:
    manifest = _read_json(run_root / "control" / "phase1_run_manifest.json")
    _verify_manifest_hash(manifest, "run manifest")
    if Path(str(manifest.get("run_root"))).resolve() != run_root:
        raise PreparationError("run manifest path does not match --run-root")
    revision_path = run_root / "control" / "provenance" / "gate-e-probe-model-revisions.json"
    revision_report = _read_json(revision_path)
    validate_revision_report_payload(revision_report, manifest, "gate-e-probe")

    score_job = manifest["stages"]["gate-e-score"]["jobs"][0]
    score_path = Path(str(score_job["output_dir"])) / "window_scores.json"
    score = _read_json(score_path)
    _verify_manifest_hash(score, "window score report")
    criterion_path = Path(manifest["inputs"]["criterion"]["snapshot_path"])
    criterion = _read_json(criterion_path)
    if _file_sha256(criterion_path) != manifest["inputs"]["criterion"]["file_sha256"]:
        raise PreparationError("criterion snapshot file SHA256 changed")
    criterion_hash = manifest_sha256(criterion)
    expected_criterion = manifest["inputs"]["criterion"]["criterion_sha256"]
    grid = _read_json(Path(manifest["inputs"]["window_grid"]["snapshot_path"]))
    _verify_manifest_hash(grid, "window grid snapshot")
    grid_hash = manifest["inputs"]["window_grid"]["manifest_sha256"]
    if criterion_hash != expected_criterion or score.get("criterion_sha256") != criterion_hash:
        raise PreparationError("score report criterion SHA256 mismatch")
    if _nested_value(score, ["window_grid", "manifest_sha256"]) != grid_hash:
        raise PreparationError("score report window-grid SHA256 mismatch")
    expected_windows = [str(item["window"]) for item in grid["windows"]]
    if _nested_value(score, ["window_grid", "candidate_windows"]) != expected_windows:
        raise PreparationError("score report candidate windows differ from frozen grid")
    full_probe_evidence = _validate_full_pool_score_inputs(manifest, score)

    freeze = {
        "schema_version": "loopscope.score-freeze.v1",
        "run_manifest_sha256": manifest["manifest_sha256"],
        "score_job_id": score_job["job_id"],
        "score_report_path": str(score_path),
        "score_report_sha256": _file_sha256(score_path),
        "score_report_manifest_sha256": score["manifest_sha256"],
        "criterion_path": str(criterion_path),
        "criterion_file_sha256": _file_sha256(criterion_path),
        "criterion_sha256": criterion_hash,
        "window_grid_manifest_sha256": grid_hash,
        "candidate_windows": expected_windows,
        "gate_e_probe_revision_report_sha256": revision_report["manifest_sha256"],
        "full_probe_evidence": full_probe_evidence,
    }
    freeze["manifest_sha256"] = manifest_sha256(freeze)
    validate_score_freeze_payload(freeze, manifest, score)
    output = run_root / "control" / "provenance" / "gate-e-score-freeze.json"
    _write_new_json(output, freeze)
    print(str(output))
    print(freeze["manifest_sha256"])
    return 0


def validate_score_freeze_payload(
    freeze: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    score_report: Mapping[str, Any],
) -> None:
    if freeze.get("schema_version") != "loopscope.score-freeze.v1":
        raise PreparationError("unsupported score-freeze schema")
    _verify_manifest_hash(freeze, "score freeze")
    if freeze.get("run_manifest_sha256") != run_manifest.get("manifest_sha256"):
        raise PreparationError("score freeze is bound to another run manifest")
    score_job = run_manifest["stages"]["gate-e-score"]["jobs"][0]
    expected_path = str(Path(str(score_job["output_dir"])) / "window_scores.json")
    if freeze.get("score_job_id") != score_job["job_id"] or freeze.get(
        "score_report_path"
    ) != expected_path:
        raise PreparationError("score freeze job/path mismatch")
    if freeze.get("score_report_manifest_sha256") != score_report.get("manifest_sha256"):
        raise PreparationError("score freeze report manifest SHA256 mismatch")
    if freeze.get("criterion_sha256") != run_manifest["inputs"]["criterion"][
        "criterion_sha256"
    ]:
        raise PreparationError("score freeze criterion SHA256 mismatch")
    if freeze.get("window_grid_manifest_sha256") != run_manifest["inputs"]["window_grid"][
        "manifest_sha256"
    ]:
        raise PreparationError("score freeze window-grid SHA256 mismatch")
    expected_evidence = _validate_full_pool_score_inputs(run_manifest, score_report)
    if freeze.get("full_probe_evidence") != expected_evidence:
        raise PreparationError("score freeze full-probe evidence mismatch")


def _validate_full_pool_score_inputs(
    run_manifest: Mapping[str, Any], score_report: Mapping[str, Any]
) -> Dict[str, Any]:
    pool_info = run_manifest["inputs"]["probe_pool"]
    pool_path = Path(str(pool_info["snapshot_path"]))
    pool_manifest_path = Path(str(pool_info["manifest_snapshot_path"]))
    frozen_pool = _read_json(pool_manifest_path)
    _verify_manifest_hash(frozen_pool, "frozen probe-pool manifest")
    _validate_pool(pool_path, frozen_pool)

    expected_count = int(pool_info["count"])
    if int(frozen_pool["count"]) != expected_count:
        raise PreparationError("frozen probe-pool count differs from run manifest")
    sample_ids = [str(value) for value in frozen_pool["sample_ids"]]
    first_record = None
    with pool_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                first_record = json.loads(line)
                break
    if not isinstance(first_record, Mapping):
        raise PreparationError("frozen probe pool has no first record")

    selection_contract = {
        "schema_version": "loopscope.probe-pool-selection.v1",
        "source": frozen_pool["source"],
        "split": frozen_pool["split"],
        "count": expected_count,
        "seed": frozen_pool["seed"],
        "sample_ids": sample_ids,
        "records": frozen_pool["records"],
        "task_group": "mmlu",
        "num_fewshot": 5,
        "uses_target_gold_labels": False,
        "fewshot_answers_present": True,
        "renderer": dict(first_record["renderer"]),
        "render_contract_sha256": first_record["renderer"]["render_contract_sha256"],
        "rendering_records": frozen_pool["rendering_records"],
        "render_contract_subset_sha256": frozen_pool["render_contract_subset_sha256"],
        "source_manifest_sha256": frozen_pool["manifest_sha256"],
    }
    expected_selected_hash = manifest_sha256(selection_contract)
    expected_source_hash = frozen_pool["manifest_sha256"]
    expected_render_hash = frozen_pool["render_contract_subset_sha256"]

    jobs = {job["job_id"]: job for job in run_manifest["stages"]["gate-e-probe"]["jobs"]}
    expected_paths = {
        "layer": str(Path(jobs["probe-layers-full"]["output_dir"]) / "probe_report.json"),
        "window": str(Path(jobs["probe-windows-full"]["output_dir"]) / "probe_report.json"),
    }
    score_paths = _nested_value(score_report, ["inputs", "paths"])
    if score_paths.get("layer_probe") != expected_paths["layer"]:
        raise PreparationError("score layer-probe path is not the frozen full-probe job output")
    if score_paths.get("window_probes") != [expected_paths["window"]]:
        raise PreparationError("score window-probe path is not the frozen full-probe job output")

    reports = {name: _read_json(Path(path)) for name, path in expected_paths.items()}
    evidence: Dict[str, Any] = {}
    for name, report in reports.items():
        report_canonical_hash = manifest_sha256(report)
        if report.get("manifest_sha256") not in (None, report_canonical_hash):
            raise PreparationError("%s probe report canonical self-hash mismatch" % name)
        probe_pool = report.get("probe_pool")
        if not isinstance(probe_pool, Mapping):
            raise PreparationError("%s full probe report lacks probe_pool" % name)
        if int(probe_pool.get("count", -1)) != expected_count or int(
            probe_pool.get("source_manifest_count", -1)
        ) != expected_count:
            raise PreparationError("%s probe is not the complete frozen pool" % name)
        if [str(value) for value in probe_pool.get("sample_ids", [])] != sample_ids:
            raise PreparationError("%s probe sample IDs differ from frozen pool" % name)
        expected_pairs = {
            "source_manifest_sha256": expected_source_hash,
            "manifest_sha256": expected_selected_hash,
            "selected_subset_sha256": expected_selected_hash,
            "source_render_contract_subset_sha256": expected_render_hash,
            "selected_render_contract_subset_sha256": expected_render_hash,
        }
        for key, expected in expected_pairs.items():
            if probe_pool.get(key) != expected:
                raise PreparationError("%s probe_pool.%s mismatch" % (name, key))
        report_path = Path(expected_paths[name])
        evidence[name] = {
            "path": str(report_path),
            "file_sha256": _file_sha256(report_path),
            "canonical_sha256": report_canonical_hash,
            "probe_pool_manifest_sha256": expected_selected_hash,
        }

    score_pool = _nested_value(score_report, ["probe_provenance", "probe_pool"])
    if int(score_pool.get("count", -1)) != expected_count or int(
        score_pool.get("source_manifest_count", -1)
    ) != expected_count:
        raise PreparationError("window score provenance is not the complete frozen pool")
    if [str(value) for value in score_pool.get("sample_ids", [])] != sample_ids:
        raise PreparationError("window score sample IDs differ from frozen pool")
    score_expected = {
        "source_manifest_sha256": expected_source_hash,
        "selected_subset_sha256": expected_selected_hash,
        "source_render_contract_subset_sha256": expected_render_hash,
        "selected_render_contract_subset_sha256": expected_render_hash,
    }
    for key, expected in score_expected.items():
        if score_pool.get(key) != expected:
            raise PreparationError("window score probe_pool.%s mismatch" % key)
    if _nested_value(score_report, ["inputs", "layer_probe_manifest_sha256"]) != expected_selected_hash:
        raise PreparationError("score layer probe manifest SHA256 mismatch")
    if _nested_value(score_report, ["inputs", "window_probe_manifest_sha256"]) != [
        expected_selected_hash
    ]:
        raise PreparationError("score window probe manifest SHA256 mismatch")

    return {
        "count": expected_count,
        "source_manifest_sha256": expected_source_hash,
        "selected_subset_sha256": expected_selected_hash,
        "render_contract_subset_sha256": expected_render_hash,
        "sample_ids_sha256": hashlib.sha256(canonical_json_bytes(sample_ids)).hexdigest(),
        "probe_reports": evidence,
    }


def _git_provenance(repo_root: Path) -> Dict[str, Any]:
    if not repo_root.is_dir():
        raise PreparationError("repo root does not exist: %s" % repo_root)

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo_root)] + list(args), text=True
        ).strip()

    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    if root != repo_root:
        raise PreparationError("Git root mismatch: expected %s, got %s" % (repo_root, root))
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "loopscope":
        raise PreparationError("LoopScope preparation requires branch loopscope, got %s" % branch)
    commit = git("rev-parse", "HEAD")
    if subprocess.call(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", BASE_COMMIT, commit]
    ) != 0:
        raise PreparationError("current commit is not descended from the frozen TFLT base")
    status = git("status", "--porcelain")
    if status:
        raise PreparationError("LoopScope checkout must be clean before run preparation")
    origin = git("remote", "get-url", "origin")
    if origin != EXPECTED_ORIGIN:
        raise PreparationError(
            "unexpected origin: expected %s, got %s" % (EXPECTED_ORIGIN, origin)
        )
    return {
        "root": str(root),
        "branch": branch,
        "commit": commit,
        "dirty": False,
        "base_commit": BASE_COMMIT,
        "origin": origin,
    }


def _validate_phase_config(config: Mapping[str, Any]) -> None:
    expected = {
        ("model", "alias"): "qwen3-1.7b-base",
        ("model", "repo_id"): "Qwen/Qwen3-1.7B-Base",
        ("evaluation", "task"): "mmlu",
        ("evaluation", "num_fewshot"): 5,
        ("evaluation", "dtype"): "float16",
        ("loop", "k"): 2,
        ("loop", "iteration_mode"): "block",
        ("loop", "strategy"): "damped_euler",
        ("loop", "alpha"): 1.0,
        ("loop", "beta"): 0.0,
        ("loop", "cache_strategy"): "last",
        ("loop", "decode_mode"): "bypass",
        ("loop", "window_width"): 4,
        ("probe_pool", "seed"): 20260710,
        ("probe_pool", "forbid_test_split"): True,
        ("probe_pool", "uses_target_gold_labels"): False,
        ("probe_pool", "fewshot_answers_present"): True,
        ("probe_pool", "task_group"): "mmlu",
        ("probe_pool", "num_fewshot"): 5,
        ("probe_pool", "prompt_mode"): "lm_eval_prerendered",
        ("probe_pool", "lm_eval_version"): "0.4.11",
        ("probe_pool", "fewshot_split"): "dev",
        ("probe_pool", "chat_template"): False,
        ("probe_pool", "multiturn"): False,
        ("probe_pool", "require_renderer_manifest"): True,
        ("probe_pool", "fixed_fewshot_ids_per_subject"): True,
    }
    for path, value in expected.items():
        actual = _nested_value(config, list(path))
        if actual != value:
            raise PreparationError(
                "phase config changed at %s: expected %r, got %r"
                % (".".join(path), value, actual)
            )


def _validate_criterion(criterion: Mapping[str, Any]) -> None:
    if criterion.get("schema_version") != "loopscope.criterion.v0":
        raise PreparationError("phase one requires criterion schema loopscope.criterion.v0")
    primary = str(criterion.get("primary_signal", ""))
    signals = criterion.get("signals")
    if not primary or not isinstance(signals, Mapping) or primary not in signals:
        raise PreparationError("criterion primary_signal is absent from signals")
    if _nested_value(criterion, ["bootstrap", "seed"]) != 20260710:
        raise PreparationError("criterion bootstrap seed must remain 20260710")


def _validate_grid(grid: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if grid.get("schema_version") != "loopscope.window-grid.v1":
        raise PreparationError("unsupported window-grid schema")
    _verify_manifest_hash(grid, "window grid")
    if int(_nested_value(grid, ["generation", "width"])) != 4:
        raise PreparationError("phase-one window grid must have inclusive width 4")
    expected_generation = {
        "width": int(_nested_value(config, ["loop", "window_width"])),
        "min_center_fraction": float(
            _nested_value(config, ["window_grid", "min_center_fraction"])
        ),
        "max_center_fraction": float(
            _nested_value(config, ["window_grid", "max_center_fraction"])
        ),
        "candidate_count_requested": int(
            _nested_value(config, ["window_grid", "candidate_count"])
        ),
        "fixed_depth_fraction": float(
            _nested_value(config, ["window_grid", "fixed_depth_fraction"])
        ),
        "random_count": int(_nested_value(config, ["window_grid", "random_in_band_count"])),
        "random_seed": int(_nested_value(config, ["window_grid", "random_seed"])),
    }
    for key, expected in expected_generation.items():
        actual = _nested_value(grid, ["generation", key])
        if actual != expected:
            raise PreparationError(
                "window-grid generation.%s differs from frozen config: %r != %r"
                % (key, actual, expected)
            )
    windows = grid.get("windows")
    if not isinstance(windows, list) or not windows:
        raise PreparationError("window grid has no candidates")
    seen = set()
    layer_count = int(grid.get("layer_count", 0))
    for item in windows:
        window = str(item.get("window", ""))
        start, end = _parse_window(window)
        if end - start + 1 != 4 or start < 0 or end >= layer_count:
            raise PreparationError("invalid candidate window: %s" % window)
        if window in seen:
            raise PreparationError("duplicate candidate window: %s" % window)
        seen.add(window)
    anchor = str(_nested_value(grid, ["anchors", "required"]))
    expected_anchor = str(_nested_value(config, ["window_grid", "required_anchor"]))
    if anchor != expected_anchor or anchor != "12:15" or anchor not in seen:
        raise PreparationError("frozen anchor 12:15 is missing from candidate grid")
    fixed = str(_nested_value(grid, ["anchors", "fixed_depth"]))
    if fixed not in seen:
        raise PreparationError("fixed-depth window is missing from candidate grid")
    random_windows = list(_nested_value(grid, ["comparison_windows", "random_in_band"]))
    if len(random_windows) != expected_generation["random_count"]:
        raise PreparationError("random-in-band comparison count differs from frozen config")
    if len(set(str(value) for value in random_windows)) != len(random_windows):
        raise PreparationError("random-in-band comparison windows contain duplicates")
    for value in random_windows:
        start, end = _parse_window(str(value))
        if end - start + 1 != 4 or start < 0 or end >= layer_count:
            raise PreparationError("invalid random-in-band comparison window: %s" % value)


def _validate_pool(pool_path: Path, manifest: Mapping[str, Any]) -> bytes:
    if not pool_path.is_file():
        raise PreparationError("probe pool does not exist: %s" % pool_path)
    if manifest.get("schema_version") != "loopscope.probe-pool-manifest.v1":
        raise PreparationError("unsupported probe-pool manifest")
    _verify_manifest_hash(manifest, "probe-pool manifest")
    if manifest.get("uses_target_gold_labels") is not False:
        raise PreparationError("probe-pool manifest must exclude target gold labels")
    if manifest.get("fewshot_answers_present") is not True:
        raise PreparationError("five-shot prompts must retain demonstration answers")
    if manifest.get("task_group") != "mmlu" or manifest.get("num_fewshot") != 5:
        raise PreparationError("probe-pool manifest must freeze MMLU five-shot rendering")
    if "test" in str(manifest.get("split", "")).lower():
        raise PreparationError("test split is forbidden for LoopScope probes")
    if int(manifest.get("seed", -1)) != 20260710:
        raise PreparationError("probe pool must use frozen seed 20260710")
    if not str(manifest.get("source", "")).strip() or not str(manifest.get("split", "")).strip():
        raise PreparationError("probe-pool manifest source/split must be non-empty")
    if int(manifest.get("count", 0)) < 4:
        raise PreparationError("probe pool must contain at least four records for Gate C")
    renderer = manifest.get("renderer")
    if not isinstance(renderer, Mapping):
        raise PreparationError("probe-pool manifest is missing renderer provenance")
    if renderer.get("lm_eval_version") != "0.4.11":
        raise PreparationError("probe pool must use lm-eval 0.4.11")
    for key in (
        "renderer_source_sha256",
        "template_sha256",
        "render_contract_sha256",
    ):
        _require_sha256(renderer.get(key), "probe-pool renderer.%s" % key)
    renderer_manifest_hash = _nested_value(
        manifest, ["renderer_manifest", "manifest_sha256"]
    )
    _require_sha256(renderer_manifest_hash, "renderer manifest SHA256")
    expected_subset_hash = _require_sha256(
        manifest.get("render_contract_subset_sha256"),
        "render contract subset SHA256",
    )

    pool_bytes = pool_path.read_bytes()
    expected_hash = _nested_value(manifest, ["output", "pool_sha256"])
    actual_hash = hashlib.sha256(pool_bytes).hexdigest()
    if expected_hash != actual_hash:
        raise PreparationError("probe pool SHA256 does not match its manifest")
    ids = []
    record_entries = []
    rendering_entries = []
    seen_ids = set()
    fixed_demos: Dict[str, Tuple[Tuple[str, ...], ...]] = {}
    manifest_source = str(manifest.get("source", ""))
    manifest_split = str(manifest.get("split", ""))
    try:
        pool_text = pool_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreparationError("probe pool must be UTF-8 JSONL") from exc
    for line_number, line in enumerate(pool_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreparationError("invalid probe-pool JSON at line %d" % line_number) from exc
        if not isinstance(item, Mapping):
            raise PreparationError("probe-pool line %d is not an object" % line_number)
        required = (
            "id",
            "text",
            "source",
            "split",
            "subject",
            "prompt_sha256",
            "task_group",
            "task_name",
            "target_doc_id",
            "num_fewshot",
            "renderer",
            "fewshot_sample_ids",
            "demonstrations",
        )
        if any(item.get(key) in (None, "") for key in required):
            raise PreparationError("probe-pool line %d is missing required fields" % line_number)
        if "test" in str(item.get("split", "")).lower():
            raise PreparationError("probe-pool line %d uses test split" % line_number)
        leaked = _forbidden_gold_paths(item)
        if leaked:
            raise PreparationError(
                "probe-pool line %d contains gold fields: %s"
                % (line_number, ", ".join(leaked))
            )
        if str(item["source"]) != manifest_source or str(item["split"]) != manifest_split:
            raise PreparationError(
                "probe-pool line %d source/split differs from manifest" % line_number
            )
        if item.get("task_group") != "mmlu" or item.get("num_fewshot") != 5:
            raise PreparationError("probe-pool line %d is not MMLU five-shot" % line_number)
        if item.get("uses_target_gold_labels") is not False:
            raise PreparationError("probe-pool line %d may contain target gold" % line_number)
        if item.get("fewshot_answers_present") is not True:
            raise PreparationError("probe-pool line %d lacks fewshot answers" % line_number)
        record_id = str(item["id"])
        if record_id in seen_ids:
            raise PreparationError("duplicate probe-pool id: %s" % record_id)
        seen_ids.add(record_id)
        prompt_hash = hashlib.sha256(str(item["text"]).encode("utf-8")).hexdigest()
        if str(item["prompt_sha256"]) != prompt_hash:
            raise PreparationError("probe-pool prompt hash mismatch for %s" % record_id)
        item_renderer = item.get("renderer")
        if not isinstance(item_renderer, Mapping):
            raise PreparationError("probe-pool renderer metadata is malformed")
        expected_renderer = {
            "renderer_entrypoint": renderer["renderer_entrypoint"],
            "lm_eval_version": "0.4.11",
            "renderer_source_sha256": renderer["renderer_source_sha256"],
            "template_sha256": renderer["template_sha256"],
            "render_contract_sha256": renderer["render_contract_sha256"],
            "render_sha256": prompt_hash,
            "renderer_manifest_sha256": renderer_manifest_hash,
        }
        if dict(item_renderer) != expected_renderer:
            raise PreparationError("probe-pool renderer contract mismatch for %s" % record_id)
        demos = item.get("demonstrations")
        if not isinstance(demos, list) or len(demos) != 5:
            raise PreparationError("probe-pool record %s needs five demonstrations" % record_id)
        normalized_demos = []
        demo_ids = []
        for demo in demos:
            if not isinstance(demo, Mapping):
                raise PreparationError("probe-pool demonstration must be an object")
            demo_required = (
                "id", "source", "split", "subject", "doc_sha256", "rendered_sha256"
            )
            if any(not str(demo.get(key, "")).strip() for key in demo_required):
                raise PreparationError("probe-pool demonstration provenance is incomplete")
            if demo["split"] != "dev" or str(demo["subject"]) != str(item["subject"]):
                raise PreparationError("probe-pool demonstration split/subject mismatch")
            _require_sha256(demo["doc_sha256"], "demonstration doc SHA256")
            _require_sha256(demo["rendered_sha256"], "demonstration render SHA256")
            normalized = {key: str(demo[key]) for key in demo_required}
            normalized_demos.append(normalized)
            demo_ids.append(normalized["id"])
        if len(set(demo_ids)) != 5 or demo_ids != list(item["fewshot_sample_ids"]):
            raise PreparationError("probe-pool five-shot IDs are not ordered and unique")
        if str(item["target_doc_id"]) in demo_ids:
            raise PreparationError("target doc appears in its demonstrations")
        demo_contract = tuple(tuple(demo[key] for key in sorted(demo)) for demo in normalized_demos)
        previous = fixed_demos.setdefault(str(item["subject"]), demo_contract)
        if previous != demo_contract:
            raise PreparationError("five-shot provenance changes within one subject")
        ids.append(record_id)
        record_entries.append({"id": record_id, "prompt_sha256": prompt_hash})
        rendering_entries.append(
            {
                "id": record_id,
                "target": {
                    "source": str(item["source"]),
                    "split": str(item["split"]),
                    "subject": str(item["subject"]),
                },
                "task_group": "mmlu",
                "task_name": str(item["task_name"]),
                "target_doc_id": str(item["target_doc_id"]),
                "num_fewshot": 5,
                "uses_target_gold_labels": False,
                "fewshot_answers_present": True,
                "renderer": dict(item_renderer),
                "fewshot_sample_ids": demo_ids,
                "demonstrations": normalized_demos,
                "prompt_sha256": prompt_hash,
            }
        )
    if ids != [str(value) for value in manifest.get("sample_ids", [])]:
        raise PreparationError("probe-pool record order does not match manifest sample_ids")
    if len(ids) != int(manifest.get("count", -1)):
        raise PreparationError("probe-pool count does not match manifest")
    if record_entries != manifest.get("records"):
        raise PreparationError("probe-pool records do not match manifest id/hash entries")
    if rendering_entries != manifest.get("rendering_records"):
        raise PreparationError("probe-pool rendering provenance differs from manifest")
    actual_subset_hash = hashlib.sha256(canonical_json_bytes(rendering_entries)).hexdigest()
    if actual_subset_hash != expected_subset_hash:
        raise PreparationError("render contract subset SHA256 mismatch")
    return pool_bytes


def _requested_run_root(args: argparse.Namespace) -> Path:
    run_base = Path(args.run_base).expanduser().resolve()
    if args.run_root:
        root = Path(args.run_root).expanduser().resolve()
    else:
        timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        _validate_timestamp(timestamp)
        root = run_base / (RUN_PREFIX + timestamp)
    if root.parent != run_base:
        raise PreparationError("run root must be a direct child of --run-base")
    name = root.name
    if not name.startswith(RUN_PREFIX):
        raise PreparationError("run root must use prefix %s" % RUN_PREFIX)
    _validate_timestamp(name[len(RUN_PREFIX) :])
    return root


def _validate_hpc2_paths(
    repo_root: Path,
    run_base: Path,
    run_root: Path,
    venv: Path,
) -> None:
    if repo_root != DEFAULT_REMOTE_REPO:
        raise PreparationError(
            "repo root must be the exact dedicated HPC2 LoopScope clone: %s"
            % DEFAULT_REMOTE_REPO
        )
    if run_base != DEFAULT_RUN_BASE:
        raise PreparationError("run base must be exactly %s" % DEFAULT_RUN_BASE)
    if run_root.parent != DEFAULT_RUN_BASE:
        raise PreparationError("run root must be a direct child of the exact HPC2 run base")
    if not run_root.name.startswith(RUN_PREFIX):
        raise PreparationError("run root basename must start with %s" % RUN_PREFIX)
    _validate_timestamp(run_root.name[len(RUN_PREFIX) :])
    if venv.parent != DEFAULT_REMOTE_REPO:
        raise PreparationError("versioned venv must be a direct child of the LoopScope clone")
    if not re.fullmatch(r"\.venv-loopscope-cu121-[0-9]{8}(?:-v[0-9]+)?", venv.name):
        raise PreparationError(
            "venv basename must be versioned as .venv-loopscope-cu121-YYYYMMDD[-vN]"
        )


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.strptime(value, "%Y%m%d-%H%M%S")
    except ValueError as exc:
        raise PreparationError("timestamp must be UTC YYYYMMDD-HHMMSS") from exc
    if parsed.strftime("%Y%m%d-%H%M%S") != value:
        raise PreparationError("timestamp must be canonical UTC YYYYMMDD-HHMMSS")


def _sentinel_windows(windows: Sequence[str], anchor: str) -> List[str]:
    ordered = sorted(windows, key=lambda value: _parse_window(value))
    values = _deduplicate([ordered[0], anchor, ordered[-1]])
    if len(values) != 3:
        raise PreparationError("candidate grid cannot provide early/anchor/late sentinels")
    return values


def _parse_window(value: str) -> Tuple[int, int]:
    try:
        left, right = value.split(":", 1)
        return int(left), int(right)
    except Exception as exc:
        raise PreparationError("invalid window string: %r" % value) from exc


def _deduplicate(values: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _forbidden_gold_paths(value: Any, path: Tuple[str, ...] = ()) -> List[str]:
    forbidden = {
        "answer", "answers", "answer_idx", "answer_index", "answer_key", "answerkey",
        "correct", "correct_answer", "correct_choice", "correct_index", "gold",
        "gold_answer", "gold_label", "label", "labels", "target",
    }
    found: List[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            current = path + (str(key),)
            if normalized in forbidden:
                found.append(".".join(current))
            found.extend(_forbidden_gold_paths(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_gold_paths(item, path + (str(index),)))
    return sorted(set(found))


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if not path.is_file():
        raise PreparationError("required file does not exist: %s" % path)
    return path


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise PreparationError("JSON artifact does not exist: %s" % path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError("could not read JSON artifact: %s" % path) from exc
    if not isinstance(payload, dict):
        raise PreparationError("JSON artifact must contain an object: %s" % path)
    return payload


def _nested_value(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise PreparationError("missing field: %s" % ".".join(keys))
        value = value[key]
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def manifest_sha256(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _verify_manifest_hash(payload: Mapping[str, Any], context: str) -> None:
    expected = payload.get("manifest_sha256")
    actual = manifest_sha256(payload)
    if not isinstance(expected, str) or expected != actual:
        raise PreparationError("%s SHA256 mismatch" % context)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PreparationError("%s must be a lowercase SHA256" % context)
    return digest


def _write_new_text(path: Path, text: str, executable: bool = False) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def _write_new_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)


def _write_new_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    _write_new_text(path, text + "\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PreparationError, FileExistsError, subprocess.CalledProcessError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
