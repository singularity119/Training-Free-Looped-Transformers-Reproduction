#!/usr/bin/env python3
"""Narrow CLI for LoopScope Phase 5 Gate B only."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def _install_isolated_tflt_namespace() -> None:
    """Avoid side-effect imports from the package initializer in GPU producers."""

    if "tflt" in sys.modules:
        raise RuntimeError("Gate B launcher requires a fresh Python process")
    root = Path(__file__).resolve().parents[2]
    package = types.ModuleType("tflt")
    package.__path__ = [str(root / "src/tflt")]
    package.__package__ = "tflt"
    sys.modules["tflt"] = package


_install_isolated_tflt_namespace()

from tflt.loopscope.phase5_acquisition import (  # noqa: E402
    AUDITED_VENV,
    AUTHORIZED_RUN_ROOT,
    REMOTE_REPO,
    acquire_membership,
    attach_manifest_sha256,
    freeze_formal,
    freeze_smoke,
    file_sha256,
    git_provenance,
    implementation_hashes,
    load_frozen_card,
    load_strict_json,
    load_strict_jsonl,
    materialize_admission,
    merge_formal,
    verify_trajectory_population,
    verify_manifest_sha256,
    write_new_json,
    write_new_text,
)
from tflt.loopscope.phase5_schema import load_json, validate_registry  # noqa: E402


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-commit", required=True)


def _launcher_text(
    *, mode: str, expected_commit: str, shard_count: Optional[int], concurrency: Optional[int]
) -> str:
    script = REMOTE_REPO / "scripts/loopscope/run_qwen4base_phase5_gate_b.py"
    common = [
        "#!/bin/bash",
        "#SBATCH --job-name=p5gb-%s" % mode,
        "#SBATCH --partition=gpu3",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --no-requeue",
        "#SBATCH --output=%s/slurm/%s-%%A_%%a.out" % (AUTHORIZED_RUN_ROOT, mode),
        "#SBATCH --error=%s/slurm/%s-%%A_%%a.err" % (AUTHORIZED_RUN_ROOT, mode),
    ]
    if mode == "smoke":
        common.append("#SBATCH --time=02:00:00")
    else:
        if shard_count is None or concurrency is None:
            raise ValueError("formal launcher requires shard_count/concurrency")
        common.extend(
            [
                "#SBATCH --time=08:00:00",
                "#SBATCH --array=0-%d%%%d" % (shard_count - 1, concurrency),
            ]
        )
    common.extend(
        [
            "set -euo pipefail",
            "module purge",
            "module load anaconda3 cuda/12.4",
            "source %s/bin/activate" % AUDITED_VENV,
            "export PYTHONDONTWRITEBYTECODE=1",
            "export PYTHONPATH=%s/src" % REMOTE_REPO,
            "export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home",
            "export TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub",
            "export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets",
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "unset HF_ENDPOINT",
            "cd %s" % REMOTE_REPO,
        ]
    )
    if mode == "smoke":
        command = (
            "python %s acquire-smoke --expected-commit %s --attempt 1"
            % (script, expected_commit)
        )
    else:
        command = (
            "python %s acquire-shard --expected-commit %s "
            "--shard-id ${SLURM_ARRAY_TASK_ID} --attempt 1"
            % (script, expected_commit)
        )
    common.append(command)
    return "\n".join(common) + "\n"


def write_launcher(
    *, mode: str, expected_commit: str, shard_count: Optional[int], concurrency: Optional[int]
) -> Dict[str, Any]:
    path = AUTHORIZED_RUN_ROOT / ("slurm/%s.sbatch" % mode)
    text = _launcher_text(
        mode=mode,
        expected_commit=expected_commit,
        shard_count=shard_count,
        concurrency=concurrency,
    )
    digest = write_new_text(path, text, executable=True)
    receipt = attach_manifest_sha256(
        {
            "schema_version": "loopscope.phase5.gate-b-launcher.v1",
            "gate": "B",
            "mode": mode,
            "launcher": str(path.relative_to(AUTHORIZED_RUN_ROOT)),
            "launcher_sha256": digest,
            "expected_commit": expected_commit,
            "shard_count": shard_count,
            "array_concurrency": concurrency,
        }
    )
    write_new_json(AUTHORIZED_RUN_ROOT / ("slurm/%s_launcher_receipt.json" % mode), receipt)
    return receipt


def record_submission(*, mode: str, job_id: str, launcher_sha256: str) -> Dict[str, Any]:
    if not job_id.isdigit():
        raise ValueError("Slurm job id must be numeric")
    receipt = attach_manifest_sha256(
        {
            "schema_version": "loopscope.phase5.gate-b-submission.v1",
            "gate": "B",
            "mode": mode,
            "job_id": job_id,
            "launcher_sha256": launcher_sha256,
            "terminal_state": None,
        }
    )
    write_new_json(AUTHORIZED_RUN_ROOT / ("slurm/%s_submission_receipt.json" % mode), receipt)
    return receipt


def record_terminal(
    *,
    mode: str,
    job_id: str,
    state: str,
    exit_code: str,
    gpu_seconds: float,
    task_count: int,
    node_list: str,
) -> Dict[str, Any]:
    if not job_id.isdigit() or state != "COMPLETED" or exit_code != "0:0":
        raise ValueError("only exact successful terminal Slurm evidence may be sealed")
    if gpu_seconds < 0.0 or task_count < 1:
        raise ValueError("terminal resource accounting values are invalid")
    receipt = attach_manifest_sha256(
        {
            "schema_version": "loopscope.phase5.gate-b-slurm-terminal.v1",
            "gate": "B",
            "mode": mode,
            "job_id": job_id,
            "state": state,
            "exit_code": exit_code,
            "gpu_seconds": gpu_seconds,
            "task_count": task_count,
            "node_list": node_list,
        }
    )
    write_new_json(AUTHORIZED_RUN_ROOT / ("slurm/%s_terminal_receipt.json" % mode), receipt)
    return receipt


def record_monitor_lifecycle(
    *, automation_id: str, cadence_minutes: int, status: str
) -> Dict[str, Any]:
    if status not in ("NOT_REQUIRED_JOB_TERMINATED_BEFORE_IDLE", "PAUSED_AFTER_TERMINAL"):
        raise ValueError("monitor lifecycle status differs")
    if cadence_minutes not in (10, 30, 60):
        raise ValueError("monitor cadence differs")
    receipt = attach_manifest_sha256(
        {
            "schema_version": "loopscope.phase5.gate-b-monitor-lifecycle.v1",
            "gate": "B",
            "executor_thread_id": "019f864e-9b57-73e1-b56c-e7e40e7ad28d",
            "planning_thread_id": "019f8604-4717-7be2-8bf8-9d4a26a3d7f7",
            "automation_id": automation_id,
            "cadence_minutes": cadence_minutes,
            "status": status,
            "authority": "bounded_read_only_status_checks_only",
        }
    )
    write_new_json(AUTHORIZED_RUN_ROOT / "slurm/monitor_lifecycle_receipt.json", receipt)
    return receipt


def _formal_membership_rows(manifest: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for shard in manifest["shards"]:
        rows.extend(dict(value) for value in shard["records"])
    rows.sort(key=lambda row: row["ordinal"])
    if [row["ordinal"] for row in rows] != list(range(1531)):
        raise ValueError("formal membership does not close to canonical 1531 order")
    return rows


def freeze_selector_files(*, expected_commit: str, argv: Sequence[str]) -> Dict[str, Any]:
    from tflt.loopscope.phase5_selector import build_selector_artifacts, selector_ranking_csv

    git = git_provenance(expected_commit)
    card = load_frozen_card()
    registry_path = REMOTE_REPO / "configs/loopscope/phase5_known_outcome_registry.json"
    registry = load_json(registry_path)
    validate_registry(registry)
    trajectory_manifest = load_strict_json(
        AUTHORIZED_RUN_ROOT / "formal/validation1531_phase5_trajectory_manifest.json"
    )
    verify_manifest_sha256(trajectory_manifest)
    trajectory_path = AUTHORIZED_RUN_ROOT / trajectory_manifest["trajectory_file"]
    if file_sha256(trajectory_path) != trajectory_manifest["trajectory_file_sha256"]:
        raise ValueError("trajectory file hash differs before selector freeze")
    records = load_strict_jsonl(trajectory_path)
    membership = load_strict_json(AUTHORIZED_RUN_ROOT / "formal/formal_shard_manifest.json")
    verify_manifest_sha256(membership)
    bindings = {
        "git_commit": expected_commit,
        "card_file_sha256": file_sha256(REMOTE_REPO / "configs/loopscope/phase5_card.json"),
        "card_manifest_sha256": card["manifest_sha256"],
        "known_registry_file_sha256": file_sha256(registry_path),
        "known_registry_manifest_sha256": registry["manifest_sha256"],
        "trajectory_file_sha256": trajectory_manifest["trajectory_file_sha256"],
        "trajectory_manifest_sha256": trajectory_manifest["manifest_sha256"],
        "formal_membership_manifest_sha256": membership["manifest_sha256"],
        "implementation_sha256": implementation_hashes(),
    }
    artifacts = build_selector_artifacts(
        records,
        card=card,
        registry=registry,
        expected_membership=_formal_membership_rows(membership),
        bindings=bindings,
        formal=True,
    )
    selector_dir = AUTHORIZED_RUN_ROOT / "selector"
    report_sha = write_new_json(selector_dir / "selector_report.json", artifacts["selector_report"])
    freeze_sha = write_new_json(selector_dir / "selector_freeze.json", artifacts["selector_freeze"])
    panel_sha = write_new_json(selector_dir / "outcome_panel.json", artifacts["outcome_panel"])
    ranking_sha = write_new_text(
        selector_dir / "selector_ranking.csv", selector_ranking_csv(artifacts["selector_report"])
    )
    report = artifacts["selector_report"]
    panel = artifacts["outcome_panel"]
    top_rows = report["published_ranking"][:6]
    summary = "\n".join(
        [
            "# LoopScope 第五阶段 Gate B selector 冻结摘要",
            "",
            "- 决策：`%s`" % report["window_decision"],
            "- selected window：`%s`" % (report["selected_window"] or "NONE"),
            "- point top-1：`%s`" % (report["point_top1_window"] or "NONE"),
            "- selection frequency：`%s`" % report["selection_frequency"],
            "- eligible 数量：`%s`" % report["eligible_count"],
            "- raw High3：`%s`" % ", ".join(panel["raw_high3"]),
            "- panel High3：`%s`" % ", ".join(panel["panel_high3"]),
            "- blind Low3：`%s`" % ", ".join(panel["blind_low3"]),
            "- 固定比较窗：`15:18`；no-loop 也已进入后续冻结面板。",
            "",
            "## 排名前六",
            "",
            *[
                "%d. `%s` score=`%.12g`, eligible=`%s`"
                % (index + 1, row["window"], row["score"], row["point_eligible"])
                for index, row in enumerate(top_rows)
            ],
            "",
            "本摘要只来自 validation trajectory 标量；未读取 test、标签、正确性或任何禁止的后续字段。",
            "",
        ]
    )
    summary_sha = write_new_text(selector_dir / "selector_summary_zh.md", summary)
    return {
        "selector_report_manifest_sha256": report["manifest_sha256"],
        "selector_freeze_manifest_sha256": artifacts["selector_freeze"]["manifest_sha256"],
        "outcome_panel_manifest_sha256": panel["manifest_sha256"],
        "file_sha256": {
            "selector_report.json": report_sha,
            "selector_freeze.json": freeze_sha,
            "outcome_panel.json": panel_sha,
            "selector_ranking.csv": ranking_sha,
            "selector_summary_zh.md": summary_sha,
        },
        "window_decision": report["window_decision"],
        "selected_window": report["selected_window"],
        "argv": list(argv),
    }


def verify_selector_files(
    *, expected_commit: str, argv: Sequence[str], trajectory_verifier_receipt: Mapping[str, Any]
) -> Dict[str, Any]:
    from tflt.loopscope.phase5_selector import verify_selector_artifacts

    git = git_provenance(expected_commit)
    card = load_frozen_card()
    registry_path = REMOTE_REPO / "configs/loopscope/phase5_known_outcome_registry.json"
    registry = load_json(registry_path)
    validate_registry(registry)
    trajectory_manifest = load_strict_json(
        AUTHORIZED_RUN_ROOT / "formal/validation1531_phase5_trajectory_manifest.json"
    )
    records = load_strict_jsonl(AUTHORIZED_RUN_ROOT / trajectory_manifest["trajectory_file"])
    membership = load_strict_json(AUTHORIZED_RUN_ROOT / "formal/formal_shard_manifest.json")
    selector_report = load_strict_json(AUTHORIZED_RUN_ROOT / "selector/selector_report.json")
    artifacts = {
        "selector_report": selector_report,
        "outcome_panel": load_strict_json(AUTHORIZED_RUN_ROOT / "selector/outcome_panel.json"),
        "selector_freeze": load_strict_json(AUTHORIZED_RUN_ROOT / "selector/selector_freeze.json"),
    }
    recomputed = verify_selector_artifacts(
        records,
        card=card,
        registry=registry,
        expected_membership=_formal_membership_rows(membership),
        bindings=selector_report["bindings"],
        artifacts=artifacts,
        formal=True,
    )
    selector_receipt = attach_manifest_sha256(
        dict(recomputed, gate="B", created_from_argv=list(argv), git=git)
    )
    write_new_json(
        AUTHORIZED_RUN_ROOT / "verification/selector_verifier_receipt.json", selector_receipt
    )
    smoke_terminal = load_strict_json(AUTHORIZED_RUN_ROOT / "slurm/smoke_terminal_receipt.json")
    formal_terminal = load_strict_json(AUTHORIZED_RUN_ROOT / "slurm/formal_terminal_receipt.json")
    monitor = load_strict_json(AUTHORIZED_RUN_ROOT / "slurm/monitor_lifecycle_receipt.json")
    for receipt in (smoke_terminal, formal_terminal, monitor):
        verify_manifest_sha256(receipt)
    if smoke_terminal["state"] != "COMPLETED" or formal_terminal["state"] != "COMPLETED":
        raise ValueError("Slurm jobs are not terminal successful")
    panel = artifacts["outcome_panel"]
    gate_receipt = attach_manifest_sha256(
        {
            "schema_version": "loopscope.phase5.gate-b-receipt.v1",
            "gate": "B",
            "result": "READY_FOR_PLANNING_AUDIT",
            "git": git,
            "trajectory_verifier_receipt_sha256": trajectory_verifier_receipt["manifest_sha256"],
            "selector_verifier_receipt_sha256": selector_receipt["manifest_sha256"],
            "trajectory_manifest_sha256": trajectory_manifest["manifest_sha256"],
            "selector_report_manifest_sha256": artifacts["selector_report"]["manifest_sha256"],
            "selector_freeze_manifest_sha256": artifacts["selector_freeze"]["manifest_sha256"],
            "outcome_panel_manifest_sha256": panel["manifest_sha256"],
            "window_decision": artifacts["selector_report"]["window_decision"],
            "selected_window": artifacts["selector_report"]["selected_window"],
            "raw_high3": panel["raw_high3"],
            "panel_high3": panel["panel_high3"],
            "blind_low3": panel["blind_low3"],
            "unique_cells": panel["unique_cells"],
            "record_count": 1531,
            "subject_count": 57,
            "forward_count": trajectory_manifest["forward_count"],
            "slurm": {
                "smoke_terminal_receipt_sha256": smoke_terminal["manifest_sha256"],
                "formal_terminal_receipt_sha256": formal_terminal["manifest_sha256"],
                "total_gpu_seconds": smoke_terminal["gpu_seconds"] + formal_terminal["gpu_seconds"],
            },
            "monitor_lifecycle_receipt_sha256": monitor["manifest_sha256"],
            "implementation_sha256": implementation_hashes(),
            "repair_loops": 1,
            "outcome_accessed": False,
            "test_split_accessed": False,
            "gate_c_entered": False,
            "argv": list(argv),
        }
    )
    write_new_json(AUTHORIZED_RUN_ROOT / "verification/gate_b_receipt.json", gate_receipt)
    return gate_receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    admission = sub.add_parser("admission")
    _common(admission)
    smoke = sub.add_parser("freeze-smoke")
    _common(smoke)
    acquire_smoke = sub.add_parser("acquire-smoke")
    _common(acquire_smoke)
    acquire_smoke.add_argument("--attempt", type=int, default=1)
    formal = sub.add_parser("freeze-formal")
    _common(formal)
    formal.add_argument("--shard-count", type=int, required=True)
    formal.add_argument("--smoke-receipt-sha256", required=True)
    acquire = sub.add_parser("acquire-shard")
    _common(acquire)
    acquire.add_argument("--shard-id", type=int, required=True)
    acquire.add_argument("--attempt", type=int, default=1)
    merge = sub.add_parser("merge")
    _common(merge)
    select = sub.add_parser("select")
    _common(select)
    verify = sub.add_parser("verify")
    _common(verify)
    launcher = sub.add_parser("write-launcher")
    _common(launcher)
    launcher.add_argument("--mode", choices=("smoke", "formal"), required=True)
    launcher.add_argument("--shard-count", type=int)
    launcher.add_argument("--concurrency", type=int)
    submit = sub.add_parser("record-submission")
    submit.add_argument("--mode", choices=("smoke", "formal"), required=True)
    submit.add_argument("--job-id", required=True)
    submit.add_argument("--launcher-sha256", required=True)
    terminal = sub.add_parser("record-terminal")
    terminal.add_argument("--mode", choices=("smoke", "formal"), required=True)
    terminal.add_argument("--job-id", required=True)
    terminal.add_argument("--state", required=True)
    terminal.add_argument("--exit-code", required=True)
    terminal.add_argument("--gpu-seconds", type=float, required=True)
    terminal.add_argument("--task-count", type=int, required=True)
    terminal.add_argument("--node-list", required=True)
    monitor = sub.add_parser("record-monitor-lifecycle")
    monitor.add_argument("--automation-id", required=True)
    monitor.add_argument("--cadence-minutes", type=int, required=True)
    monitor.add_argument("--status", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    full_argv = list(sys.argv if argv is None else [sys.argv[0], *argv])
    if args.command == "admission":
        result = materialize_admission(expected_commit=args.expected_commit, argv=full_argv)
    elif args.command == "freeze-smoke":
        result = freeze_smoke(expected_commit=args.expected_commit, argv=full_argv)
    elif args.command == "acquire-smoke":
        result = acquire_membership(
            mode="smoke",
            expected_commit=args.expected_commit,
            argv=full_argv,
            attempt=args.attempt,
        )
    elif args.command == "freeze-formal":
        result = freeze_formal(
            expected_commit=args.expected_commit,
            shard_count=args.shard_count,
            smoke_receipt_sha256=args.smoke_receipt_sha256,
            argv=full_argv,
        )
    elif args.command == "acquire-shard":
        result = acquire_membership(
            mode="formal",
            expected_commit=args.expected_commit,
            shard_id=args.shard_id,
            attempt=args.attempt,
            argv=full_argv,
        )
    elif args.command == "merge":
        result = merge_formal(expected_commit=args.expected_commit, argv=full_argv)
    elif args.command == "select":
        result = freeze_selector_files(expected_commit=args.expected_commit, argv=full_argv)
    elif args.command == "verify":
        trajectory = verify_trajectory_population()
        trajectory_path = AUTHORIZED_RUN_ROOT / "verification/trajectory_verifier_receipt.json"
        write_new_json(trajectory_path, trajectory)
        result = verify_selector_files(
            expected_commit=args.expected_commit,
            argv=full_argv,
            trajectory_verifier_receipt=trajectory,
        )
    elif args.command == "write-launcher":
        result = write_launcher(
            mode=args.mode,
            expected_commit=args.expected_commit,
            shard_count=args.shard_count,
            concurrency=args.concurrency,
        )
    elif args.command == "record-submission":
        result = record_submission(
            mode=args.mode, job_id=args.job_id, launcher_sha256=args.launcher_sha256
        )
    elif args.command == "record-terminal":
        result = record_terminal(
            mode=args.mode,
            job_id=args.job_id,
            state=args.state,
            exit_code=args.exit_code,
            gpu_seconds=args.gpu_seconds,
            task_count=args.task_count,
            node_list=args.node_list,
        )
    elif args.command == "record-monitor-lifecycle":
        result = record_monitor_lifecycle(
            automation_id=args.automation_id,
            cadence_minutes=args.cadence_minutes,
            status=args.status,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
