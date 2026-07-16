"""Sealed full-outcome execution for LoopScope Phase 3 Gate P3-E.

P3-E has three irreversible boundaries:

* ``prepare_run`` freezes the exact missing-12 full matrix and immutable
  launchers without opening any outcome file.
* ``seal_completed_sources`` admits unsealing only after all twelve Slurm
  array tasks are ``COMPLETED/0:0`` and outcome-free identity/provenance
  checks close exactly 57 tasks and 14,042 ordered identities per cell.
* ``unseal_analyze_verify`` creates a write-once unseal marker, consumes the
  baseline + historical-13 + blind-12 exactly once, and emits both the all-25
  analysis and an independently recomputed verifier receipt.

The module is intentionally side-car only.  It never changes the evaluator,
renderer, loop implementation, selector, card, or historical artifacts.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import shlex
import stat
import struct
import subprocess
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase3_p3c import (
    HISTORICAL_WINDOWS,
    MODEL_REVISION,
    TEST_MANIFEST_NAME,
    TEST_METADATA_NAME,
    _load_outcome_cell,
    _phase1_ordered_sample_identity,
    _validate_cell_command,
    _validate_cell_revision,
    _verify_result_sample_content,
    load_strict_json,
    load_strict_jsonl,
    validate_test_metadata_records,
)
from tflt.loopscope.phase3_schema import (
    PHASE3_CARD_BYTE_SHA256,
    attach_manifest_sha256,
    canonical_json_bytes,
    file_sha256,
    load_phase3_card,
    verify_manifest_sha256,
)


GATE = "P3-E"
EXECUTOR_THREAD_ID = "019f6a4f-e4c6-7170-add5-1c513c19aa16"
PLANNING_THREAD_ID = "019f670d-24ca-7ad0-b92e-64438aa04ef1"
AUTHORIZED_BASE_COMMIT = "1793df195518b823a0886249050a204bc17dd31e"

REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUTHORIZED_RUN_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3e-20260716T094417Z"
)
P3C_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3c-20260716T064601Z"
)
P3D_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase3-p3d-20260716T085629Z"
)
PHASE1_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers/"
    "runs/loopscope-qwen17-mmlu-phase1-20260711-053022"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"

SELECTOR_FILE_SHA256 = "8e0737d68cdf3c4cdb59a6f28df1e25932a156d0c944c2caa2e33ce63f8b2eef"
RETROSPECTIVE_FILE_SHA256 = "ef91c08f085ce89489515bc7c35eaa8075fc48384d9fd76c06fefb36dd97c147"
MISSING12_FILE_SHA256 = "0b3f12c6beb8070f70d9ab6def1918fa1aa35e8ff26303273d322f440aae25f0"
P3D_SUMMARY_FILE_SHA256 = "8cd9ab9f96044eae9ed93ec5004cf56c34e6ff0adc7b124ffe5a673ce0f5eb18"
EXPECTED_TEST_ORDERED_IDENTITY_SHA256 = (
    "872c9f6bd7f40e3c6fa40b13cc862033379ac0a9cbc439a1593653945f3d521d"
)

BLIND_WINDOWS = (
    "0:3",
    "1:4",
    "2:5",
    "3:6",
    "5:8",
    "7:10",
    "10:13",
    "16:19",
    "19:22",
    "21:24",
    "23:26",
    "24:27",
)
ALL_WINDOWS = tuple("%d:%d" % (start, start + 3) for start in range(25))
BLIND_SCORABLE_RANKING = ("7:10", "19:22", "16:19", "10:13", "5:8")
BLIND_TOP_M = ("7:10", "19:22")
BLIND_BOTTOM_M = ("10:13", "5:8")
SELECTED_VARIANT = "CONSENSUS"
GLOBAL_TOP1 = "12:15"

FULL_MANIFEST_NAME = "p3e_full_manifest.json"
PREOUTCOME_FREEZE_NAME = "p3e_preoutcome_freeze_receipt.json"
SUBMISSION_NAME = "p3e_submission_receipt.json"
SOURCE_MANIFEST_NAME = "blind12_full_source_manifest.json"
RESOURCE_NAME = "p3e_scheduler_throttle_resource_accounting.json"
UNSEAL_MARKER_NAME = "p3e_unseal_once.json"
ANALYSIS_NAME = "width4_all25_analysis.json"
VERIFIER_NAME = "width4_all25_verifier_receipt.json"

IMPLEMENTATION_RELATIVE_PATHS = (
    "src/tflt/loopscope/phase3_p3e.py",
    "scripts/loopscope/run_qwen17_phase3_p3e.py",
    "tests/test_loopscope_phase3_p3e.py",
)


class P3EError(ValueError):
    """Fail-closed P3-E contract, provenance, or one-shot violation."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def implementation_hashes() -> Dict[str, str]:
    root = repository_root()
    result = {}
    for relative in IMPLEMENTATION_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file():
            raise P3EError("P3-E implementation path is missing: %s" % relative)
        result[relative] = file_sha256(path)
    return result


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root)] + list(args),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise P3EError("git command failed: %s" % completed.stderr.strip())
    return completed.stdout.strip()


def git_provenance(expected_commit: str, *, require_remote_root: bool) -> Dict[str, Any]:
    root = repository_root().resolve()
    if require_remote_root and root != REMOTE_REPO:
        raise P3EError("P3-E remote action is outside the dedicated clone")
    branch = _git(root, "symbolic-ref", "--short", "HEAD")
    head = _git(root, "rev-parse", "HEAD")
    origin = _git(root, "rev-parse", "refs/remotes/origin/loopscope")
    dirty = bool(_git(root, "status", "--porcelain"))
    if branch != "loopscope":
        raise P3EError("P3-E requires branch loopscope")
    if head != str(expected_commit) or origin != str(expected_commit):
        raise P3EError("P3-E HEAD/origin differs from the expected implementation commit")
    if dirty:
        raise P3EError("P3-E requires a clean working tree")
    return {
        "root": str(root),
        "branch": branch,
        "commit": head,
        "origin_loopscope": origin,
        "dirty": False,
    }


def assert_authorized_paths(
    card_path: Path, p3c_root: Path, p3d_root: Path, p3e_root: Path
) -> None:
    exact = {
        Path(card_path).resolve(): REMOTE_REPO / "configs/loopscope/phase3_card.json",
        Path(p3c_root).resolve(): P3C_ROOT,
        Path(p3d_root).resolve(): P3D_ROOT,
        Path(p3e_root).resolve(): AUTHORIZED_RUN_ROOT,
    }
    for actual, expected in exact.items():
        if actual != expected:
            raise P3EError("P3-E path differs from exact authorization: %s" % actual)


def write_new_json(path: Path, payload: Any) -> str:
    path = Path(path)
    text = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def write_new_text(path: Path, text: str, *, executable: bool = False) -> str:
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return file_sha256(path)


def _file_metadata(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise P3EError("required source is missing or symlinked: %s" % path)
    info = path.stat()
    return {
        "path": str(path),
        "size_bytes": info.st_size,
        "file_sha256": file_sha256(path),
    }


def _window_start(window: str) -> int:
    return int(str(window).split(":", 1)[0])


def _load_frozen_inputs(p3c_root: Path, p3d_root: Path) -> Dict[str, Any]:
    expected = {
        "selector_freeze.json": SELECTOR_FILE_SHA256,
        "retrospective_adjudication.json": RETROSPECTIVE_FILE_SHA256,
        "missing12_frozen_predictions.json": MISSING12_FILE_SHA256,
    }
    loaded = {}
    for name, digest in expected.items():
        path = Path(p3c_root) / name
        if file_sha256(path) != digest:
            raise P3EError("frozen P3-C hash mismatch: %s" % name)
        payload = load_strict_json(path)
        verify_manifest_sha256(payload)
        loaded[name] = payload
    summary_path = Path(p3d_root) / "summary/p3d_engineering_summary.json"
    if file_sha256(summary_path) != P3D_SUMMARY_FILE_SHA256:
        raise P3EError("P3-D engineering summary hash mismatch")
    summary = load_strict_json(summary_path)
    predictions = loaded["missing12_frozen_predictions.json"]
    if tuple(predictions.get("blind12_universe", ())) != BLIND_WINDOWS:
        raise P3EError("P3-C blind-12 membership/order differs")
    if (
        predictions.get("selected_variant") != SELECTED_VARIANT
        or tuple(row.get("window") for row in predictions.get("scorable_ranking", ()))
        != BLIND_SCORABLE_RANKING
        or predictions.get("frozen_top1") != BLIND_TOP_M[0]
        or predictions.get("blind12_outcome_values_consumed") is not False
    ):
        raise P3EError("P3-C frozen blind ranking differs")
    if tuple(summary.get("windows", ())) != BLIND_WINDOWS:
        raise P3EError("P3-D engineering closure is not the exact blind-12")
    d2 = summary.get("closure", {}).get("D2-limit5", {})
    if (
        d2.get("cells") != 12
        or d2.get("tasks_per_cell") != 57
        or d2.get("samples_per_cell") != 285
        or d2.get("same_ordered_identities_all_cells_and_phase1_anchor") is not True
    ):
        raise P3EError("P3-D engineering closure is incomplete")
    return {
        "selector": loaded["selector_freeze.json"],
        "retrospective": loaded["retrospective_adjudication.json"],
        "predictions": predictions,
        "p3d_summary": summary,
    }


def _cell_argv(output_dir: Path, window: str) -> List[str]:
    return [
        "python",
        "-m",
        "tflt.eval_runner",
        "--model",
        "qwen3-1.7b-base",
        "--revision",
        MODEL_REVISION,
        "--tasks",
        "mmlu",
        "--output-dir",
        str(output_dir),
        "--num-fewshot",
        "5",
        "--batch-size",
        "auto",
        "--dtype",
        "float16",
        "--loop",
        "--window",
        window,
        "--k",
        "2",
        "--iteration-mode",
        "block",
        "--strategy",
        "damped_euler",
        "--alpha",
        "1.0",
        "--beta",
        "0.0",
        "--cache-strategy",
        "last",
        "--decode-mode",
        "bypass",
    ]


def build_full_manifest(
    *,
    run_root: Path,
    expected_commit: str,
    partition: str,
    qos: Optional[str],
    account: Optional[str],
    array_throttle: int,
    time_limit: str,
    created_at_utc: str,
    hashes: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Build and validate the outcome-free exact missing-12 manifest."""

    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT:
        raise P3EError("full manifest run root differs from authorization")
    if not partition or not time_limit:
        raise P3EError("scheduler partition/time must be explicit")
    if not isinstance(array_throttle, int) or isinstance(array_throttle, bool):
        raise P3EError("array throttle must be an integer")
    if array_throttle < 1 or array_throttle > 12:
        raise P3EError("array throttle must be in 1..12")
    cells = []
    for index, window in enumerate(BLIND_WINDOWS):
        output_dir = Path(run_root) / (
            "full/cell-%02d-window-%s" % (index, window.replace(":", "-"))
        )
        cells.append(
            {
                "index": index,
                "window": window,
                "output_dir": str(output_dir),
                "launcher": "launch/cell-%02d.sh" % index,
                "argv": _cell_argv(output_dir, window),
                "automatic_retry": False,
                "replacement_budget": 1,
                "replacement_admission": (
                    "only before any model sample/scientific output and only when no "
                    "results/claim exists; preserve original job/log and exact argv"
                ),
            }
        )
    manifest: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3e-full-manifest.v1",
        "artifact_role": "p3e_exact_missing12_full_preoutcome_freeze",
        "gate": GATE,
        "created_at_utc": created_at_utc,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "run_root": str(Path(run_root)),
        "git": {
            "repo": str(REMOTE_REPO),
            "branch": "loopscope",
            "commit": str(expected_commit),
            "dirty": False,
        },
        "implementation_sha256": dict(hashes or implementation_hashes()),
        "frozen_inputs": {
            "card_file_sha256": PHASE3_CARD_BYTE_SHA256,
            "selector_freeze_file_sha256": SELECTOR_FILE_SHA256,
            "retrospective_adjudication_file_sha256": RETROSPECTIVE_FILE_SHA256,
            "missing12_predictions_file_sha256": MISSING12_FILE_SHA256,
            "p3d_engineering_summary_file_sha256": P3D_SUMMARY_FILE_SHA256,
            "selected_variant": SELECTED_VARIANT,
            "global_top1": GLOBAL_TOP1,
            "blind_scorable_ranking": list(BLIND_SCORABLE_RANKING),
            "blind_top_m": list(BLIND_TOP_M),
            "blind_bottom_m": list(BLIND_BOTTOM_M),
        },
        "recipe": {
            "model": "qwen3-1.7b-base",
            "model_repo": "Qwen/Qwen3-1.7B-Base",
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "task": "mmlu",
            "num_fewshot": 5,
            "dtype": "float16",
            "k": 2,
            "fixed_horizon": True,
            "iteration_mode": "block",
            "strategy": "damped_euler",
            "alpha": 1.0,
            "beta": 0.0,
            "cache_strategy": "last",
            "decode_mode": "bypass",
            "limit": None,
        },
        "scheduler": {
            "submission_kind": "one_slurm_array_task_per_full_cell",
            "partition": partition,
            "qos": qos,
            "account": account,
            "array": "0-11%%%d" % array_throttle,
            "initial_throttle": array_throttle,
            "cpus_per_task": 8,
            "memory": "64G",
            "gres": "gpu:a40:1",
            "time_limit": time_limit,
            "requeue": False,
        },
        "cells": cells,
        "cell_count": 12,
        "window_order": list(BLIND_WINDOWS),
        "sealed_completion": {
            "required_terminal_tasks": 12,
            "required_state": "COMPLETED",
            "required_exit_code": "0:0",
            "required_tasks_per_cell": 57,
            "required_samples_per_cell": 14042,
            "required_phase1_ordered_identity_sha256": EXPECTED_TEST_ORDERED_IDENTITY_SHA256,
            "partial_outcome_viewing": False,
        },
        "analysis": {
            "unseal_process_count": 1,
            "bootstrap_method": "joint_paired_sample_bootstrap",
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 20260710,
            "joint_index_reuse_all_windows_and_contrasts": True,
            "exact_mcnemar_windows": 25,
            "holm_family_size": 25,
            "holm_familywise_alpha": 0.05,
        },
        "outcome_values_consumed": False,
    }
    attach_manifest_sha256(manifest)
    validate_full_manifest(manifest)
    return manifest


def validate_full_manifest(manifest: Mapping[str, Any]) -> None:
    verify_manifest_sha256(manifest)
    if (
        manifest.get("schema_version") != "loopscope.phase3.p3e-full-manifest.v1"
        or manifest.get("gate") != GATE
        or manifest.get("executor_thread_id") != EXECUTOR_THREAD_ID
        or Path(str(manifest.get("run_root", ""))).resolve() != AUTHORIZED_RUN_ROOT
        or manifest.get("outcome_values_consumed") is not False
    ):
        raise P3EError("P3-E full manifest identity/role differs")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != 12:
        raise P3EError("P3-E manifest must contain exactly 12 cells")
    if [cell.get("index") for cell in cells] != list(range(12)):
        raise P3EError("P3-E cell indices differ")
    if tuple(cell.get("window") for cell in cells) != BLIND_WINDOWS:
        raise P3EError("P3-E exact missing-12 membership/order differs")
    if tuple(manifest.get("window_order", ())) != BLIND_WINDOWS:
        raise P3EError("P3-E window order differs")
    for cell in cells:
        expected_dir = AUTHORIZED_RUN_ROOT / (
            "full/cell-%02d-window-%s"
            % (cell["index"], cell["window"].replace(":", "-"))
        )
        if Path(str(cell.get("output_dir", ""))).resolve() != expected_dir:
            raise P3EError("P3-E cell output path differs")
        expected_argv = _cell_argv(expected_dir, cell["window"])
        if cell.get("argv") != expected_argv or "--limit" in expected_argv:
            raise P3EError("P3-E cell full argv differs")
        if cell.get("automatic_retry") is not False:
            raise P3EError("P3-E automatic retry must be false")
    frozen = manifest.get("frozen_inputs", {})
    expected_hashes = {
        "card_file_sha256": PHASE3_CARD_BYTE_SHA256,
        "selector_freeze_file_sha256": SELECTOR_FILE_SHA256,
        "retrospective_adjudication_file_sha256": RETROSPECTIVE_FILE_SHA256,
        "missing12_predictions_file_sha256": MISSING12_FILE_SHA256,
        "p3d_engineering_summary_file_sha256": P3D_SUMMARY_FILE_SHA256,
    }
    if any(frozen.get(key) != value for key, value in expected_hashes.items()):
        raise P3EError("P3-E frozen input hash differs")


def _common_launcher_environment() -> str:
    return """set -euo pipefail
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset HF_ENDPOINT || true
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH={repo}/src
cd {repo}
""".format(repo=shlex.quote(str(REMOTE_REPO)))


def _cell_launcher_text(cell: Mapping[str, Any]) -> str:
    output_dir = shlex.quote(str(cell["output_dir"]))
    argv = list(cell["argv"])
    command = '"%s/bin/python" %s\n' % (
        AUDITED_VENV,
        shlex.join([str(value) for value in argv[1:]]),
    )
    return (
        "#!/usr/bin/env bash\n"
        + _common_launcher_environment()
        + "if [[ -e %s ]]; then\n" % output_dir
        + "  echo 'refusing to reuse an existing P3-E cell output' >&2\n"
        + "  exit 73\n"
        + "fi\n"
        + "exec "
        + command
    )


def _array_runner_text() -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", 'case "${SLURM_ARRAY_TASK_ID:?}" in']
    for index in range(12):
        lines.append(
            "  %d) exec %s ;;"
            % (index, shlex.quote(str(AUTHORIZED_RUN_ROOT / ("launch/cell-%02d.sh" % index))))
        )
    lines.extend(["  *) echo 'invalid P3-E array task id' >&2; exit 64 ;;", "esac", ""])
    return "\n".join(lines)


def _submit_text(manifest: Mapping[str, Any]) -> str:
    scheduler = manifest["scheduler"]
    argv = [
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--job-name=loopscope-p3e-full",
        "--partition=%s" % scheduler["partition"],
        "--gres=%s" % scheduler["gres"],
        "--cpus-per-task=%d" % scheduler["cpus_per_task"],
        "--mem=%s" % scheduler["memory"],
        "--time=%s" % scheduler["time_limit"],
        "--array=%s" % scheduler["array"],
        "--no-requeue",
        "--output=%s" % (AUTHORIZED_RUN_ROOT / "slurm/loopscope-p3e-full-%A_%a.out"),
        "--error=%s" % (AUTHORIZED_RUN_ROOT / "slurm/loopscope-p3e-full-%A_%a.err"),
    ]
    if scheduler.get("qos"):
        argv.append("--qos=%s" % scheduler["qos"])
    if scheduler.get("account"):
        argv.append("--account=%s" % scheduler["account"])
    argv.append(str(AUTHORIZED_RUN_ROOT / "launch/array_runner.sh"))
    return "#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n" % shlex.join(argv)


def prepare_run(
    *,
    card_path: Path,
    p3c_root: Path,
    p3d_root: Path,
    p3e_root: Path,
    expected_commit: str,
    partition: str,
    qos: Optional[str],
    account: Optional[str],
    array_throttle: int,
    time_limit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    assert_authorized_paths(card_path, p3c_root, p3d_root, p3e_root)
    if Path(p3e_root).exists():
        raise FileExistsError("authorized P3-E write-once run root already exists")
    git = git_provenance(expected_commit, require_remote_root=True)
    load_phase3_card(card_path)
    frozen = _load_frozen_inputs(p3c_root, p3d_root)
    hashes = implementation_hashes()
    manifest = build_full_manifest(
        run_root=p3e_root,
        expected_commit=expected_commit,
        partition=partition,
        qos=qos,
        account=account,
        array_throttle=array_throttle,
        time_limit=time_limit,
        created_at_utc=utc_now(),
        hashes=hashes,
    )
    root = Path(p3e_root)
    root.mkdir()
    for name in ("launch", "slurm", "full"):
        (root / name).mkdir()
    manifest_file_sha = write_new_json(root / FULL_MANIFEST_NAME, manifest)
    launcher_hashes = {}
    for cell in manifest["cells"]:
        relative = cell["launcher"]
        launcher_hashes[relative] = write_new_text(
            root / relative, _cell_launcher_text(cell), executable=True
        )
    launcher_hashes["launch/array_runner.sh"] = write_new_text(
        root / "launch/array_runner.sh", _array_runner_text(), executable=True
    )
    launcher_hashes["launch/submit.sh"] = write_new_text(
        root / "launch/submit.sh", _submit_text(manifest), executable=True
    )
    freeze: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3e-preoutcome-freeze.v1",
        "artifact_role": "p3e_preoutcome_manifest_and_immutable_launcher_freeze",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "argv": list(argv),
        "card_file_sha256": PHASE3_CARD_BYTE_SHA256,
        "selector_freeze_file_sha256": SELECTOR_FILE_SHA256,
        "retrospective_adjudication_file_sha256": RETROSPECTIVE_FILE_SHA256,
        "missing12_predictions_file_sha256": MISSING12_FILE_SHA256,
        "p3d_engineering_summary_file_sha256": P3D_SUMMARY_FILE_SHA256,
        "full_manifest_file_sha256": manifest_file_sha,
        "full_manifest_internal_sha256": manifest["manifest_sha256"],
        "launcher_sha256": launcher_hashes,
        "cell_count": len(manifest["cells"]),
        "window_order": list(BLIND_WINDOWS),
        "selected_variant": frozen["predictions"]["selected_variant"],
        "blind_scorable_ranking": [
            row["window"] for row in frozen["predictions"]["scorable_ranking"]
        ],
        "metadata_only": True,
        "outcome_values_consumed": False,
        "status": "PREOUTCOME_FREEZE_PASS",
    }
    attach_manifest_sha256(freeze)
    write_new_json(root / PREOUTCOME_FREEZE_NAME, freeze)
    return freeze


def record_submission(
    *, p3e_root: Path, expected_commit: str, job_id: str, argv: Sequence[str]
) -> Dict[str, Any]:
    if Path(p3e_root).resolve() != AUTHORIZED_RUN_ROOT:
        raise P3EError("submission root differs from authorization")
    git = git_provenance(expected_commit, require_remote_root=True)
    if not re.fullmatch(r"[0-9]+", str(job_id)):
        raise P3EError("Slurm job id must be decimal")
    manifest = load_strict_json(Path(p3e_root) / FULL_MANIFEST_NAME)
    validate_full_manifest(manifest)
    receipt: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3e-submission.v1",
        "artifact_role": "p3e_single_full_array_submission_receipt",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "git": git,
        "job_id": str(job_id),
        "array": manifest["scheduler"]["array"],
        "cell_count": 12,
        "full_manifest_file_sha256": file_sha256(Path(p3e_root) / FULL_MANIFEST_NAME),
        "full_manifest_internal_sha256": manifest["manifest_sha256"],
        "argv": list(argv),
        "automatic_retry": False,
        "outcome_values_consumed": False,
    }
    attach_manifest_sha256(receipt)
    write_new_json(Path(p3e_root) / SUBMISSION_NAME, receipt)
    return receipt


SACCT_FIELDS = (
    "JobIDRaw",
    "State",
    "ExitCode",
    "Partition",
    "NodeList",
    "ElapsedRaw",
    "AllocTRES",
    "Submit",
    "Start",
    "End",
)


def query_sacct(job_id: str) -> List[Dict[str, str]]:
    completed = subprocess.run(
        [
            "/opt/slurm/bin/sacct",
            "-j",
            str(job_id),
            "--noheader",
            "--parsable2",
            "-o",
            ",".join(SACCT_FIELDS),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise P3EError("sacct failed: %s" % completed.stderr.strip())
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if values and values[-1] == "":
            values.pop()
        if len(values) != len(SACCT_FIELDS):
            raise P3EError("unexpected sacct field count")
        rows.append(dict(zip(SACCT_FIELDS, values)))
    return rows


def validate_sealed_scheduler_rows(
    rows: Sequence[Mapping[str, Any]], job_id: str
) -> List[Dict[str, str]]:
    """Reject partial/running/failed sets before any result artifact is opened."""

    task_pattern = re.compile(r"^%s_([0-9]+)$" % re.escape(str(job_id)))
    tasks = {}
    for raw in rows:
        raw_id = str(raw.get("JobIDRaw", ""))
        match = task_pattern.fullmatch(raw_id)
        if not match:
            continue
        index = int(match.group(1))
        if index in tasks:
            raise P3EError("duplicate Slurm task row")
        tasks[index] = {key: str(raw.get(key, "")) for key in SACCT_FIELDS}
    if set(tasks) != set(range(12)):
        raise P3EError("sealed completion requires the exact 12 Slurm array tasks")
    for index, row in tasks.items():
        state_value = row["State"].split()[0].rstrip("+")
        if state_value != "COMPLETED" or row["ExitCode"] != "0:0":
            raise P3EError(
                "sealed completion rejected task %d state=%s exit=%s"
                % (index, row["State"], row["ExitCode"])
            )
    return [tasks[index] for index in range(12)]


def _validate_submission(root: Path, job_id: str, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    receipt = load_strict_json(root / SUBMISSION_NAME)
    verify_manifest_sha256(receipt)
    if (
        receipt.get("job_id") != str(job_id)
        or receipt.get("full_manifest_internal_sha256") != manifest["manifest_sha256"]
        or receipt.get("automatic_retry") is not False
    ):
        raise P3EError("submission receipt differs from the sealed job")
    return receipt


def _inspect_identity_only_cell(
    cell: Mapping[str, Any],
    test_records: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    output_dir = Path(str(cell["output_dir"])).resolve()
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise P3EError("full cell output directory is missing or symlinked")
    command_path = output_dir / "command_args.json"
    revision_path = output_dir / "model_revision.json"
    results_path = output_dir / "results.json"
    # File existence/non-emptiness is checked before the first JSON parse.
    for path in (command_path, revision_path, results_path):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise P3EError("sealed full artifact is absent/empty/symlinked: %s" % path)
    command = load_strict_json(command_path)
    revision = load_strict_json(revision_path)
    _validate_cell_command(command, str(cell["window"]), output_dir, card)
    _validate_cell_revision(revision)
    payload = load_strict_json(results_path)
    legacy_identity = _phase1_ordered_sample_identity(payload)
    if legacy_identity != {
        "task_count": 57,
        "sample_count": 14042,
        "ordered_identity_sha256": EXPECTED_TEST_ORDERED_IDENTITY_SHA256,
    }:
        raise P3EError("full cell legacy identity does not close 57/14,042")
    expected_by_pair_id = {}
    for record in test_records:
        identity = record["identity"]
        pair_id = "%s:%s" % (identity["task"], identity["doc_id"])
        if pair_id in expected_by_pair_id:
            raise P3EError("test metadata contains duplicate evaluator pair id")
        expected_by_pair_id[pair_id] = record
    _verify_result_sample_content(payload, expected_by_pair_id)
    sidecars = sorted(output_dir.glob("samples_*.jsonl"), key=lambda path: path.name)
    result = {
        "cell": str(cell["window"]),
        "window": str(cell["window"]),
        "index": int(cell["index"]),
        "output_dir": str(output_dir),
        "job_id": None,
        "command_args": _file_metadata(command_path),
        "model_revision": _file_metadata(revision_path),
        "results": _file_metadata(results_path),
        "sample_sidecars": [_file_metadata(path) for path in sidecars],
        "task_count": 57,
        "sample_count": 14042,
        "phase1_ordered_sample_identity_sha256": EXPECTED_TEST_ORDERED_IDENTITY_SHA256,
        "canonical_test_identity_exact": True,
        "renderer_source_doc_hash_exact": True,
        "recipe": {
            "loop": True,
            "k": 2,
            "fixed_horizon": True,
            "step_size": 0.5,
            "total_horizon": 1.0,
            "iteration_mode": "block",
            "strategy": "damped_euler",
            "alpha": 1.0,
            "beta": 0.0,
            "cache_strategy": "last",
            "decode_mode": "bypass",
        },
    }
    del payload
    gc.collect()
    return result


def _allocated_gpu_count(alloc_tres: str) -> int:
    match = re.search(r"(?:^|,)gres/gpu=([0-9]+)(?:,|$)", str(alloc_tres))
    return int(match.group(1)) if match else 1


def _resource_receipt(
    *,
    manifest: Mapping[str, Any],
    job_id: str,
    scheduler_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    elapsed = []
    gpu_seconds = 0
    for row in scheduler_rows:
        try:
            seconds = int(row["ElapsedRaw"])
        except (KeyError, ValueError) as exc:
            raise P3EError("Slurm ElapsedRaw is not an integer") from exc
        elapsed.append(seconds)
        gpu_seconds += seconds * _allocated_gpu_count(row.get("AllocTRES", ""))
    receipt: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3e-resource-accounting.v1",
        "artifact_role": "p3e_scheduler_throttle_resource_accounting",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "job_id": str(job_id),
        "full_manifest_internal_sha256": manifest["manifest_sha256"],
        "initial_array_throttle": manifest["scheduler"]["initial_throttle"],
        "throttle_history": [
            {
                "kind": "initial_submission",
                "array": manifest["scheduler"]["array"],
                "science_manifest_unchanged": True,
            }
        ],
        "terminal_task_count": len(scheduler_rows),
        "completed_0_0": sum(
            row["State"].split()[0].rstrip("+") == "COMPLETED"
            and row["ExitCode"] == "0:0"
            for row in scheduler_rows
        ),
        "elapsed_raw_seconds_sum": sum(elapsed),
        "gpu_hours_allocated_from_elapsed": gpu_seconds / 3600.0,
        "partitions": sorted({row.get("Partition", "") for row in scheduler_rows}),
        "nodes": sorted({row.get("NodeList", "") for row in scheduler_rows}),
        "alloc_tres": sorted({row.get("AllocTRES", "") for row in scheduler_rows}),
        "tasks": [dict(row) for row in scheduler_rows],
        "scientific_argv_config_hash_unchanged": True,
        "outcome_values_consumed": False,
        "status": "TERMINAL_RESOURCE_ACCOUNTING_PASS",
    }
    attach_manifest_sha256(receipt)
    return receipt


def seal_completed_sources(
    *,
    card_path: Path,
    p3c_root: Path,
    p3d_root: Path,
    p3e_root: Path,
    expected_commit: str,
    job_id: str,
    argv: Sequence[str],
    sacct_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create the source allowlist only after exact terminal and identity closure."""

    assert_authorized_paths(card_path, p3c_root, p3d_root, p3e_root)
    git = git_provenance(expected_commit, require_remote_root=True)
    root = Path(p3e_root)
    for name in (SOURCE_MANIFEST_NAME, RESOURCE_NAME, UNSEAL_MARKER_NAME, ANALYSIS_NAME, VERIFIER_NAME):
        if (root / name).exists():
            raise FileExistsError("refusing a second sealed/unseal artifact: %s" % name)
    manifest = load_strict_json(root / FULL_MANIFEST_NAME)
    validate_full_manifest(manifest)
    if manifest["git"]["commit"] != expected_commit:
        raise P3EError("full manifest implementation commit differs")
    _load_frozen_inputs(p3c_root, p3d_root)
    _validate_submission(root, job_id, manifest)
    terminal_rows = validate_sealed_scheduler_rows(
        list(sacct_rows) if sacct_rows is not None else query_sacct(job_id), job_id
    )
    # Prove all result artifacts exist and are non-empty before opening any JSON.
    expected_dirs = {Path(cell["output_dir"]).resolve() for cell in manifest["cells"]}
    actual_dirs = {path.resolve() for path in (root / "full").iterdir() if path.is_dir()}
    if actual_dirs != expected_dirs:
        raise P3EError("full output directory set has missing/extra cells")
    for cell in manifest["cells"]:
        for name in ("command_args.json", "model_revision.json", "results.json"):
            path = Path(cell["output_dir"]) / name
            if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
                raise P3EError("sealed completion lacks a non-empty full artifact")
    card = load_phase3_card(card_path)
    test_records = validate_test_metadata_records(
        load_strict_jsonl(Path(p3c_root) / TEST_METADATA_NAME), card
    )
    test_manifest = load_strict_json(Path(p3c_root) / TEST_MANIFEST_NAME)
    verify_manifest_sha256(test_manifest)
    entries = []
    for cell, scheduler_row in zip(manifest["cells"], terminal_rows):
        entry = _inspect_identity_only_cell(cell, test_records, card)
        entry["job_id"] = "%s_%d" % (job_id, cell["index"])
        entry["scheduler"] = dict(scheduler_row)
        entries.append(entry)
    resource = _resource_receipt(
        manifest=manifest, job_id=job_id, scheduler_rows=terminal_rows
    )
    resource_file_sha = write_new_json(root / RESOURCE_NAME, resource)
    source: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3e-blind12-full-source-manifest.v1",
        "artifact_role": "p3e_sealed_complete_blind12_full_exact_source_allowlist",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "argv": list(argv),
        "job_id": str(job_id),
        "full_manifest_file_sha256": file_sha256(root / FULL_MANIFEST_NAME),
        "full_manifest_internal_sha256": manifest["manifest_sha256"],
        "submission_file_sha256": file_sha256(root / SUBMISSION_NAME),
        "resource_accounting_file_sha256": resource_file_sha,
        "resource_accounting_internal_sha256": resource["manifest_sha256"],
        "card_file_sha256": PHASE3_CARD_BYTE_SHA256,
        "selector_freeze_file_sha256": SELECTOR_FILE_SHA256,
        "retrospective_adjudication_file_sha256": RETROSPECTIVE_FILE_SHA256,
        "missing12_predictions_file_sha256": MISSING12_FILE_SHA256,
        "p3d_engineering_summary_file_sha256": P3D_SUMMARY_FILE_SHA256,
        "window_order": list(BLIND_WINDOWS),
        "cells": entries,
        "cell_count": 12,
        "terminal_completed_0_0": 12,
        "task_count_per_cell": 57,
        "sample_count_per_cell": 14042,
        "phase1_ordered_sample_identity_sha256": EXPECTED_TEST_ORDERED_IDENTITY_SHA256,
        "canonical_test_ordered_identity_sha256": test_manifest["ordered_identity_sha256"],
        "same_ordered_identity_all_blind_baseline_historical": True,
        "recipe_revision_renderer_closure": True,
        "sealed_completion_proven_before_result_json_parse": True,
        "result_identity_and_source_doc_fields_accessed_after_terminal_proof": True,
        "accuracy_answer_correctness_metric_values_accessed": False,
        "outcome_values_consumed": False,
        "unseal_authorized": True,
        "status": "SEALED_12_OF_12_COMPLETION_AND_IDENTITY_PASS",
    }
    attach_manifest_sha256(source)
    write_new_json(root / SOURCE_MANIFEST_NAME, source)
    return source


def exact_mcnemar_p(wrong_to_right: int, right_to_wrong: int) -> float:
    for value in (wrong_to_right, right_to_wrong):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise P3EError("McNemar counts must be non-negative integers")
    discordant = wrong_to_right + right_to_wrong
    if discordant == 0:
        return 1.0
    low = min(wrong_to_right, right_to_wrong)
    logs = []
    current = -discordant * math.log(2.0)
    logs.append(current)
    for index in range(1, low + 1):
        current += math.log(discordant - index + 1) - math.log(index)
        logs.append(current)
    peak = max(logs)
    log_tail = peak + math.log(math.fsum(math.exp(value - peak) for value in logs))
    return min(1.0, math.exp(math.log(2.0) + log_tail))


def holm_step_down_25(
    raw_p_by_window: Mapping[str, float], *, alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    if set(raw_p_by_window) != set(ALL_WINDOWS):
        raise P3EError("Holm family must be the exact all-25 width-4 set")
    order_index = {window: index for index, window in enumerate(ALL_WINDOWS)}
    ranked = sorted(
        ALL_WINDOWS,
        key=lambda window: (float(raw_p_by_window[window]), order_index[window]),
    )
    running = 0.0
    result = {}
    for rank, window in enumerate(ranked, start=1):
        raw = float(raw_p_by_window[window])
        if not math.isfinite(raw) or raw < 0.0 or raw > 1.0:
            raise P3EError("Holm raw p-value is invalid")
        adjusted = max(running, min(1.0, (26 - rank) * raw))
        running = adjusted
        result[window] = {
            "raw_p": raw,
            "adjusted_p": adjusted,
            "significant": adjusted < alpha,
            "holm_rank": rank,
        }
    return {window: result[window] for window in ALL_WINDOWS}


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise P3EError("quantile requires values")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _index_bytes(indices: Sequence[int]) -> bytes:
    packed = array("I", indices)
    if packed.itemsize != 4:
        return b"".join(struct.pack("<I", index) for index in indices)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _validate_difference_matrix(
    differences_by_window: Mapping[str, Sequence[int]],
) -> Tuple[int, List[List[int]]]:
    if tuple(differences_by_window) != ALL_WINDOWS:
        raise P3EError("joint bootstrap window order must be exact all-25")
    lengths = {len(values) for values in differences_by_window.values()}
    if len(lengths) != 1 or next(iter(lengths), 0) <= 0:
        raise P3EError("joint bootstrap differences must have one positive length")
    count = next(iter(lengths))
    rows = []
    for window in ALL_WINDOWS:
        row = []
        for value in differences_by_window[window]:
            if isinstance(value, bool) or int(value) != value or int(value) not in (-1, 0, 1):
                raise P3EError("paired accuracy difference must be -1, 0, or 1")
            row.append(int(value))
        rows.append(row)
    return count, rows


def joint_paired_bootstrap(
    differences_by_window: Mapping[str, Sequence[int]],
    *,
    replicates: int,
    seed: int,
    force_python: bool = False,
) -> Dict[str, Any]:
    """Use one ``Random`` stream and the same sampled indices for all contrasts."""

    count, rows = _validate_difference_matrix(differences_by_window)
    if not isinstance(replicates, int) or replicates < 1:
        raise P3EError("bootstrap replicates must be positive")
    top_indices = [ALL_WINDOWS.index(window) for window in BLIND_TOP_M]
    bottom_indices = [ALL_WINDOWS.index(window) for window in BLIND_BOTTOM_M]
    g_values = [
        0.5 * (rows[top_indices[0]][index] + rows[top_indices[1]][index])
        - 0.5 * (rows[bottom_indices[0]][index] + rows[bottom_indices[1]][index])
        for index in range(count)
    ]
    rng = random.Random(int(seed))
    stream_hash = hashlib.sha256()
    estimates = {window: [] for window in ALL_WINDOWS}
    g_estimates = []
    numpy_module = None
    if not force_python and count * replicates * len(ALL_WINDOWS) >= 1_000_000:
        try:
            import numpy as numpy_module  # type: ignore
        except ImportError:
            numpy_module = None
    if numpy_module is not None:
        matrix = numpy_module.asarray(rows, dtype=numpy_module.int8)
        g_vector = numpy_module.asarray(g_values, dtype=numpy_module.float64)
        for _ in range(replicates):
            indices = [rng.randrange(count) for _ in range(count)]
            stream_hash.update(_index_bytes(indices))
            sampled_counts = numpy_module.bincount(indices, minlength=count)
            points = matrix.dot(sampled_counts) / float(count)
            g_point = float(g_vector.dot(sampled_counts) / float(count))
            for window, point in zip(ALL_WINDOWS, points.tolist()):
                estimates[window].append(float(point))
            g_estimates.append(g_point)
    else:
        for _ in range(replicates):
            indices = [rng.randrange(count) for _ in range(count)]
            stream_hash.update(_index_bytes(indices))
            for window, row in zip(ALL_WINDOWS, rows):
                estimates[window].append(
                    math.fsum(row[index] for index in indices) / float(count)
                )
            g_estimates.append(
                math.fsum(g_values[index] for index in indices) / float(count)
            )
    windows = {}
    for window in ALL_WINDOWS:
        values = estimates[window]
        windows[window] = {
            "point_delta_accuracy_fraction": math.fsum(
                differences_by_window[window]
            )
            / float(count),
            "bootstrap_mean_fraction": math.fsum(values) / float(replicates),
            "percentile_95_ci_fraction": [
                _quantile(values, 0.025),
                _quantile(values, 0.975),
            ],
        }
    g_point = math.fsum(g_values) / float(count)
    g_ci = [_quantile(g_estimates, 0.025), _quantile(g_estimates, 0.975)]
    return {
        "method": "joint_paired_sample_bootstrap",
        "sample_count": count,
        "replicates": replicates,
        "seed": int(seed),
        "iteration_order": "replicate-major then draw-major over canonical test identity order",
        "index_stream_sha256": stream_hash.hexdigest(),
        "joint_index_reuse_all_windows_and_g_blind": True,
        "windows": windows,
        "g_blind": {
            "top_m": list(BLIND_TOP_M),
            "bottom_m": list(BLIND_BOTTOM_M),
            "point_fraction": g_point,
            "percentile_95_ci_fraction": g_ci,
            "replicate_mean_fraction": math.fsum(g_estimates) / float(replicates),
        },
    }


def _load_source_bundle(
    card_path: Path, p3c_root: Path, p3e_root: Path
) -> Tuple[Dict[str, bool], Dict[str, List[int]], Dict[str, Any]]:
    card = load_phase3_card(card_path)
    test_records = validate_test_metadata_records(
        load_strict_jsonl(Path(p3c_root) / TEST_METADATA_NAME), card
    )
    test_manifest = load_strict_json(Path(p3c_root) / TEST_MANIFEST_NAME)
    verify_manifest_sha256(test_manifest)
    allowlist = load_strict_json(Path(p3c_root) / "historical13_source_allowlist_receipt.json")
    verify_manifest_sha256(allowlist)
    historical_entries = {entry["cell"]: entry for entry in allowlist["cells"]}
    source = load_strict_json(Path(p3e_root) / SOURCE_MANIFEST_NAME)
    verify_manifest_sha256(source)
    if (
        source.get("status") != "SEALED_12_OF_12_COMPLETION_AND_IDENTITY_PASS"
        or source.get("unseal_authorized") is not True
        or source.get("outcome_values_consumed") is not False
        or tuple(source.get("window_order", ())) != BLIND_WINDOWS
    ):
        raise P3EError("blind-12 source manifest does not admit one-shot unseal")
    blind_entries = {entry["cell"]: entry for entry in source["cells"]}
    if set(blind_entries) != set(BLIND_WINDOWS):
        raise P3EError("blind-12 source manifest membership differs")
    baseline_map, baseline_summary = _load_outcome_cell(
        historical_entries["baseline"], test_records, test_manifest
    )
    baseline_order = list(baseline_map)
    if len(baseline_order) != 14042:
        raise P3EError("baseline does not contain exactly 14,042 identities")
    baseline_values = [bool(baseline_map[key]) for key in baseline_order]
    differences: Dict[str, List[int]] = {}
    summaries = {"baseline": baseline_summary, "windows": {}}
    for window in ALL_WINDOWS:
        entry = (
            historical_entries[window]
            if window in historical_entries
            else blind_entries[window]
        )
        candidate_map, summary = _load_outcome_cell(entry, test_records, test_manifest)
        if list(candidate_map) != baseline_order:
            raise P3EError("all-25 canonical identity order differs from baseline")
        candidate_values = [bool(candidate_map[key]) for key in baseline_order]
        differences[window] = [
            int(candidate) - int(baseline)
            for baseline, candidate in zip(baseline_values, candidate_values)
        ]
        summary["accuracy_fraction"] = sum(candidate_values) / 14042.0
        summaries["windows"][window] = summary
        del candidate_map, candidate_values
        gc.collect()
    return baseline_map, differences, summaries


def _selector_context(selector: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        variant = selector["analysis"]["variants"][SELECTED_VARIANT]
        deployment = variant["scopes"]["deployment"]
        blind = variant["scopes"]["blind_enrichment"]
    except (KeyError, TypeError) as exc:
        raise P3EError("selector freeze lacks selected-variant contexts") from exc
    if deployment.get("point_top1_window") != GLOBAL_TOP1:
        raise P3EError("frozen deployment top-1 differs")
    ranking = deployment.get("published_ranking")
    if not isinstance(ranking, list) or not ranking:
        raise P3EError("frozen deployment ranking is missing")
    if tuple(row.get("window") for row in blind.get("published_ranking", ())) != BLIND_SCORABLE_RANKING:
        raise P3EError("frozen blind ranking differs")
    window_rows = {row["window"]: row for row in variant["windows"]}
    return {
        "support_starts": list(variant["support_starts"]),
        "point_eligible_starts": list(deployment["point_eligible_starts"]),
        "deployment_ranking": [row["window"] for row in ranking],
        "deployment_top3": [row["window"] for row in ranking[:3]],
        "blind_ranking": list(BLIND_SCORABLE_RANKING),
        "window_rows": window_rows,
    }


def _window_statistics(
    baseline_values: Sequence[bool],
    differences: Mapping[str, Sequence[int]],
    bootstrap: Mapping[str, Any],
    selector_context: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    count = len(baseline_values)
    baseline_accuracy = sum(bool(value) for value in baseline_values) / float(count)
    raw_p = {}
    transitions = {}
    accuracy = {}
    for window in ALL_WINDOWS:
        row_counts = {
            "right_to_right": 0,
            "wrong_to_right": 0,
            "right_to_wrong": 0,
            "wrong_to_wrong": 0,
        }
        candidate_correct = 0
        for baseline, difference in zip(baseline_values, differences[window]):
            candidate = bool(int(baseline) + int(difference))
            candidate_correct += int(candidate)
            if baseline and candidate:
                row_counts["right_to_right"] += 1
            elif not baseline and candidate:
                row_counts["wrong_to_right"] += 1
            elif baseline and not candidate:
                row_counts["right_to_wrong"] += 1
            else:
                row_counts["wrong_to_wrong"] += 1
        transitions[window] = row_counts
        accuracy[window] = candidate_correct / float(count)
        raw_p[window] = exact_mcnemar_p(
            row_counts["wrong_to_right"], row_counts["right_to_wrong"]
        )
    holm = holm_step_down_25(raw_p)
    outcome_order = sorted(ALL_WINDOWS, key=lambda w: (-accuracy[w], _window_start(w)))
    outcome_rank = {window: rank for rank, window in enumerate(outcome_order, start=1)}
    selector_rank = {
        window: rank
        for rank, window in enumerate(selector_context["deployment_ranking"], start=1)
    }
    rows = []
    for window in ALL_WINDOWS:
        start = _window_start(window)
        selector_row = selector_context["window_rows"].get(window)
        interval = bootstrap["windows"][window]["percentile_95_ci_fraction"]
        rows.append(
            {
                "window": window,
                "start": start,
                "sample_count": count,
                "accuracy_fraction": accuracy[window],
                "delta_accuracy_fraction": accuracy[window] - baseline_accuracy,
                "delta_accuracy_pp": (accuracy[window] - baseline_accuracy) * 100.0,
                "paired_ci95_fraction": list(interval),
                "paired_ci95_pp": [float(value) * 100.0 for value in interval],
                "transitions": transitions[window],
                "exact_mcnemar_p": raw_p[window],
                "holm_adjusted_p": holm[window]["adjusted_p"],
                "holm_significant": holm[window]["significant"],
                "holm_rank": holm[window]["holm_rank"],
                "outcome_rank": outcome_rank[window],
                "selector_scorable": selector_row is not None,
                "frozen_selector_rank": selector_rank.get(window),
                "frozen_selector_score": None if selector_row is None else selector_row["score"],
                "frozen_point_eligible": (
                    False if selector_row is None else bool(selector_row["point_eligible"])
                ),
            }
        )
    best = accuracy[outcome_order[0]]
    oracle_windows = [window for window in ALL_WINDOWS if accuracy[window] == best]
    oracle = {
        "windows": oracle_windows,
        "unique": len(oracle_windows) == 1,
        "accuracy_fraction": best,
        "ranked_windows": outcome_order,
        "all_inside_selected_variant_support": all(
            _window_start(window) in selector_context["support_starts"]
            for window in oracle_windows
        ),
    }
    return rows, oracle


def classify_width4(
    *,
    evidence_complete: bool,
    retrospective_recovered: bool,
    selected_window_row: Mapping[str, Any],
    oracle_windows: Sequence[str],
    oracle_inside_support: bool,
    deployment_top3: Sequence[str],
    false_positives_excluded: bool,
    blind_label: str,
) -> Tuple[str, Dict[str, bool]]:
    """Apply the frozen first-match label order without tuning."""

    selected_negative_supported = (
        float(selected_window_row["delta_accuracy_fraction"]) < 0.0
        and bool(selected_window_row["holm_significant"])
    )
    top1_hit = GLOBAL_TOP1 in set(oracle_windows)
    top3_hit = bool(set(deployment_top3) & set(oracle_windows))
    selected_positive = float(selected_window_row["delta_accuracy_fraction"]) > 0.0
    blind_supported = blind_label == "SUPPORTED"
    checks = {
        "evidence_complete": bool(evidence_complete),
        "retrospective_recovered": bool(retrospective_recovered),
        "selected_negative_holm_supported": selected_negative_supported,
        "oracle_inside_support": bool(oracle_inside_support),
        "frozen_top1_hits_oracle": top1_hit,
        "frozen_top3_hits_oracle": top3_hit,
        "selected_point_gain_positive": selected_positive,
        "primary_false_positives_excluded": bool(false_positives_excluded),
        "blind_enrichment_supported": blind_supported,
    }
    if not evidence_complete:
        label = "WIDTH4_INCONCLUSIVE"
    elif selected_negative_supported:
        label = "WIDTH4_HARMFUL_SELECTION"
    elif retrospective_recovered and not oracle_inside_support:
        label = "WIDTH4_COVERAGE_FAILURE"
    elif (
        retrospective_recovered
        and oracle_inside_support
        and top1_hit
        and selected_positive
        and false_positives_excluded
        and blind_supported
    ):
        label = "WIDTH4_WITHIN_CELL_FEASIBILITY_SUPPORTED"
    elif top3_hit or blind_supported:
        label = "WIDTH4_RANKING_SIGNAL_ONLY"
    elif retrospective_recovered:
        label = "WIDTH4_RETROSPECTIVE_RECOVERY_ONLY"
    else:
        label = "WIDTH4_NOT_SUPPORTED"
    return label, checks


def _build_analysis(
    *,
    baseline_map: Mapping[str, bool],
    differences: Mapping[str, Sequence[int]],
    summaries: Mapping[str, Any],
    selector: Mapping[str, Any],
    retrospective: Mapping[str, Any],
    source: Mapping[str, Any],
    git: Mapping[str, Any],
    argv: Sequence[str],
    replicates: int,
    seed: int,
) -> Dict[str, Any]:
    baseline_values = [bool(baseline_map[key]) for key in baseline_map]
    selector_context = _selector_context(selector)
    bootstrap = joint_paired_bootstrap(
        differences, replicates=replicates, seed=seed
    )
    rows, oracle = _window_statistics(
        baseline_values, differences, bootstrap, selector_context
    )
    by_window = {row["window"]: row for row in rows}
    g_ci = bootstrap["g_blind"]["percentile_95_ci_fraction"]
    if g_ci[0] > 0.0:
        blind_label = "SUPPORTED"
    elif g_ci[1] < 0.0:
        blind_label = "REFUTED"
    else:
        blind_label = "INCONCLUSIVE"
    false_positives_excluded = not (
        {4, 6} & set(selector_context["point_eligible_starts"])
    )
    label, label_checks = classify_width4(
        evidence_complete=True,
        retrospective_recovered=(
            retrospective.get("science_label") == "RETROSPECTIVE_TOP1_RECOVERY"
            and retrospective.get("selected_variant") == SELECTED_VARIANT
        ),
        selected_window_row=by_window[GLOBAL_TOP1],
        oracle_windows=oracle["windows"],
        oracle_inside_support=oracle["all_inside_selected_variant_support"],
        deployment_top3=selector_context["deployment_top3"],
        false_positives_excluded=false_positives_excluded,
        blind_label=blind_label,
    )
    oracle_accuracy = float(oracle["accuracy_fraction"])
    diagnostics = {}
    for window in (GLOBAL_TOP1, BLIND_TOP_M[0]):
        row = by_window[window]
        diagnostics[window] = {
            "outcome_rank": row["outcome_rank"],
            "accuracy_fraction": row["accuracy_fraction"],
            "delta_accuracy_pp": row["delta_accuracy_pp"],
            "regret_to_observed_oracle_pp": (
                oracle_accuracy - float(row["accuracy_fraction"])
            )
            * 100.0,
            "holm_significant": row["holm_significant"],
        }
    accepted = [
        row for row in rows if row["frozen_point_eligible"]
    ]
    harmful = [
        row["window"] for row in accepted if row["delta_accuracy_fraction"] < 0.0
    ]
    harmful_supported = [
        row["window"]
        for row in accepted
        if row["delta_accuracy_fraction"] < 0.0 and row["holm_significant"]
    ]
    analysis: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3e-all25-analysis.v1",
        "artifact_role": "p3e_one_shot_all25_width4_outcome_analysis",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": dict(git),
        "argv": list(argv),
        "implementation_sha256": implementation_hashes(),
        "card_file_sha256": PHASE3_CARD_BYTE_SHA256,
        "selector_freeze_file_sha256": SELECTOR_FILE_SHA256,
        "retrospective_adjudication_file_sha256": RETROSPECTIVE_FILE_SHA256,
        "missing12_predictions_file_sha256": MISSING12_FILE_SHA256,
        "blind12_source_manifest_internal_sha256": source["manifest_sha256"],
        "unseal_count": 1,
        "sample_count": 14042,
        "baseline": {
            "accuracy_fraction": sum(baseline_values) / 14042.0,
            "source": summaries["baseline"],
        },
        "bootstrap": bootstrap,
        "blind_primary_endpoint": {
            "n": 5,
            "m": 2,
            "top_m": list(BLIND_TOP_M),
            "bottom_m": list(BLIND_BOTTOM_M),
            "g_blind_fraction": bootstrap["g_blind"]["point_fraction"],
            "g_blind_pp": bootstrap["g_blind"]["point_fraction"] * 100.0,
            "ci95_fraction": list(g_ci),
            "ci95_pp": [float(value) * 100.0 for value in g_ci],
            "science_result": blind_label,
        },
        "all25_windows": rows,
        "holm_family": {
            "method": "Holm step-down",
            "family_size": 25,
            "familywise_alpha": 0.05,
        },
        "observed_all25_oracle": oracle,
        "selector": {
            "selected_variant": SELECTED_VARIANT,
            "support_starts": selector_context["support_starts"],
            "frozen_global_top1": GLOBAL_TOP1,
            "frozen_global_top3": selector_context["deployment_top3"],
            "blind_scorable_ranking": selector_context["blind_ranking"],
            "top3_oracle_hit": bool(
                set(selector_context["deployment_top3"]) & set(oracle["windows"])
            ),
            "diagnostics": diagnostics,
        },
        "harmful_acceptance": {
            "accepted_point_eligible_windows": [row["window"] for row in accepted],
            "harmful_point_windows": harmful,
            "holm_supported_harmful_windows": harmful_supported,
        },
        "abstention_coverage": {
            "selected_variant_scorable_windows": len(selector_context["support_starts"]),
            "all25_total_windows": 25,
            "support_fraction": len(selector_context["support_starts"]) / 25.0,
            "point_eligible_accepted_windows": len(accepted),
            "point_eligible_acceptance_fraction": len(accepted) / 25.0,
            "blind_scorable_windows": 5,
            "blind_total_windows": 12,
            "blind_scorable_fraction": 5.0 / 12.0,
            "oracle_inside_support": oracle["all_inside_selected_variant_support"],
        },
        "width4_science_label": label,
        "width4_label_checks": label_checks,
        "evidence_complete": True,
        "analysis_rerun_authorized": False,
        "status": "ALL25_ANALYSIS_COMPLETE",
    }
    attach_manifest_sha256(analysis)
    return analysis


def _independent_verify(
    *,
    analysis: Mapping[str, Any],
    baseline_map: Mapping[str, bool],
    differences: Mapping[str, Sequence[int]],
    selector: Mapping[str, Any],
    retrospective: Mapping[str, Any],
    replicates: int,
    seed: int,
) -> Dict[str, Any]:
    """Recompute every decision-critical statistic from exact source maps."""

    verify_manifest_sha256(analysis)
    reference_bootstrap = joint_paired_bootstrap(
        differences,
        replicates=replicates,
        seed=seed,
        force_python=(len(baseline_map) < 100),
    )
    baseline_values = [bool(baseline_map[key]) for key in baseline_map]
    selector_context = _selector_context(selector)
    rows, oracle = _window_statistics(
        baseline_values, differences, reference_bootstrap, selector_context
    )
    serialized_rows = analysis.get("all25_windows")
    if canonical_json_bytes(rows) != canonical_json_bytes(serialized_rows):
        raise P3EError("verifier all-25 window statistics differ")
    if canonical_json_bytes(oracle) != canonical_json_bytes(
        analysis.get("observed_all25_oracle")
    ):
        raise P3EError("verifier observed oracle differs")
    if canonical_json_bytes(reference_bootstrap) != canonical_json_bytes(
        analysis.get("bootstrap")
    ):
        raise P3EError("verifier joint bootstrap differs")
    g_ci = reference_bootstrap["g_blind"]["percentile_95_ci_fraction"]
    blind_label = (
        "SUPPORTED"
        if g_ci[0] > 0.0
        else "REFUTED"
        if g_ci[1] < 0.0
        else "INCONCLUSIVE"
    )
    by_window = {row["window"]: row for row in rows}
    false_positives_excluded = not (
        {4, 6} & set(selector_context["point_eligible_starts"])
    )
    label, checks = classify_width4(
        evidence_complete=True,
        retrospective_recovered=(
            retrospective.get("science_label") == "RETROSPECTIVE_TOP1_RECOVERY"
            and retrospective.get("selected_variant") == SELECTED_VARIANT
        ),
        selected_window_row=by_window[GLOBAL_TOP1],
        oracle_windows=oracle["windows"],
        oracle_inside_support=oracle["all_inside_selected_variant_support"],
        deployment_top3=selector_context["deployment_top3"],
        false_positives_excluded=false_positives_excluded,
        blind_label=blind_label,
    )
    if label != analysis.get("width4_science_label") or canonical_json_bytes(checks) != canonical_json_bytes(
        analysis.get("width4_label_checks")
    ):
        raise P3EError("verifier frozen width-4 first-match label differs")
    return {
        "all25_identity_and_point_statistics_recomputed": True,
        "joint_bootstrap_recomputed": True,
        "index_stream_sha256": reference_bootstrap["index_stream_sha256"],
        "g_blind_recomputed": True,
        "exact_mcnemar_25_recomputed": True,
        "holm_25_recomputed": True,
        "oracle_rank_regret_recomputed": True,
        "harmful_acceptance_coverage_recomputed": True,
        "first_match_label_recomputed": True,
        "blind_science_result": blind_label,
        "width4_science_label": label,
    }


def unseal_analyze_verify(
    *,
    card_path: Path,
    p3c_root: Path,
    p3d_root: Path,
    p3e_root: Path,
    expected_commit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    assert_authorized_paths(card_path, p3c_root, p3d_root, p3e_root)
    git = git_provenance(expected_commit, require_remote_root=True)
    root = Path(p3e_root)
    for name in (UNSEAL_MARKER_NAME, ANALYSIS_NAME, VERIFIER_NAME):
        if (root / name).exists():
            raise FileExistsError("P3-E unseal/analyzer/verifier is write-once: %s" % name)
    source = load_strict_json(root / SOURCE_MANIFEST_NAME)
    verify_manifest_sha256(source)
    if source.get("unseal_authorized") is not True:
        raise P3EError("sealed source manifest does not authorize unseal")
    _load_frozen_inputs(p3c_root, p3d_root)
    marker: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3e-unseal-once.v1",
        "artifact_role": "p3e_irreversible_single_unseal_marker",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "git": git,
        "blind12_source_manifest_file_sha256": file_sha256(root / SOURCE_MANIFEST_NAME),
        "blind12_source_manifest_internal_sha256": source["manifest_sha256"],
        "unseal_count": 1,
        "analysis_rerun_authorized": False,
    }
    attach_manifest_sha256(marker)
    write_new_json(root / UNSEAL_MARKER_NAME, marker)
    # No repair/rerun is permitted after this point.  Any exception is material.
    selector = load_strict_json(Path(p3c_root) / "selector_freeze.json")
    verify_manifest_sha256(selector)
    retrospective = load_strict_json(
        Path(p3c_root) / "retrospective_adjudication.json"
    )
    verify_manifest_sha256(retrospective)
    baseline_map, differences, summaries = _load_source_bundle(
        card_path, p3c_root, p3e_root
    )
    analysis = _build_analysis(
        baseline_map=baseline_map,
        differences=differences,
        summaries=summaries,
        selector=selector,
        retrospective=retrospective,
        source=source,
        git=git,
        argv=argv,
        replicates=2000,
        seed=20260710,
    )
    analysis_file_sha = write_new_json(root / ANALYSIS_NAME, analysis)
    serialized_analysis = load_strict_json(root / ANALYSIS_NAME)
    # The verifier starts from a fresh source load rather than trusting the
    # analyzer's in-memory correctness maps or derived difference matrix.
    del baseline_map, differences, summaries
    gc.collect()
    verifier_baseline_map, verifier_differences, verifier_summaries = _load_source_bundle(
        card_path, p3c_root, p3e_root
    )
    del verifier_summaries
    verification = _independent_verify(
        analysis=serialized_analysis,
        baseline_map=verifier_baseline_map,
        differences=verifier_differences,
        selector=selector,
        retrospective=retrospective,
        replicates=2000,
        seed=20260710,
    )
    verifier: Dict[str, Any] = {
        "schema_version": "loopscope.phase3.p3e-all25-verifier.v1",
        "artifact_role": "p3e_independent_all25_decision_verifier_receipt",
        "gate": GATE,
        "created_at_utc": utc_now(),
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "analysis_file_sha256": analysis_file_sha,
        "analysis_internal_sha256": analysis["manifest_sha256"],
        "blind12_source_manifest_file_sha256": file_sha256(root / SOURCE_MANIFEST_NAME),
        "blind12_source_manifest_internal_sha256": source["manifest_sha256"],
        "unseal_marker_file_sha256": file_sha256(root / UNSEAL_MARKER_NAME),
        "unseal_count": 1,
        "verification": verification,
        "implementation_sha256": implementation_hashes(),
        "status": "PASS",
    }
    attach_manifest_sha256(verifier)
    write_new_json(root / VERIFIER_NAME, verifier)
    del verifier_baseline_map, verifier_differences
    gc.collect()
    return verifier


__all__ = [
    "ALL_WINDOWS",
    "AUTHORIZED_RUN_ROOT",
    "BLIND_BOTTOM_M",
    "BLIND_TOP_M",
    "BLIND_WINDOWS",
    "P3EError",
    "build_full_manifest",
    "classify_width4",
    "exact_mcnemar_p",
    "holm_step_down_25",
    "joint_paired_bootstrap",
    "prepare_run",
    "query_sacct",
    "record_submission",
    "seal_completed_sources",
    "unseal_analyze_verify",
    "validate_full_manifest",
    "validate_sealed_scheduler_rows",
]
