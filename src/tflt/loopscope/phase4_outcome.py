"""Sealed Phase 4 P4-C outcome acquisition.

This module owns only the frozen eight-cell launcher, an outcome-free smoke
audit, and identity/completion receipts.  It never parses ``results.json``.
P4-D is the only consumer allowed to unseal outcome values.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase4_schema import (
    canonical_json_bytes,
    file_sha256,
    load_phase4_card,
    semantic_sha256,
)


GATE = "P4-C"
EXECUTOR_THREAD_ID = "019f6d14-bf10-73c0-8220-2b5a38fd22b9"
PLANNING_THREAD_ID = "019f6bf0-d1ec-7d53-ba06-391e9db2bd88"
AUTHORIZED_BASE_COMMIT = "ffaa20eaf1c68727b0107ccb1c50ac58001dede2"
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
AUTHORIZED_RUN_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase4-p4c-20260716T223844Z"
)
P4B_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/"
    "runs/phase4-p4b-20260716T204154Z"
)

POLICY_BYTE_SHA256 = "9784d91d48813932feb547436f4d79677aa847e70242e917780bbe4e6064fc73"
CONTROL_BYTE_SHA256 = "a377a348f6df5468a9e564ee7d9de189906bb0d14f16f4efc00918d349208a7c"
CARD_BYTE_SHA256 = "980955386907a1865699808219da1379031c1395d9cdb77029585cf266f160f8"
P4A_VERIFIER_MANIFEST_SHA256 = "a78b58cf1c54eae6133ab0bd1833c641b281be02cf0c3b62ecbb47eb113906a8"
SOURCE_MANIFEST_FILE_SHA256 = "173aeb1f7a8663652975ec4d0faf0998d9a017e9fb4ee30e149f23dc1a2e944f"
SELECTOR_REPORT_FILE_SHA256 = "a941ccb15fafc98062b895918c2d61791d55ef36d62b15794c7f7dcef2639d35"
SELECTOR_FREEZE_FILE_SHA256 = "a4cc48a9b82b3d89b962cdefb55b6e4133c08a0aff3431f0de72c3750fc062c8"
SELECTOR_RECEIPT_FILE_SHA256 = "05fb8184a3ec6b98c354d75635685feec0747ce302e13cb70bcc6c538eebc521"
PANEL_FILE_SHA256 = "02148127ef7b634106b1ee1044f591012d3543563e94344583480a7c3061eb79"
PANEL_MANIFEST_SHA256 = "abc65342e5444638d3073ce780ec37d8d72eb7c34d87ac809bdb29a24056d094"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
RENDERER_SHA256 = "74ab409c4e4c96e4351fbe6122d519f64dd3e11381a676957631e858923cc9fc"
TASK_YAML_SHA256 = "0271e3fdbbb0e8df5b6909786600c1292da29842086b521a1181954941156f94"
DEFAULT_TEMPLATE_SHA256 = "356e937a958288fafd8f07b03da7fd4825977e9bb3d62136931c9f649b600647"
DATASET_INFO_SHA256 = "ce81f7ec00c6701dc737889a3171a9dd32d2521914bc1e5029a3f8bb28df9aab"
TEST_ARROW_SHA256 = "4c741944e3b53719b433be6e7916e4ab9e8ff33f70e9c365614b4e317c0476bb"
VALIDATION_ARROW_SHA256 = "5730c2a9cd07a1ee5f70f4cd1940e43da1df7602c21977b7dc7ef51d496bdf00"

SOURCE_MANIFEST = P4B_ROOT / "source/phase4_shared12032_source_manifest.json"
SOURCE_RECORDS = P4B_ROOT / "source/shared12032.jsonl"
PANEL_PATH = P4B_ROOT / "freeze/phase4_outcome_panel_manifest.json"
SELECTOR_REPORT = P4B_ROOT / "freeze/phase4_selector_report.json"
SELECTOR_FREEZE = P4B_ROOT / "freeze/phase4_selector_freeze.json"
SELECTOR_RECEIPT = P4B_ROOT / "freeze/phase4_selector_verifier_receipt.json"
DATASET_CACHE_SNAPSHOT = Path(
    "/hpc2hdd/home/xhuang225/shared/datasets/TIGER-Lab___mmlu-pro/default/0.0.0/"
    "b189ec765aa7ed75c8acfea42df31fdae71f97be"
)
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")

SMOKE_TASK = "mmlu_pro_business"
FULL_TASK = "mmlu_pro"
BATCH_SIZE = "64"
SEEDS = {
    "random_seed": 0,
    "numpy_random_seed": 1234,
    "torch_random_seed": 1234,
    "fewshot_random_seed": 1234,
}
SMOKE_IDENTITIES: Tuple[str, ...] = (
    "70:business:ori_mmlu-business_ethics:7967613730e252a49314cfb3dfb30134f09b9e55c0b8404c6a3f6a51d51ac5f7",
    "71:business:ori_mmlu-business_ethics:38b3336c68915a29da56adae72fb61b9cf3ce0900286b113ef5968d74f720ed1",
    "72:business:ori_mmlu-business_ethics:6855dca4e7384aeaf6b277ef4b5be07bdb8d660523bde14137cfa1ea4d283db4",
    "73:business:ori_mmlu-business_ethics:c9ce33ffc01b747d15dccadf744f9c5ea48baebed23c541dd4428de2c6a5f039",
    "74:business:ori_mmlu-business_ethics:10ffbd0ff765474d880aefc144418a8e1cdce2c51b749945f32d1f0d2cd85dec",
)

# (cell id, role, window).  The order itself is part of the authorization.
CELLS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("baseline_no_loop", "baseline", None),
    ("fixed_15_18", "fixed_comparator", "15:18"),
    ("blind_high_6_9", "blind_high", "6:9"),
    ("blind_high_10_13", "blind_high", "10:13"),
    ("blind_high_25_28", "blind_high", "25:28"),
    ("blind_low_4_7", "blind_low", "4:7"),
    ("blind_low_5_8", "blind_low", "5:8"),
    ("blind_low_22_25", "blind_low", "22:25"),
)
IMPLEMENTATION_PATHS = (
    "src/tflt/loopscope/phase4_outcome.py",
    "scripts/loopscope/run_qwen4_phase4_p4c.py",
    "tests/test_loopscope_phase4_outcome.py",
)
SMOKE_MANIFEST_NAME = "phase4_p4c_smoke_manifest.json"
SMOKE_SUBMISSION_NAME = "phase4_p4c_smoke_submission.json"
SMOKE_RECEIPT_NAME = "phase4_outcome_smoke_summary.json"
FULL_MANIFEST_NAME = "phase4_p4c_full_launch_manifest.json"
FULL_SUBMISSION_NAME = "phase4_p4c_full_submission.json"
COMPLETION_NAME = "phase4_outcome_completion_receipt.json"


class P4CError(ValueError):
    """Fail-closed P4-C contract or provenance error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise P4CError("git provenance command failed: %s" % completed.stderr.strip())
    return completed.stdout.strip()


def git_provenance(expected_commit: str, *, remote_required: bool) -> Dict[str, Any]:
    root = repository_root().resolve()
    if remote_required and root != REMOTE_REPO:
        raise P4CError("remote action is outside the dedicated LoopScope clone")
    observed = {
        "repo": str(root),
        "branch": _git(root, "symbolic-ref", "--short", "HEAD"),
        "commit": _git(root, "rev-parse", "HEAD"),
        "origin_loopscope": _git(root, "rev-parse", "origin/loopscope"),
        "dirty": bool(_git(root, "status", "--porcelain")),
    }
    if (
        observed["branch"] != "loopscope"
        or observed["commit"] != expected_commit
        or observed["origin_loopscope"] != expected_commit
        or observed["dirty"]
    ):
        raise P4CError("Git provenance is not clean/exact at the expected pushed commit")
    return observed


def implementation_hashes() -> Dict[str, str]:
    root = repository_root()
    hashes = {}
    for relative in IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            raise P4CError("missing P4-C implementation path: %s" % relative)
        hashes[relative] = file_sha256(path)
    return hashes


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise P4CError("expected JSON object: %s" % path)
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def _write_new_text(path: Path, text: str, *, executable: bool = False) -> str:
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if executable:
        Path(path).chmod(Path(path).stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return file_sha256(path)


def _attach_manifest(value: Dict[str, Any]) -> Dict[str, Any]:
    value["manifest_sha256"] = semantic_sha256(value)
    return value


def _verify_manifest(value: Mapping[str, Any]) -> None:
    expected = value.get("manifest_sha256")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if expected != semantic_sha256(body):
        raise P4CError("internal manifest SHA256 mismatch")


def _load_card(card_path: Path) -> Dict[str, Any]:
    path = Path(card_path).resolve()
    if file_sha256(path) != CARD_BYTE_SHA256:
        raise P4CError("frozen Phase 4 card byte hash differs")
    card = load_phase4_card(path)
    receipt = _read_json(path.parent / "phase4_card_verifier_receipt.json")
    if receipt.get("manifest_sha256") != P4A_VERIFIER_MANIFEST_SHA256:
        raise P4CError("P4-A verifier manifest identity differs")
    if remote_required_control_path(path):
        control = path.parents[3] / ".planning/loopscope_phase4_control.md"
        if file_sha256(control) != CONTROL_BYTE_SHA256:
            raise P4CError("Phase 4 control byte hash differs")
    return card


def remote_required_control_path(card_path: Path) -> bool:
    """The mutable control exists beside the local clone, not in the HPC clone."""

    return (card_path.parents[3] / ".planning/loopscope_phase4_control.md").is_file()


def validate_frozen_inputs() -> Dict[str, Any]:
    expected = {
        SOURCE_MANIFEST: SOURCE_MANIFEST_FILE_SHA256,
        SELECTOR_REPORT: SELECTOR_REPORT_FILE_SHA256,
        SELECTOR_FREEZE: SELECTOR_FREEZE_FILE_SHA256,
        SELECTOR_RECEIPT: SELECTOR_RECEIPT_FILE_SHA256,
        PANEL_PATH: PANEL_FILE_SHA256,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise P4CError("frozen P4-B file hash differs: %s" % path.name)
    panel = _read_json(PANEL_PATH)
    if panel.get("manifest_sha256") != PANEL_MANIFEST_SHA256:
        raise P4CError("embedded P4-B panel manifest differs")
    source = _read_json(SOURCE_MANIFEST)
    _verify_manifest(source)
    if source.get("record_count") != 12032 or source.get("dataset_revision") != DATASET_REVISION:
        raise P4CError("P4-B shared source population identity differs")
    return {"panel": panel, "source": source}


def _cell_argv(mode: str, run_root: Path, index: int, expected_commit: str) -> List[str]:
    return [
        "python",
        "scripts/loopscope/run_qwen4_phase4_p4c.py",
        "run-cell",
        "--mode",
        mode,
        "--run-root",
        str(run_root),
        "--cell-index",
        str(index),
        "--expected-commit",
        expected_commit,
        "--batch-size",
        BATCH_SIZE,
    ]


def build_launch_manifest(
    *,
    mode: str,
    run_root: Path,
    expected_commit: str,
    partition: str,
    gres: str,
    qos: Optional[str],
    account: Optional[str],
    array_throttle: int,
    time_limit: str,
    created_at_utc: str,
    hashes: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise P4CError("mode must be smoke or full")
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT:
        raise P4CError("P4-C run root differs from exact authorization")
    if not partition or not gres or not time_limit:
        raise P4CError("scheduler partition/gres/time must be explicit")
    if not isinstance(array_throttle, int) or not 1 <= array_throttle <= 8:
        raise P4CError("array throttle must be in 1..8")
    task = SMOKE_TASK if mode == "smoke" else FULL_TASK
    limit = 5 if mode == "smoke" else None
    cells = []
    for index, (cell_id, role, window) in enumerate(CELLS):
        output_dir = Path(run_root) / mode / ("cell-%02d-%s" % (index, cell_id))
        cells.append(
            {
                "index": index,
                "cell_id": cell_id,
                "role": role,
                "window": window,
                "loop_enabled": window is not None,
                "output_dir": str(output_dir),
                "launcher": "%s/launch/cell-%02d.sh" % (mode, index),
                "argv": _cell_argv(mode, Path(run_root), index, expected_commit),
                "automatic_retry": False,
            }
        )
    value: Dict[str, Any] = {
        "schema_version": "loopscope.phase4.p4c-%s-launch.v1" % mode,
        "artifact_role": "p4c_frozen_eight_cell_%s_launch" % mode,
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "created_at_utc": created_at_utc,
        "run_root": str(Path(run_root)),
        "git": {"branch": "loopscope", "commit": expected_commit, "dirty": False},
        "implementation_sha256": dict(hashes or implementation_hashes()),
        "frozen_inputs": {
            "policy_file_sha256": POLICY_BYTE_SHA256,
            "control_file_sha256": CONTROL_BYTE_SHA256,
            "card_file_sha256": CARD_BYTE_SHA256,
            "p4a_verifier_manifest_sha256": P4A_VERIFIER_MANIFEST_SHA256,
            "p4b_source_manifest_file_sha256": SOURCE_MANIFEST_FILE_SHA256,
            "p4b_selector_report_file_sha256": SELECTOR_REPORT_FILE_SHA256,
            "p4b_selector_freeze_file_sha256": SELECTOR_FREEZE_FILE_SHA256,
            "p4b_selector_receipt_file_sha256": SELECTOR_RECEIPT_FILE_SHA256,
            "p4b_panel_file_sha256": PANEL_FILE_SHA256,
            "p4b_panel_manifest_sha256": PANEL_MANIFEST_SHA256,
        },
        "recipe": {
            "model": "qwen3-4b-instruct-2507",
            "model_repo": "Qwen/Qwen3-4B-Instruct-2507",
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "task": task,
            "logical_task_group": FULL_TASK,
            "dataset_revision": DATASET_REVISION,
            "num_fewshot": 5,
            "apply_chat_template": False,
            "fewshot_as_multiturn": True,
            "generation_kwargs": {
                "until": ["Question:"],
                "max_gen_toks": 2048,
                "do_sample": False,
                "temperature": 0.0,
            },
            "metric": "exact_match,custom-extract",
            "batch_size": BATCH_SIZE,
            "seeds": dict(SEEDS),
            "limit": limit,
            "window_width": 4,
            "k": 3,
            "iteration_mode": "block",
            "strategy": "euler",
            "alpha": 1.0,
            "step_size": 1.0 / 3.0,
            "total_horizon": 1.0,
            "beta": 0.0,
            "cache_strategy": "first",
            "decode_mode": "full",
        },
        "population": {
            "source_manifest_path": str(SOURCE_MANIFEST),
            "source_manifest_file_sha256": SOURCE_MANIFEST_FILE_SHA256,
            "source_records_path": str(SOURCE_RECORDS),
            "required_full_count": 12032,
            "smoke_identities": list(SMOKE_IDENTITIES),
            "smoke_identity_order_sha256": semantic_sha256(list(SMOKE_IDENTITIES)),
        },
        "scheduler": {
            "partition": partition,
            "gres": gres,
            "qos": qos,
            "account": account,
            "array": "0-7%%%d" % array_throttle,
            "array_throttle": array_throttle,
            "cpus_per_task": 8,
            "memory": "64G",
            "time_limit": time_limit,
            "requeue": False,
        },
        "cells": cells,
        "cell_count": 8,
        "information_barrier": {
            "results_parse_authorized": False,
            "samples_parse_authorized": False,
            "completion_stat_sha_only": True,
            "p4d_unseal_authorized": False,
        },
        "outcome_values_consumed": False,
    }
    _attach_manifest(value)
    validate_launch_manifest(value, mode=mode)
    return value


def validate_launch_manifest(value: Mapping[str, Any], *, mode: str) -> None:
    _verify_manifest(value)
    if (
        value.get("gate") != GATE
        or value.get("executor_thread_id") != EXECUTOR_THREAD_ID
        or value.get("run_root") != str(AUTHORIZED_RUN_ROOT)
        or value.get("cell_count") != 8
        or value.get("outcome_values_consumed") is not False
    ):
        raise P4CError("P4-C manifest identity differs")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != 8:
        raise P4CError("P4-C manifest must contain exactly eight cells")
    observed = [
        (cell.get("cell_id"), cell.get("role"), cell.get("window")) for cell in cells
    ]
    if observed != list(CELLS) or [cell.get("index") for cell in cells] != list(range(8)):
        raise P4CError("P4-C cell membership/order differs from the frozen panel")
    recipe = value.get("recipe", {})
    expected = {
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "num_fewshot": 5,
        "batch_size": BATCH_SIZE,
        "k": 3,
        "iteration_mode": "block",
        "strategy": "euler",
        "alpha": 1.0,
        "cache_strategy": "first",
        "decode_mode": "full",
        "window_width": 4,
    }
    if any(recipe.get(key) != item for key, item in expected.items()):
        raise P4CError("P4-C frozen recipe drift")
    if recipe.get("seeds") != SEEDS or recipe.get("apply_chat_template") is not False:
        raise P4CError("P4-C evaluator seed/chat contract drift")
    if mode == "smoke":
        if recipe.get("task") != SMOKE_TASK or recipe.get("limit") != 5:
            raise P4CError("P4-C smoke task/limit differs")
    elif mode == "full":
        if recipe.get("task") != FULL_TASK or recipe.get("limit") is not None:
            raise P4CError("P4-C full task/limit differs")
    else:
        raise P4CError("unknown manifest mode")
    frozen = value.get("frozen_inputs", {})
    expected_hashes = {
        "policy_file_sha256": POLICY_BYTE_SHA256,
        "control_file_sha256": CONTROL_BYTE_SHA256,
        "card_file_sha256": CARD_BYTE_SHA256,
        "p4b_source_manifest_file_sha256": SOURCE_MANIFEST_FILE_SHA256,
        "p4b_panel_file_sha256": PANEL_FILE_SHA256,
        "p4b_panel_manifest_sha256": PANEL_MANIFEST_SHA256,
    }
    if any(frozen.get(key) != item for key, item in expected_hashes.items()):
        raise P4CError("P4-C frozen upstream hash drift")
    for cell in cells:
        window = cell["window"]
        if window is None:
            if cell.get("loop_enabled") is not False:
                raise P4CError("baseline must disable loop")
        else:
            start, end = (int(item) for item in window.split(":"))
            if end - start + 1 != 4 or cell.get("loop_enabled") is not True:
                raise P4CError("variable-width or disabled loop cell is forbidden")


def _common_environment() -> str:
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
export PYTHONHASHSEED=0
export PYTHONPATH={repo}/src
cd {repo}
""".format(repo=shlex.quote(str(REMOTE_REPO)))


def _launcher_text(cell: Mapping[str, Any]) -> str:
    output = shlex.quote(str(cell["output_dir"]))
    command = shlex.join([str(AUDITED_VENV / "bin/python"), *cell["argv"][1:]])
    return (
        "#!/usr/bin/env bash\n"
        + _common_environment()
        + "if [[ -e %s ]]; then echo 'refusing existing P4-C cell output' >&2; exit 73; fi\n" % output
        + "exec %s\n" % command
    )


def _array_runner_text(mode: str) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", 'case "${SLURM_ARRAY_TASK_ID:?}" in']
    for index in range(8):
        path = AUTHORIZED_RUN_ROOT / mode / "launch" / ("cell-%02d.sh" % index)
        lines.append("  %d) exec %s ;;" % (index, shlex.quote(str(path))))
    lines.extend(["  *) echo 'invalid P4-C array task id' >&2; exit 64 ;;", "esac", ""])
    return "\n".join(lines)


def _submit_text(mode: str, manifest: Mapping[str, Any]) -> str:
    scheduler = manifest["scheduler"]
    args = [
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--job-name=loopscope-p4c-%s" % mode,
        "--partition=%s" % scheduler["partition"],
        "--gres=%s" % scheduler["gres"],
        "--cpus-per-task=%d" % scheduler["cpus_per_task"],
        "--mem=%s" % scheduler["memory"],
        "--time=%s" % scheduler["time_limit"],
        "--array=%s" % scheduler["array"],
        "--no-requeue",
        "--output=%s" % (AUTHORIZED_RUN_ROOT / mode / ("slurm/p4c-%s-%%A_%%a.out" % mode)),
        "--error=%s" % (AUTHORIZED_RUN_ROOT / mode / ("slurm/p4c-%s-%%A_%%a.err" % mode)),
    ]
    if scheduler.get("qos"):
        args.append("--qos=%s" % scheduler["qos"])
    if scheduler.get("account"):
        args.append("--account=%s" % scheduler["account"])
    args.append(str(AUTHORIZED_RUN_ROOT / mode / "launch/array_runner.sh"))
    return "#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n" % shlex.join(args)


def _materialize_launch(mode: str, manifest: Mapping[str, Any]) -> Dict[str, str]:
    root = AUTHORIZED_RUN_ROOT / mode
    for name in ("launch", "slurm"):
        (root / name).mkdir(parents=True, exist_ok=False)
    hashes = {}
    manifest_name = SMOKE_MANIFEST_NAME if mode == "smoke" else FULL_MANIFEST_NAME
    hashes[manifest_name] = _write_new_json(root / manifest_name, manifest)
    for cell in manifest["cells"]:
        relative = Path(cell["launcher"]).relative_to(mode)
        hashes[str(relative)] = _write_new_text(root / relative, _launcher_text(cell), executable=True)
    hashes["launch/array_runner.sh"] = _write_new_text(
        root / "launch/array_runner.sh", _array_runner_text(mode), executable=True
    )
    hashes["launch/submit.sh"] = _write_new_text(
        root / "launch/submit.sh", _submit_text(mode, manifest), executable=True
    )
    return hashes


def prepare_smoke(
    *, card_path: Path, run_root: Path, expected_commit: str, partition: str, gres: str,
    qos: Optional[str], account: Optional[str], array_throttle: int, time_limit: str,
) -> Dict[str, Any]:
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT or Path(run_root).exists():
        raise P4CError("authorized P4-C write-once root is wrong or already exists")
    git = git_provenance(expected_commit, remote_required=True)
    if file_sha256(REMOTE_REPO / "AGENTS.md") != POLICY_BYTE_SHA256:
        raise P4CError("remote AGENTS policy hash differs")
    _load_card(card_path)
    validate_frozen_inputs()
    manifest = build_launch_manifest(
        mode="smoke", run_root=run_root, expected_commit=expected_commit,
        partition=partition, gres=gres, qos=qos, account=account,
        array_throttle=array_throttle, time_limit=time_limit,
        created_at_utc=utc_now(),
    )
    Path(run_root).mkdir(parents=False, exist_ok=False)
    hashes = _materialize_launch("smoke", manifest)
    freeze = _attach_manifest({
        "schema_version": "loopscope.phase4.p4c-smoke-freeze.v1",
        "gate": GATE, "created_at_utc": utc_now(), "git": git,
        "smoke_manifest_file_sha256": hashes[SMOKE_MANIFEST_NAME],
        "smoke_manifest_internal_sha256": manifest["manifest_sha256"],
        "launcher_sha256": hashes, "cell_count": 8,
        "outcome_values_consumed": False, "status": "SMOKE_PREPARED",
    })
    _write_new_json(Path(run_root) / "phase4_p4c_preoutcome_freeze.json", freeze)
    return freeze


def freeze_full(
    *, run_root: Path, expected_commit: str, partition: str, gres: str,
    qos: Optional[str], account: Optional[str], array_throttle: int, time_limit: str,
) -> Dict[str, Any]:
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT:
        raise P4CError("full freeze root differs from authorization")
    git_provenance(expected_commit, remote_required=True)
    smoke = _read_json(Path(run_root) / SMOKE_RECEIPT_NAME)
    _verify_manifest(smoke)
    if smoke.get("status") != "PASS" or smoke.get("cell_count") != 8:
        raise P4CError("full launch requires exact eight-cell smoke PASS")
    if (Path(run_root) / "full").exists():
        raise FileExistsError("full launch is already frozen")
    validate_frozen_inputs()
    manifest = build_launch_manifest(
        mode="full", run_root=run_root, expected_commit=expected_commit,
        partition=partition, gres=gres, qos=qos, account=account,
        array_throttle=array_throttle, time_limit=time_limit,
        created_at_utc=utc_now(),
    )
    hashes = _materialize_launch("full", manifest)
    freeze = _attach_manifest({
        "schema_version": "loopscope.phase4.p4c-full-freeze.v1",
        "gate": GATE, "created_at_utc": utc_now(),
        "full_manifest_file_sha256": hashes[FULL_MANIFEST_NAME],
        "full_manifest_internal_sha256": manifest["manifest_sha256"],
        "smoke_receipt_file_sha256": file_sha256(Path(run_root) / SMOKE_RECEIPT_NAME),
        "launcher_sha256": hashes, "batch_size": BATCH_SIZE,
        "cell_count": 8, "outcome_values_consumed": False,
        "status": "FULL_FROZEN",
    })
    _write_new_json(Path(run_root) / "phase4_p4c_full_freeze.json", freeze)
    return freeze


class CountOnlyCollector:
    """Wrapper audit sink that persists no tensors, logits, tokens, or responses."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def record(self, event: str, payload: Mapping[str, Any]) -> None:
        self.counts[event] += 1
        if event == "wrapper_forward" and payload.get("bypass"):
            self.counts["bypass_true"] += 1

    def record_tensor_diff(self, name: str, before: Any, after: Any) -> None:
        return None


def _assert_runtime_files() -> None:
    from lm_eval.tasks.mmlu_pro import utils

    checks = {
        Path(utils.__file__).resolve(): RENDERER_SHA256,
        Path(utils.__file__).resolve().parent / "_mmlu_pro.yaml": TASK_YAML_SHA256,
        Path(utils.__file__).resolve().parent / "_default_template_yaml": DEFAULT_TEMPLATE_SHA256,
        DATASET_CACHE_SNAPSHOT / "dataset_info.json": DATASET_INFO_SHA256,
        DATASET_CACHE_SNAPSHOT / "mmlu-pro-test.arrow": TEST_ARROW_SHA256,
        DATASET_CACHE_SNAPSHOT / "mmlu-pro-validation.arrow": VALIDATION_ARROW_SHA256,
    }
    for path, expected in checks.items():
        if file_sha256(path) != expected:
            raise P4CError("runtime task/dataset hash differs: %s" % path.name)


def _first_smoke_sources() -> List[Dict[str, Any]]:
    records = []
    with SOURCE_RECORDS.open("r", encoding="utf-8") as handle:
        for _ in range(5):
            line = handle.readline()
            if not line:
                raise P4CError("shared source has fewer than five rows")
            records.append(json.loads(line))
    if tuple(row.get("canonical_identity") for row in records) != SMOKE_IDENTITIES:
        raise P4CError("first five canonical smoke identities/order differ")
    if any(row.get("category") != "business" for row in records):
        raise P4CError("frozen five-row smoke subset is no longer the business task prefix")
    return records


def _safe_target(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: row[key]
        for key in ("question_id", "category", "src", "question", "ordered_options")
    }


def _identity_sidecar(mode: str, manifest: Mapping[str, Any], cell: Mapping[str, Any]) -> Dict[str, Any]:
    return _attach_manifest({
        "schema_version": "loopscope.phase4.p4c-cell-identity.v1",
        "gate": GATE, "mode": mode, "cell_id": cell["cell_id"],
        "role": cell["role"], "window": cell["window"],
        "loop_enabled": cell["loop_enabled"],
        "source_manifest_path": str(SOURCE_MANIFEST),
        "source_manifest_file_sha256": SOURCE_MANIFEST_FILE_SHA256,
        "source_records_path": str(SOURCE_RECORDS),
        "source_records_file_sha256": manifest["population"]["source_records_file_sha256"],
        "expected_population_count": 5 if mode == "smoke" else 12032,
        "expected_smoke_identities": list(SMOKE_IDENTITIES) if mode == "smoke" else None,
        "recipe": manifest["recipe"],
        "outcome_values_consumed": False,
    })


def run_cell(
    *, mode: str, run_root: Path, cell_index: int, expected_commit: str, batch_size: str
) -> None:
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT or expected_commit == AUTHORIZED_BASE_COMMIT:
        raise P4CError("run-cell requires the post-implementation authorized commit/root")
    if batch_size != BATCH_SIZE or cell_index not in range(8):
        raise P4CError("run-cell batch/index differs from frozen launch")
    git = git_provenance(expected_commit, remote_required=True)
    manifest_path = Path(run_root) / mode / (
        SMOKE_MANIFEST_NAME if mode == "smoke" else FULL_MANIFEST_NAME
    )
    manifest = _read_json(manifest_path)
    validate_launch_manifest(manifest, mode=mode)
    cell = manifest["cells"][cell_index]
    output_dir = Path(cell["output_dir"])
    output_dir.mkdir(parents=False, exist_ok=False)
    _assert_runtime_files()
    frozen = validate_frozen_inputs()
    source_records_sha = file_sha256(SOURCE_RECORDS)
    if source_records_sha != frozen["source"].get("record_file_sha256"):
        raise P4CError("shared source record file hash differs")
    manifest["population"]["source_records_file_sha256"] = source_records_sha

    command_args = {
        "mode": mode, "cell_index": cell_index, "cell_id": cell["cell_id"],
        "role": cell["role"], "window": cell["window"],
        "loop_enabled": cell["loop_enabled"], "batch_size": batch_size,
        "recipe": manifest["recipe"], "git": git,
        "launch_manifest_sha256": manifest["manifest_sha256"],
    }
    _write_new_json(output_dir / "command_args.json", command_args)
    _write_new_json(output_dir / "env.json", {
        key: os.environ.get(key)
        for key in (
            "HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "CUDA_VISIBLE_DEVICES",
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "PYTHONHASHSEED",
        )
    })
    identity = _identity_sidecar(mode, manifest, cell)
    _write_new_json(output_dir / "identity_receipt.json", identity)

    try:
        import torch
        from datasets import DownloadMode, load_dataset
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
        from lm_eval.tasks.mmlu_pro import utils as renderer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tflt.config import LoopConfig
        from tflt.loopscope.phase4_runtime import (
            render_exact_prefix, tokenization_metadata, validation_demos_by_category,
        )
        from tflt.loopscope.revisions import (
            load_tokenizer_with_resolved_commit, strict_revision_closure,
        )
        from tflt.wrapper import apply_loop_wrapper
    except Exception as exc:
        raise P4CError("audited P4-C runtime imports failed") from exc
    if not torch.cuda.is_available():
        raise P4CError("P4-C cell requires a visible CUDA GPU")

    load_kwargs = {
        "revision": MODEL_REVISION, "local_files_only": True, "trust_remote_code": True,
    }
    tokenizer, tokenizer_commit = load_tokenizer_with_resolved_commit(
        AutoTokenizer, "Qwen/Qwen3-4B-Instruct-2507", load_kwargs
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", torch_dtype=torch.bfloat16, **load_kwargs
    ).to("cuda")
    model.eval()
    closure = strict_revision_closure(model, tokenizer_commit, MODEL_REVISION)
    _write_new_json(output_dir / "model_revision.json", {
        "schema_version": "loopscope.model-tokenizer-revision.v1",
        "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
        "model_commit": closure["model_commit"],
        "tokenizer_commit": closure["tokenizer_commit"],
        "manifest_commit": closure["manifest_commit"], "match": closure["match"],
    })

    collector = CountOnlyCollector()
    if cell["loop_enabled"]:
        config = LoopConfig.from_window_string(
            model_alias="qwen3-4b-instruct-2507", window=cell["window"], k=3,
            iteration_mode="block", strategy="euler", alpha=1.0, beta=0.0,
            cache_strategy="first", decode_mode="full",
            audit_collector=collector if mode == "smoke" else None,
        )
        apply_loop_wrapper(model, config)

    if mode == "smoke":
        validation = load_dataset(
            "TIGER-Lab/MMLU-Pro", revision=DATASET_REVISION, split="validation",
            cache_dir=str(HF_DATASETS_CACHE),
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        )
        demos = validation_demos_by_category(validation)
        audit_rows = []
        for source in _first_smoke_sources():
            prefix = render_exact_prefix(_safe_target(source), demos[source["category"]], renderer)
            metadata = tokenization_metadata(tokenizer, prefix)
            for key in (
                "rendered_prefix_sha256", "rendered_token_ids_sha256",
                "attention_mask_sha256", "last_effective_prefix_token_index",
            ):
                if metadata[key] != source[key]:
                    raise P4CError("smoke prefix/source tokenization drift at %s" % key)
            before = dict(collector.counts)
            encoded = tokenizer(prefix, return_tensors="pt")
            encoded = {key: item.to("cuda") for key, item in encoded.items()}
            with torch.inference_mode():
                produced = model(**encoded, use_cache=True, logits_to_keep=1)
            del produced, encoded
            after = dict(collector.counts)
            body = after.get("body_call", 0) - before.get("body_call", 0)
            wrapper = after.get("wrapper_forward", 0) - before.get("wrapper_forward", 0)
            stash = after.get("stash_pass", 0) - before.get("stash_pass", 0)
            if cell["loop_enabled"]:
                if (body, wrapper, stash) != (3, 1, 1):
                    raise P4CError("loop smoke audit did not observe exactly 3/1/1 calls")
            elif (body, wrapper, stash) != (0, 0, 0):
                raise P4CError("baseline smoke unexpectedly observed loop activity")
            audit_rows.append({
                "canonical_identity": source["canonical_identity"],
                "loop_enabled": cell["loop_enabled"],
                "operator_body_calls": body, "wrapper_forward_calls": wrapper,
                "stash_passes": stash,
            })
        audit = _attach_manifest({
            "schema_version": "loopscope.phase4.p4c-loop-audit.v1",
            "gate": GATE, "cell_id": cell["cell_id"], "window": cell["window"],
            "decode_mode": "full", "k": 3, "rows": audit_rows,
            "same_five_canonical_identities": [row["canonical_identity"] for row in audit_rows]
            == list(SMOKE_IDENTITIES),
            "outcome_values_consumed": False,
        })
        _write_new_json(output_dir / "loop_audit_receipt.json", audit)

    lm = HFLM(pretrained=model, tokenizer=tokenizer, trust_remote_code=True, batch_size=batch_size)
    result = evaluator.simple_evaluate(
        model=lm,
        tasks=[manifest["recipe"]["task"]],
        num_fewshot=5,
        batch_size=batch_size,
        limit=manifest["recipe"]["limit"],
        log_samples=True,
        apply_chat_template=False,
        fewshot_as_multiturn=True,
        task_manager=TaskManager(),
        random_seed=SEEDS["random_seed"],
        numpy_random_seed=SEEDS["numpy_random_seed"],
        torch_random_seed=SEEDS["torch_random_seed"],
        fewshot_random_seed=SEEDS["fewshot_random_seed"],
    )
    results_path = output_dir / "results.json"
    with results_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    metadata = _file_metadata(results_path)
    completion = _attach_manifest({
        "schema_version": "loopscope.phase4.p4c-cell-completion.v1",
        "gate": GATE, "mode": mode, "cell_id": cell["cell_id"],
        "result": metadata, "identity_receipt_sha256": file_sha256(output_dir / "identity_receipt.json"),
        "command_args_sha256": file_sha256(output_dir / "command_args.json"),
        "model_revision_sha256": file_sha256(output_dir / "model_revision.json"),
        "loop_audit_receipt_sha256": (
            file_sha256(output_dir / "loop_audit_receipt.json") if mode == "smoke" else None
        ),
        "exit_state": "PRODUCER_COMPLETED", "outcome_values_consumed": False,
    })
    _write_new_json(output_dir / "producer_completion.json", completion)


def _file_metadata(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise P4CError("sealed result is absent, empty, or symlinked")
    return {"path": str(path), "size_bytes": path.stat().st_size, "file_sha256": file_sha256(path)}


SACCT_FIELDS = (
    "JobID", "JobIDRaw", "State", "ExitCode", "Partition", "NodeList",
    "ElapsedRaw", "AllocTRES", "Submit", "Start", "End",
)


def query_sacct(job_id: str) -> List[Dict[str, str]]:
    completed = subprocess.run(
        ["/opt/slurm/bin/sacct", "-j", str(job_id), "--array", "--noheader", "--parsable2",
         "-o", ",".join(SACCT_FIELDS)],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise P4CError("sacct failed: %s" % completed.stderr.strip())
    rows = []
    for line in completed.stdout.splitlines():
        values = line.split("|")
        if values and values[-1] == "":
            values.pop()
        if len(values) == len(SACCT_FIELDS):
            rows.append(dict(zip(SACCT_FIELDS, values)))
    return rows


def validate_scheduler_rows(rows: Sequence[Mapping[str, Any]], job_id: str) -> List[Dict[str, str]]:
    pattern = re.compile(r"^%s_([0-7])$" % re.escape(str(job_id)))
    tasks: Dict[int, Dict[str, str]] = {}
    for raw in rows:
        match = pattern.fullmatch(str(raw.get("JobID", "")))
        if match:
            index = int(match.group(1))
            if index in tasks:
                raise P4CError("duplicate scheduler array row")
            tasks[index] = {key: str(raw.get(key, "")) for key in SACCT_FIELDS}
    if set(tasks) != set(range(8)):
        raise P4CError("sealed completion requires exact eight scheduler tasks")
    for index, row in tasks.items():
        if row["State"].split()[0].rstrip("+") != "COMPLETED" or row["ExitCode"] != "0:0":
            raise P4CError("task %d is not COMPLETED/0:0" % index)
    return [tasks[index] for index in range(8)]


def record_submission(
    *, mode: str, run_root: Path, expected_commit: str, job_id: str
) -> Dict[str, Any]:
    if not re.fullmatch(r"[0-9]+", str(job_id)):
        raise P4CError("Slurm job id must be decimal")
    git = git_provenance(expected_commit, remote_required=True)
    root = Path(run_root) / mode
    manifest_name = SMOKE_MANIFEST_NAME if mode == "smoke" else FULL_MANIFEST_NAME
    submission_name = SMOKE_SUBMISSION_NAME if mode == "smoke" else FULL_SUBMISSION_NAME
    manifest = _read_json(root / manifest_name)
    validate_launch_manifest(manifest, mode=mode)
    receipt = _attach_manifest({
        "schema_version": "loopscope.phase4.p4c-%s-submission.v1" % mode,
        "gate": GATE, "mode": mode, "created_at_utc": utc_now(), "git": git,
        "job_id": str(job_id), "array": manifest["scheduler"]["array"],
        "scheduler": manifest["scheduler"], "cell_count": 8,
        "launch_manifest_file_sha256": file_sha256(root / manifest_name),
        "launch_manifest_internal_sha256": manifest["manifest_sha256"],
        "automatic_retry": False, "outcome_values_consumed": False,
    })
    _write_new_json(root / submission_name, receipt)
    return receipt


def _close_cells(mode: str, manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    closed = []
    for cell in manifest["cells"]:
        root = Path(cell["output_dir"])
        completion = _read_json(root / "producer_completion.json")
        _verify_manifest(completion)
        metadata = _file_metadata(root / "results.json")
        if completion.get("result") != metadata or completion.get("cell_id") != cell["cell_id"]:
            raise P4CError("sealed producer completion/result identity differs")
        identity = _read_json(root / "identity_receipt.json")
        _verify_manifest(identity)
        if identity.get("cell_id") != cell["cell_id"] or identity.get("recipe") != manifest["recipe"]:
            raise P4CError("cell identity receipt differs from frozen launch")
        audit_meta = None
        if mode == "smoke":
            audit = _read_json(root / "loop_audit_receipt.json")
            _verify_manifest(audit)
            rows = audit.get("rows")
            if not isinstance(rows, list) or len(rows) != 5:
                raise P4CError("smoke audit must close exact five identities")
            expected_body = 0 if cell["window"] is None else 3
            if any(row.get("operator_body_calls") != expected_body for row in rows):
                raise P4CError("smoke operator body-call closure differs")
            audit_meta = {
                "path": str(root / "loop_audit_receipt.json"),
                "file_sha256": file_sha256(root / "loop_audit_receipt.json"),
                "operator_body_calls_per_identity": [row["operator_body_calls"] for row in rows],
            }
        closed.append({
            "index": cell["index"], "cell_id": cell["cell_id"], "role": cell["role"],
            "window": cell["window"], "loop_enabled": cell["loop_enabled"],
            "argv": cell["argv"], "identity_receipt_path": str(root / "identity_receipt.json"),
            "identity_receipt_sha256": file_sha256(root / "identity_receipt.json"),
            "command_args_path": str(root / "command_args.json"),
            "command_args_sha256": file_sha256(root / "command_args.json"),
            "model_revision_path": str(root / "model_revision.json"),
            "model_revision_sha256": file_sha256(root / "model_revision.json"),
            "result": metadata, "audit": audit_meta, "completion_state": "COMPLETED_SEALED",
        })
    return closed


def seal_stage(
    *, mode: str, run_root: Path, expected_commit: str, job_id: str
) -> Dict[str, Any]:
    if Path(run_root).resolve() != AUTHORIZED_RUN_ROOT:
        raise P4CError("seal root differs from authorization")
    git = git_provenance(expected_commit, remote_required=True)
    root = Path(run_root) / mode
    manifest_name = SMOKE_MANIFEST_NAME if mode == "smoke" else FULL_MANIFEST_NAME
    submission_name = SMOKE_SUBMISSION_NAME if mode == "smoke" else FULL_SUBMISSION_NAME
    output_name = SMOKE_RECEIPT_NAME if mode == "smoke" else COMPLETION_NAME
    output_path = Path(run_root) / output_name
    if output_path.exists():
        raise FileExistsError("sealed stage receipt already exists")
    manifest = _read_json(root / manifest_name)
    validate_launch_manifest(manifest, mode=mode)
    submission = _read_json(root / submission_name)
    _verify_manifest(submission)
    if submission.get("job_id") != str(job_id):
        raise P4CError("submission/job identity differs")
    scheduler_rows = validate_scheduler_rows(query_sacct(job_id), job_id)
    cells = _close_cells(mode, manifest)
    receipt = _attach_manifest({
        "schema_version": "loopscope.phase4.p4c-%s-completion.v1" % mode,
        "artifact_role": "p4c_%s_identity_and_stat_sha_completion" % mode,
        "gate": GATE, "mode": mode, "created_at_utc": utc_now(), "git": git,
        "job_id": str(job_id), "scheduler_rows": scheduler_rows,
        "scheduler": manifest["scheduler"], "run_root": str(run_root),
        "launch_manifest_path": str(root / manifest_name),
        "launch_manifest_file_sha256": file_sha256(root / manifest_name),
        "launch_manifest_internal_sha256": manifest["manifest_sha256"],
        "source_manifest_file_sha256": SOURCE_MANIFEST_FILE_SHA256,
        "panel_file_sha256": PANEL_FILE_SHA256,
        "panel_manifest_sha256": PANEL_MANIFEST_SHA256,
        "recipe": manifest["recipe"], "cells": cells, "cell_count": 8,
        "all_terminal_exit_zero": True, "all_results_nonempty_stat_sha_closed": True,
        "results_json_parsed": False, "samples_parsed": False,
        "outcome_values_consumed": False, "p4d_unseal_performed": False,
        "status": "PASS" if mode == "smoke" else "SEALED_COMPLETE",
    })
    _write_new_json(output_path, receipt)
    return receipt


__all__ = [
    "AUTHORIZED_RUN_ROOT", "BATCH_SIZE", "CELLS", "P4CError", "SEEDS",
    "SMOKE_IDENTITIES", "build_launch_manifest", "freeze_full", "prepare_smoke",
    "record_submission", "run_cell", "seal_stage", "validate_launch_manifest",
]
