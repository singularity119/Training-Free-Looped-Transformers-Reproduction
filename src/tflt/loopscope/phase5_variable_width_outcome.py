"""Sealed acquisition and combined one-shot analysis for LoopScope Phase 5 Gate F.

The original four variable-width cells and supplemental ``15:19`` cell are
derived from the exact Gate E score artifact. Before both formal structural
seals close, this module only projects canonical sample identities and hashes
evaluator payloads. Outcome parsing is confined to the combined one-shot
analysis after exact (4 + 1) x 14,042 closure.
"""

from __future__ import annotations

import csv
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tflt.loopscope.analysis import AnalysisError, extract_accuracy, extract_correctness_samples
from tflt.loopscope.phase3_schema import canonical_json_bytes, file_sha256, ordered_identity_sha256
from tflt.loopscope.phase5_outcome import (
    AUDITED_VENV,
    DATASET_REVISION,
    MODEL_REPO,
    MODEL_REVISION,
    REMOTE_REPO,
    SMOKE_IDENTITIES,
    TEST_MANIFEST_FILE_SHA256,
    TEST_MANIFEST_INTERNAL_SHA256,
    TEST_MANIFEST_PATH,
    TEST_METADATA_FILE_SHA256,
    TEST_METADATA_PATH,
    TEST_ORDERED_IDENTITY_SHA256,
    WORKSPACE,
    _load_test_metadata,
    _smoke_prompts,
    _validate_loop_audit,
    project_identity_only,
)


GATE = "F"
EXECUTOR_THREAD_ID = "019f8eb5-2f08-7c80-afc2-b4ae52464162"
PLANNING_THREAD_ID = "019f8604-4717-7be2-8bf8-9d4a26a3d7f7"
REMOTE_SOURCE_COMMIT = "f33a37a4e5c0cfc461184a3471277897c77cf83b"
RUN_PREFIX = "phase5-gate-f-multiwidth-top4-outcome-"
SUPPLEMENT_RUN_PREFIX = "phase5-gate-f-supplement-15-19-outcome-"
COMBINED_RUN_PREFIX = "phase5-gate-f-combined-top4-plus-15-19-"
CARD_RELATIVE = "configs/loopscope/phase5_multiwidth_top4_outcome_card.json"
IMPLEMENTATION_PATHS = (
    CARD_RELATIVE,
    "src/tflt/loopscope/phase5_variable_width_outcome.py",
    "scripts/loopscope/run_qwen4base_phase5_gate_f.py",
    "tests/test_loopscope_phase5_gate_f.py",
)
MANIFEST_NAMES = {
    "smoke": "phase5_gate_f_smoke_launch_manifest.json",
    "formal": "phase5_gate_f_formal_launch_manifest.json",
}
SUBMISSION_NAMES = {
    "smoke": "phase5_gate_f_smoke_submission.json",
    "formal": "phase5_gate_f_formal_submission.json",
}
SEAL_NAMES = {
    "smoke": "phase5_gate_f_smoke_seal.json",
    "formal": "phase5_gate_f_formal_completion_seal.json",
}
SACCT_FIELDS = (
    "JobID",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "AllocTRES",
    "Partition",
    "NodeList",
)
OUTPUT_NAMES = {
    "unseal": "phase5_gate_f_unseal_once.json",
    "analysis": "phase5_gate_f_analysis.json",
    "table": "phase5_gate_f_window_metrics.csv",
    "bootstrap": "phase5_gate_f_bootstrap_digest.json",
    "verifier": "phase5_gate_f_verifier_receipt.json",
    "summary": "phase5_gate_f_summary_zh.md",
    "terminal": "phase5_gate_f_terminal_receipt.json",
}
SUPPLEMENT_MANIFEST = "phase5_gate_f_supplement_formal_launch_manifest.json"
SUPPLEMENT_SUBMISSION = "phase5_gate_f_supplement_formal_submission.json"
SUPPLEMENT_SEAL = "phase5_gate_f_supplement_formal_completion_seal.json"


class Phase5GateFError(ValueError):
    """Fail-closed Gate F contract, provenance, barrier, or analysis error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _strict_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=lambda item: (_ for _ in ()).throw(
                Phase5GateFError("non-finite JSON constant: %s" % item)
            ),
        )
    if not isinstance(value, dict):
        raise Phase5GateFError("expected JSON object: %s" % path)
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    body = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(body).hexdigest()


def _write_new_text(path: Path, value: str, *, executable: bool = False) -> str:
    body = value.encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return hashlib.sha256(body).hexdigest()


def _attach_manifest(value: Dict[str, Any]) -> Dict[str, Any]:
    value["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def _verify_manifest(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    expected = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if value.get("manifest_sha256") != expected:
        raise Phase5GateFError("internal manifest SHA256 mismatch")


def _run_root(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.parent != (WORKSPACE / "runs") or not resolved.name.startswith(RUN_PREFIX):
        raise Phase5GateFError("Gate F run root is outside the authorized fresh-root prefix")
    if not re.fullmatch(r"phase5-gate-f-multiwidth-top4-outcome-[0-9]{8}T[0-9]{6}Z", resolved.name):
        raise Phase5GateFError("Gate F run root timestamp format differs")
    return resolved


def _supplement_run_root(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.parent != (WORKSPACE / "runs"):
        raise Phase5GateFError("Gate F supplement root is outside the authorized runs root")
    if not re.fullmatch(
        r"phase5-gate-f-supplement-15-19-outcome-[0-9]{8}T[0-9]{6}Z",
        resolved.name,
    ):
        raise Phase5GateFError("Gate F supplement root timestamp/prefix differs")
    return resolved


def _combined_run_root(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.parent != (WORKSPACE / "runs"):
        raise Phase5GateFError("Gate F combined root is outside the authorized runs root")
    if not re.fullmatch(
        r"phase5-gate-f-combined-top4-plus-15-19-[0-9]{8}T[0-9]{6}Z",
        resolved.name,
    ):
        raise Phase5GateFError("Gate F combined root timestamp/prefix differs")
    return resolved


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise Phase5GateFError("git provenance command failed: %s" % completed.stderr.strip())
    return completed.stdout.strip()


def remote_source_provenance(expected_commit: str) -> Dict[str, Any]:
    if expected_commit != REMOTE_SOURCE_COMMIT:
        raise Phase5GateFError("remote source commit differs from frozen Gate F source")
    observed = {
        "repo": str(REMOTE_REPO),
        "branch": _git(REMOTE_REPO, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git(REMOTE_REPO, "rev-parse", "HEAD"),
        "origin_loopscope": _git(REMOTE_REPO, "rev-parse", "origin/loopscope"),
        "dirty": bool(_git(REMOTE_REPO, "status", "--porcelain")),
    }
    if (
        observed["branch"] != "loopscope"
        or observed["commit"] != expected_commit
        or observed["origin_loopscope"] != expected_commit
        or observed["dirty"]
    ):
        raise Phase5GateFError("immutable HPC2 source checkout provenance differs")
    return observed


def implementation_hashes() -> Dict[str, str]:
    root = repository_root()
    result: Dict[str, str] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            raise Phase5GateFError("missing staged Gate F path: %s" % relative)
        result[relative] = file_sha256(path)
    return result


def validate_extension_card(card: Mapping[str, Any]) -> None:
    if (
        card.get("schema_version") != "loopscope.phase5.gate-f-card.v1"
        or card.get("card")
        != "PHASE5_POST_TERMINAL_MULTIWIDTH_TOP4_OUTCOME_EXPLORATORY_V1"
        or card.get("status") != "PLANNING_FROZEN_GATE_F"
        or card.get("gate") != GATE
        or card.get("executor_thread_id") != EXECUTOR_THREAD_ID
        or card.get("planning_thread_id") != PLANNING_THREAD_ID
    ):
        raise Phase5GateFError("Gate F card identity differs")
    cells = card.get("frozen_cells")
    if not isinstance(cells, list) or len(cells) != 4:
        raise Phase5GateFError("Gate F card requires exactly four cells")
    if [row.get("array_index") for row in cells] != list(range(4)):
        raise Phase5GateFError("Gate F array order differs")
    expected_windows = [(1, 5, 13, "13:17"), (2, 6, 15, "15:20"), (3, 6, 12, "12:17"), (4, 3, 15, "15:17")]
    observed = [
        (row.get("gate_e_rank"), row.get("width"), row.get("start"), row.get("window"))
        for row in cells
    ]
    if observed != expected_windows:
        raise Phase5GateFError("Gate F card membership differs")
    supplemental = card.get("supplemental_cell")
    if not isinstance(supplemental, dict):
        raise Phase5GateFError("Gate F supplemental cell is absent")
    expected_supplemental = {
        "array_index": 0,
        "gate_e_rank": 11,
        "width_rank": 2,
        "role": "gate_e_rank11_supplement",
        "width": 5,
        "start": 15,
        "window": "15:19",
        "score": 3.067738720279154,
    }
    if canonical_json_bytes(supplemental) != canonical_json_bytes(expected_supplemental):
        raise Phase5GateFError("Gate F supplemental cell differs")
    recipe = card.get("recipe", {})
    required_recipe = {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "dataset": "cais/mmlu",
        "dataset_revision": DATASET_REVISION,
        "task_group": "mmlu",
        "split": "test",
        "num_fewshot": 5,
        "lm_eval_version": "0.4.11",
        "metric": "acc,none",
        "k": 3,
        "operator_body_calls_per_prefill": 3,
        "iteration_mode": "block",
        "strategy_cli": "euler",
        "strategy_kernel": "damped_euler_alias",
        "alpha": 1.0,
        "beta": 0.0,
        "cache_strategy": "first",
        "decode_mode": "bypass",
        "batch_size": 16,
    }
    if any(recipe.get(key) != value for key, value in required_recipe.items()):
        raise Phase5GateFError("Gate F card recipe drift")
    if (
        recipe.get("step_size") != 1.0 / 3.0
        or recipe.get("total_horizon") != 1.0
        or card.get("population", {}).get("record_count") != 14042
        or card.get("population", {}).get("subject_count") != 57
        or card.get("population", {}).get("ordered_identity_sha256")
        != TEST_ORDERED_IDENTITY_SHA256
        or card.get("bootstrap", {}).get("replicates") != 2000
        or card.get("bootstrap", {}).get("seed") != 20260723
        or card.get("bootstrap", {}).get(
            "shared_draw_stream_across_baseline_and_five_cells"
        )
        is not True
        or card.get("claim_boundary", {}).get("prospective_selected_window_success_forbidden")
        is not True
    ):
        raise Phase5GateFError("Gate F card population/bootstrap/claim boundary drift")


def derive_top4(score_payload: Mapping[str, Any], card: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Derive, then verify, the exact Gate E global ranks 1-4."""

    if (
        score_payload.get("manifest_sha256")
        != card["source"]["gate_e_score_manifest_sha256"]
        or score_payload.get("outcome_values_consumed") is not False
        or score_payload.get("test_split_consumed") is not False
        or score_payload.get("post_terminal_exploratory_no_selection") is not True
    ):
        raise Phase5GateFError("Gate E score artifact contract differs")
    rows = score_payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 42:
        raise Phase5GateFError("Gate E score artifact must contain 42 rows")
    ranked = sorted(rows, key=lambda row: int(row.get("global_rank", -1)))
    if [row.get("global_rank") for row in ranked] != list(range(1, 43)):
        raise Phase5GateFError("Gate E global ranking is not a complete permutation")
    derived = []
    for array_index, row in enumerate(ranked[:4]):
        start = int(row["start"])
        width = int(row["width"])
        normalized = {
            "array_index": array_index,
            "gate_e_rank": int(row["global_rank"]),
            "role": "gate_e_rank%d" % int(row["global_rank"]),
            "width": width,
            "start": start,
            "window": str(row["window"]),
            "score": float(row["score"]),
        }
        if normalized["window"] != "%d:%d" % (start, start + width - 1):
            raise Phase5GateFError("Gate E top4 window/width encoding differs")
        derived.append(normalized)
    if canonical_json_bytes(derived) != canonical_json_bytes(card["frozen_cells"]):
        raise Phase5GateFError("Gate E-derived top4 differs from frozen Gate F card")
    return derived


def derive_supplemental_cell(
    score_payload: Mapping[str, Any], card: Mapping[str, Any]
) -> Dict[str, Any]:
    """Derive and verify the exact Gate E rank-11 / width-rank-2 cell."""

    if (
        score_payload.get("manifest_sha256")
        != card["source"]["gate_e_score_manifest_sha256"]
        or score_payload.get("outcome_values_consumed") is not False
        or score_payload.get("test_split_consumed") is not False
        or score_payload.get("post_terminal_exploratory_no_selection") is not True
    ):
        raise Phase5GateFError("Gate E score artifact contract differs")
    rows = score_payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 42:
        raise Phase5GateFError("Gate E score artifact must contain 42 rows")
    matches = [row for row in rows if row.get("global_rank") == 11]
    if len(matches) != 1:
        raise Phase5GateFError("Gate E rank-11 membership is not unique")
    row = matches[0]
    derived = {
        "array_index": 0,
        "gate_e_rank": int(row["global_rank"]),
        "width_rank": int(row["width_rank"]),
        "role": "gate_e_rank11_supplement",
        "width": int(row["width"]),
        "start": int(row["start"]),
        "window": str(row["window"]),
        "score": float(row["score"]),
    }
    if (
        derived["width_rank"] != 2
        or derived["width"] != 5
        or derived["start"] != 15
        or derived["window"] != "15:19"
        or derived["window"]
        != "%d:%d" % (derived["start"], derived["start"] + derived["width"] - 1)
    ):
        raise Phase5GateFError("Gate E rank-11 supplemental identity differs")
    if canonical_json_bytes(derived) != canonical_json_bytes(card.get("supplemental_cell")):
        raise Phase5GateFError(
            "Gate E-derived rank-11 supplemental cell differs from frozen card"
        )
    return derived


def validate_frozen_inputs(card: Mapping[str, Any]) -> Dict[str, Any]:
    validate_extension_card(card)
    source = card["source"]
    required = (
        (Path(source["gate_e_score_path"]), source["gate_e_score_file_sha256"]),
        (Path(source["gate_e_receipt_path"]), source["gate_e_receipt_file_sha256"]),
        (Path(source["gate_c_completion_path"]), source["gate_c_completion_file_sha256"]),
        (Path(source["baseline_results_path"]), source["baseline_results_file_sha256"]),
        (TEST_MANIFEST_PATH, TEST_MANIFEST_FILE_SHA256),
        (TEST_METADATA_PATH, TEST_METADATA_FILE_SHA256),
    )
    for path, digest in required:
        if not path.is_file() or path.is_symlink() or file_sha256(path) != digest:
            raise Phase5GateFError("frozen input byte hash differs: %s" % path)
    baseline = Path(source["baseline_results_path"])
    if baseline.stat().st_size != source["baseline_results_size_bytes"]:
        raise Phase5GateFError("canonical baseline stat differs")
    score = _strict_json(Path(source["gate_e_score_path"]))
    cells = derive_top4(score, card)
    supplemental_cell = derive_supplemental_cell(score, card)
    gate_e_receipt = _strict_json(Path(source["gate_e_receipt_path"]))
    _verify_manifest(gate_e_receipt)
    if (
        gate_e_receipt.get("result") != "READY_FOR_PLANNING_AUDIT"
        or gate_e_receipt.get("outcome_values_consumed") is not False
        or gate_e_receipt.get("test_split_consumed") is not False
    ):
        raise Phase5GateFError("Gate E receipt closure differs")
    completion = _strict_json(Path(source["gate_c_completion_path"]))
    _verify_manifest(completion)
    if (
        completion.get("manifest_sha256") != source["gate_c_completion_manifest_sha256"]
        or completion.get("status") != "SEALED_COMPLETE_READY_FOR_PLANNING_AUDIT"
        or completion.get("outcome_values_consumed") is not False
        or completion.get("identity_closure", {}).get("ordered_identity_sha256_all_cells")
        != TEST_ORDERED_IDENTITY_SHA256
    ):
        raise Phase5GateFError("Gate C completion closure differs")
    baseline_row = next((row for row in completion.get("cells", []) if row.get("role") == "baseline"), None)
    if (
        baseline_row is None
        or baseline_row.get("results", {}).get("path") != str(baseline)
        or baseline_row.get("results", {}).get("size_bytes") != baseline.stat().st_size
        or baseline_row.get("results", {}).get("file_sha256") != file_sha256(baseline)
    ):
        raise Phase5GateFError("Gate C baseline reuse binding differs")
    test_manifest = _strict_json(TEST_MANIFEST_PATH)
    if (
        test_manifest.get("manifest_sha256") != TEST_MANIFEST_INTERNAL_SHA256
        or test_manifest.get("record_count") != 14042
        or test_manifest.get("subject_count") != 57
        or test_manifest.get("ordered_identity_sha256") != TEST_ORDERED_IDENTITY_SHA256
    ):
        raise Phase5GateFError("test identity manifest closure differs")
    _load_test_metadata()
    return {
        "cells": cells,
        "supplemental_cell": supplemental_cell,
        "baseline": {
            "path": str(baseline),
            "size_bytes": baseline.stat().st_size,
            "file_sha256": file_sha256(baseline),
        },
        "input_file_sha256": {str(path): digest for path, digest in required},
    }


def build_eval_argv(mode: str, output_dir: Path, window: str, batch_size: str) -> List[str]:
    tasks = "mmlu_abstract_algebra,mmlu_anatomy" if mode == "smoke" else "mmlu"
    argv = [
        "python",
        "-m",
        "tflt.eval_runner",
        "--model",
        MODEL_REPO,
        "--revision",
        MODEL_REVISION,
        "--tasks",
        tasks,
        "--output-dir",
        str(output_dir),
        "--num-fewshot",
        "5",
        "--batch-size",
        batch_size,
        "--dtype",
        "bfloat16",
    ]
    if mode == "smoke":
        argv.extend(("--limit", "2"))
    argv.extend(
        (
            "--loop",
            "--window",
            window,
            "--k",
            "3",
            "--iteration-mode",
            "block",
            "--strategy",
            "euler",
            "--alpha",
            "1.0",
            "--beta",
            "0.0",
            "--cache-strategy",
            "first",
            "--decode-mode",
            "bypass",
        )
    )
    return argv


def _cell_argv(mode: str, run_root: Path, index: int, source_commit: str) -> List[str]:
    return [
        "python",
        "scripts/loopscope/run_qwen4base_phase5_gate_f.py",
        "run-cell",
        "--mode",
        mode,
        "--run-root",
        str(run_root),
        "--cell-index",
        str(index),
        "--expected-source-commit",
        source_commit,
    ]


def build_launch_manifest(
    *,
    mode: str,
    run_root: Path,
    implementation_commit: str,
    expected_source_commit: str,
    partition: str,
    gres: str,
    qos: Optional[str],
    account: Optional[str],
    array_throttle: int,
    time_limit: str,
    batch_size: str,
    created_at_utc: str,
    card: Mapping[str, Any],
    score_payload: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
) -> Dict[str, Any]:
    root = _run_root(run_root)
    if mode not in {"smoke", "formal"}:
        raise Phase5GateFError("mode must be smoke or formal")
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise Phase5GateFError("local implementation commit must be a full SHA1")
    if expected_source_commit != REMOTE_SOURCE_COMMIT:
        raise Phase5GateFError("remote source commit differs")
    if not partition or not gres or not time_limit or batch_size != "16":
        raise Phase5GateFError("scheduler/batch parameters differ")
    if not isinstance(array_throttle, int) or not 1 <= array_throttle <= 4:
        raise Phase5GateFError("array throttle must be in 1..4")
    cells = []
    for row in derive_top4(score_payload, card):
        cell_id = "%s_w%d_%s" % (row["role"], row["width"], row["window"].replace(":", "_"))
        cell_root = root / mode / ("cell-%02d-%s" % (row["array_index"], cell_id))
        cells.append(
            {
                **row,
                "cell_id": cell_id,
                "cell_root": str(cell_root),
                "eval_output_dir": str(cell_root / "eval"),
                "argv": _cell_argv(mode, root, row["array_index"], expected_source_commit),
                "eval_argv": build_eval_argv(mode, cell_root / "eval", row["window"], batch_size),
                "automatic_retry": False,
            }
        )
    value: Dict[str, Any] = {
        "schema_version": "loopscope.phase5.gate-f-%s-launch.v1" % mode,
        "artifact_role": "phase5_gate_f_frozen_four_cell_%s" % mode,
        "gate": GATE,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "created_at_utc": created_at_utc,
        "run_root": str(root),
        "implementation": {
            "local_commit": implementation_commit,
            "staged_root": str(repository_root()),
            "staged_file_sha256": dict(implementation_sha256),
        },
        "remote_source": {"branch": "loopscope", "commit": expected_source_commit, "dirty": False},
        "frozen_inputs": dict(card["source"]),
        "recipe": {
            **dict(card["recipe"]),
            "limit": 2 if mode == "smoke" else None,
            "only_window_start_end_width_vary": True,
        },
        "population": {
            "test_manifest_path": str(TEST_MANIFEST_PATH),
            "test_metadata_path": str(TEST_METADATA_PATH),
            "required_count": 4 if mode == "smoke" else 14042,
            "required_subject_count": 2 if mode == "smoke" else 57,
            "ordered_identity_sha256": (
                ordered_identity_sha256(
                    [
                        {"task": task, "doc_id": doc_id, "doc_hash": doc_hash}
                        for task, doc_id, doc_hash in SMOKE_IDENTITIES
                    ]
                )
                if mode == "smoke"
                else TEST_ORDERED_IDENTITY_SHA256
            ),
        },
        "scheduler": {
            "partition": partition,
            "gres": gres,
            "qos": qos,
            "account": account,
            "array": "0-3%%%d" % array_throttle,
            "array_throttle": array_throttle,
            "cpus_per_task": 8,
            "memory": "64G",
            "time_limit": time_limit,
            "requeue": False,
        },
        "cells": cells,
        "cell_count": 4,
        "baseline_reuse": dict(card["source"]),
        "information_barrier": {
            "producer_identity_projection_only": True,
            "sealer_results_parse_authorized": False,
            "partial_cell_outcome_inspection_authorized": False,
            "accuracy_or_gain_computation_authorized_before_formal_seal": False,
            "one_shot_analysis_after_exact_closure": True,
        },
        "outcome_values_consumed": False,
    }
    _attach_manifest(value)
    validate_launch_manifest(value, mode=mode, card=card)
    return value


def validate_launch_manifest(value: Mapping[str, Any], *, mode: str, card: Mapping[str, Any]) -> None:
    _verify_manifest(value)
    if (
        value.get("gate") != GATE
        or value.get("executor_thread_id") != EXECUTOR_THREAD_ID
        or value.get("planning_thread_id") != PLANNING_THREAD_ID
        or value.get("cell_count") != 4
        or value.get("outcome_values_consumed") is not False
    ):
        raise Phase5GateFError("Gate F launch identity differs")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != 4:
        raise Phase5GateFError("Gate F requires exactly four cells")
    observed = [
        {
            key: row.get(key)
            for key in ("array_index", "gate_e_rank", "role", "width", "start", "window", "score")
        }
        for row in cells
    ]
    if canonical_json_bytes(observed) != canonical_json_bytes(card["frozen_cells"]):
        raise Phase5GateFError("Gate F launch membership/order differs")
    recipe = value.get("recipe", {})
    for key, expected in card["recipe"].items():
        if recipe.get(key) != expected:
            raise Phase5GateFError("Gate F launch recipe drift: %s" % key)
    required_count = 4 if mode == "smoke" else 14042
    if (
        recipe.get("limit") != (2 if mode == "smoke" else None)
        or value.get("population", {}).get("required_count") != required_count
    ):
        raise Phase5GateFError("Gate F mode/population differs")
    for cell in cells:
        start, end = (int(item) for item in cell["window"].split(":"))
        if (
            end - start + 1 != cell["width"]
            or cell["eval_argv"]
            != build_eval_argv(mode, Path(cell["eval_output_dir"]), cell["window"], "16")
            or cell.get("automatic_retry") is not False
        ):
            raise Phase5GateFError("Gate F cell width/argv/retry drift")


def _common_environment(run_root: Path) -> str:
    staged = _run_root(run_root) / "staged"
    return """set -euo pipefail
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
unset HF_ENDPOINT || true
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PYTHONPATH={staged}/src:{source}/src
cd {staged}
""".format(staged=shlex.quote(str(staged)), source=shlex.quote(str(REMOTE_REPO)))


def _launcher_text(cell: Mapping[str, Any], run_root: Path) -> str:
    cell_root = shlex.quote(str(cell["cell_root"]))
    command = shlex.join([str(AUDITED_VENV / "bin/python"), *cell["argv"][1:]])
    return (
        "#!/usr/bin/env bash\n"
        + _common_environment(run_root)
        + "if [[ -e %s ]]; then echo 'refusing existing Gate F cell output' >&2; exit 73; fi\n"
        % cell_root
        + "exec %s\n" % command
    )


def _array_runner_text(mode: str, run_root: Path) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", 'case "${SLURM_ARRAY_TASK_ID:?}" in']
    for index in range(4):
        path = _run_root(run_root) / mode / "launch" / ("cell-%02d.sh" % index)
        lines.append("  %d) exec %s ;;" % (index, shlex.quote(str(path))))
    lines.extend(["  *) echo 'invalid Gate F array task id' >&2; exit 64 ;;", "esac", ""])
    return "\n".join(lines)


def _submit_text(mode: str, manifest: Mapping[str, Any]) -> str:
    scheduler = manifest["scheduler"]
    run_root = Path(manifest["run_root"])
    args = [
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--job-name=loopscope-p5f-%s" % mode,
        "--partition=%s" % scheduler["partition"],
        "--gres=%s" % scheduler["gres"],
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=%s" % scheduler["time_limit"],
        "--array=%s" % scheduler["array"],
        "--no-requeue",
        "--output=%s" % (run_root / mode / ("slurm/p5f-%s-%%A_%%a.out" % mode)),
        "--error=%s" % (run_root / mode / ("slurm/p5f-%s-%%A_%%a.err" % mode)),
    ]
    if scheduler.get("qos"):
        args.append("--qos=%s" % scheduler["qos"])
    if scheduler.get("account"):
        args.append("--account=%s" % scheduler["account"])
    args.append(str(run_root / mode / "launch/array_runner.sh"))
    return "#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n" % shlex.join(args)


def _materialize_launch(mode: str, manifest: Mapping[str, Any]) -> Dict[str, str]:
    run_root = Path(manifest["run_root"])
    root = run_root / mode
    (root / "launch").mkdir(parents=True, exist_ok=False)
    (root / "slurm").mkdir(parents=True, exist_ok=False)
    hashes = {MANIFEST_NAMES[mode]: _write_new_json(root / MANIFEST_NAMES[mode], manifest)}
    for cell in manifest["cells"]:
        path = root / "launch" / ("cell-%02d.sh" % cell["array_index"])
        hashes[str(path.relative_to(root))] = _write_new_text(
            path, _launcher_text(cell, run_root), executable=True
        )
    hashes["launch/array_runner.sh"] = _write_new_text(
        root / "launch/array_runner.sh",
        _array_runner_text(mode, run_root),
        executable=True,
    )
    hashes["launch/submit.sh"] = _write_new_text(
        root / "launch/submit.sh", _submit_text(mode, manifest), executable=True
    )
    return hashes


def _load_card() -> Dict[str, Any]:
    card = _strict_json(repository_root() / CARD_RELATIVE)
    validate_extension_card(card)
    return card


def prepare_smoke(
    *,
    run_root: Path,
    implementation_commit: str,
    expected_source_commit: str,
    partition: str,
    gres: str,
    qos: Optional[str],
    account: Optional[str],
    array_throttle: int,
    time_limit: str,
    batch_size: str,
) -> Dict[str, Any]:
    root = _run_root(run_root)
    if root.exists() and any(root.iterdir()):
        allowed = {"staged"}
        if {item.name for item in root.iterdir()} != allowed:
            raise Phase5GateFError("fresh Gate F root contains unexpected entries")
    root.mkdir(parents=False, exist_ok=True)
    card = _load_card()
    git = remote_source_provenance(expected_source_commit)
    frozen = validate_frozen_inputs(card)
    impl = implementation_hashes()
    manifest = build_launch_manifest(
        mode="smoke",
        run_root=root,
        implementation_commit=implementation_commit,
        expected_source_commit=expected_source_commit,
        partition=partition,
        gres=gres,
        qos=qos,
        account=account,
        array_throttle=array_throttle,
        time_limit=time_limit,
        batch_size=batch_size,
        created_at_utc=utc_now(),
        card=card,
        score_payload=_strict_json(Path(card["source"]["gate_e_score_path"])),
        implementation_sha256=impl,
    )
    hashes = _materialize_launch("smoke", manifest)
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-preoutcome-freeze.v1",
            "gate": GATE,
            "git": git,
            "implementation_commit": implementation_commit,
            "implementation_sha256": impl,
            "derived_cells": frozen["cells"],
            "baseline_reuse_stat_sha_only": frozen["baseline"],
            "launch_manifest_sha256": hashes[MANIFEST_NAMES["smoke"]],
            "launch_manifest_internal_sha256": manifest["manifest_sha256"],
            "launcher_sha256": hashes,
            "outcome_values_consumed": False,
            "status": "SMOKE_PREPARED",
        }
    )
    _write_new_json(root / "phase5_gate_f_preoutcome_freeze.json", receipt)
    return receipt


def freeze_formal(
    *,
    run_root: Path,
    implementation_commit: str,
    expected_source_commit: str,
    partition: str,
    gres: str,
    qos: Optional[str],
    account: Optional[str],
    array_throttle: int,
    time_limit: str,
    batch_size: str,
) -> Dict[str, Any]:
    root = _run_root(run_root)
    remote_source_provenance(expected_source_commit)
    card = _load_card()
    validate_frozen_inputs(card)
    smoke = _strict_json(root / SEAL_NAMES["smoke"])
    _verify_manifest(smoke)
    if smoke.get("status") != "NON_SCIENTIFIC_LIMIT_SMOKE_PASS" or smoke.get("cell_count") != 4:
        raise Phase5GateFError("formal launch requires exact four-cell smoke PASS")
    if (root / "formal").exists():
        raise FileExistsError("formal launch is already frozen")
    impl = implementation_hashes()
    manifest = build_launch_manifest(
        mode="formal",
        run_root=root,
        implementation_commit=implementation_commit,
        expected_source_commit=expected_source_commit,
        partition=partition,
        gres=gres,
        qos=qos,
        account=account,
        array_throttle=array_throttle,
        time_limit=time_limit,
        batch_size=batch_size,
        created_at_utc=utc_now(),
        card=card,
        score_payload=_strict_json(Path(card["source"]["gate_e_score_path"])),
        implementation_sha256=impl,
    )
    hashes = _materialize_launch("formal", manifest)
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-formal-freeze.v1",
            "gate": GATE,
            "implementation_commit": implementation_commit,
            "implementation_sha256": impl,
            "launch_manifest_sha256": hashes[MANIFEST_NAMES["formal"]],
            "launch_manifest_internal_sha256": manifest["manifest_sha256"],
            "smoke_seal_file_sha256": file_sha256(root / SEAL_NAMES["smoke"]),
            "launcher_sha256": hashes,
            "outcome_values_consumed": False,
            "status": "FORMAL_FROZEN",
        }
    )
    _write_new_json(root / "phase5_gate_f_formal_freeze.json", receipt)
    return receipt


def record_submission(*, mode: str, run_root: Path, job_id: str) -> Dict[str, Any]:
    root = _run_root(run_root)
    if not re.fullmatch(r"[0-9]+", str(job_id)):
        raise Phase5GateFError("submission job id must be decimal")
    manifest_path = root / mode / MANIFEST_NAMES[mode]
    manifest = _strict_json(manifest_path)
    validate_launch_manifest(manifest, mode=mode, card=_load_card())
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-%s-submission.v1" % mode,
            "gate": GATE,
            "mode": mode,
            "job_id": str(job_id),
            "submitted_at_utc": utc_now(),
            "launch_manifest_file_sha256": file_sha256(manifest_path),
            "launch_manifest_internal_sha256": manifest["manifest_sha256"],
            "automatic_retry": False,
            "outcome_values_consumed": False,
        }
    )
    _write_new_json(root / mode / SUBMISSION_NAMES[mode], receipt)
    return receipt


def _audit_command(cell: Mapping[str, Any], output_dir: Path, prompts: Sequence[Mapping[str, str]]) -> List[str]:
    argv = [
        str(AUDITED_VENV / "bin/python"),
        "-m",
        "tflt.cli",
        "audit-loop-effect",
        "--model",
        MODEL_REPO,
        "--revision",
        MODEL_REVISION,
        "--output-dir",
        str(output_dir),
        "--window",
        str(cell["window"]),
        "--k",
        "3",
        "--iteration-mode",
        "block",
        "--strategy",
        "euler",
        "--alpha",
        "1.0",
        "--beta",
        "0.0",
        "--cache-strategy",
        "first",
        "--decode-mode",
        "bypass",
        "--dtype",
        "bfloat16",
        "--device",
        "cuda",
    ]
    for prompt in prompts:
        argv.extend(("--prompt", str(prompt["text"])))
    return argv


def run_cell(
    *, mode: str, run_root: Path, cell_index: int, expected_source_commit: str
) -> None:
    root = _run_root(run_root)
    if cell_index not in range(4):
        raise Phase5GateFError("run-cell index differs")
    git = remote_source_provenance(expected_source_commit)
    card = _load_card()
    validate_frozen_inputs(card)
    manifest = _strict_json(root / mode / MANIFEST_NAMES[mode])
    validate_launch_manifest(manifest, mode=mode, card=card)
    cell = manifest["cells"][cell_index]
    cell_root = Path(cell["cell_root"])
    cell_root.mkdir(parents=False, exist_ok=False)
    _write_new_json(
        cell_root / "producer_command.json",
        {
            "gate": GATE,
            "mode": mode,
            "cell": {
                key: cell[key]
                for key in (
                    "array_index",
                    "gate_e_rank",
                    "role",
                    "width",
                    "start",
                    "window",
                    "score",
                    "eval_argv",
                )
            },
            "git": git,
            "launch_manifest_sha256": manifest["manifest_sha256"],
            "outcome_values_consumed": False,
        },
    )
    smoke_prompts = _smoke_prompts() if mode == "smoke" else []
    if mode == "smoke":
        audit_dir = cell_root / "audit"
        completed = subprocess.run(_audit_command(cell, audit_dir, smoke_prompts), check=False)
        if completed.returncode:
            raise Phase5GateFError("loop-effect audit failed with exit %d" % completed.returncode)
        structural = _validate_loop_audit(
            _strict_json(audit_dir / "audit_report.json"), str(cell["window"])
        )
    else:
        structural = {"formal_loop_audit_not_persisted": True}
    structural_receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-cell-structural.v1",
            "mode": mode,
            "cell_id": cell["cell_id"],
            "width": cell["width"],
            "window": cell["window"],
            "structural": structural,
            "fixed_smoke_prompt_identity": [row["identity"] for row in smoke_prompts],
            "fixed_smoke_prompt_render_sha256": [row["render_sha256"] for row in smoke_prompts],
            "outcome_values_consumed": False,
        }
    )
    _write_new_json(cell_root / "structural_receipt.json", structural_receipt)
    eval_argv = [str(AUDITED_VENV / "bin/python"), *cell["eval_argv"][1:]]
    completed = subprocess.run(eval_argv, check=False)
    if completed.returncode:
        raise Phase5GateFError("lm-eval producer failed with exit %d" % completed.returncode)
    eval_dir = Path(cell["eval_output_dir"])
    results_path = eval_dir / "results.json"
    command_path = eval_dir / "command_args.json"
    revision_path = eval_dir / "model_revision.json"
    for path in (results_path, command_path, revision_path):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise Phase5GateFError("producer artifact is absent/empty/symlinked: %s" % path)
    all_records = _load_test_metadata()
    if mode == "smoke":
        expected_keys = set(SMOKE_IDENTITIES)
        expected_records = [
            row
            for row in all_records
            if (
                str(row["identity"]["task"]),
                str(row["identity"]["doc_id"]),
                str(row["identity"]["doc_hash"]),
            )
            in expected_keys
        ]
    else:
        expected_records = all_records
    result = _strict_json(results_path)
    projected = project_identity_only(result, expected_records)
    del result
    identity = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-cell-identity.v1",
            "mode": mode,
            "cell_id": cell["cell_id"],
            "gate_e_rank": cell["gate_e_rank"],
            "width": cell["width"],
            "window": cell["window"],
            "record_count": len(projected),
            "subject_count": len({row["subject"] for row in projected}),
            "ordered_identity_sha256": ordered_identity_sha256(
                [row["identity"] for row in projected]
            ),
            "records": projected,
            "projection_allowlist": [
                "task",
                "doc_id",
                "doc_hash",
                "subject",
                "split",
                "canonical_ordinal",
            ],
            "outcome_fields_accessed": False,
            "outcome_values_consumed": False,
        }
    )
    identity_path = cell_root / "identity_sidecar.json"
    _write_new_json(identity_path, identity)
    completion = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-cell-completion.v1",
            "gate": GATE,
            "mode": mode,
            "cell_id": cell["cell_id"],
            "results": {
                "path": str(results_path),
                "size_bytes": results_path.stat().st_size,
                "file_sha256": file_sha256(results_path),
            },
            "command_args_file_sha256": file_sha256(command_path),
            "model_revision_file_sha256": file_sha256(revision_path),
            "identity_sidecar_file_sha256": file_sha256(identity_path),
            "structural_receipt_file_sha256": file_sha256(
                cell_root / "structural_receipt.json"
            ),
            "exit_state": "PRODUCER_COMPLETED",
            "outcome_values_consumed": False,
        }
    )
    _write_new_json(cell_root / "producer_completion.json", completion)


def query_sacct(job_id: str) -> List[Dict[str, str]]:
    completed = subprocess.run(
        [
            "/opt/slurm/bin/sacct",
            "-j",
            str(job_id),
            "--array",
            "--noheader",
            "--parsable2",
            "--format=" + ",".join(SACCT_FIELDS),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise Phase5GateFError("sacct failed: %s" % completed.stderr.strip())
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) > len(SACCT_FIELDS) and values[-1] == "":
            values = values[:-1]
        if len(values) != len(SACCT_FIELDS):
            raise Phase5GateFError("unexpected sacct field count")
        rows.append(dict(zip(SACCT_FIELDS, values)))
    return rows


def validate_scheduler_rows(rows: Sequence[Mapping[str, Any]], job_id: str) -> List[Dict[str, str]]:
    if not re.fullmatch(r"[0-9]+", str(job_id)):
        raise Phase5GateFError("Slurm job id must be decimal")
    pattern = re.compile(r"^%s_([0-9]+)$" % re.escape(str(job_id)))
    tasks: Dict[int, Dict[str, str]] = {}
    for raw in rows:
        match = pattern.fullmatch(str(raw.get("JobID", "")))
        if not match:
            continue
        index = int(match.group(1))
        if index in tasks:
            raise Phase5GateFError("scheduler rows contain duplicate array task")
        tasks[index] = {key: str(raw.get(key, "")) for key in SACCT_FIELDS}
    if set(tasks) != set(range(4)):
        raise Phase5GateFError("sealed completion requires exact four scheduler tasks")
    for index, row in tasks.items():
        state = row["State"].split()[0].rstrip("+")
        if state != "COMPLETED" or row["ExitCode"] != "0:0":
            raise Phase5GateFError(
                "scheduler task %d is not COMPLETED/0:0: %s %s"
                % (index, row["State"], row["ExitCode"])
            )
    return [tasks[index] for index in range(4)]


def _gpu_count(alloc_tres: str) -> int:
    matches = re.findall(r"(?:^|,)gres/gpu(?::[^=,]+)?=([0-9]+)(?:,|$)", alloc_tres)
    if len(matches) > 1:
        raise Phase5GateFError("ambiguous GPU allocation TRES")
    return int(matches[0]) if matches else 0


def validate_smoke_structural_info(info: Mapping[str, Any]) -> None:
    calls = info.get("operator_body_calls_per_prompt")
    if (
        not isinstance(calls, list)
        or not calls
        or any(value != 3 for value in calls)
        or info.get("prompt_count") != len(calls)
        or info.get("restore_allclose_all_prompts") is not True
        or info.get("overall_decision") != "loop_effective_logits_changed"
    ):
        raise Phase5GateFError("smoke loop structural closure differs")


def _close_cells(mode: str, manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    entries = []
    for cell in manifest["cells"]:
        root = Path(cell["cell_root"])
        completion_path = root / "producer_completion.json"
        identity_path = root / "identity_sidecar.json"
        structural_path = root / "structural_receipt.json"
        for path in (completion_path, identity_path, structural_path):
            if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
                raise Phase5GateFError("cell closure artifact is absent/empty/symlinked")
        completion = _strict_json(completion_path)
        identity = _strict_json(identity_path)
        structural = _strict_json(structural_path)
        _verify_manifest(completion)
        _verify_manifest(identity)
        _verify_manifest(structural)
        if (
            completion.get("cell_id") != cell["cell_id"]
            or completion.get("exit_state") != "PRODUCER_COMPLETED"
            or identity.get("cell_id") != cell["cell_id"]
            or identity.get("record_count") != manifest["population"]["required_count"]
            or identity.get("subject_count")
            != manifest["population"]["required_subject_count"]
            or identity.get("ordered_identity_sha256")
            != manifest["population"]["ordered_identity_sha256"]
            or identity.get("outcome_fields_accessed") is not False
            or identity.get("outcome_values_consumed") is not False
        ):
            raise Phase5GateFError("cell identity/completion closure differs")
        results = completion.get("results", {})
        results_path = Path(str(results.get("path", "")))
        if (
            results_path != Path(cell["eval_output_dir"]) / "results.json"
            or not results_path.is_file()
            or results_path.is_symlink()
            or results_path.stat().st_size != results.get("size_bytes")
            or file_sha256(results_path) != results.get("file_sha256")
        ):
            raise Phase5GateFError("sealed results stat/SHA differs")
        if mode == "smoke":
            validate_smoke_structural_info(structural.get("structural", {}))
        entries.append(
            {
                "array_index": cell["array_index"],
                "cell_id": cell["cell_id"],
                "gate_e_rank": cell["gate_e_rank"],
                "role": cell["role"],
                "width": cell["width"],
                "start": cell["start"],
                "window": cell["window"],
                "score": cell["score"],
                "results": dict(results),
                "identity_sidecar": {
                    "path": str(identity_path),
                    "file_sha256": file_sha256(identity_path),
                    "manifest_sha256": identity["manifest_sha256"],
                },
                "producer_completion_file_sha256": file_sha256(completion_path),
                "structural_receipt_file_sha256": file_sha256(structural_path),
            }
        )
    return entries


def seal_stage(
    *,
    mode: str,
    run_root: Path,
    expected_source_commit: str,
    job_id: str,
    scheduler_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    root = _run_root(run_root)
    git = remote_source_provenance(expected_source_commit)
    card = _load_card()
    validate_frozen_inputs(card)
    output_path = root / SEAL_NAMES[mode]
    if output_path.exists():
        raise FileExistsError("sealed stage receipt already exists")
    manifest_path = root / mode / MANIFEST_NAMES[mode]
    manifest = _strict_json(manifest_path)
    validate_launch_manifest(manifest, mode=mode, card=card)
    submission = _strict_json(root / mode / SUBMISSION_NAMES[mode])
    _verify_manifest(submission)
    if submission.get("job_id") != str(job_id):
        raise Phase5GateFError("submission/job identity differs")
    terminal = validate_scheduler_rows(
        list(scheduler_rows) if scheduler_rows is not None else query_sacct(job_id),
        job_id,
    )
    entries = _close_cells(mode, manifest)
    total_gpu_seconds = 0
    for row in terminal:
        try:
            elapsed = int(row["ElapsedRaw"])
        except ValueError as exc:
            raise Phase5GateFError("Slurm ElapsedRaw is invalid") from exc
        total_gpu_seconds += elapsed * _gpu_count(row["AllocTRES"])
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-%s-seal.v1" % mode,
            "artifact_role": "gate_f_%s_identity_scheduler_stat_sha_closure" % mode,
            "gate": GATE,
            "mode": mode,
            "created_at_utc": utc_now(),
            "git": git,
            "job_id": str(job_id),
            "scheduler_rows": terminal,
            "launch_manifest_path": str(manifest_path),
            "launch_manifest_file_sha256": file_sha256(manifest_path),
            "launch_manifest_internal_sha256": manifest["manifest_sha256"],
            "cell_count": 4,
            "cells": entries,
            "identity_closure": {
                "record_count_per_cell": manifest["population"]["required_count"],
                "subject_count_per_cell": manifest["population"]["required_subject_count"],
                "ordered_identity_sha256_all_cells": manifest["population"][
                    "ordered_identity_sha256"
                ],
                "missing_duplicate_extra": 0,
            },
            "resource": {
                "total_gpu_seconds": total_gpu_seconds,
                "total_gpu_hours": total_gpu_seconds / 3600.0,
            },
            "outcome_values_consumed": False,
            "accuracy_or_gain_computed": False,
            "results_payload_parsed_by_sealer": False,
            "one_shot_analysis_performed": False,
            "status": (
                "NON_SCIENTIFIC_LIMIT_SMOKE_PASS"
                if mode == "smoke"
                else "SEALED_COMPLETE_READY_FOR_ONE_SHOT_ANALYSIS"
            ),
        }
    )
    _write_new_json(output_path, receipt)
    return receipt


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    numbers = sorted(float(value) for value in values)
    if not numbers or not 0.0 <= quantile <= 1.0:
        raise Phase5GateFError("invalid percentile input")
    position = quantile * (len(numbers) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return numbers[lower]
    weight = position - lower
    return numbers[lower] * (1.0 - weight) + numbers[upper] * weight


def exact_mcnemar_p(wrong_to_correct: int, correct_to_wrong: int) -> float:
    values = (wrong_to_correct, correct_to_wrong)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise Phase5GateFError("McNemar counts must be non-negative integers")
    total = sum(values)
    if not total:
        return 1.0
    tail = min(values)
    logs = [
        math.lgamma(total + 1)
        - math.lgamma(index + 1)
        - math.lgamma(total - index + 1)
        - total * math.log(2.0)
        for index in range(tail + 1)
    ]
    peak = max(logs)
    one_sided = math.exp(peak) * math.fsum(math.exp(value - peak) for value in logs)
    return min(1.0, 2.0 * one_sided)


def _bootstrap_summary(values: Sequence[float]) -> Dict[str, Any]:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {
        "bootstrap_mean_fraction": mean,
        "bootstrap_standard_error_ddof1": math.sqrt(variance),
        "percentile_95_ci_fraction": [
            linear_percentile(values, 0.025),
            linear_percentile(values, 0.975),
        ],
    }


def joint_paired_bootstrap(
    correctness: np.ndarray,
    subjects: Sequence[str],
    *,
    replicates: int = 2000,
    seed: int = 20260723,
) -> Dict[str, Any]:
    matrix = np.asarray(correctness, dtype=np.int8)
    if matrix.shape != (len(subjects), 6) or len(subjects) < 2:
        raise Phase5GateFError("bootstrap matrix shape differs")
    if not np.logical_or(matrix == 0, matrix == 1).all():
        raise Phase5GateFError("bootstrap matrix is not binary")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise Phase5GateFError("bootstrap replicates must be >=2")
    groups = {
        subject: [index for index, value in enumerate(subjects) if value == subject]
        for subject in sorted(set(subjects))
    }
    rng = random.Random(seed)
    digest = hashlib.sha256()
    estimates = np.empty((replicates, 6), dtype=np.float64)
    for replicate in range(replicates):
        drawn: List[int] = []
        for subject in sorted(groups):
            indices = groups[subject]
            drawn.extend(indices[rng.randrange(len(indices))] for _ in indices)
        digest.update(struct.pack("<%dI" % len(drawn), *drawn))
        estimates[replicate] = matrix[np.asarray(drawn, dtype=np.int64)].mean(axis=0)
    gains = estimates[:, 1:] - estimates[:, [0]]
    return {
        "method": "subject_stratified_joint_paired_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "prng": "python_stdlib_random.Random",
        "random_api": "randrange(n_s)",
        "subject_order": sorted(groups),
        "subject_sizes": {key: len(groups[key]) for key in sorted(groups)},
        "draw_iteration_order": "replicate_major_subject_ascending_identity_canonical",
        "draw_digest_representation": "little_endian_uint32_global_indices",
        "draw_index_sha256": digest.hexdigest(),
        "same_joint_draw_stream_baseline_and_five_cells": True,
        "standard_error": "sample_sd_ddof_1",
        "percentile_method": "linear_interpolation_at_q_times_R_minus_1",
        "cell_gain_summaries": [
            _bootstrap_summary(gains[:, index].tolist()) for index in range(5)
        ],
    }


def _transitions(baseline: np.ndarray, candidate: np.ndarray) -> Dict[str, int]:
    return {
        "wrong_to_correct": int(np.logical_and(baseline == 0, candidate == 1).sum()),
        "correct_to_wrong": int(np.logical_and(baseline == 1, candidate == 0).sum()),
        "correct_to_correct": int(np.logical_and(baseline == 1, candidate == 1).sum()),
        "wrong_to_wrong": int(np.logical_and(baseline == 0, candidate == 0).sum()),
    }


def _load_correctness(result_path: Path, metadata: Sequence[Mapping[str, Any]]) -> List[bool]:
    payload = _strict_json(result_path)
    project_identity_only(payload, metadata)
    try:
        correctness = extract_correctness_samples(payload, task="mmlu", metric="acc,none")
        aggregate = extract_accuracy(payload, task="mmlu", metric="acc,none")
    except AnalysisError as exc:
        raise Phase5GateFError("lm-eval outcome payload cannot close acc,none") from exc
    ordered = []
    for row in metadata:
        identity = row["identity"]
        key = "%s:%s" % (identity["task"], identity["doc_id"])
        if key not in correctness:
            raise Phase5GateFError("outcome correctness lacks canonical identity")
        ordered.append(bool(correctness[key]))
    expected = {
        "%s:%s" % (row["identity"]["task"], row["identity"]["doc_id"]) for row in metadata
    }
    if set(correctness) != expected:
        raise Phase5GateFError("outcome correctness has missing or extra identity")
    sample_accuracy = math.fsum(ordered) / len(ordered)
    if not math.isclose(sample_accuracy, aggregate, rel_tol=0.0, abs_tol=1e-12):
        raise Phase5GateFError("aggregate and per-sample acc,none differ")
    del payload, correctness
    gc.collect()
    return ordered


def _correctness_matrix(
    baseline_path: Path,
    completion: Mapping[str, Any],
    metadata: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    paths = [baseline_path] + [Path(row["results"]["path"]) for row in completion["cells"]]
    columns = [_load_correctness(path, metadata) for path in paths]
    matrix = np.column_stack([np.asarray(column, dtype=np.int8) for column in columns])
    if matrix.shape != (14042, 5):
        raise Phase5GateFError("outcome matrix does not close 14,042 x 5")
    return matrix


def build_statistical_analysis(
    *,
    correctness: np.ndarray,
    identities: Sequence[Mapping[str, Any]],
    subjects: Sequence[str],
    cells: Sequence[Mapping[str, Any]],
    replicates: int = 2000,
    seed: int = 20260723,
) -> Dict[str, Any]:
    matrix = np.asarray(correctness, dtype=np.int8)
    if matrix.shape != (len(identities), 6) or len(identities) != len(subjects):
        raise Phase5GateFError("analysis input shape differs")
    if len(cells) != 5:
        raise Phase5GateFError("analysis requires exactly five cells")
    bootstrap = joint_paired_bootstrap(matrix, subjects, replicates=replicates, seed=seed)
    accuracy = matrix.mean(axis=0, dtype=np.float64)
    rows: List[Dict[str, Any]] = [
        {
            "matrix_index": 0,
            "role": "canonical_baseline_reuse",
            "gate_e_rank": None,
            "score": None,
            "width": None,
            "window": None,
            "sample_count": len(matrix),
            "accuracy_fraction": float(accuracy[0]),
            "accuracy_percent": float(accuracy[0] * 100.0),
            "gain_fraction": 0.0,
            "gain_percentage_points": 0.0,
            "paired_percentile_95_ci_fraction": [0.0, 0.0],
            "paired_percentile_95_ci_percentage_points": [0.0, 0.0],
            "transitions_vs_baseline": {
                "wrong_to_correct": 0,
                "correct_to_wrong": 0,
                "correct_to_correct": int(matrix[:, 0].sum()),
                "wrong_to_wrong": int(len(matrix) - matrix[:, 0].sum()),
            },
            "exact_two_sided_mcnemar_p": 1.0,
        }
    ]
    for offset, cell in enumerate(cells, start=1):
        gain = float(accuracy[offset] - accuracy[0])
        ci = bootstrap["cell_gain_summaries"][offset - 1][
            "percentile_95_ci_fraction"
        ]
        transitions = _transitions(matrix[:, 0], matrix[:, offset])
        rows.append(
            {
                "matrix_index": offset,
                "role": cell["role"],
                "gate_e_rank": cell["gate_e_rank"],
                "score": cell["score"],
                "width": cell["width"],
                "window": cell["window"],
                "sample_count": len(matrix),
                "accuracy_fraction": float(accuracy[offset]),
                "accuracy_percent": float(accuracy[offset] * 100.0),
                "gain_fraction": gain,
                "gain_percentage_points": gain * 100.0,
                "paired_percentile_95_ci_fraction": list(ci),
                "paired_percentile_95_ci_percentage_points": [
                    value * 100.0 for value in ci
                ],
                "transitions_vs_baseline": transitions,
                "exact_two_sided_mcnemar_p": exact_mcnemar_p(
                    transitions["wrong_to_correct"],
                    transitions["correct_to_wrong"],
                ),
            }
        )
    return {
        "schema_version": "loopscope.phase5.gate-f-analysis.v1",
        "artifact_role": "one_shot_post_terminal_multiwidth_top4_plus_15_19_outcome_analysis",
        "gate": GATE,
        "label": "POST_TERMINAL_MULTIWIDTH_TOP4_PLUS_15_19_OUTCOME_EXPLORATORY",
        "population": {
            "record_count": len(matrix),
            "subject_count": len(set(subjects)),
            "matrix_shape": [len(matrix), 6],
            "ordered_identity_sha256": ordered_identity_sha256(identities),
            "subject_counts": dict(sorted(Counter(subjects).items())),
            "missing_duplicate_extra": 0,
        },
        "metric": {
            "primary": "acc,none",
            "internal_unit": "fraction",
            "report_unit": "percentage_points",
        },
        "cells": rows,
        "bootstrap": bootstrap,
        "claim_boundary": {
            "post_terminal_exploratory": True,
            "prospective_selected_window_success_claimed": False,
            "phase5_terminal_label_unchanged": "H5_ABSTAIN_WITH_RANKING_RESULT",
            "gate_e_eligible_count_remains_zero": True,
        },
        "persistence_safety": {
            "raw_payload_copied": False,
            "row_level_correctness_persisted": False,
            "baseline_rerun": False,
            "supplemental_fifth_cell_run": True,
            "sixth_candidate_run": False,
        },
    }


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    import io

    output = io.StringIO()
    fields = (
        "gate_e_rank",
        "score",
        "width",
        "window",
        "accuracy_percent",
        "gain_percentage_points",
        "ci95_low_pp",
        "ci95_high_pp",
        "wrong_to_correct",
        "correct_to_wrong",
        "correct_to_correct",
        "wrong_to_wrong",
        "exact_two_sided_mcnemar_p",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        ci = row["paired_percentile_95_ci_percentage_points"]
        transitions = row["transitions_vs_baseline"]
        writer.writerow(
            {
                "gate_e_rank": row["gate_e_rank"],
                "score": row["score"],
                "width": row["width"],
                "window": row["window"],
                "accuracy_percent": row["accuracy_percent"],
                "gain_percentage_points": row["gain_percentage_points"],
                "ci95_low_pp": ci[0],
                "ci95_high_pp": ci[1],
                **transitions,
                "exact_two_sided_mcnemar_p": row["exact_two_sided_mcnemar_p"],
            }
        )
    return output.getvalue()


def _summary_zh(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# LoopScope 第五阶段 Gate F 多宽度 Top4 + 15:19 全量 Outcome",
        "",
        "标签：`POST_TERMINAL_MULTIWIDTH_TOP4_OUTCOME_EXPLORATORY`。",
        "",
        "本结果是在 Phase 5 outcome 已解封后的探索性扩展，不是 prospective selected-window success；"
        "原终态 `H5_ABSTAIN_WITH_RANKING_RESULT` 保持不变。",
        "",
        "| Gate E rank | width | window | score | accuracy | gain vs baseline (pp) | 95% CI (pp) | W→C | C→W | McNemar p |",
        "|---:|---:|:---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for row in analysis["cells"][1:]:
        ci = row["paired_percentile_95_ci_percentage_points"]
        transitions = row["transitions_vs_baseline"]
        lines.append(
            "| {rank} | {width} | `{window}` | {score:.6f} | {acc:.6f}% | {gain:+.6f} | "
            "[{low:+.6f}, {high:+.6f}] | {wc} | {cw} | {p:.6g} |".format(
                rank=row["gate_e_rank"],
                width=row["width"],
                window=row["window"],
                score=row["score"],
                acc=row["accuracy_percent"],
                gain=row["gain_percentage_points"],
                low=ci[0],
                high=ci[1],
                wc=transitions["wrong_to_correct"],
                cw=transitions["correct_to_wrong"],
                p=row["exact_two_sided_mcnemar_p"],
            )
        )
    baseline = analysis["cells"][0]
    lines.extend(
        [
            "",
            "Canonical baseline 由原始 Gate C 逐样本 artifact 重新计算，未重跑："
            "`{:.6f}%`。".format(baseline["accuracy_percent"]),
            "",
            "所有统计基于 14,042 条相同 ordered identities、57 subjects，"
            "并使用 2,000 次 subject-stratified joint paired bootstrap（seed=20260723）。",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_formal_completion(root: Path, card: Mapping[str, Any]) -> Dict[str, Any]:
    completion_path = root / SEAL_NAMES["formal"]
    completion = _strict_json(completion_path)
    _verify_manifest(completion)
    identity = completion.get("identity_closure", {})
    if (
        completion.get("status") != "SEALED_COMPLETE_READY_FOR_ONE_SHOT_ANALYSIS"
        or completion.get("cell_count") != 4
        or completion.get("outcome_values_consumed") is not False
        or completion.get("accuracy_or_gain_computed") is not False
        or completion.get("results_payload_parsed_by_sealer") is not False
        or identity.get("record_count_per_cell") != 14042
        or identity.get("subject_count_per_cell") != 57
        or identity.get("ordered_identity_sha256_all_cells")
        != TEST_ORDERED_IDENTITY_SHA256
        or identity.get("missing_duplicate_extra") != 0
    ):
        raise Phase5GateFError("formal structural completion seal differs")
    frozen = validate_frozen_inputs(card)
    observed = [
        {
            key: row.get(key)
            for key in ("array_index", "gate_e_rank", "role", "width", "start", "window", "score")
        }
        for row in completion.get("cells", [])
    ]
    if canonical_json_bytes(observed) != canonical_json_bytes(frozen["cells"]):
        raise Phase5GateFError("formal completion cell membership/order differs")
    for row in completion["cells"]:
        result = row.get("results", {})
        path = Path(str(result.get("path", "")))
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != result.get("size_bytes")
            or file_sha256(path) != result.get("file_sha256")
        ):
            raise Phase5GateFError("formal sealed result stat/SHA differs")
    return completion


def run_one_shot_analysis(
    *,
    run_root: Path,
    expected_source_commit: str,
    implementation_commit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    raise Phase5GateFError(
        "four-cell subset analysis is forbidden after the 15:19 supplement authorization; "
        "use the combined baseline-plus-five analysis"
    )
    # The unreachable legacy body remains as immutable-contract reference for the
    # already launched four-cell root. The supplemental implementation never calls it.
    root = _run_root(run_root)
    analysis_root = root / "analysis"
    if analysis_root.exists() or any((root / name).exists() for name in OUTPUT_NAMES.values()):
        raise FileExistsError("Gate F analysis outputs already exist; refusing second unseal")
    card = _load_card()
    git = remote_source_provenance(expected_source_commit)
    frozen_before = validate_frozen_inputs(card)
    completion = _validate_formal_completion(root, card)
    analysis_root.mkdir(parents=False, exist_ok=False)
    raw_hashes = {
        "baseline": frozen_before["baseline"]["file_sha256"],
        **{
            row["cell_id"]: row["results"]["file_sha256"] for row in completion["cells"]
        },
    }
    unseal = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-unseal-once.v1",
            "gate": GATE,
            "created_at_utc": utc_now(),
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "implementation_commit": implementation_commit,
            "argv": list(argv),
            "formal_completion_file_sha256": file_sha256(root / SEAL_NAMES["formal"]),
            "sealed_result_file_sha256": raw_hashes,
            "unseal_count": 1,
            "analysis_count": 1,
        }
    )
    unseal_sha = _write_new_json(root / OUTPUT_NAMES["unseal"], unseal)
    metadata = _load_test_metadata()
    identities = [row["identity"] for row in metadata]
    subjects = [row["subject"] for row in metadata]
    matrix = _correctness_matrix(
        Path(card["source"]["baseline_results_path"]), completion, metadata
    )
    analysis = build_statistical_analysis(
        correctness=matrix,
        identities=identities,
        subjects=subjects,
        cells=completion["cells"],
    )
    analysis_sha = _write_new_json(analysis_root / OUTPUT_NAMES["analysis"], analysis)
    table_sha = _write_new_text(
        analysis_root / OUTPUT_NAMES["table"], _csv_text(analysis["cells"][1:])
    )
    bootstrap_digest = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-bootstrap-digest.v1",
            "gate": GATE,
            "replicates": analysis["bootstrap"]["replicates"],
            "seed": analysis["bootstrap"]["seed"],
            "draw_index_sha256": analysis["bootstrap"]["draw_index_sha256"],
            "cell_gain_summaries": analysis["bootstrap"]["cell_gain_summaries"],
        }
    )
    bootstrap_sha = _write_new_json(
        analysis_root / OUTPUT_NAMES["bootstrap"], bootstrap_digest
    )
    del matrix
    gc.collect()

    verify_matrix = _correctness_matrix(
        Path(card["source"]["baseline_results_path"]),
        _validate_formal_completion(root, card),
        _load_test_metadata(),
    )
    verify_analysis = build_statistical_analysis(
        correctness=verify_matrix,
        identities=identities,
        subjects=subjects,
        cells=completion["cells"],
    )
    if canonical_json_bytes(verify_analysis) != canonical_json_bytes(analysis):
        raise Phase5GateFError("independent verifier recomputation differs")
    del verify_matrix
    gc.collect()
    frozen_after = validate_frozen_inputs(card)
    if canonical_json_bytes(frozen_after) != canonical_json_bytes(frozen_before):
        raise Phase5GateFError("protected frozen inputs changed during analysis")
    verifier = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-verifier-receipt.v1",
            "gate": GATE,
            "created_at_utc": utc_now(),
            "result": "PASS",
            "verifier_count": 1,
            "checks": {
                "exact_14042_x_5_identity_equality": True,
                "baseline_recomputed_from_original_raw_artifact": True,
                "four_accuracies_recomputed": True,
                "four_paired_gains_and_ci_recomputed": True,
                "four_transition_tables_and_mcnemar_recomputed": True,
                "one_unseal_one_analysis": True,
                "gate_e_top4_membership_unchanged": True,
                "protected_prior_artifacts_unchanged": True,
                "prospective_selection_success_not_claimed": True,
            },
            "analysis_file_sha256": analysis_sha,
            "raw_result_file_sha256": raw_hashes,
        }
    )
    verifier_sha = _write_new_json(
        analysis_root / OUTPUT_NAMES["verifier"], verifier
    )
    summary_sha = _write_new_text(
        analysis_root / OUTPUT_NAMES["summary"], _summary_zh(analysis)
    )
    terminal = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-terminal-receipt.v1",
            "gate": GATE,
            "created_at_utc": utc_now(),
            "status": "GATE_F_READY_FOR_PLANNING_AUDIT",
            "label": "POST_TERMINAL_MULTIWIDTH_TOP4_OUTCOME_EXPLORATORY",
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "planning_thread_id": PLANNING_THREAD_ID,
            "git": git,
            "implementation_commit": implementation_commit,
            "implementation_sha256": implementation_hashes(),
            "run_root": str(root),
            "smoke_job_id": _strict_json(root / "smoke" / SUBMISSION_NAMES["smoke"])[
                "job_id"
            ],
            "formal_job_id": completion["job_id"],
            "scheduler_rows": completion["scheduler_rows"],
            "resource": completion["resource"],
            "identity_closure": completion["identity_closure"],
            "baseline_reuse": frozen_after["baseline"],
            "unseal_count": 1,
            "analysis_count": 1,
            "verifier_count": 1,
            "verifier_result": "PASS",
            "outputs": {
                OUTPUT_NAMES["unseal"]: unseal_sha,
                OUTPUT_NAMES["analysis"]: analysis_sha,
                OUTPUT_NAMES["table"]: table_sha,
                OUTPUT_NAMES["bootstrap"]: bootstrap_sha,
                OUTPUT_NAMES["verifier"]: verifier_sha,
                OUTPUT_NAMES["summary"]: summary_sha,
            },
            "claim_boundary": analysis["claim_boundary"],
            "actions_not_taken": {
                "baseline_rerun": True,
                "comparator_15_18_rerun": True,
                "fifth_candidate_or_extra_sweep": True,
                "outcome_informed_repair": True,
                "public_git_push": True,
                "automatic_next_gate": True,
            },
            "requested_decision": "PASS / PASS_WITH_FIXES / BLOCK",
        }
    )
    terminal_sha = _write_new_json(
        analysis_root / OUTPUT_NAMES["terminal"], terminal
    )
    return {
        "status": terminal["status"],
        "label": terminal["label"],
        "run_root": str(root),
        "analysis_root": str(analysis_root),
        "terminal_receipt_file_sha256": terminal_sha,
        "verifier_result": "PASS",
        "unseal_count": 1,
        "analysis_count": 1,
        "verifier_count": 1,
        "cells": analysis["cells"],
    }


def build_supplement_launch_manifest(
    *,
    run_root: Path,
    implementation_commit: str,
    expected_source_commit: str,
    partition: str,
    gres: str,
    qos: Optional[str],
    account: Optional[str],
    time_limit: str,
    batch_size: str,
    created_at_utc: str,
    card: Mapping[str, Any],
    score_payload: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
) -> Dict[str, Any]:
    root = _supplement_run_root(run_root)
    validate_extension_card(card)
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise Phase5GateFError("supplement implementation commit must be a full SHA1")
    if expected_source_commit != REMOTE_SOURCE_COMMIT:
        raise Phase5GateFError("supplement remote source commit differs")
    if not partition or not gres or not time_limit or batch_size != "16":
        raise Phase5GateFError("supplement scheduler/batch parameters differ")
    row = derive_supplemental_cell(score_payload, card)
    cell_id = "gate_e_rank11_supplement_w5_15_19"
    cell_root = root / "formal" / ("cell-00-" + cell_id)
    cell = {
        **row,
        "cell_id": cell_id,
        "cell_root": str(cell_root),
        "eval_output_dir": str(cell_root / "eval"),
        "argv": [
            "python",
            "scripts/loopscope/run_qwen4base_phase5_gate_f.py",
            "run-supplement-cell",
            "--run-root",
            str(root),
            "--expected-source-commit",
            expected_source_commit,
        ],
        "eval_argv": build_eval_argv("formal", cell_root / "eval", row["window"], batch_size),
        "automatic_retry": False,
    }
    value = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-supplement-formal-launch.v1",
            "artifact_role": "phase5_gate_f_frozen_single_supplement_formal",
            "gate": GATE,
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "planning_thread_id": PLANNING_THREAD_ID,
            "created_at_utc": created_at_utc,
            "run_root": str(root),
            "implementation": {
                "local_commit": implementation_commit,
                "staged_root": str(repository_root()),
                "staged_file_sha256": dict(implementation_sha256),
            },
            "remote_source": {
                "branch": "loopscope",
                "commit": expected_source_commit,
                "dirty": False,
            },
            "frozen_inputs": dict(card["source"]),
            "recipe": {
                **dict(card["recipe"]),
                "limit": None,
                "only_window_start_end_width_vary": True,
            },
            "population": {
                "test_manifest_path": str(TEST_MANIFEST_PATH),
                "test_metadata_path": str(TEST_METADATA_PATH),
                "required_count": 14042,
                "required_subject_count": 57,
                "ordered_identity_sha256": TEST_ORDERED_IDENTITY_SHA256,
            },
            "scheduler": {
                "partition": partition,
                "gres": gres,
                "qos": qos,
                "account": account,
                "array": "0-0%1",
                "array_throttle": 1,
                "cpus_per_task": 8,
                "memory": "64G",
                "time_limit": time_limit,
                "requeue": False,
            },
            "cells": [cell],
            "cell_count": 1,
            "baseline_reuse": dict(card["source"]),
            "information_barrier": {
                "producer_identity_projection_only": True,
                "sealer_results_parse_authorized": False,
                "partial_cell_outcome_inspection_authorized": False,
                "combined_analysis_requires_original_and_supplement_seals": True,
            },
            "outcome_values_consumed": False,
        }
    )
    validate_supplement_launch_manifest(value, card=card)
    return value


def validate_supplement_launch_manifest(
    value: Mapping[str, Any], *, card: Mapping[str, Any]
) -> None:
    _verify_manifest(value)
    if (
        value.get("schema_version")
        != "loopscope.phase5.gate-f-supplement-formal-launch.v1"
        or value.get("gate") != GATE
        or value.get("executor_thread_id") != EXECUTOR_THREAD_ID
        or value.get("planning_thread_id") != PLANNING_THREAD_ID
        or value.get("cell_count") != 1
        or value.get("outcome_values_consumed") is not False
        or value.get("scheduler", {}).get("array") != "0-0%1"
        or value.get("scheduler", {}).get("requeue") is not False
    ):
        raise Phase5GateFError("Gate F supplement launch identity differs")
    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise Phase5GateFError("Gate F supplement requires exactly one cell")
    expected = card["supplemental_cell"]
    observed = {
        key: cells[0].get(key)
        for key in (
            "array_index",
            "gate_e_rank",
            "width_rank",
            "role",
            "width",
            "start",
            "window",
            "score",
        )
    }
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise Phase5GateFError("Gate F supplement launch membership differs")
    for key, expected_value in card["recipe"].items():
        if value.get("recipe", {}).get(key) != expected_value:
            raise Phase5GateFError("Gate F supplement recipe drift: %s" % key)
    cell = cells[0]
    if (
        cell["eval_argv"]
        != build_eval_argv("formal", Path(cell["eval_output_dir"]), "15:19", "16")
        or cell.get("automatic_retry") is not False
        or value.get("population", {}).get("required_count") != 14042
        or value.get("population", {}).get("required_subject_count") != 57
        or value.get("population", {}).get("ordered_identity_sha256")
        != TEST_ORDERED_IDENTITY_SHA256
    ):
        raise Phase5GateFError("Gate F supplement argv/population differs")


def _supplement_environment(run_root: Path) -> str:
    root = _supplement_run_root(run_root)
    staged = root / "staged"
    return """set -euo pipefail
export HF_HOME=/hpc2hdd/home/xhuang225/shared/hf_home
export TRANSFORMERS_CACHE=/hpc2hdd/home/xhuang225/shared/hf_home/hub
export HF_DATASETS_CACHE=/hpc2hdd/home/xhuang225/shared/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
unset HF_ENDPOINT || true
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PYTHONPATH={staged}/src:{source}/src
cd {staged}
""".format(staged=shlex.quote(str(staged)), source=shlex.quote(str(REMOTE_REPO)))


def _materialize_supplement_launch(manifest: Mapping[str, Any]) -> Dict[str, str]:
    root = _supplement_run_root(Path(manifest["run_root"]))
    formal = root / "formal"
    (formal / "launch").mkdir(parents=True, exist_ok=False)
    (formal / "slurm").mkdir(parents=True, exist_ok=False)
    hashes = {
        SUPPLEMENT_MANIFEST: _write_new_json(formal / SUPPLEMENT_MANIFEST, manifest)
    }
    cell = manifest["cells"][0]
    command = shlex.join([str(AUDITED_VENV / "bin/python"), *cell["argv"][1:]])
    cell_text = (
        "#!/usr/bin/env bash\n"
        + _supplement_environment(root)
        + "if [[ -e %s ]]; then echo 'refusing existing Gate F supplement cell output' >&2; exit 73; fi\n"
        % shlex.quote(cell["cell_root"])
        + "exec %s\n" % command
    )
    hashes["launch/cell-00.sh"] = _write_new_text(
        formal / "launch/cell-00.sh", cell_text, executable=True
    )
    runner = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        '[[ "${SLURM_ARRAY_TASK_ID:?}" == "0" ]] || { echo "invalid supplement task id" >&2; exit 64; }\n'
        "exec %s\n" % shlex.quote(str(formal / "launch/cell-00.sh"))
    )
    hashes["launch/array_runner.sh"] = _write_new_text(
        formal / "launch/array_runner.sh", runner, executable=True
    )
    scheduler = manifest["scheduler"]
    args = [
        "/opt/slurm/bin/sbatch",
        "--parsable",
        "--job-name=loopscope-p5f-supplement",
        "--partition=%s" % scheduler["partition"],
        "--gres=%s" % scheduler["gres"],
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=%s" % scheduler["time_limit"],
        "--array=0-0%1",
        "--no-requeue",
        "--output=%s" % (formal / "slurm/p5f-supplement-%A_%a.out"),
        "--error=%s" % (formal / "slurm/p5f-supplement-%A_%a.err"),
    ]
    if scheduler.get("qos"):
        args.append("--qos=%s" % scheduler["qos"])
    if scheduler.get("account"):
        args.append("--account=%s" % scheduler["account"])
    args.append(str(formal / "launch/array_runner.sh"))
    hashes["launch/submit.sh"] = _write_new_text(
        formal / "launch/submit.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\nexec %s\n" % shlex.join(args),
        executable=True,
    )
    return hashes


def prepare_supplement_formal(
    *,
    run_root: Path,
    implementation_commit: str,
    expected_source_commit: str,
    partition: str,
    gres: str,
    qos: Optional[str],
    account: Optional[str],
    time_limit: str,
    batch_size: str,
) -> Dict[str, Any]:
    root = _supplement_run_root(run_root)
    if root.exists() and {item.name for item in root.iterdir()} != {"staged"}:
        raise Phase5GateFError("fresh supplement root contains unexpected entries")
    root.mkdir(parents=False, exist_ok=True)
    card = _load_card()
    git = remote_source_provenance(expected_source_commit)
    frozen = validate_frozen_inputs(card)
    impl = implementation_hashes()
    manifest = build_supplement_launch_manifest(
        run_root=root,
        implementation_commit=implementation_commit,
        expected_source_commit=expected_source_commit,
        partition=partition,
        gres=gres,
        qos=qos,
        account=account,
        time_limit=time_limit,
        batch_size=batch_size,
        created_at_utc=utc_now(),
        card=card,
        score_payload=_strict_json(Path(card["source"]["gate_e_score_path"])),
        implementation_sha256=impl,
    )
    hashes = _materialize_supplement_launch(manifest)
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-supplement-preoutcome-freeze.v1",
            "gate": GATE,
            "git": git,
            "implementation_commit": implementation_commit,
            "implementation_sha256": impl,
            "derived_cell": frozen["supplemental_cell"],
            "baseline_reuse_stat_sha_only": frozen["baseline"],
            "launch_manifest_sha256": hashes[SUPPLEMENT_MANIFEST],
            "launch_manifest_internal_sha256": manifest["manifest_sha256"],
            "launcher_sha256": hashes,
            "gpu_smoke_skipped": True,
            "gpu_smoke_skip_basis": "original_width5_runtime_semantics_closed_and_cpu_dry_run",
            "outcome_values_consumed": False,
            "status": "SUPPLEMENT_FORMAL_FROZEN",
        }
    )
    _write_new_json(root / "phase5_gate_f_supplement_preoutcome_freeze.json", receipt)
    return receipt


def record_supplement_submission(*, run_root: Path, job_id: str) -> Dict[str, Any]:
    root = _supplement_run_root(run_root)
    if not re.fullmatch(r"[0-9]+", str(job_id)):
        raise Phase5GateFError("supplement job id must be decimal")
    manifest_path = root / "formal" / SUPPLEMENT_MANIFEST
    manifest = _strict_json(manifest_path)
    validate_supplement_launch_manifest(manifest, card=_load_card())
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-supplement-submission.v1",
            "gate": GATE,
            "job_id": str(job_id),
            "submitted_at_utc": utc_now(),
            "launch_manifest_file_sha256": file_sha256(manifest_path),
            "launch_manifest_internal_sha256": manifest["manifest_sha256"],
            "automatic_retry": False,
            "outcome_values_consumed": False,
        }
    )
    _write_new_json(root / "formal" / SUPPLEMENT_SUBMISSION, receipt)
    return receipt


def run_supplement_cell(*, run_root: Path, expected_source_commit: str) -> None:
    root = _supplement_run_root(run_root)
    git = remote_source_provenance(expected_source_commit)
    card = _load_card()
    validate_frozen_inputs(card)
    manifest = _strict_json(root / "formal" / SUPPLEMENT_MANIFEST)
    validate_supplement_launch_manifest(manifest, card=card)
    cell = manifest["cells"][0]
    cell_root = Path(cell["cell_root"])
    cell_root.mkdir(parents=False, exist_ok=False)
    _write_new_json(
        cell_root / "producer_command.json",
        {
            "gate": GATE,
            "mode": "formal",
            "cell": {
                key: cell[key]
                for key in (
                    "array_index",
                    "gate_e_rank",
                    "width_rank",
                    "role",
                    "width",
                    "start",
                    "window",
                    "score",
                    "eval_argv",
                )
            },
            "git": git,
            "launch_manifest_sha256": manifest["manifest_sha256"],
            "outcome_values_consumed": False,
        },
    )
    structural = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-cell-structural.v1",
            "mode": "formal",
            "cell_id": cell["cell_id"],
            "width": 5,
            "window": "15:19",
            "structural": {"formal_loop_audit_not_persisted": True},
            "fixed_smoke_prompt_identity": [],
            "fixed_smoke_prompt_render_sha256": [],
            "outcome_values_consumed": False,
        }
    )
    _write_new_json(cell_root / "structural_receipt.json", structural)
    eval_argv = [str(AUDITED_VENV / "bin/python"), *cell["eval_argv"][1:]]
    completed = subprocess.run(eval_argv, check=False)
    if completed.returncode:
        raise Phase5GateFError(
            "supplement lm-eval producer failed with exit %d" % completed.returncode
        )
    eval_dir = Path(cell["eval_output_dir"])
    results_path = eval_dir / "results.json"
    command_path = eval_dir / "command_args.json"
    revision_path = eval_dir / "model_revision.json"
    for path in (results_path, command_path, revision_path):
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise Phase5GateFError("supplement producer artifact is absent/empty/symlinked")
    result = _strict_json(results_path)
    projected = project_identity_only(result, _load_test_metadata())
    del result
    identity = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-cell-identity.v1",
            "mode": "formal",
            "cell_id": cell["cell_id"],
            "gate_e_rank": 11,
            "width": 5,
            "window": "15:19",
            "record_count": len(projected),
            "subject_count": len({row["subject"] for row in projected}),
            "ordered_identity_sha256": ordered_identity_sha256(
                [row["identity"] for row in projected]
            ),
            "records": projected,
            "projection_allowlist": [
                "task",
                "doc_id",
                "doc_hash",
                "subject",
                "split",
                "canonical_ordinal",
            ],
            "outcome_fields_accessed": False,
            "outcome_values_consumed": False,
        }
    )
    identity_path = cell_root / "identity_sidecar.json"
    _write_new_json(identity_path, identity)
    completion = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-cell-completion.v1",
            "gate": GATE,
            "mode": "formal",
            "cell_id": cell["cell_id"],
            "results": {
                "path": str(results_path),
                "size_bytes": results_path.stat().st_size,
                "file_sha256": file_sha256(results_path),
            },
            "command_args_file_sha256": file_sha256(command_path),
            "model_revision_file_sha256": file_sha256(revision_path),
            "identity_sidecar_file_sha256": file_sha256(identity_path),
            "structural_receipt_file_sha256": file_sha256(
                cell_root / "structural_receipt.json"
            ),
            "exit_state": "PRODUCER_COMPLETED",
            "outcome_values_consumed": False,
        }
    )
    _write_new_json(cell_root / "producer_completion.json", completion)


def _validate_single_scheduler_rows(
    rows: Sequence[Mapping[str, Any]], job_id: str
) -> List[Dict[str, str]]:
    pattern = re.compile(r"^%s_0$" % re.escape(str(job_id)))
    tasks = [
        {key: str(row.get(key, "")) for key in SACCT_FIELDS}
        for row in rows
        if pattern.fullmatch(str(row.get("JobID", "")))
    ]
    if len(tasks) != 1:
        raise Phase5GateFError("supplement seal requires exact one scheduler task")
    state = tasks[0]["State"].split()[0].rstrip("+")
    if state != "COMPLETED" or tasks[0]["ExitCode"] != "0:0":
        raise Phase5GateFError("supplement scheduler task is not COMPLETED/0:0")
    return tasks


def seal_supplement(
    *,
    run_root: Path,
    expected_source_commit: str,
    job_id: str,
    scheduler_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    root = _supplement_run_root(run_root)
    git = remote_source_provenance(expected_source_commit)
    card = _load_card()
    validate_frozen_inputs(card)
    output_path = root / SUPPLEMENT_SEAL
    if output_path.exists():
        raise FileExistsError("supplement seal already exists")
    manifest_path = root / "formal" / SUPPLEMENT_MANIFEST
    manifest = _strict_json(manifest_path)
    validate_supplement_launch_manifest(manifest, card=card)
    submission = _strict_json(root / "formal" / SUPPLEMENT_SUBMISSION)
    _verify_manifest(submission)
    if submission.get("job_id") != str(job_id):
        raise Phase5GateFError("supplement submission/job identity differs")
    terminal = _validate_single_scheduler_rows(
        list(scheduler_rows) if scheduler_rows is not None else query_sacct(job_id),
        job_id,
    )
    entries = _close_cells("formal", manifest)
    elapsed = int(terminal[0]["ElapsedRaw"])
    total_gpu_seconds = elapsed * _gpu_count(terminal[0]["AllocTRES"])
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-supplement-formal-seal.v1",
            "artifact_role": "gate_f_supplement_identity_scheduler_stat_sha_closure",
            "gate": GATE,
            "created_at_utc": utc_now(),
            "git": git,
            "job_id": str(job_id),
            "scheduler_rows": terminal,
            "launch_manifest_path": str(manifest_path),
            "launch_manifest_file_sha256": file_sha256(manifest_path),
            "launch_manifest_internal_sha256": manifest["manifest_sha256"],
            "cell_count": 1,
            "cells": entries,
            "identity_closure": {
                "record_count_per_cell": 14042,
                "subject_count_per_cell": 57,
                "ordered_identity_sha256_all_cells": TEST_ORDERED_IDENTITY_SHA256,
                "missing_duplicate_extra": 0,
            },
            "resource": {
                "total_gpu_seconds": total_gpu_seconds,
                "total_gpu_hours": total_gpu_seconds / 3600.0,
            },
            "outcome_values_consumed": False,
            "accuracy_or_gain_computed": False,
            "results_payload_parsed_by_sealer": False,
            "one_shot_analysis_performed": False,
            "status": "SEALED_COMPLETE_READY_FOR_COMBINED_ANALYSIS",
        }
    )
    _write_new_json(output_path, receipt)
    return receipt


def _validate_supplement_completion(
    root: Path, card: Mapping[str, Any]
) -> Dict[str, Any]:
    completion = _strict_json(_supplement_run_root(root) / SUPPLEMENT_SEAL)
    _verify_manifest(completion)
    identity = completion.get("identity_closure", {})
    if (
        completion.get("status") != "SEALED_COMPLETE_READY_FOR_COMBINED_ANALYSIS"
        or completion.get("cell_count") != 1
        or completion.get("outcome_values_consumed") is not False
        or completion.get("accuracy_or_gain_computed") is not False
        or completion.get("results_payload_parsed_by_sealer") is not False
        or identity.get("record_count_per_cell") != 14042
        or identity.get("subject_count_per_cell") != 57
        or identity.get("ordered_identity_sha256_all_cells")
        != TEST_ORDERED_IDENTITY_SHA256
        or identity.get("missing_duplicate_extra") != 0
    ):
        raise Phase5GateFError("supplement formal structural seal differs")
    expected = derive_supplemental_cell(
        _strict_json(Path(card["source"]["gate_e_score_path"])), card
    )
    row = completion.get("cells", [None])[0]
    observed = {
        key: row.get(key)
        for key in ("array_index", "gate_e_rank", "role", "width", "start", "window", "score")
    }
    expected_without_width_rank = {
        key: expected[key]
        for key in ("array_index", "gate_e_rank", "role", "width", "start", "window", "score")
    }
    if canonical_json_bytes(observed) != canonical_json_bytes(expected_without_width_rank):
        raise Phase5GateFError("supplement completion membership differs")
    result = row.get("results", {})
    path = Path(str(result.get("path", "")))
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != result.get("size_bytes")
        or file_sha256(path) != result.get("file_sha256")
    ):
        raise Phase5GateFError("supplement sealed result stat/SHA differs")
    return completion


def _combined_protected_hashes(
    card: Mapping[str, Any],
    original_root: Path,
    supplement_root: Path,
    original: Mapping[str, Any],
    supplement: Mapping[str, Any],
) -> Dict[str, str]:
    paths = [
        Path(card["source"]["gate_e_score_path"]),
        Path(card["source"]["gate_e_receipt_path"]),
        Path(card["source"]["gate_c_completion_path"]),
        Path(card["source"]["baseline_results_path"]),
        TEST_MANIFEST_PATH,
        TEST_METADATA_PATH,
        _run_root(original_root) / SEAL_NAMES["formal"],
        _supplement_run_root(supplement_root) / SUPPLEMENT_SEAL,
    ]
    paths.extend(Path(row["results"]["path"]) for row in original["cells"])
    paths.extend(Path(row["results"]["path"]) for row in supplement["cells"])
    return {str(path): file_sha256(path) for path in paths}


def _combined_correctness_matrix(
    baseline_path: Path,
    cells: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    if len(cells) != 5:
        raise Phase5GateFError("combined matrix requires exactly five new cells")
    paths = [baseline_path] + [Path(row["results"]["path"]) for row in cells]
    columns = [_load_correctness(path, metadata) for path in paths]
    matrix = np.column_stack([np.asarray(column, dtype=np.int8) for column in columns])
    if matrix.shape != (14042, 6):
        raise Phase5GateFError("combined outcome matrix does not close 14,042 x 6")
    return matrix


def run_combined_one_shot_analysis(
    *,
    original_run_root: Path,
    supplement_run_root: Path,
    combined_run_root: Path,
    expected_source_commit: str,
    implementation_commit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    original_root = _run_root(original_run_root)
    supplement_root = _supplement_run_root(supplement_run_root)
    combined_root = _combined_run_root(combined_run_root)
    if combined_root.exists():
        raise FileExistsError("combined aggregation root already exists")
    card = _load_card()
    git = remote_source_provenance(expected_source_commit)
    frozen = validate_frozen_inputs(card)
    original = _validate_formal_completion(original_root, card)
    supplement = _validate_supplement_completion(supplement_root, card)
    cells = list(original["cells"]) + list(supplement["cells"])
    protected_before = _combined_protected_hashes(
        card, original_root, supplement_root, original, supplement
    )
    combined_root.mkdir(parents=False, exist_ok=False)
    analysis_root = combined_root / "analysis"
    analysis_root.mkdir(parents=False, exist_ok=False)
    raw_hashes = {
        "baseline": frozen["baseline"]["file_sha256"],
        **{row["cell_id"]: row["results"]["file_sha256"] for row in cells},
    }
    unseal = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-combined-unseal-once.v1",
            "gate": GATE,
            "created_at_utc": utc_now(),
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "implementation_commit": implementation_commit,
            "argv": list(argv),
            "original_formal_completion_file_sha256": file_sha256(
                original_root / SEAL_NAMES["formal"]
            ),
            "supplement_formal_completion_file_sha256": file_sha256(
                supplement_root / SUPPLEMENT_SEAL
            ),
            "sealed_result_file_sha256": raw_hashes,
            "unseal_count": 1,
            "analysis_count": 1,
        }
    )
    unseal_sha = _write_new_json(combined_root / OUTPUT_NAMES["unseal"], unseal)
    metadata = _load_test_metadata()
    identities = [row["identity"] for row in metadata]
    subjects = [row["subject"] for row in metadata]
    matrix = _combined_correctness_matrix(
        Path(card["source"]["baseline_results_path"]), cells, metadata
    )
    analysis = build_statistical_analysis(
        correctness=matrix,
        identities=identities,
        subjects=subjects,
        cells=cells,
    )
    analysis_sha = _write_new_json(analysis_root / OUTPUT_NAMES["analysis"], analysis)
    table_sha = _write_new_text(
        analysis_root / OUTPUT_NAMES["table"], _csv_text(analysis["cells"][1:])
    )
    bootstrap_digest = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-combined-bootstrap-digest.v1",
            "gate": GATE,
            "replicates": analysis["bootstrap"]["replicates"],
            "seed": analysis["bootstrap"]["seed"],
            "draw_index_sha256": analysis["bootstrap"]["draw_index_sha256"],
            "cell_gain_summaries": analysis["bootstrap"]["cell_gain_summaries"],
        }
    )
    bootstrap_sha = _write_new_json(
        analysis_root / OUTPUT_NAMES["bootstrap"], bootstrap_digest
    )
    del matrix
    gc.collect()

    verify_matrix = _combined_correctness_matrix(
        Path(card["source"]["baseline_results_path"]), cells, _load_test_metadata()
    )
    verify_analysis = build_statistical_analysis(
        correctness=verify_matrix,
        identities=identities,
        subjects=subjects,
        cells=cells,
    )
    if canonical_json_bytes(verify_analysis) != canonical_json_bytes(analysis):
        raise Phase5GateFError("combined verifier recomputation differs")
    del verify_matrix
    gc.collect()
    protected_after = _combined_protected_hashes(
        card,
        original_root,
        supplement_root,
        _validate_formal_completion(original_root, card),
        _validate_supplement_completion(supplement_root, card),
    )
    if protected_after != protected_before:
        raise Phase5GateFError("protected inputs changed during combined analysis")
    verifier = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-combined-verifier-receipt.v1",
            "gate": GATE,
            "created_at_utc": utc_now(),
            "result": "PASS",
            "verifier_count": 1,
            "checks": {
                "exact_14042_x_6_identity_equality": True,
                "baseline_recomputed_from_original_raw_artifact": True,
                "five_accuracies_recomputed": True,
                "five_paired_gains_and_ci_recomputed": True,
                "five_transition_tables_and_mcnemar_recomputed": True,
                "one_combined_unseal_one_analysis": True,
                "gate_e_top4_plus_rank11_membership_unchanged": True,
                "protected_prior_artifacts_unchanged": True,
                "prospective_selection_success_not_claimed": True,
            },
            "analysis_file_sha256": analysis_sha,
            "raw_result_file_sha256": raw_hashes,
        }
    )
    verifier_sha = _write_new_json(analysis_root / OUTPUT_NAMES["verifier"], verifier)
    summary_sha = _write_new_text(
        analysis_root / OUTPUT_NAMES["summary"], _summary_zh(analysis)
    )
    terminal = _attach_manifest(
        {
            "schema_version": "loopscope.phase5.gate-f-combined-terminal-receipt.v1",
            "gate": GATE,
            "created_at_utc": utc_now(),
            "status": "GATE_F_READY_FOR_PLANNING_AUDIT",
            "label": analysis["label"],
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "planning_thread_id": PLANNING_THREAD_ID,
            "git": git,
            "implementation_commit": implementation_commit,
            "implementation_sha256": implementation_hashes(),
            "combined_run_root": str(combined_root),
            "original_four": {
                "run_root": str(original_root),
                "formal_job_id": original["job_id"],
                "scheduler_rows": original["scheduler_rows"],
                "resource": original["resource"],
                "identity_closure": original["identity_closure"],
                "formal_seal_file_sha256": file_sha256(
                    original_root / SEAL_NAMES["formal"]
                ),
            },
            "supplement_15_19": {
                "run_root": str(supplement_root),
                "formal_job_id": supplement["job_id"],
                "scheduler_rows": supplement["scheduler_rows"],
                "resource": supplement["resource"],
                "identity_closure": supplement["identity_closure"],
                "formal_seal_file_sha256": file_sha256(
                    supplement_root / SUPPLEMENT_SEAL
                ),
            },
            "baseline_reuse": frozen["baseline"],
            "unseal_count": 1,
            "analysis_count": 1,
            "verifier_count": 1,
            "verifier_result": "PASS",
            "outputs": {
                OUTPUT_NAMES["unseal"]: unseal_sha,
                OUTPUT_NAMES["analysis"]: analysis_sha,
                OUTPUT_NAMES["table"]: table_sha,
                OUTPUT_NAMES["bootstrap"]: bootstrap_sha,
                OUTPUT_NAMES["verifier"]: verifier_sha,
                OUTPUT_NAMES["summary"]: summary_sha,
            },
            "claim_boundary": analysis["claim_boundary"],
            "actions_not_taken": {
                "baseline_rerun": True,
                "comparator_15_18_rerun": True,
                "sixth_candidate_or_extra_sweep": True,
                "four_cell_subset_analysis": True,
                "outcome_informed_repair": True,
                "public_git_push": True,
                "automatic_next_gate": True,
            },
            "requested_decision": "PASS / PASS_WITH_FIXES / BLOCK",
        }
    )
    terminal_sha = _write_new_json(
        analysis_root / OUTPUT_NAMES["terminal"], terminal
    )
    return {
        "status": terminal["status"],
        "label": terminal["label"],
        "combined_run_root": str(combined_root),
        "analysis_root": str(analysis_root),
        "terminal_receipt_file_sha256": terminal_sha,
        "verifier_result": "PASS",
        "unseal_count": 1,
        "analysis_count": 1,
        "verifier_count": 1,
        "cells": analysis["cells"],
    }


__all__ = [
    "Phase5GateFError",
    "build_eval_argv",
    "build_launch_manifest",
    "build_supplement_launch_manifest",
    "build_statistical_analysis",
    "derive_top4",
    "derive_supplemental_cell",
    "exact_mcnemar_p",
    "freeze_formal",
    "joint_paired_bootstrap",
    "prepare_smoke",
    "record_submission",
    "record_supplement_submission",
    "run_cell",
    "run_combined_one_shot_analysis",
    "run_one_shot_analysis",
    "run_supplement_cell",
    "prepare_supplement_formal",
    "seal_stage",
    "seal_supplement",
    "validate_extension_card",
    "validate_launch_manifest",
    "validate_scheduler_rows",
    "validate_supplement_launch_manifest",
    "validate_smoke_structural_info",
]
