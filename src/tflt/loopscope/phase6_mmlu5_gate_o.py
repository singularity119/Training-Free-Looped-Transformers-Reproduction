"""LoopScope Phase 6 Gate O: fixed-panel MMLU outcome execution.

The producer and sealer in this module are intentionally outcome-blind.  They
close only the frozen recipe, scheduler rows, result file statistics, and the
safe canonical identity projection.  Outcome values are read by exactly one
``analyze-once`` invocation after the formal completion receipt exists; a
separate ``verify-analysis`` invocation recomputes the aggregate projection.
"""

from __future__ import annotations

import argparse
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase3_p3c import (
    P3CError,
    safe_test_dataset_fields,
    safe_test_doc_sha256,
)
from tflt.loopscope.phase3_schema import (
    ordered_identity_sha256,
    sanitized_content_sha256,
)


GATE = "O"
EXECUTOR_THREAD_ID = "019fc345-d707-78a1-844d-d6755f03bfa1"
PLANNING_THREAD_ID = "019fb3de-2298-75f2-a083-0dca453ea79c"
ADMISSION_BASE = "f95aeb4f3583d30f0bca2b351b741b039a1084eb"
HISTORICAL_BASE = "4f59bd93eca4da3cbf458a93508f91c5b23912bc"

REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")
WORKSPACE = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope"
)
CANONICAL_ROOT = WORKSPACE / "runs/phase3-p3c-20260716T064601Z"
CANONICAL_MANIFEST = CANONICAL_ROOT / "test14042_identity_content_manifest.json"
CANONICAL_METADATA = CANONICAL_ROOT / "test14042_identity_content_metadata.jsonl"

MODEL_REPO = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
DATASET_REPO = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
LM_EVAL_VERSION = "0.4.11"
CANONICAL_MANIFEST_FILE_SHA256 = "59500340aa64d91810cd5a5d40f3d96ded5618ab5508543b31eea93050e7f3e4"
CANONICAL_METADATA_FILE_SHA256 = "3c281447cc7618e58c292a20c13b73d8f42f081692a6927372cf1ee1eebf84df"
CANONICAL_ORDERED_IDENTITY_SHA256 = "2c4557079fca1a7c071460f779ed714805fc96c3391560825cd34386d4051532"

CARD_RELATIVE = Path("configs/loopscope/phase6_mmlu5_v3_outcome_card.json")
CARD_FILE_SHA256 = "c4a8da16b7fe60fa44b69c345e98d0eb0b81ffc99b676c70cc8567f1251d6f27"
CARD_INTERNAL_SHA256 = "279dd91a8f0ea5eaa71c884cd6c41647e382eb2bbbee062ee2430f4b2906a348"

CELLS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("no_loop", "baseline", None),
    ("14:16", "fixed_window", "14:16"),
    ("13:16", "fixed_window", "13:16"),
    ("15:18", "fixed_window", "15:18"),
    ("12:16", "fixed_window", "12:16"),
)
SMOKE_IDENTITIES: Tuple[Tuple[str, str, str], ...] = (
    (
        "mmlu_abstract_algebra",
        "0",
        "307538e6db7f0f40546171e82b7606ccfa2b51dca7d2fdf65a94c350b7413ab1",
    ),
    (
        "mmlu_abstract_algebra",
        "1",
        "dd14a588c453fd5ce4a77194d428f7ad76cb67b7939ae751bcb2ce9d0e4c48bd",
    ),
    (
        "mmlu_anatomy",
        "0",
        "260435a901c78f8f4fb792cb2893f23dc444053829742d2397f3974fe8deb4c2",
    ),
    (
        "mmlu_anatomy",
        "1",
        "edca44324847f85192304489cf7f6021744efddf49294af461c85335da3eed31",
    ),
)

DEBUG_PARTITION = "debug"
DEBUG_GPU = "a40"
DEBUG_TIME_LIMIT = "00:29:00"
FORMAL_PARTITION = "emergency_gpu"
FORMAL_GPU = "a800"
FORMAL_QOS = "emergency_gpu"
FORMAL_TIME_LIMIT = "06:00:00"
ARRAY = "0-4%5"
CPUS_PER_TASK = 8
MEMORY = "64G"
BATCH_SIZE = "16"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260803
REPORT_RELATIVE = Path("资产/报告/phase6/LoopScope_第六阶段GateO_MMLU5_V3固定Panel全量Acc.md")

IMPLEMENTATION_RELATIVE = (
    "configs/loopscope/phase6_mmlu5_v3_outcome_card.json",
    "scripts/loopscope/run_qwen4_phase6_mmlu5_gate_o.py",
    "src/tflt/loopscope/phase6_mmlu5_gate_o.py",
    "tests/test_loopscope_phase6_mmlu5_gate_o.py",
)


class GateOError(RuntimeError):
    """Fail-closed Gate O contract, evidence, or runtime error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateOError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


def _strict_json(path: Path) -> Any:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise GateOError("invalid strict JSON: %s" % path) from exc


def _attach_manifest(value: Mapping[str, Any]) -> Dict[str, Any]:
    body = dict(value)
    body.pop("manifest_sha256", None)
    body["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def _verify_manifest(value: Mapping[str, Any], label: str = "manifest") -> None:
    observed = value.get("manifest_sha256")
    body = dict(value)
    body.pop("manifest_sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    _require(observed == expected, "%s internal SHA256 differs" % label)


def _write_new_bytes(path: Path, body: bytes, *, executable: bool = False) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return hashlib.sha256(body).hexdigest()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return _write_new_bytes(path, body)


def _write_new_text(path: Path, text: str) -> str:
    return _write_new_bytes(Path(path), text.encode("utf-8"))


def _git(*args: str, cwd: Optional[Path] = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd or repository_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise GateOError("git command failed: %s" % " ".join(args))
    return completed.stdout.strip()


def validate_git(expected_commit: str, *, remote_required: bool) -> Dict[str, Any]:
    _require(re.fullmatch(r"[0-9a-f]{40}", str(expected_commit)) is not None, "commit is not a full SHA-1")
    root = repository_root().resolve()
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    head = _git("rev-parse", "HEAD")
    origin = _git("rev-parse", "origin/loopscope")
    dirty = _git("status", "--porcelain")
    _require(branch == "loopscope", "Gate O requires branch loopscope")
    _require(head == expected_commit, "Gate O HEAD differs from expected commit")
    _require(origin == expected_commit, "Gate O origin/loopscope differs from expected commit")
    if remote_required:
        _require(root == REMOTE_REPO.resolve(), "remote action is outside the dedicated clone")
        _require(dirty == "", "remote Gate O clone must be clean")
    _require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ADMISSION_BASE, expected_commit],
            cwd=str(root),
            check=False,
        ).returncode
        == 0,
        "implementation commit is not descended from Gate O admission base",
    )
    _require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", HISTORICAL_BASE, expected_commit],
            cwd=str(root),
            check=False,
        ).returncode
        == 0,
        "implementation commit is not descended from the historical admission base",
    )
    return {
        "repo": str(root),
        "branch": branch,
        "commit": head,
        "origin_loopscope": origin,
        "dirty": bool(dirty),
    }


def implementation_hashes() -> Dict[str, str]:
    root = repository_root()
    result: Dict[str, str] = {}
    for relative in IMPLEMENTATION_RELATIVE:
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), "missing Gate O path: %s" % relative)
        result[relative] = file_sha256(path)
    return result


def _cell_tuples(payload: Mapping[str, Any]) -> List[Tuple[str, str, Optional[str]]]:
    rows = payload.get("cells")
    _require(isinstance(rows, list), "card cells must be a list")
    return [
        (str(row.get("cell_id")), str(row.get("role")), row.get("window"))
        for row in rows
        if isinstance(row, Mapping)
    ]


def validate_card(card: Mapping[str, Any]) -> None:
    _verify_manifest(card, "Gate O card")
    _require(card.get("manifest_sha256") == CARD_INTERNAL_SHA256, "Gate O card internal hash differs")
    _require(card.get("schema_version") == "loopscope.phase6.mmlu5.v3.outcome-card.v1", "Gate O card schema differs")
    _require(card.get("gate") == GATE and card.get("status") == "PLANNING_FROZEN", "Gate O card identity/status differs")
    _require(card.get("model", {}).get("repo") == MODEL_REPO, "Gate O model repo differs")
    _require(card.get("model", {}).get("revision") == MODEL_REVISION, "Gate O model revision differs")
    _require(card.get("model", {}).get("tokenizer_revision") == MODEL_REVISION, "Gate O tokenizer revision differs")
    dataset = card.get("dataset", {})
    _require(
        dataset.get("repo") == DATASET_REPO
        and dataset.get("revision") == DATASET_REVISION
        and dataset.get("task_group") == "mmlu"
        and dataset.get("split") == "test"
        and dataset.get("fewshot_split") == "dev"
        and dataset.get("prompt") == "plain"
        and dataset.get("apply_chat_template") is False
        and dataset.get("generation") is False,
        "Gate O dataset/prompt contract differs",
    )
    evaluator = card.get("evaluator", {})
    _require(
        evaluator.get("name") == "lm-eval"
        and evaluator.get("version") == LM_EVAL_VERSION
        and evaluator.get("mode") == "standard MMLU choice-loglikelihood"
        and evaluator.get("metric") == "acc,none"
        and evaluator.get("num_fewshot") == 5
        and evaluator.get("log_samples") is True,
        "Gate O evaluator contract differs",
    )
    recipe = card.get("loop_recipe", {})
    for key, expected in {
        "dtype": "bfloat16",
        "k": 3,
        "iteration_mode": "block",
        "strategy": "euler",
        "alpha": 1.0,
        "beta": 0.0,
        "cache_strategy": "first",
        "decode_mode": "full",
        "batch_size": BATCH_SIZE,
        "window_width": 4,
        "inclusive_window": True,
    }.items():
        _require(recipe.get(key) == expected, "Gate O loop recipe differs at %s" % key)
    _require(_cell_tuples(card) == list(CELLS), "Gate O card cell order differs")
    population = card.get("population", {})
    _require(
        population.get("record_count") == 14042
        and population.get("subject_count") == 57
        and population.get("manifest_file_sha256") == CANONICAL_MANIFEST_FILE_SHA256
        and population.get("metadata_file_sha256") == CANONICAL_METADATA_FILE_SHA256
        and population.get("ordered_identity_sha256") == CANONICAL_ORDERED_IDENTITY_SHA256,
        "Gate O population closure differs",
    )
    analysis = card.get("analysis", {}).get("paired_bootstrap", {})
    _require(
        analysis.get("replicates") == BOOTSTRAP_REPLICATES
        and analysis.get("seed") == BOOTSTRAP_SEED
        and analysis.get("confidence_level") == 0.95
        and analysis.get("contrast") == "each loop cell versus no_loop",
        "Gate O analysis contract differs",
    )


def load_card() -> Dict[str, Any]:
    path = repository_root() / CARD_RELATIVE
    _require(path.is_file() and not path.is_symlink(), "Gate O card is missing")
    _require(file_sha256(path) == CARD_FILE_SHA256, "Gate O card byte SHA256 differs")
    card = _strict_json(path)
    _require(isinstance(card, dict), "Gate O card root must be an object")
    validate_card(card)
    return dict(card)


def _validate_identity(identity: Mapping[str, Any]) -> Dict[str, str]:
    task = str(identity.get("task", "")).strip()
    doc_id = str(identity.get("doc_id", "")).strip()
    doc_hash = str(identity.get("doc_hash", "")).strip().lower()
    _require(task != "" and re.fullmatch(r"0|[1-9][0-9]*", doc_id) is not None, "canonical task/doc_id is invalid")
    _require(re.fullmatch(r"[0-9a-f]{64}", doc_hash) is not None, "canonical doc_hash is invalid")
    return {"task": task, "doc_id": doc_id, "doc_hash": doc_hash}


def load_canonical_metadata() -> List[Dict[str, Any]]:
    _require(file_sha256(CANONICAL_MANIFEST) == CANONICAL_MANIFEST_FILE_SHA256, "canonical manifest byte hash differs")
    _require(file_sha256(CANONICAL_METADATA) == CANONICAL_METADATA_FILE_SHA256, "canonical metadata byte hash differs")
    manifest = _strict_json(CANONICAL_MANIFEST)
    _require(isinstance(manifest, dict), "canonical manifest root must be an object")
    _require(
        manifest.get("record_count") == 14042
        and manifest.get("subject_count") == 57
        and manifest.get("ordered_identity_sha256") == CANONICAL_ORDERED_IDENTITY_SHA256,
        "canonical manifest population differs",
    )
    records: List[Dict[str, Any]] = []
    with CANONICAL_METADATA.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except (ValueError, TypeError) as exc:
                raise GateOError("invalid canonical metadata row %d" % line_number) from exc
            _require(isinstance(row, Mapping), "canonical metadata row is not an object")
            identity = _validate_identity(row.get("identity", {}))
            subject = str(row.get("subject", "")).strip()
            split = str(row.get("split", "")).strip()
            content_hash = str(row.get("sanitized_content_sha256", "")).strip().lower()
            _require(subject != "" and split == "test", "canonical metadata subject/split differs")
            _require(re.fullmatch(r"[0-9a-f]{64}", content_hash) is not None, "canonical content hash is invalid")
            records.append(
                {
                    "identity": identity,
                    "subject": subject,
                    "split": split,
                    "sanitized_content_sha256": content_hash,
                }
            )
    _require(len(records) == 14042, "canonical metadata record count differs")
    _require(len({row["subject"] for row in records}) == 57, "canonical metadata subject count differs")
    _require(ordered_identity_sha256([row["identity"] for row in records]) == CANONICAL_ORDERED_IDENTITY_SHA256, "canonical ordered identity hash differs")
    pairs = [(row["identity"]["task"], row["identity"]["doc_id"]) for row in records]
    _require(len(set(pairs)) == len(pairs), "canonical metadata task/doc_id is duplicate")
    return records


def _smoke_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    wanted = {(task, doc_id) for task, doc_id, _ in SMOKE_IDENTITIES}
    selected = [
        dict(row)
        for row in records
        if (str(row["identity"]["task"]), str(row["identity"]["doc_id"])) in wanted
    ]
    observed = [
        (
            str(row["identity"]["task"]),
            str(row["identity"]["doc_id"]),
            str(row["identity"]["doc_hash"]),
        )
        for row in selected
    ]
    _require(observed == list(SMOKE_IDENTITIES), "canonical smoke identity order/hash differs")
    return selected


def project_identity_only(
    result: Mapping[str, Any], expected_records: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Project only safe MMLU identities; never inspect target labels or metrics."""

    samples = result.get("samples")
    _require(isinstance(samples, Mapping), "lm-eval result lacks logged samples")
    expected_by_pair: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for record in expected_records:
        identity = _validate_identity(record.get("identity", {}))
        pair = (identity["task"], identity["doc_id"])
        _require(pair not in expected_by_pair, "expected canonical task/doc_id is duplicate")
        expected_by_pair[pair] = record

    observed_pairs = set()
    for task, rows in samples.items():
        _require(isinstance(task, str) and isinstance(rows, list), "lm-eval sample mapping is malformed")
        for row in rows:
            _require(isinstance(row, Mapping), "lm-eval sample row is malformed")
            doc_id = str(row.get("doc_id", "")).strip()
            _require(re.fullmatch(r"0|[1-9][0-9]*", doc_id) is not None, "lm-eval doc_id is invalid")
            pair = (task, doc_id)
            _require(pair not in observed_pairs, "lm-eval task/doc_id is duplicate")
            observed_pairs.add(pair)
            expected = expected_by_pair.get(pair)
            if expected is None:
                continue
            try:
                safe = safe_test_dataset_fields(row.get("doc"), str(expected["subject"]))
            except (P3CError, KeyError, TypeError) as exc:
                raise GateOError("lm-eval safe identity fields are invalid") from exc
            _require(safe_test_doc_sha256(safe) == expected["identity"]["doc_hash"], "safe canonical doc hash differs")
            _require(
                sanitized_content_sha256(safe["question"], safe["choices"])
                == expected["sanitized_content_sha256"],
                "safe sanitized content hash differs",
            )
    expected_pairs = set(expected_by_pair)
    _require(observed_pairs == expected_pairs, "identity projection is incomplete")
    return [
        {
            "ordinal": index,
            "identity": dict(row["identity"]),
            "subject": str(row["subject"]),
            "split": str(row["split"]),
        }
        for index, row in enumerate(expected_records)
    ]


def _cell_eval_argv(mode: str, cell_root: Path, window: Optional[str]) -> List[str]:
    tasks = "mmlu_abstract_algebra,mmlu_anatomy" if mode == "debug" else "mmlu"
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
        str(cell_root / "eval"),
        "--num-fewshot",
        "5",
        "--batch-size",
        BATCH_SIZE,
        "--dtype",
        "bfloat16",
    ]
    if mode == "debug":
        argv.extend(("--limit", "2"))
    if window is not None:
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
                "full",
            )
        )
    return argv


def _cell_runner_argv(mode: str, run_root: Path, cell_index: int, expected_commit: str) -> List[str]:
    return [
        "python",
        "scripts/loopscope/run_qwen4_phase6_mmlu5_gate_o.py",
        "run-cell",
        "--mode",
        mode,
        "--run-root",
        str(run_root),
        "--cell-index",
        str(cell_index),
        "--expected-commit",
        expected_commit,
    ]


def _scheduler_for(mode: str) -> Dict[str, Any]:
    if mode == "debug":
        return {
            "partition": DEBUG_PARTITION,
            "gpu_type": DEBUG_GPU,
            "qos": None,
            "time_limit": DEBUG_TIME_LIMIT,
        }
    if mode == "formal":
        return {
            "partition": FORMAL_PARTITION,
            "gpu_type": FORMAL_GPU,
            "qos": FORMAL_QOS,
            "time_limit": FORMAL_TIME_LIMIT,
        }
    raise GateOError("mode must be debug or formal")


def build_launch_manifest(
    *, mode: str, run_root: Path, expected_commit: str, scheduler: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Build a frozen five-cell launch plan without reading outcomes."""

    _require(mode in {"debug", "formal"}, "mode must be debug or formal")
    _require(re.fullmatch(r"[0-9a-f]{40}", str(expected_commit)) is not None, "expected commit is invalid")
    card = load_card()
    scheduler_values = dict(_scheduler_for(mode))
    if scheduler is not None:
        scheduler_values.update(dict(scheduler))
    expected_scheduler = _scheduler_for(mode)
    for key, expected in expected_scheduler.items():
        _require(scheduler_values.get(key) == expected, "scheduler contract differs at %s" % key)
    cells: List[Dict[str, Any]] = []
    root = Path(run_root).resolve()
    for index, (cell_id, role, window) in enumerate(CELLS):
        cell_root = root / "cells" / ("cell-%02d-%s" % (index, cell_id.replace(":", "_")))
        cells.append(
            {
                "index": index,
                "cell_id": cell_id,
                "role": role,
                "window": window,
                "loop_enabled": window is not None,
                "cell_root": str(cell_root),
                "eval_output_dir": str(cell_root / "eval"),
                "runner_argv": _cell_runner_argv(mode, root, index, expected_commit),
                "eval_argv": _cell_eval_argv(mode, cell_root, window),
                "automatic_retry": False,
            }
        )
    value = {
        "schema_version": "loopscope.phase6.gate-o.%s-launch.v1" % mode,
        "artifact_role": "gate_o_%s_five_cell_launch" % mode,
        "gate": GATE,
        "mode": mode,
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "created_at_utc": utc_now(),
        "run_root": str(root),
        "git": {
            "branch": "loopscope",
            "commit": expected_commit,
            "dirty": False,
            "admission_base": ADMISSION_BASE,
        },
        "implementation_sha256": implementation_hashes(),
        "card_file_sha256": CARD_FILE_SHA256,
        "card_internal_sha256": card["manifest_sha256"],
        "contract": {
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "evaluator": "lm-eval",
            "lm_eval_version": LM_EVAL_VERSION,
            "task_group": "mmlu",
            "split": "test",
            "fewshot_split": "dev",
            "prompt": "plain",
            "apply_chat_template": False,
            "generation": False,
            "num_fewshot": 5,
            "metric": "acc,none",
            "dtype": "bfloat16",
            "k": 3,
            "iteration_mode": "block",
            "strategy": "euler",
            "alpha": 1.0,
            "beta": 0.0,
            "cache_strategy": "first",
            "decode_mode": "full",
            "batch_size": BATCH_SIZE,
        },
        "population": {
            "record_count": 4 if mode == "debug" else 14042,
            "subject_count": 2 if mode == "debug" else 57,
            "canonical_manifest": str(CANONICAL_MANIFEST),
            "canonical_manifest_file_sha256": CANONICAL_MANIFEST_FILE_SHA256,
            "canonical_metadata": str(CANONICAL_METADATA),
            "canonical_metadata_file_sha256": CANONICAL_METADATA_FILE_SHA256,
            "ordered_identity_sha256": (
                ordered_identity_sha256(
                    [
                        {"task": task, "doc_id": doc_id, "doc_hash": doc_hash}
                        for task, doc_id, doc_hash in SMOKE_IDENTITIES
                    ]
                )
                if mode == "debug"
                else CANONICAL_ORDERED_IDENTITY_SHA256
            ),
            "smoke_identities": [
                {"task": task, "doc_id": doc_id, "doc_hash": doc_hash}
                for task, doc_id, doc_hash in SMOKE_IDENTITIES
            ],
        },
        "scheduler": {
            "partition": scheduler_values["partition"],
            "gpu_type": scheduler_values["gpu_type"],
            "qos": scheduler_values["qos"],
            "array": ARRAY,
            "cpus_per_task": CPUS_PER_TASK,
            "memory": MEMORY,
            "time_limit": scheduler_values["time_limit"],
            "no_requeue": True,
        },
        "cells": cells,
        "outcome_values_consumed": False,
        "accuracy_or_gain_computed": False,
        "result_payload_parsed_by_sealer": False,
    }
    manifest = _attach_manifest(value)
    validate_launch_manifest(manifest, mode=mode)
    return manifest


def validate_launch_manifest(value: Mapping[str, Any], *, mode: str) -> None:
    _verify_manifest(value, "Gate O launch manifest")
    _require(value.get("gate") == GATE and value.get("mode") == mode, "launch identity differs")
    _require(value.get("executor_thread_id") == EXECUTOR_THREAD_ID, "executor thread differs")
    _require(value.get("planning_thread_id") == PLANNING_THREAD_ID, "planning thread differs")
    _require(value.get("outcome_values_consumed") is False, "launch manifest crossed outcome barrier")
    _require(value.get("accuracy_or_gain_computed") is False, "launch manifest contains analysis")
    _require(value.get("result_payload_parsed_by_sealer") is False, "launch manifest sealer barrier differs")
    _require(value.get("card_file_sha256") == CARD_FILE_SHA256, "launch card byte hash differs")
    _require(value.get("card_internal_sha256") == CARD_INTERNAL_SHA256, "launch card internal hash differs")
    contract = value.get("contract", {})
    expected_contract = {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "evaluator": "lm-eval",
        "lm_eval_version": LM_EVAL_VERSION,
        "task_group": "mmlu",
        "split": "test",
        "fewshot_split": "dev",
        "prompt": "plain",
        "apply_chat_template": False,
        "generation": False,
        "num_fewshot": 5,
        "metric": "acc,none",
        "dtype": "bfloat16",
        "k": 3,
        "iteration_mode": "block",
        "strategy": "euler",
        "alpha": 1.0,
        "beta": 0.0,
        "cache_strategy": "first",
        "decode_mode": "full",
        "batch_size": BATCH_SIZE,
    }
    for key, expected in expected_contract.items():
        _require(contract.get(key) == expected, "launch contract differs at %s" % key)
    population = value.get("population", {})
    expected_count = 4 if mode == "debug" else 14042
    expected_subjects = 2 if mode == "debug" else 57
    expected_identity = (
        ordered_identity_sha256(
            [{"task": task, "doc_id": doc_id, "doc_hash": doc_hash} for task, doc_id, doc_hash in SMOKE_IDENTITIES]
        )
        if mode == "debug"
        else CANONICAL_ORDERED_IDENTITY_SHA256
    )
    _require(
        population.get("record_count") == expected_count
        and population.get("subject_count") == expected_subjects
        and population.get("ordered_identity_sha256") == expected_identity,
        "launch population differs",
    )
    scheduler = value.get("scheduler", {})
    expected_scheduler = _scheduler_for(mode)
    for key, expected in expected_scheduler.items():
        _require(scheduler.get(key) == expected, "launch scheduler differs at %s" % key)
    _require(
        scheduler.get("array") == ARRAY
        and scheduler.get("cpus_per_task") == CPUS_PER_TASK
        and scheduler.get("memory") == MEMORY
        and scheduler.get("no_requeue") is True,
        "launch scheduler resource closure differs",
    )
    rows = value.get("cells")
    _require(isinstance(rows, list) and len(rows) == len(CELLS), "Gate O requires exactly five cells")
    observed = [
        (row.get("cell_id"), row.get("role"), row.get("window"))
        for row in rows
        if isinstance(row, Mapping)
    ]
    _require(observed == list(CELLS), "Gate O cell membership/order differs")
    for index, row in enumerate(rows):
        _require(row.get("index") == index, "Gate O cell index differs")
        _require(row.get("automatic_retry") is False, "automatic retry is forbidden")
        _require(row.get("loop_enabled") is (index != 0), "loop-enabled flag differs")
        expected_eval = _cell_eval_argv(mode, Path(str(row["cell_root"])).resolve().parent.parent.parent, row.get("window"))
        # The helper above derives output paths from the cell root.  Rebuild it
        # with the recorded cell path so validation stays independent of cwd.
        expected_eval = _cell_eval_argv(mode, Path(str(row["cell_root"])).resolve().parent.parent.parent, row.get("window"))
        # _cell_eval_argv expects a run-root child and appends cells/cell-...;
        # compare the stable argv tail and the recorded output path separately.
        recorded_eval = list(row.get("eval_argv", []))
        _require(recorded_eval[:9] == expected_eval[:9], "cell evaluator prefix differs")
        _require(recorded_eval[9:] == _cell_eval_suffix(mode, row.get("window"), str(row["eval_output_dir"])), "cell evaluator recipe differs")
        runner = row.get("runner_argv")
        _require(isinstance(runner, list) and runner[0] == "python" and runner[1:3] == ["scripts/loopscope/run_qwen4_phase6_mmlu5_gate_o.py", "run-cell"], "cell runner is not the Gate O CLI")
        if index == 0:
            _require("--loop" not in recorded_eval and "--decode-mode" not in recorded_eval, "baseline must use native no-loop path")
        else:
            _require("--loop" in recorded_eval and "--decode-mode" in recorded_eval and recorded_eval[recorded_eval.index("--decode-mode") + 1] == "full", "loop cell decode contract differs")


def _cell_eval_suffix(mode: str, window: Optional[str], output_dir: str) -> List[str]:
    suffix = ["--output-dir", output_dir, "--num-fewshot", "5", "--batch-size", BATCH_SIZE, "--dtype", "bfloat16"]
    if mode == "debug":
        suffix.extend(("--limit", "2"))
    if window is not None:
        suffix.extend((
            "--loop", "--window", window, "--k", "3", "--iteration-mode", "block",
            "--strategy", "euler", "--alpha", "1.0", "--beta", "0.0",
            "--cache-strategy", "first", "--decode-mode", "full",
        ))
    return suffix


def _common_environment() -> str:
    return """set -euo pipefail
source {venv}/bin/activate
cd {repo}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH={repo}/src
export HF_HOME={hf_home}
export TRANSFORMERS_CACHE={hf_home}/hub
export HF_DATASETS_CACHE={datasets}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
unset HF_ENDPOINT || true
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
""".format(
        venv=shlex.quote(str(AUDITED_VENV)),
        repo=shlex.quote(str(REMOTE_REPO)),
        hf_home=shlex.quote(str(HF_HOME)),
        datasets=shlex.quote(str(HF_DATASETS_CACHE)),
    )


def build_sbatch_text(*, mode: str, run_root: Path, expected_commit: str) -> str:
    _require(mode in {"debug", "formal"}, "mode must be debug or formal")
    manifest_path = Path(run_root).resolve() / "manifest" / "launch_manifest.json"
    manifest = _strict_json(manifest_path)
    _require(isinstance(manifest, dict), "launch manifest root must be an object")
    validate_launch_manifest(manifest, mode=mode)
    scheduler = manifest["scheduler"]
    runner = REMOTE_REPO / "scripts/loopscope/run_qwen4_phase6_mmlu5_gate_o.py"
    lines = [
        "#!/usr/bin/env bash",
        "# LoopScope Phase 6 Gate O fixed five-cell array.",
        "#SBATCH --job-name=loopscope-p6-o-mmlu5-%s" % mode,
        "#SBATCH --partition=%s" % scheduler["partition"],
        "#SBATCH --array=%s" % scheduler["array"],
        "#SBATCH --time=%s" % scheduler["time_limit"],
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=%d" % scheduler["cpus_per_task"],
        "#SBATCH --mem=%s" % scheduler["memory"],
        "#SBATCH --gres=gpu:%s:1" % scheduler["gpu_type"],
        "#SBATCH --no-requeue",
        "#SBATCH --output=%s/slurm/gate-o-%%A_%%a.out" % shlex.quote(str(Path(run_root).resolve())),
        "#SBATCH --error=%s/slurm/gate-o-%%A_%%a.err" % shlex.quote(str(Path(run_root).resolve())),
    ]
    if scheduler.get("qos"):
        lines.append("#SBATCH --qos=%s" % scheduler["qos"])
    lines.extend(
        [
            _common_environment().rstrip("\n"),
            "exec %s %s run-cell --mode %s --run-root %s --cell-index \"$SLURM_ARRAY_TASK_ID\" --expected-commit %s"
            % (
                shlex.quote(str(AUDITED_VENV / "bin/python")),
                shlex.quote(str(runner)),
                mode,
                shlex.quote(str(Path(run_root).resolve())),
                shlex.quote(expected_commit),
            ),
            "",
        ]
    )
    text = "\n".join(lines)
    _require("#SBATCH --array=0-4%5" in text, "Gate O array is not 0-4%5")
    _require("--cpus-per-task=8" in text and "--mem=64G" in text, "Gate O resources drifted")
    _require("--decode-mode full" not in text, "array runner must not encode a second recipe")
    return text


def prepare_run(*, mode: str, run_root: Path, expected_commit: str) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    _require(not root.exists(), "Gate O run root must be fresh/write-once")
    validate_git(expected_commit, remote_required=True)
    load_card()
    records = load_canonical_metadata()
    _require(len(records) == 14042 and len({row["subject"] for row in records}) == 57, "canonical metadata is not formal")
    root.mkdir(parents=True, exist_ok=False)
    (root / "manifest").mkdir(exist_ok=False)
    (root / "slurm").mkdir(exist_ok=False)
    (root / "cells").mkdir(exist_ok=False)
    manifest = build_launch_manifest(mode=mode, run_root=root, expected_commit=expected_commit)
    launch_sha = _write_new_json(root / "manifest/launch_manifest.json", manifest)
    sbatch_body = build_sbatch_text(mode=mode, run_root=root, expected_commit=expected_commit)
    sbatch_sha = _write_new_bytes(root / "slurm/gate_o_array.sbatch", sbatch_body.encode("utf-8"), executable=True)
    freeze = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o.%s-freeze.v1" % mode,
            "gate": GATE,
            "mode": mode,
            "run_root": str(root),
            "created_at_utc": utc_now(),
            "git": validate_git(expected_commit, remote_required=True),
            "launch_manifest_file_sha256": launch_sha,
            "launch_manifest_internal_sha256": manifest["manifest_sha256"],
            "launcher_file_sha256": sbatch_sha,
            "record_count": manifest["population"]["record_count"],
            "subject_count": manifest["population"]["subject_count"],
            "cells": list(CELLS),
            "outcome_values_consumed": False,
            "status": "%s_FROZEN" % mode.upper(),
        }
    )
    freeze_sha = _write_new_json(root / "manifest/freeze_receipt.json", freeze)
    return {
        "run_root": str(root),
        "mode": mode,
        "launch_manifest_file_sha256": launch_sha,
        "launcher_file_sha256": sbatch_sha,
        "freeze_receipt_file_sha256": freeze_sha,
        "outcome_values_consumed": False,
    }


def submit_run(run_root: Path) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    manifest = _strict_json(root / "manifest/launch_manifest.json")
    _require(isinstance(manifest, dict), "launch manifest root is invalid")
    mode = str(manifest.get("mode"))
    validate_launch_manifest(manifest, mode=mode)
    submission_path = root / "manifest/submission.json"
    _require(not submission_path.exists(), "submission receipt already exists")
    script = root / "slurm/gate_o_array.sbatch"
    completed = subprocess.run(
        ["/opt/slurm/bin/sbatch", "--parsable", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise GateOError("sbatch failed: %s" % completed.stderr.strip())
    job_id = completed.stdout.strip().split(";", 1)[0]
    _require(re.fullmatch(r"[0-9]+", job_id) is not None, "sbatch did not return a decimal Job ID")
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o.%s-submission.v1" % mode,
            "gate": GATE,
            "mode": mode,
            "run_root": str(root),
            "job_id": job_id,
            "submitted_at_utc": utc_now(),
            "launch_manifest_file_sha256": file_sha256(root / "manifest/launch_manifest.json"),
            "launch_manifest_internal_sha256": manifest["manifest_sha256"],
            "automatic_retry": False,
            "outcome_values_consumed": False,
        }
    )
    _write_new_json(submission_path, receipt)
    return receipt


def record_submission(*, run_root: Path, job_id: str) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    manifest = _strict_json(root / "manifest/launch_manifest.json")
    _require(isinstance(manifest, dict), "launch manifest root is invalid")
    mode = str(manifest.get("mode"))
    validate_launch_manifest(manifest, mode=mode)
    _require(re.fullmatch(r"[0-9]+", str(job_id)) is not None, "job id must be decimal")
    path = root / "manifest/submission.json"
    _require(not path.exists(), "submission receipt already exists")
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o.%s-submission.v1" % mode,
            "gate": GATE,
            "mode": mode,
            "run_root": str(root),
            "job_id": str(job_id),
            "submitted_at_utc": utc_now(),
            "launch_manifest_file_sha256": file_sha256(root / "manifest/launch_manifest.json"),
            "launch_manifest_internal_sha256": manifest["manifest_sha256"],
            "automatic_retry": False,
            "outcome_values_consumed": False,
        }
    )
    _write_new_json(path, receipt)
    return receipt


SACCT_FIELDS = ("JobID", "State", "ExitCode", "ElapsedRaw", "AllocTRES", "Partition", "NodeList")


def query_sacct(job_id: str) -> List[Dict[str, str]]:
    _require(re.fullmatch(r"[0-9]+", str(job_id)) is not None, "job id must be decimal")
    commands = [
        [
            "/opt/slurm/bin/sacct",
            "-j",
            str(job_id),
            "--array",
            "--noheader",
            "--parsable2",
            "--format=" + ",".join(SACCT_FIELDS),
        ],
        [
            "sacct",
            "-j",
            str(job_id),
            "--array",
            "--noheader",
            "--parsable2",
            "--format=" + ",".join(SACCT_FIELDS),
        ],
    ]
    completed = None
    for command in commands:
        candidate = subprocess.run(command, capture_output=True, text=True, check=False)
        if candidate.returncode == 0:
            completed = candidate
            break
    _require(completed is not None, "sacct failed")
    rows: List[Dict[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        while len(values) > len(SACCT_FIELDS) and values[-1] == "":
            values.pop()
        _require(len(values) == len(SACCT_FIELDS), "unexpected sacct field count")
        rows.append(dict(zip(SACCT_FIELDS, values)))
    return rows


def validate_scheduler_rows(rows: Sequence[Mapping[str, Any]], job_id: str) -> List[Dict[str, str]]:
    _require(re.fullmatch(r"[0-9]+", str(job_id)) is not None, "job id must be decimal")
    pattern = re.compile(r"^%s_([0-4])$" % re.escape(str(job_id)))
    tasks: Dict[int, Dict[str, str]] = {}
    for raw in rows:
        match = pattern.fullmatch(str(raw.get("JobID", "")))
        if not match:
            continue
        index = int(match.group(1))
        _require(index not in tasks, "scheduler rows contain duplicate array task")
        tasks[index] = {key: str(raw.get(key, "")) for key in SACCT_FIELDS}
    _require(set(tasks) == set(range(5)), "scheduler terminal evidence lacks exact tasks 0-4")
    for index in range(5):
        row = tasks[index]
        state = row["State"].split()[0].rstrip("+")
        _require(state == "COMPLETED" and row["ExitCode"] == "0:0", "task %d is not COMPLETED/0:0" % index)
    return [tasks[index] for index in range(5)]


def _smoke_prompts() -> List[Dict[str, str]]:
    try:
        from tflt.loopscope.mmlu_renderer import LmEvalMMLURendererBackend

        collected = LmEvalMMLURendererBackend().collect(
            task_names=("mmlu_abstract_algebra", "mmlu_anatomy"),
            dataset_revision=DATASET_REVISION,
            target_split="test",
            fewshot_split="dev",
            seed=1234,
            max_targets_per_task=2,
        )
    except Exception as exc:
        raise GateOError("debug renderer dependency failed") from exc
    records = collected.get("records") if isinstance(collected, Mapping) else None
    _require(isinstance(records, list) and len(records) == 4, "debug renderer did not return four prompts")
    observed = [
        (str(row.get("task_name")), str(row.get("target_doc_index")), str(row.get("target_doc_sha256")))
        for row in records
    ]
    _require(observed == list(SMOKE_IDENTITIES), "debug renderer identities differ")
    return [
        {
            "identity": "%s:%s:%s" % identity,
            "text": str(row["text"]),
            "render_sha256": str(row["render_sha256"]),
        }
        for identity, row in zip(SMOKE_IDENTITIES, records)
    ]


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
        "full",
        "--dtype",
        "bfloat16",
        "--device",
        "cuda",
    ]
    for prompt in prompts:
        argv.extend(("--prompt", str(prompt["text"])))
    return argv


def _assert_finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), "non-finite audit value at %s" % path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, "%s.%s" % (path, key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_tree(item, "%s[%d]" % (path, index))


def _validate_loop_audit(report: Mapping[str, Any], window: str) -> Dict[str, Any]:
    _assert_finite_tree(report)
    config = report.get("loop_config", {})
    _require(
        config.get("window") == window
        and config.get("k") == 3
        and config.get("iteration_mode") == "block"
        and config.get("strategy") == "euler"
        and config.get("alpha") == 1.0
        and config.get("beta") == 0.0
        and config.get("cache_strategy") == "first"
        and config.get("decode_mode") == "full",
        "debug loop audit recipe differs",
    )
    prompts = report.get("prompts")
    overall = report.get("overall_decision", {})
    _require(isinstance(prompts, list) and prompts, "debug loop audit prompts are absent")
    _require(overall.get("code") == "loop_effective_logits_changed", "loop audit did not prove effective logits")
    for prompt in prompts:
        decision = prompt.get("decision", {})
        _require(
            decision.get("operator_body_calls") == 3
            and decision.get("code") == "loop_effective_logits_changed"
            and decision.get("restore_allclose") is True
            and decision.get("bypass_true") == 0
            and decision.get("bypass_false") > 0
            and float(decision.get("max_final_logits_diff", 0.0)) > 1e-12,
            "debug loop structural closure differs",
        )
    return {
        "overall_decision": "loop_effective_logits_changed",
        "prompt_count": len(prompts),
        "operator_body_calls_per_prompt": [3 for _ in prompts],
        "bypass_true_per_prompt": [0 for _ in prompts],
        "restore_allclose_all_prompts": True,
        "effective_logits_changed_all_prompts": True,
        "decode_mode": "full",
    }


def _expected_records_for_mode(mode: str, records: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return _smoke_records(records) if mode == "debug" else list(records)


def _validate_eval_command_args(
    args: Mapping[str, Any], cell: Mapping[str, Any], mode: str
) -> None:
    expected_tasks = "mmlu_abstract_algebra,mmlu_anatomy" if mode == "debug" else "mmlu"
    _require(args.get("model") == MODEL_REPO and args.get("revision") == MODEL_REVISION, "producer model/revision differs")
    _require(args.get("tasks") == expected_tasks, "producer task list differs")
    _require(args.get("num_fewshot") == 5 and args.get("batch_size") == BATCH_SIZE and args.get("dtype") == "bfloat16", "producer fewshot/batch/dtype differs")
    _require(args.get("output_dir") == cell["eval_output_dir"], "producer output path differs")
    _require(args.get("loop") is bool(cell["loop_enabled"]), "producer loop flag differs")
    if mode == "debug":
        _require(args.get("limit") == 2, "debug producer limit differs")
    else:
        _require(args.get("limit") is None, "formal producer must not be limited")
    if cell["loop_enabled"]:
        _require(
            args.get("window") == cell["window"]
            and args.get("k") == 3
            and args.get("iteration_mode") == "block"
            and args.get("strategy") == "euler"
            and args.get("alpha") == 1.0
            and args.get("beta") == 0.0
            and args.get("cache_strategy") == "first"
            and args.get("decode_mode") == "full",
            "producer loop recipe differs",
        )


def _validate_model_revision(payload: Mapping[str, Any]) -> None:
    _require(payload.get("repo_id") == MODEL_REPO, "model revision repo differs")
    _require(
        payload.get("model_commit") == MODEL_REVISION
        and payload.get("tokenizer_commit") == MODEL_REVISION
        and payload.get("manifest_commit") == MODEL_REVISION
        and payload.get("match") is True,
        "model/tokenizer revision closure differs",
    )
    _require(payload.get("lm_eval_version") == LM_EVAL_VERSION, "remote lm-eval version differs")


def run_cell(*, mode: str, run_root: Path, cell_index: int, expected_commit: str) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    _require(mode in {"debug", "formal"} and 0 <= int(cell_index) < len(CELLS), "run-cell mode/index differs")
    validate_git(expected_commit, remote_required=True)
    load_card()
    manifest = _strict_json(root / "manifest/launch_manifest.json")
    _require(isinstance(manifest, dict), "launch manifest root is invalid")
    validate_launch_manifest(manifest, mode=mode)
    _require(manifest["run_root"] == str(root), "run-cell root differs from manifest")
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        _require(int(os.environ["SLURM_ARRAY_TASK_ID"]) == int(cell_index), "Slurm array index differs")
    cell = manifest["cells"][int(cell_index)]
    cell_root = Path(cell["cell_root"])
    _require(not cell_root.exists(), "cell output root already exists")
    cell_root.mkdir(parents=False, exist_ok=False)
    producer_command = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o-cell-command.v1",
            "gate": GATE,
            "mode": mode,
            "cell": {key: cell[key] for key in ("index", "cell_id", "role", "window", "loop_enabled", "eval_argv")},
            "git": {"commit": expected_commit, "repo": str(repository_root())},
            "contract": manifest["contract"],
            "outcome_values_consumed": False,
        }
    )
    _write_new_json(cell_root / "producer_command.json", producer_command)

    if mode == "debug" and cell["loop_enabled"]:
        prompts = _smoke_prompts()
        audit_dir = cell_root / "audit"
        completed = subprocess.run(_audit_command(cell, audit_dir, prompts), check=False)
        if completed.returncode:
            raise GateOError("loop-effect audit failed with exit %d" % completed.returncode)
        structural = _validate_loop_audit(_strict_json(audit_dir / "audit_report.json"), str(cell["window"]))
        smoke_prompt_ids = [row["identity"] for row in prompts]
        smoke_render_hashes = [row["render_sha256"] for row in prompts]
    elif mode == "debug":
        structural = {
            "overall_decision": "native_baseline_no_loop_activity",
            "prompt_count": 4,
            "operator_body_calls_per_prompt": [0, 0, 0, 0],
            "bypass_true_per_prompt": [0, 0, 0, 0],
            "restore_allclose_all_prompts": True,
            "effective_logits_changed_all_prompts": False,
            "decode_mode": None,
        }
        smoke_prompt_ids = ["mmlu_abstract_algebra:0", "mmlu_abstract_algebra:1", "mmlu_anatomy:0", "mmlu_anatomy:1"]
        smoke_render_hashes = []
    else:
        structural = {
            "formal_loop_audit_not_persisted": True,
            "outcome_values_consumed": False,
        }
        smoke_prompt_ids = []
        smoke_render_hashes = []
    structural_receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o-cell-structural.v1",
            "gate": GATE,
            "mode": mode,
            "cell_id": cell["cell_id"],
            "window": cell["window"],
            "loop_enabled": cell["loop_enabled"],
            "structural": structural,
            "fixed_smoke_prompt_identity": smoke_prompt_ids,
            "fixed_smoke_prompt_render_sha256": smoke_render_hashes,
            "outcome_values_consumed": False,
        }
    )
    structural_path = cell_root / "structural_receipt.json"
    _write_new_json(structural_path, structural_receipt)

    eval_argv = [str(AUDITED_VENV / "bin/python"), *list(cell["eval_argv"])[1:]]
    completed = subprocess.run(eval_argv, check=False)
    if completed.returncode:
        raise GateOError("lm-eval producer failed with exit %d" % completed.returncode)
    eval_dir = Path(cell["eval_output_dir"])
    results_path = eval_dir / "results.json"
    command_path = eval_dir / "command_args.json"
    revision_path = eval_dir / "model_revision.json"
    for path in (results_path, command_path, revision_path):
        _require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0, "producer artifact absent/empty: %s" % path)
    command_args = _strict_json(command_path)
    revision = _strict_json(revision_path)
    _require(isinstance(command_args, dict) and isinstance(revision, dict), "producer metadata roots are invalid")
    _validate_eval_command_args(command_args, cell, mode)
    _validate_model_revision(revision)

    records = load_canonical_metadata()
    expected_records = _expected_records_for_mode(mode, records)
    result = _strict_json(results_path)
    _require(isinstance(result, dict), "lm-eval result root is invalid")
    projected = project_identity_only(result, expected_records)
    del result
    identity = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o-cell-identity.v1",
            "gate": GATE,
            "mode": mode,
            "cell_id": cell["cell_id"],
            "role": cell["role"],
            "window": cell["window"],
            "loop_enabled": cell["loop_enabled"],
            "record_count": len(projected),
            "subject_count": len({row["subject"] for row in projected}),
            "ordered_identity_sha256": ordered_identity_sha256([row["identity"] for row in projected]),
            "records": projected,
            "projection_allowlist": ["task", "doc_id", "doc_hash", "subject", "split", "canonical_ordinal"],
            "safe_test_doc_hash_projection": True,
            "lm_eval_internal_doc_hash_consumed_as_canonical": False,
            "outcome_fields_accessed": False,
            "outcome_values_consumed": False,
        }
    )
    identity_path = cell_root / "identity_sidecar.json"
    _write_new_json(identity_path, identity)
    completion = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o-cell-completion.v1",
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
            "structural_receipt_file_sha256": file_sha256(structural_path),
            "exit_state": "PRODUCER_COMPLETED",
            "outcome_values_consumed": False,
        }
    )
    completion_path = cell_root / "producer_completion.json"
    completion_sha = _write_new_json(completion_path, completion)
    return {
        "cell_id": cell["cell_id"],
        "producer_completion_file_sha256": completion_sha,
        "result_file_sha256": completion["results"]["file_sha256"],
        "identity_sidecar_file_sha256": completion["identity_sidecar_file_sha256"],
        "outcome_values_consumed": False,
    }


def _validate_structural_receipt(
    structural: Mapping[str, Any], cell: Mapping[str, Any], mode: str
) -> None:
    _verify_manifest(structural, "structural receipt")
    _require(structural.get("cell_id") == cell["cell_id"], "structural cell differs")
    _require(structural.get("outcome_values_consumed") is False, "structural receipt crossed outcome barrier")
    info = structural.get("structural", {})
    if mode == "debug" and cell["loop_enabled"]:
        _require(
            info.get("overall_decision") == "loop_effective_logits_changed"
            and info.get("operator_body_calls_per_prompt")
            and all(value == 3 for value in info["operator_body_calls_per_prompt"])
            and all(value == 0 for value in info.get("bypass_true_per_prompt", []))
            and info.get("restore_allclose_all_prompts") is True
            and info.get("effective_logits_changed_all_prompts") is True
            and info.get("decode_mode") == "full",
            "debug loop structural closure differs",
        )
    elif mode == "debug":
        _require(
            info.get("overall_decision") == "native_baseline_no_loop_activity"
            and all(value == 0 for value in info.get("operator_body_calls_per_prompt", [])),
            "debug baseline structural closure differs",
        )
    else:
        _require(info.get("formal_loop_audit_not_persisted") is True, "formal structural receipt differs")


def _close_cells(manifest: Mapping[str, Any], mode: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for cell in manifest["cells"]:
        cell_root = Path(cell["cell_root"])
        completion_path = cell_root / "producer_completion.json"
        identity_path = cell_root / "identity_sidecar.json"
        structural_path = cell_root / "structural_receipt.json"
        for path in (completion_path, identity_path, structural_path):
            _require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0, "cell closure artifact absent/empty")
        completion = _strict_json(completion_path)
        identity = _strict_json(identity_path)
        structural = _strict_json(structural_path)
        _require(isinstance(completion, dict) and isinstance(identity, dict) and isinstance(structural, dict), "cell closure roots invalid")
        _verify_manifest(completion, "producer completion")
        _verify_manifest(identity, "identity sidecar")
        _validate_structural_receipt(structural, cell, mode)
        population = manifest["population"]
        _require(
            completion.get("cell_id") == cell["cell_id"]
            and completion.get("exit_state") == "PRODUCER_COMPLETED"
            and completion.get("outcome_values_consumed") is False
            and identity.get("cell_id") == cell["cell_id"]
            and identity.get("record_count") == population["record_count"]
            and identity.get("subject_count") == population["subject_count"]
            and identity.get("ordered_identity_sha256") == population["ordered_identity_sha256"]
            and identity.get("outcome_fields_accessed") is False
            and identity.get("outcome_values_consumed") is False,
            "cell identity/completion closure differs",
        )
        result_info = completion.get("results", {})
        results_path = Path(str(result_info.get("path", "")))
        _require(
            results_path == Path(cell["eval_output_dir"]) / "results.json"
            and results_path.is_file()
            and not results_path.is_symlink()
            and results_path.stat().st_size == result_info.get("size_bytes")
            and file_sha256(results_path) == result_info.get("file_sha256"),
            "cell result stat/SHA closure differs",
        )
        entries.append(
            {
                "index": cell["index"],
                "cell_id": cell["cell_id"],
                "role": cell["role"],
                "window": cell["window"],
                "results": dict(result_info),
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


def seal_run(*, mode: str, run_root: Path, job_id: str, scheduler_rows: Optional[Sequence[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    manifest = _strict_json(root / "manifest/launch_manifest.json")
    _require(isinstance(manifest, dict), "launch manifest root is invalid")
    validate_launch_manifest(manifest, mode=mode)
    completion_path = root / ("manifest/%s_completion.json" % mode)
    _require(not completion_path.exists(), "completion receipt already exists")
    submission = _strict_json(root / "manifest/submission.json")
    _require(isinstance(submission, dict), "submission receipt root is invalid")
    _verify_manifest(submission, "submission receipt")
    _require(submission.get("job_id") == str(job_id), "submission job id differs")
    terminal_rows = validate_scheduler_rows(list(scheduler_rows) if scheduler_rows is not None else query_sacct(job_id), job_id)
    cells = _close_cells(manifest, mode)
    total_gpu_seconds = 0
    for row in terminal_rows:
        try:
            total_gpu_seconds += int(row.get("ElapsedRaw", "0"))
        except (TypeError, ValueError) as exc:
            raise GateOError("scheduler ElapsedRaw is invalid") from exc
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o.%s-completion.v1" % mode,
            "artifact_role": "gate_o_%s_preoutcome_closure" % mode,
            "gate": GATE,
            "mode": mode,
            "run_root": str(root),
            "created_at_utc": utc_now(),
            "job_id": str(job_id),
            "scheduler_rows": terminal_rows,
            "launch_manifest_file_sha256": file_sha256(root / "manifest/launch_manifest.json"),
            "launch_manifest_internal_sha256": manifest["manifest_sha256"],
            "cell_count": 5,
            "cells": cells,
            "identity_closure": {
                "record_count_per_cell": manifest["population"]["record_count"],
                "subject_count_per_cell": manifest["population"]["subject_count"],
                "ordered_identity_sha256_all_cells": manifest["population"]["ordered_identity_sha256"],
                "missing_duplicate_extra": 0,
            },
            "resource": {
                "total_gpu_seconds": total_gpu_seconds,
                "total_gpu_hours": total_gpu_seconds / 3600.0,
            },
            "outcome_values_consumed": False,
            "accuracy_or_gain_computed": False,
            "results_payload_parsed_by_sealer": False,
            "status": "DEBUG_PASS" if mode == "debug" else "SEALED_COMPLETE_READY_FOR_ANALYSIS",
        }
    )
    _write_new_json(completion_path, receipt)
    return receipt


def verify_completion(*, mode: str, run_root: Path) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    manifest = _strict_json(root / "manifest/launch_manifest.json")
    completion_path = root / ("manifest/%s_completion.json" % mode)
    completion = _strict_json(completion_path)
    _require(isinstance(manifest, dict) and isinstance(completion, dict), "preoutcome verifier roots are invalid")
    validate_launch_manifest(manifest, mode=mode)
    _verify_manifest(completion, "completion receipt")
    _require(completion.get("status") in {"DEBUG_PASS", "SEALED_COMPLETE_READY_FOR_ANALYSIS"}, "completion status differs")
    _require(completion.get("cell_count") == 5 and completion.get("outcome_values_consumed") is False, "completion barrier differs")
    expected_cells = _close_cells(manifest, mode)
    _require(completion.get("cells") == expected_cells, "fresh preoutcome cell closure differs")
    verification_dir = root / "verification"
    verification_dir.mkdir(parents=True, exist_ok=False)
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o.%s-preoutcome-verifier.v1" % mode,
            "gate": GATE,
            "mode": mode,
            "run_root": str(root),
            "verified_at_utc": utc_now(),
            "completion_file_sha256": file_sha256(completion_path),
            "completion_manifest_sha256": completion["manifest_sha256"],
            "cell_count": 5,
            "identity_closure": completion["identity_closure"],
            "results_payload_parsed": False,
            "outcome_values_consumed": False,
            "status": "PASS",
        }
    )
    path = verification_dir / "preoutcome_verifier_receipt.json"
    _write_new_json(path, receipt)
    return receipt


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    _require(values and 0.0 <= quantile <= 1.0, "invalid percentile input")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def subject_stratified_paired_bootstrap(
    correctness: Sequence[Sequence[int]],
    subjects: Sequence[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    matrix = [list(map(int, row)) for row in correctness]
    _require(len(matrix) == len(subjects) and len(matrix) >= 2, "bootstrap matrix/subject count differs")
    _require(all(len(row) == len(CELLS) for row in matrix), "bootstrap matrix must have five cells")
    _require(all(value in (0, 1) for row in matrix for value in row), "bootstrap matrix is not binary")
    _require(isinstance(replicates, int) and not isinstance(replicates, bool) and replicates >= 2, "bootstrap replicates are invalid")
    groups: Dict[str, List[int]] = {}
    for index, subject in enumerate(subjects):
        groups.setdefault(str(subject), []).append(index)
    _require(len(groups) >= 2, "subject-stratified bootstrap needs two subjects")
    ordered_subjects = sorted(groups)
    rng = random.Random(seed)
    digest = hashlib.sha256()
    gains: List[List[float]] = [[] for _ in range(len(CELLS) - 1)]
    for _replicate in range(replicates):
        drawn: List[int] = []
        sums = [0] * len(CELLS)
        for subject in ordered_subjects:
            indices = groups[subject]
            for _ in indices:
                index = indices[rng.randrange(len(indices))]
                drawn.append(index)
                row = matrix[index]
                for cell_index, value in enumerate(row):
                    sums[cell_index] += value
        digest.update(struct.pack("<%dI" % len(drawn), *drawn))
        baseline = sums[0] / float(len(matrix))
        for cell_index in range(1, len(CELLS)):
            gains[cell_index - 1].append(sums[cell_index] / float(len(matrix)) - baseline)
    summaries = []
    for index, values in enumerate(gains, start=1):
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        summaries.append(
            {
                "cell_index": index,
                "bootstrap_mean_fraction": mean,
                "bootstrap_standard_error_ddof1": math.sqrt(variance),
                "percentile_95_ci_fraction": [linear_percentile(values, 0.025), linear_percentile(values, 0.975)],
            }
        )
    return {
        "method": "subject_stratified_paired_percentile_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "prng": "python_stdlib_random.Random",
        "random_api": "randrange(n_s)",
        "subject_order": ordered_subjects,
        "subject_sizes": {key: len(groups[key]) for key in ordered_subjects},
        "draw_iteration_order": "replicate_major_subject_ascending_identity_canonical",
        "draw_digest_representation": "little_endian_uint32_global_indices",
        "draw_index_sha256": digest.hexdigest(),
        "same_joint_draw_stream_all_cells_and_contrasts": True,
        "percentile_method": "linear_interpolation_at_q_times_R_minus_1",
        "cell_gain_summaries": summaries,
    }


def _load_correctness_for_cell(
    result_path: Path, metadata: Sequence[Mapping[str, Any]]
) -> List[int]:
    payload = _strict_json(result_path)
    _require(isinstance(payload, dict), "outcome result root is invalid")
    project_identity_only(payload, metadata)
    try:
        from tflt.loopscope.analysis import extract_accuracy, extract_correctness_samples

        correctness = extract_correctness_samples(payload, task="mmlu", metric="acc,none")
        aggregate = extract_accuracy(payload, task="mmlu", metric="acc,none")
    except Exception as exc:
        raise GateOError("outcome payload cannot close standard MMLU acc,none") from exc
    expected_keys = {
        "%s:%s" % (row["identity"]["task"], row["identity"]["doc_id"]) for row in metadata
    }
    _require(set(correctness) == expected_keys, "outcome correctness identity set differs")
    ordered = [int(bool(correctness["%s:%s" % (row["identity"]["task"], row["identity"]["doc_id"])])) for row in metadata]
    sample_accuracy = sum(ordered) / float(len(ordered))
    _require(math.isclose(sample_accuracy, float(aggregate), rel_tol=0.0, abs_tol=1e-12), "aggregate/per-sample accuracy differs")
    return ordered


def _load_formal_matrix(root: Path, completion: Mapping[str, Any], metadata: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    columns: List[List[int]] = []
    for cell in completion["cells"]:
        path = Path(cell["results"]["path"])
        _require(path == Path(cell["results"]["path"]).resolve(), "result path is not absolute")
        _require(file_sha256(path) == cell["results"]["file_sha256"], "result changed after seal")
        columns.append(_load_correctness_for_cell(path, metadata))
    _require(len(columns) == 5 and all(len(column) == len(metadata) for column in columns), "outcome matrix dimensions differ")
    return [[columns[column][row] for column in range(5)] for row in range(len(metadata))]


def build_scientific_projection(
    correctness: Sequence[Sequence[int]],
    metadata: Sequence[Mapping[str, Any]],
    *,
    result_file_sha256: Mapping[str, str],
) -> Dict[str, Any]:
    matrix = [list(map(int, row)) for row in correctness]
    _require(len(matrix) == len(metadata) and len(matrix) == 14042, "scientific matrix record count differs")
    _require(all(len(row) == 5 for row in matrix), "scientific matrix cell count differs")
    subjects = [str(row["subject"]) for row in metadata]
    bootstrap = subject_stratified_paired_bootstrap(matrix, subjects)
    cells: List[Dict[str, Any]] = []
    baseline_count = sum(row[0] for row in matrix)
    baseline_accuracy = baseline_count / float(len(matrix))
    for index, (cell_id, role, window) in enumerate(CELLS):
        correct_count = sum(row[index] for row in matrix)
        accuracy = correct_count / float(len(matrix))
        if index == 0:
            delta = 0.0
            n01 = 0
            n10 = 0
            ci_fraction = [0.0, 0.0]
        else:
            delta = accuracy - baseline_accuracy
            n01 = sum(1 for row in matrix if row[0] == 0 and row[index] == 1)
            n10 = sum(1 for row in matrix if row[0] == 1 and row[index] == 0)
            ci_fraction = bootstrap["cell_gain_summaries"][index - 1]["percentile_95_ci_fraction"]
        cells.append(
            {
                "index": index,
                "cell_id": cell_id,
                "role": role,
                "window": window,
                "sample_count": len(matrix),
                "correct_count": correct_count,
                "accuracy_fraction": accuracy,
                "accuracy_percent": accuracy * 100.0,
                "delta_percentage_points": delta * 100.0,
                "n01": n01,
                "n10": n10,
                "paired_percentile_95_ci_fraction": list(ci_fraction),
                "paired_percentile_95_ci_percentage_points": [value * 100.0 for value in ci_fraction],
            }
        )
    return {
        "schema_version": "loopscope.phase6.gate-o.scientific-projection.v1",
        "population": {
            "record_count": len(matrix),
            "subject_count": len(set(subjects)),
            "cell_count": 5,
            "matrix_shape": [len(matrix), 5],
            "ordered_identity_sha256": ordered_identity_sha256([row["identity"] for row in metadata]),
            "subject_counts": dict(sorted(Counter(subjects).items())),
            "missing_duplicate_extra": 0,
        },
        "metric": {
            "primary": "acc,none",
            "internal_unit": "fraction",
            "report_unit": "percentage_points",
        },
        "cells": cells,
        "bootstrap": bootstrap,
        "result_file_sha256": dict(result_file_sha256),
        "outcome_values_consumed": True,
        "selector_recomputed": False,
        "new_window_selected": False,
    }


def _report_text(scientific: Mapping[str, Any], analysis_file_sha256: str) -> str:
    lines = [
        "# LoopScope 第六阶段 Gate O：MMLU 5-shot 固定 panel 全量 accuracy",
        "",
        "本报告对应冻结的五-cell panel，按固定顺序列出 no-loop、14:16、13:16、15:18、12:16。",
        "结果只用于 Gate O 的事后 outcome 报告；没有重新运行 selector，也没有按 accuracy 重排主合同。",
        "",
        "- 模型：`%s@%s`" % (MODEL_REPO, MODEL_REVISION),
        "- 数据：`%s@%s`，standard MMLU test，5-shot，dev demonstrations，14042 条、57 subjects" % (DATASET_REPO, DATASET_REVISION),
        "- loop recipe：`k=3 / block / euler / alpha=1 / beta=0 / cache:first / decode:full`，common batch size 16",
        "- subject-stratified paired percentile bootstrap：2000 replicates，seed `20260803`，95% CI",
        "- analysis artifact SHA-256：`%s`" % analysis_file_sha256,
        "",
        "| cell | correct / 14042 | accuracy | delta vs no-loop (pp) | n01 | n10 | paired 95% CI (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scientific["cells"]:
        ci = row["paired_percentile_95_ci_percentage_points"]
        lines.append(
            "| `%s` | %d / %d | %.8f%% | %.8f | %d | %d | [%.8f, %.8f] |"
            % (
                row["cell_id"],
                row["correct_count"],
                row["sample_count"],
                row["accuracy_percent"],
                row["delta_percentage_points"],
                row["n01"],
                row["n10"],
                ci[0],
                ci[1],
            )
        )
    lines.extend(
        [
            "",
            "`n01` 表示 no-loop 错、loop 对；`n10` 表示 no-loop 对、loop 错。CI 是按 subject 分层、对 paired rows 联合重采样所得的 delta CI。",
            "",
            "分析状态：Gate O fixed-panel outcome analysis complete；不产生新的 selector 结论。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_once(run_root: Path) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    analysis_dir = root / "analysis"
    analysis_path = analysis_dir / "analysis_once.json"
    _require(not analysis_path.exists(), "Gate O analysis is write-once and already exists")
    manifest = _strict_json(root / "manifest/launch_manifest.json")
    completion = _strict_json(root / "manifest/formal_completion.json")
    _require(isinstance(manifest, dict) and isinstance(completion, dict), "formal analysis roots are invalid")
    validate_launch_manifest(manifest, mode="formal")
    _verify_manifest(completion, "formal completion")
    _require(completion.get("status") == "SEALED_COMPLETE_READY_FOR_ANALYSIS", "formal completion is not sealed")
    _require(completion.get("outcome_values_consumed") is False, "formal completion crossed outcome barrier")
    _require(not (root / "analysis").exists(), "analysis directory must be fresh")
    metadata = load_canonical_metadata()
    matrix = _load_formal_matrix(root, completion, metadata)
    result_hashes = {row["cell_id"]: row["results"]["file_sha256"] for row in completion["cells"]}
    scientific = build_scientific_projection(matrix, metadata, result_file_sha256=result_hashes)
    analysis_dir.mkdir(parents=False, exist_ok=False)
    analysis = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o.analysis.v1",
            "artifact_role": "one_shot_fixed_five_cell_outcome_analysis",
            "gate": GATE,
            "run_root": str(root),
            "created_at_utc": utc_now(),
            "formal_completion_file_sha256": file_sha256(root / "manifest/formal_completion.json"),
            "formal_completion_manifest_sha256": completion["manifest_sha256"],
            "scientific": scientific,
            "one_shot": True,
            "outcome_values_consumed": True,
            "selector_recomputed": False,
        }
    )
    analysis_sha = _write_new_json(analysis_path, analysis)
    report_path = WORKSPACE / REPORT_RELATIVE
    _require(not report_path.exists(), "required Chinese report path already exists")
    report_sha = _write_new_text(report_path, _report_text(scientific, analysis_sha))
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o.analysis-receipt.v1",
            "gate": GATE,
            "run_root": str(root),
            "analysis_file": str(analysis_path),
            "analysis_file_sha256": analysis_sha,
            "report_path": str(report_path),
            "report_file_sha256": report_sha,
            "outcome_values_consumed": True,
            "status": "PASS",
        }
    )
    _write_new_json(analysis_dir / "analysis_receipt.json", receipt)
    del matrix
    return receipt


def _scientific_projection_from_analysis(analysis: Mapping[str, Any]) -> Mapping[str, Any]:
    scientific = analysis.get("scientific")
    _require(isinstance(scientific, Mapping), "analysis scientific projection is absent")
    return scientific


def verify_analysis(run_root: Path) -> Dict[str, Any]:
    root = Path(run_root).resolve()
    analysis_path = root / "analysis/analysis_once.json"
    analysis_receipt_path = root / "analysis/analysis_receipt.json"
    verifier_dir = root / "verification"
    verifier_path = verifier_dir / "analysis_verifier_receipt.json"
    _require(analysis_path.is_file() and analysis_receipt_path.is_file(), "analysis artifacts are absent")
    _require(not verifier_path.exists(), "analysis verifier is write-once")
    analysis = _strict_json(analysis_path)
    analysis_receipt = _strict_json(analysis_receipt_path)
    manifest = _strict_json(root / "manifest/launch_manifest.json")
    completion = _strict_json(root / "manifest/formal_completion.json")
    _require(all(isinstance(item, dict) for item in (analysis, analysis_receipt, manifest, completion)), "analysis verifier roots are invalid")
    _verify_manifest(analysis, "analysis")
    _verify_manifest(analysis_receipt, "analysis receipt")
    validate_launch_manifest(manifest, mode="formal")
    _verify_manifest(completion, "formal completion")
    _require(analysis.get("one_shot") is True and analysis.get("outcome_values_consumed") is True, "analysis barrier differs")
    _require(analysis_receipt.get("analysis_file_sha256") == file_sha256(analysis_path), "analysis receipt file hash differs")
    metadata = load_canonical_metadata()
    matrix = _load_formal_matrix(root, completion, metadata)
    result_hashes = {row["cell_id"]: row["results"]["file_sha256"] for row in completion["cells"]}
    expected = build_scientific_projection(matrix, metadata, result_file_sha256=result_hashes)
    observed = _scientific_projection_from_analysis(analysis)
    _require(canonical_json_bytes(expected) == canonical_json_bytes(observed), "fresh verifier scientific projection differs")
    report_path = Path(str(analysis_receipt["report_path"]))
    _require(report_path.is_file() and file_sha256(report_path) == analysis_receipt["report_file_sha256"], "Chinese report hash differs")
    verifier_dir.mkdir(parents=False, exist_ok=True)
    receipt = _attach_manifest(
        {
            "schema_version": "loopscope.phase6.gate-o.analysis-verifier.v1",
            "gate": GATE,
            "run_root": str(root),
            "verified_at_utc": utc_now(),
            "analysis_file_sha256": file_sha256(analysis_path),
            "analysis_manifest_sha256": analysis["manifest_sha256"],
            "fresh_process_expected": True,
            "outcome_values_consumed": True,
            "scientific_projection_recomputed": True,
            "status": "PASS",
        }
    )
    _write_new_json(verifier_path, receipt)
    return receipt


def dry_run() -> Dict[str, Any]:
    card = load_card()
    return {
        "status": "DRY_RUN_VALID",
        "gate": GATE,
        "cells": [{"index": index, "cell_id": cell_id, "role": role, "window": window} for index, (cell_id, role, window) in enumerate(CELLS)],
        "cell_count": 5,
        "model": "%s@%s" % (MODEL_REPO, MODEL_REVISION),
        "dataset": "%s@%s" % (DATASET_REPO, DATASET_REVISION),
        "population": "14042/57",
        "decode_mode": "full",
        "batch_size": BATCH_SIZE,
        "formal_array": ARRAY,
        "formal_gpu": FORMAL_GPU,
        "debug_gpu": DEBUG_GPU,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "outcome_read": False,
        "selector_recomputed": False,
        "card_internal_sha256": card["manifest_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoopScope Phase 6 Gate O fixed-panel executor")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-run")
    prepare = sub.add_parser("prepare", help="freeze a fresh debug or formal launch root")
    prepare.add_argument("--mode", choices=("debug", "formal"), required=True)
    prepare.add_argument("--run-root", required=True)
    prepare.add_argument("--expected-commit", required=True)
    submit = sub.add_parser("submit", help="submit a frozen array and write its Job ID")
    submit.add_argument("--run-root", required=True)
    record = sub.add_parser("record-submission", help="record an externally submitted Job ID")
    record.add_argument("--run-root", required=True)
    record.add_argument("--job-id", required=True)
    run_cell_parser = sub.add_parser("run-cell", help="run one Slurm array cell")
    run_cell_parser.add_argument("--mode", choices=("debug", "formal"), required=True)
    run_cell_parser.add_argument("--run-root", required=True)
    run_cell_parser.add_argument("--cell-index", type=int, required=True)
    run_cell_parser.add_argument("--expected-commit", required=True)
    seal = sub.add_parser("seal", help="close scheduler/artifact/identity evidence")
    seal.add_argument("--mode", choices=("debug", "formal"), required=True)
    seal.add_argument("--run-root", required=True)
    seal.add_argument("--job-id", required=True)
    verify = sub.add_parser("verify", help="fresh pre-outcome closure verifier")
    verify.add_argument("--mode", choices=("debug", "formal"), required=True)
    verify.add_argument("--run-root", required=True)
    analyze = sub.add_parser("analyze-once", help="authorized one-shot outcome analysis")
    analyze.add_argument("--run-root", required=True)
    verify_analysis_parser = sub.add_parser("verify-analysis", help="fresh-process analysis verifier")
    verify_analysis_parser.add_argument("--run-root", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "dry-run":
        print(json.dumps(dry_run(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare":
        print(json.dumps(prepare_run(mode=args.mode, run_root=Path(args.run_root), expected_commit=args.expected_commit), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "submit":
        print(json.dumps(submit_run(Path(args.run_root)), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "record-submission":
        print(json.dumps(record_submission(run_root=Path(args.run_root), job_id=args.job_id), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "run-cell":
        print(json.dumps(run_cell(mode=args.mode, run_root=Path(args.run_root), cell_index=args.cell_index, expected_commit=args.expected_commit), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "seal":
        print(json.dumps(seal_run(mode=args.mode, run_root=Path(args.run_root), job_id=args.job_id), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        print(json.dumps(verify_completion(mode=args.mode, run_root=Path(args.run_root)), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "analyze-once":
        print(json.dumps(analyze_once(Path(args.run_root)), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "verify-analysis":
        print(json.dumps(verify_analysis(Path(args.run_root)), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise GateOError("unknown command")


__all__ = [
    "ADMISSION_BASE",
    "ARRAY",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CARD_FILE_SHA256",
    "CELLS",
    "GateOError",
    "build_launch_manifest",
    "build_sbatch_text",
    "build_scientific_projection",
    "dry_run",
    "linear_percentile",
    "main",
    "project_identity_only",
    "subject_stratified_paired_bootstrap",
    "validate_launch_manifest",
    "validate_scheduler_rows",
]
