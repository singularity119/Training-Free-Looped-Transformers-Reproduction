#!/usr/bin/env python3
"""Execute the exact outcome-blind LoopScope Phase 4 Gate P4-B workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _install_isolated_tflt_namespace() -> None:
    if "tflt" in sys.modules:
        raise RuntimeError("P4-B producer requires a fresh isolated Python process")
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src/tflt"
    package = types.ModuleType("tflt")
    package.__package__ = "tflt"
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    package.__file__ = str(package_root)
    sys.modules["tflt"] = package


_install_isolated_tflt_namespace()

from tflt.loopscope.phase4_acquisition import (
    AUTHORIZED_RUN_ROOT,
    SHARD_RECEIPT_SCHEMA,
    assert_authorized_run_root,
    build_shard_manifest,
    extract_known_outcome_windows,
    load_strict_json,
    load_strict_jsonl,
    merge_completed_shards,
    require_non_degenerate_width4_scores,
    resolve_run_child,
    wrap_outcome_panel,
    write_new_json,
    write_new_jsonl,
)
from tflt.loopscope.phase4_runtime import (
    CATEGORIES,
    Phase4RuntimeError,
    acquire_prefix_scalar_record,
    deterministic_option_permutation,
    load_native_prefix_runtime,
    permute_safe_target,
    project_target_row,
    project_test_dataset,
    proportional_category_quotas,
    render_exact_prefix,
    select_option_control_identities,
    template_safe_targets,
    tokenization_closure,
    tokenization_metadata,
    validation_demos_by_category,
)
from tflt.loopscope.phase4_schema import (
    canonical_identity,
    file_sha256,
    load_phase4_card,
    sanitized_content_sha256,
    scan_forbidden_fields,
    semantic_sha256,
    validate_shared_identity_contract,
    validate_source_record,
)
from tflt.loopscope.phase4_selector import (
    EDGE_KEYS,
    STRICT_KEYS,
    analyze_selector,
    build_outcome_panel,
    verify_selector_report,
)


EXECUTOR_THREAD_ID = "019f6ca9-ef16-7a53-8949-e47d89fd2dc2"
CARD_BYTE_SHA256 = "980955386907a1865699808219da1379031c1395d9cdb77029585cf266f160f8"
CONTROL_BYTE_SHA256 = "d33dd0fd0c064bfd3e49f56aac49f92ee18d80ef9da88d91f24e2e26b891932b"
RECEIPT_BYTE_SHA256 = "fdf7c21c6ea31fa83c082e8f694daea95e2fd1a97bdc6e8a338ee6768956e63b"
NON_REUSE_BYTE_SHA256 = "725d13e2465f8a6abcffd1a8074b569335c6b408426c48ab781d47bb39795a2e"
DATASET_FINGERPRINT = "32dc8126417ec7ecd00c98b5ce877ad6e56df5560330c71f4b599e2f5c1d45a2"
DATASET_FINGERPRINT_COMPONENTS = {
    "dataset_info_sha256": "ce81f7ec00c6701dc737889a3171a9dd32d2521914bc1e5029a3f8bb28df9aab",
    "test_arrow_sha256": "4c741944e3b53719b433be6e7916e4ab9e8ff33f70e9c365614b4e317c0476bb",
    "validation_arrow_sha256": "5730c2a9cd07a1ee5f70f4cd1940e43da1df7602c21977b7dc7ef51d496bdf00",
}
RENDERER_SHA256 = "74ab409c4e4c96e4351fbe6122d519f64dd3e11381a676957631e858923cc9fc"
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")
DATASET_CACHE_SNAPSHOT = (
    HF_DATASETS_CACHE
    / "TIGER-Lab___mmlu-pro/default/0.0.0/b189ec765aa7ed75c8acfea42df31fdae71f97be"
)
_DATASET_CLOSURE_VERIFIED = False


class P4BError(Phase4RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(command: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        list(command), cwd=str(cwd), text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise P4BError("local provenance command failed: %s" % command[0])
    return completed.stdout.strip()


def _git_provenance(expected_commit: str) -> Dict[str, Any]:
    root = _repo_root()
    head = _run(("git", "rev-parse", "HEAD"), cwd=root)
    branch = _run(("git", "symbolic-ref", "--short", "HEAD"), cwd=root)
    origin = _run(("git", "rev-parse", "origin/loopscope"), cwd=root)
    dirty = bool(_run(("git", "status", "--porcelain"), cwd=root))
    if head != expected_commit or origin != expected_commit or branch != "loopscope" or dirty:
        raise P4BError("Git provenance is not clean loopscope at the expected pushed commit")
    return {"branch": branch, "commit": head, "origin_loopscope": origin, "dirty": False}


def _card(card_path: Path) -> Dict[str, Any]:
    card_path = card_path.resolve()
    if file_sha256(card_path) != CARD_BYTE_SHA256:
        raise P4BError("frozen Phase 4 card byte hash differs")
    root = card_path.parents[2]
    fixed = {
        root / "configs/loopscope/phase4_card_verifier_receipt.json": RECEIPT_BYTE_SHA256,
        root / "configs/loopscope/phase4_provenance_non_reuse.json": NON_REUSE_BYTE_SHA256,
    }
    for path, expected in fixed.items():
        if file_sha256(path) != expected:
            raise P4BError("frozen upstream/control byte hash differs")
    # The mutable control plane intentionally lives outside the Git clone.  It
    # is present and hash-checked in the local executor workspace; the dedicated
    # HPC clone binds the same hash through this committed constant and every
    # Gate artifact without fabricating a remote planning-file copy.
    control_path = root.parent / ".planning/loopscope_phase4_control.md"
    if control_path.exists() and file_sha256(control_path) != CONTROL_BYTE_SHA256:
        raise P4BError("frozen upstream/control byte hash differs")
    return load_phase4_card(card_path)


def _assert_offline_environment() -> None:
    required = {
        "HF_HOME": str(HF_HOME),
        "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
        "HF_DATASETS_CACHE": str(HF_DATASETS_CACHE),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    for key, expected in required.items():
        if os.environ.get(key) != expected:
            raise P4BError("audited offline environment variable differs: %s" % key)
    if os.environ.get("HF_ENDPOINT"):
        raise P4BError("HF_ENDPOINT substitution is forbidden")


def _load_renderer() -> Any:
    from lm_eval.tasks.mmlu_pro import utils

    if file_sha256(Path(utils.__file__).resolve()) != RENDERER_SHA256:
        raise P4BError("installed MMLU-Pro renderer source hash differs")
    return utils


def _load_dataset(card: Mapping[str, Any], split: str) -> Any:
    global _DATASET_CLOSURE_VERIFIED

    from datasets import DownloadMode, load_dataset

    if not _DATASET_CLOSURE_VERIFIED:
        observed_components = {
            "dataset_info_sha256": file_sha256(DATASET_CACHE_SNAPSHOT / "dataset_info.json"),
            "test_arrow_sha256": file_sha256(DATASET_CACHE_SNAPSHOT / "mmlu-pro-test.arrow"),
            "validation_arrow_sha256": file_sha256(
                DATASET_CACHE_SNAPSHOT / "mmlu-pro-validation.arrow"
            ),
        }
        if observed_components != DATASET_FINGERPRINT_COMPONENTS:
            raise P4BError("cached dataset snapshot component hash differs")
        if semantic_sha256(observed_components) != DATASET_FINGERPRINT:
            raise P4BError("cached dataset snapshot composite fingerprint differs")
        _DATASET_CLOSURE_VERIFIED = True

    value = load_dataset(
        card["task"]["dataset"],
        revision=card["task"]["dataset_revision"],
        split=split,
        cache_dir=str(HF_DATASETS_CACHE),
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    if split == "test" and len(value) != 12032:
        raise P4BError("cached test dataset population count differs")
    if split == "validation" and len(value) != 70:
        raise P4BError("cached validation demonstration count differs")
    return value


def _load_tokenizer(card: Mapping[str, Any]) -> Any:
    from transformers import AutoTokenizer
    from tflt.loopscope.revisions import load_tokenizer_with_resolved_commit

    tokenizer, commit = load_tokenizer_with_resolved_commit(
        AutoTokenizer,
        card["model"]["repo"],
        {
            "revision": card["model"]["tokenizer_revision"],
            "local_files_only": True,
            "trust_remote_code": True,
        },
    )
    if commit != card["model"]["tokenizer_revision"]:
        raise P4BError("tokenizer revision closure differs")
    return tokenizer


def _source_record(
    target: Mapping[str, Any], metadata: Mapping[str, Any], card: Mapping[str, Any]
) -> Dict[str, Any]:
    record = {
        "schema_version": "loopscope.phase4.shared-source-record.v1",
        "dataset_revision": card["task"]["dataset_revision"],
        "split": "test",
        "question_id": target["question_id"],
        "category": target["category"],
        "src": target["src"],
        "question": target["question"],
        "ordered_options": list(target["ordered_options"]),
        "canonical_identity": canonical_identity(target),
        "sanitized_content_sha256": sanitized_content_sha256(
            target["question"], target["ordered_options"]
        ),
        "rendered_prefix_sha256": metadata["rendered_prefix_sha256"],
        "rendered_token_ids_sha256": metadata["rendered_token_ids_sha256"],
        "attention_mask_sha256": metadata["attention_mask_sha256"],
        "last_effective_prefix_token_index": metadata[
            "last_effective_prefix_token_index"
        ],
        "renderer_provenance": {
            "task_alias": card["task"]["task_alias"],
            "renderer_source_sha256": RENDERER_SHA256,
            "rendered_prefix_contract_sha256": card["provenance_closure"]["values"][
                "rendered_prefix_contract_sha256"
            ],
            "chat_template_sha256": card["provenance_closure"]["values"][
                "chat_template_sha256"
            ],
            "lm_eval_version": card["provenance_closure"]["values"]["lm_eval_version"],
            "num_fewshot": 5,
            "fewshot_split": "validation",
            "apply_chat_template": False,
            "tokenizer_revision": card["model"]["tokenizer_revision"],
        },
    }
    validate_source_record(record)
    return record


def _control_record(
    *,
    target: Mapping[str, Any],
    identity: str,
    metadata: Mapping[str, Any],
    kind: str,
    original_source_sha256: Optional[str] = None,
    permutation: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "schema_version": "loopscope.phase4.p4b-control-source.v1",
        "kind": kind,
        "canonical_identity": identity,
        "question_id": target["question_id"],
        "category": target["category"],
        "src": target["src"],
        "question": target["question"],
        "ordered_options": list(target["ordered_options"]),
        "rendered_prefix_sha256": metadata["rendered_prefix_sha256"],
        "rendered_token_ids_sha256": metadata["rendered_token_ids_sha256"],
        "attention_mask_sha256": metadata["attention_mask_sha256"],
        "last_effective_prefix_token_index": metadata[
            "last_effective_prefix_token_index"
        ],
        "last_effective_prefix_token_sha256": metadata[
            "last_effective_prefix_token_sha256"
        ],
        "sequence_length": metadata["sequence_length"],
        "original_source_sha256": original_source_sha256,
        "permutation": None if permutation is None else list(permutation),
    }
    scan_forbidden_fields(record)
    return record


def prepare_source(
    *, card_path: Path, run_root: Path, expected_commit: str, argv: Sequence[str]
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = assert_authorized_run_root(run_root, must_exist=False)
    git = _git_provenance(expected_commit)
    card = _card(card_path)
    renderer = _load_renderer()
    tokenizer = _load_tokenizer(card)
    validation = _load_dataset(card, "validation")
    demos = validation_demos_by_category(validation)
    projected = project_test_dataset(_load_dataset(card, "test"))
    sources: List[Dict[str, Any]] = []
    source_token_closures: Dict[str, Dict[str, Any]] = {}
    for row in projected:
        target = project_target_row(row)
        prefix = render_exact_prefix(target, demos[target["category"]], renderer)
        metadata = tokenization_metadata(tokenizer, prefix)
        source = _source_record(target, metadata, card)
        sources.append(source)
        source_token_closures[source["canonical_identity"]] = {
            "canonical_identity": source["canonical_identity"],
            **tokenization_closure(metadata),
            "sequence_length": metadata["sequence_length"],
        }
    sources.sort(
        key=lambda row: (
            int(str(row["question_id"]).strip()),
            row["sanitized_content_sha256"],
        )
    )
    validate_shared_identity_contract(sources, sources, require_full_population=True)
    if len(sources) != card["task"]["shared_population_count"]:
        raise P4BError("shared source population count differs")

    counts = Counter(row["category"] for row in sources)
    quotas = proportional_category_quotas(counts, 512)
    selected_ids = set(select_option_control_identities(sources, quotas))
    option_records: List[Dict[str, Any]] = []
    source_by_identity = {row["canonical_identity"]: row for row in sources}
    for source in sources:
        if source["canonical_identity"] not in selected_ids:
            continue
        target = {
            key: source[key]
            for key in ("question_id", "category", "src", "question", "ordered_options")
        }
        permutation = deterministic_option_permutation(source["canonical_identity"])
        target = permute_safe_target(target, permutation)
        prefix = render_exact_prefix(target, demos[target["category"]], renderer)
        option_records.append(
            _control_record(
                target=target,
                identity=source["canonical_identity"],
                metadata=tokenization_metadata(tokenizer, prefix),
                kind="option",
                original_source_sha256=semantic_sha256(source),
                permutation=permutation,
            )
        )
    if len(option_records) != 512:
        raise P4BError("option control membership is not exactly 512")

    template_records: List[Dict[str, Any]] = []
    for target in template_safe_targets():
        prefix = render_exact_prefix(target, demos[target["category"]], renderer)
        template_records.append(
            _control_record(
                target=target,
                identity="template:%02d:%s" % (target["question_id"], target["category"]),
                metadata=tokenization_metadata(tokenizer, prefix),
                kind="template",
            )
        )

    smoke_sources = []
    seen_categories = set()
    for source in sources:
        if source["category"] not in seen_categories:
            smoke_sources.append(source)
            seen_categories.add(source["category"])
        if len(smoke_sources) == 4:
            break
    smoke_options = []
    smoke_templates = []
    template_by_category = {row["category"]: row for row in template_records}
    for source in smoke_sources:
        target = {
            key: source[key]
            for key in ("question_id", "category", "src", "question", "ordered_options")
        }
        permutation = deterministic_option_permutation(source["canonical_identity"])
        target = permute_safe_target(target, permutation)
        prefix = render_exact_prefix(target, demos[target["category"]], renderer)
        smoke_options.append(
            _control_record(
                target=target,
                identity=source["canonical_identity"],
                metadata=tokenization_metadata(tokenizer, prefix),
                kind="option",
                original_source_sha256=semantic_sha256(source),
                permutation=permutation,
            )
        )
        smoke_templates.append(template_by_category[source["category"]])

    # Create the write-once Gate root only after every source/control closure has
    # passed in memory.  A data/renderer admission failure must leave it absent.
    root.mkdir(parents=True, exist_ok=False)
    for relative in ("source", "smoke", "formal", "controls", "freeze", "slurm"):
        (root / relative).mkdir()
    ordered_token_closures = [
        source_token_closures[row["canonical_identity"]] for row in sources
    ]
    source_sha = write_new_jsonl(root / "source/shared12032.jsonl", sources)
    token_closure_sha = write_new_jsonl(
        root / "source/shared12032_token_closure.jsonl", ordered_token_closures
    )
    option_sha = write_new_jsonl(root / "source/option512.jsonl", option_records)
    template_sha = write_new_jsonl(root / "source/template14.jsonl", template_records)
    smoke_option_sha = write_new_jsonl(root / "source/smoke_option4.jsonl", smoke_options)
    source_manifest = {
        "schema_version": "loopscope.phase4.p4b-source-manifest.v1",
        "gate": "P4-B",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "created_at_utc": _utc_now(),
        "run_root": str(root),
        "git": git,
        "card_sha256": CARD_BYTE_SHA256,
        "control_sha256": CONTROL_BYTE_SHA256,
        "dataset_revision": card["task"]["dataset_revision"],
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "dataset_fingerprint_components": dict(DATASET_FINGERPRINT_COMPONENTS),
        "datasets_runtime_projected_fingerprint": str(
            getattr(projected, "_fingerprint", "")
        ),
        "model_repo": card["model"]["repo"],
        "model_revision": card["model"]["revision"],
        "tokenizer_revision": card["model"]["tokenizer_revision"],
        "task_alias": card["task"]["task_alias"],
        "record_count": len(sources),
        "category_counts": {category: counts[category] for category in CATEGORIES},
        "ordered_identity_sha256": semantic_sha256(
            [row["canonical_identity"] for row in sources]
        ),
        "record_file": "source/shared12032.jsonl",
        "record_file_sha256": source_sha,
        "token_closure_file": "source/shared12032_token_closure.jsonl",
        "token_closure_file_sha256": token_closure_sha,
        "token_closure_record_count": len(ordered_token_closures),
        "option_control": {
            "record_count": 512,
            "category_quotas": quotas,
            "record_file": "source/option512.jsonl",
            "record_file_sha256": option_sha,
            "selection_rule": "category_largest_remainder_then_identity_sha256_ascending",
            "permutation_rule": "identity_sha256_per_index_sha256_ascending_nonidentity",
        },
        "template_control": {
            "record_count": 14,
            "record_file": "source/template14.jsonl",
            "record_file_sha256": template_sha,
            "category_order": list(CATEGORIES),
        },
        "argv": list(argv),
    }
    source_manifest["manifest_sha256"] = semantic_sha256(source_manifest)
    write_new_json(root / "source/phase4_shared12032_source_manifest.json", source_manifest)
    smoke_manifest = {
        "schema_version": "loopscope.phase4.p4b-smoke-manifest.v1",
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "original_identities": [row["canonical_identity"] for row in smoke_sources],
        "option_identities": [row["canonical_identity"] for row in smoke_options],
        "template_identities": [row["canonical_identity"] for row in smoke_templates],
        "matched_categories": [row["category"] for row in smoke_sources],
        "smoke_option_file_sha256": smoke_option_sha,
        "counts": {"original": 4, "option": 4, "template": 4},
    }
    smoke_manifest["manifest_sha256"] = semantic_sha256(smoke_manifest)
    write_new_json(root / "smoke/membership.json", smoke_manifest)
    del validation, demos, projected, tokenizer, source_by_identity
    return {"source_manifest_sha256": source_manifest["manifest_sha256"], "count": 12032}


def _load_live_sources(root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    manifest = load_strict_json(root / "source/phase4_shared12032_source_manifest.json")
    sources = load_strict_jsonl(root / "source/shared12032.jsonl")
    validate_shared_identity_contract(sources, sources, require_full_population=True)
    if file_sha256(root / "source/shared12032.jsonl") != manifest["record_file_sha256"]:
        raise P4BError("shared source file hash differs")
    return sources, manifest


def _safe_target_from_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    target = {
        key: record[key]
        for key in ("question_id", "category", "src", "question", "ordered_options")
    }
    scan_forbidden_fields(target)
    return target


def _load_original_token_closures(
    root: Path, source_manifest: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    path = root / str(source_manifest["token_closure_file"])
    if file_sha256(path) != source_manifest["token_closure_file_sha256"]:
        raise P4BError("shared token-closure sidecar hash differs")
    rows = load_strict_jsonl(path)
    if len(rows) != 12032 or len({row.get("canonical_identity") for row in rows}) != 12032:
        raise P4BError("shared token-closure sidecar membership differs")
    return {row["canonical_identity"]: row for row in rows}


def _token_closure(
    record: Mapping[str, Any],
    original_closures: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    if record.get("schema_version") == "loopscope.phase4.shared-source-record.v1":
        value = original_closures.get(str(record["canonical_identity"]))
        if value is None:
            raise P4BError("shared token-closure identity is missing")
        for key in (
            "rendered_prefix_sha256",
            "rendered_token_ids_sha256",
            "attention_mask_sha256",
            "last_effective_prefix_token_index",
        ):
            if value.get(key) != record.get(key):
                raise P4BError("shared source/token-closure drift at %s" % key)
    else:
        value = record
    return tokenization_closure(value)


def _producer_provenance(
    *,
    source: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
    expected_commit: str,
    mode: str,
    expected_tokenization: Mapping[str, Any],
) -> Dict[str, Any]:
    closure = {
        key: source[key]
        for key in (
            "canonical_identity",
            "sanitized_content_sha256",
            "rendered_prefix_sha256",
            "rendered_token_ids_sha256",
            "attention_mask_sha256",
        )
    } if source.get("schema_version") == "loopscope.phase4.shared-source-record.v1" else None
    payload = {
        "gate": "P4-B",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "git_commit": expected_commit,
        "card_sha256": CARD_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "model_revision": card["model"]["revision"],
        "dataset_revision": card["task"]["dataset_revision"],
        "mode": mode,
        "forward_type": "one_native_no_loop_prefix_forward",
        "use_cache": False,
        "projection_dtype": "float32",
        "target_generation_count": 0,
        "source_closure": closure,
        "input_closure": {
            "canonical_identity": source["canonical_identity"],
            **dict(expected_tokenization),
            "original_source_sha256": source.get("original_source_sha256"),
        },
    }
    scan_forbidden_fields(payload, allow_options=False)
    return payload


def _runtime_evidence(runtime: Any, facts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    torch = runtime.torch
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "model_class": runtime.model.__class__.__name__,
        "tokenizer_class": runtime.tokenizer.__class__.__name__,
        "decoder_layers": runtime.layer_count,
        "dtype": "bfloat16",
        "device_name": str(properties.name),
        "device_memory_bytes": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "revision_closure": dict(runtime.revision_closure),
        "forward_count": len(facts),
        "target_generation_count": 0,
        "maximum_final_projection_difference": max(
            float(row["final_projection_max_abs_difference"]) for row in facts
        ),
    }


def acquire_smoke(
    *, card_path: Path, run_root: Path, expected_commit: str, argv: Sequence[str]
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = assert_authorized_run_root(run_root, must_exist=True)
    _git_provenance(expected_commit)
    card = _card(card_path)
    sources, source_manifest = _load_live_sources(root)
    original_token_closures = _load_original_token_closures(root, source_manifest)
    smoke = load_strict_json(root / "smoke/membership.json")
    if (root / "smoke/phase4_prefix_smoke_report.json").exists():
        raise FileExistsError("smoke already completed")
    source_map = {row["canonical_identity"]: row for row in sources}
    option_map = {
        row["canonical_identity"]: row
        for row in load_strict_jsonl(root / "source/smoke_option4.jsonl")
    }
    template_map = {
        row["canonical_identity"]: row
        for row in load_strict_jsonl(root / "source/template14.jsonl")
    }
    groups = {
        "original": [source_map[value] for value in smoke["original_identities"]],
        "option": [option_map[value] for value in smoke["option_identities"]],
        "template": [template_map[value] for value in smoke["template_identities"]],
    }
    renderer = _load_renderer()
    demos = validation_demos_by_category(_load_dataset(card, "validation"))
    runtime = load_native_prefix_runtime(card)
    all_facts: List[Dict[str, Any]] = []
    output_hashes = {}
    completed_groups: Dict[str, List[Dict[str, Any]]] = {}
    for kind, rows in groups.items():
        records = []
        for row in rows:
            prefix = render_exact_prefix(
                _safe_target_from_record(row), demos[row["category"]], renderer
            )
            expected_tokenization = _token_closure(row, original_token_closures)
            record, fact = acquire_prefix_scalar_record(
                runtime,
                prefix=prefix,
                canonical_identity=row["canonical_identity"],
                category=row["category"],
                expected_tokenization=expected_tokenization,
                producer_provenance=_producer_provenance(
                    source=row,
                    source_manifest=source_manifest,
                    card=card,
                    expected_commit=expected_commit,
                    mode="smoke_%s" % kind,
                    expected_tokenization=expected_tokenization,
                ),
                d36_tolerance=card["numeric_contract"]["d36_abs_tolerance"],
            )
            records.append(record)
            all_facts.append(dict(fact, kind=kind, canonical_identity=row["canonical_identity"]))
        completed_groups[kind] = records
    for kind, records in completed_groups.items():
        output_hashes[kind] = write_new_jsonl(root / ("smoke/%s_records.jsonl" % kind), records)
    report = {
        "schema_version": "loopscope.phase4.p4b-smoke-report.v1",
        "status": "PASS",
        "gate": "P4-B",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "completed_at_utc": _utc_now(),
        "git_commit": expected_commit,
        "card_sha256": CARD_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "smoke_manifest_sha256": smoke["manifest_sha256"],
        "counts": {"original": 4, "option": 4, "template": 4, "total": 12},
        "matched_original_option_identities": smoke["original_identities"] == smoke["option_identities"],
        "matched_template_categories": smoke["matched_categories"],
        "boundaries": list(range(37)),
        "target_generation_count": 0,
        "use_cache": False,
        "scalar_reductions_finite": True,
        "d36_tolerance": card["numeric_contract"]["d36_abs_tolerance"],
        "top_k_invariants": True,
        "output_file_sha256": output_hashes,
        "per_record_facts": all_facts,
        "runtime": _runtime_evidence(runtime, all_facts),
        "argv": list(argv),
    }
    scan_forbidden_fields(report, allow_options=False)
    report["manifest_sha256"] = semantic_sha256(report)
    write_new_json(root / "smoke/phase4_prefix_smoke_report.json", report)
    return {"status": "PASS", "report_sha256": report["manifest_sha256"]}


def freeze_shards(
    *,
    card_path: Path,
    run_root: Path,
    expected_commit: str,
    original_shards: int,
    option_shards: int,
    template_shards: int,
    argv: Sequence[str],
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = assert_authorized_run_root(run_root, must_exist=True)
    git = _git_provenance(expected_commit)
    _card(card_path)
    smoke = load_strict_json(root / "smoke/phase4_prefix_smoke_report.json")
    if smoke.get("status") != "PASS" or smoke.get("target_generation_count") != 0:
        raise P4BError("formal shard freeze requires a passing real smoke")
    sources, source_manifest = _load_live_sources(root)
    by_kind = {
        "original": sources,
        "option": load_strict_jsonl(root / "source/option512.jsonl"),
        "template": load_strict_jsonl(root / "source/template14.jsonl"),
    }
    shard_counts = {
        "original": original_shards,
        "option": option_shards,
        "template": template_shards,
    }
    hashes = {}
    for kind, rows in by_kind.items():
        membership = [
            {"canonical_identity": row["canonical_identity"], "category": row["category"]}
            for row in rows
        ]
        manifest = build_shard_manifest(
            membership,
            kind=kind,
            shard_count=shard_counts[kind],
            run_root=root,
            bindings={
                "git_commit": expected_commit,
                "card_sha256": CARD_BYTE_SHA256,
                "source_manifest_sha256": source_manifest["manifest_sha256"],
                "smoke_report_sha256": smoke["manifest_sha256"],
                "argv_sha256": semantic_sha256(list(argv)),
            },
        )
        path = root / ("formal/%s_shard_manifest.json" % kind)
        write_new_json(path, manifest)
        hashes[kind] = manifest["manifest_sha256"]
    receipt = {
        "schema_version": "loopscope.phase4.p4b-shard-freeze.v1",
        "status": "FROZEN_AFTER_SMOKE",
        "created_at_utc": _utc_now(),
        "git": git,
        "shard_counts": shard_counts,
        "manifest_sha256": hashes,
        "argv": list(argv),
    }
    receipt["freeze_sha256"] = semantic_sha256(receipt)
    write_new_json(root / "formal/shard_freeze.json", receipt)
    return receipt


def _records_for_kind(root: Path, kind: str) -> List[Dict[str, Any]]:
    paths = {
        "original": root / "source/shared12032.jsonl",
        "option": root / "source/option512.jsonl",
        "template": root / "source/template14.jsonl",
    }
    manifest = load_strict_json(root / "source/phase4_shared12032_source_manifest.json")
    expected_hashes = {
        "original": manifest["record_file_sha256"],
        "option": manifest["option_control"]["record_file_sha256"],
        "template": manifest["template_control"]["record_file_sha256"],
    }
    if file_sha256(paths[kind]) != expected_hashes[kind]:
        raise P4BError("frozen %s source/control file hash differs" % kind)
    return load_strict_jsonl(paths[kind])


def acquire_shard(
    *,
    kind: str,
    shard_id: int,
    card_path: Path,
    run_root: Path,
    expected_commit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = assert_authorized_run_root(run_root, must_exist=True)
    _git_provenance(expected_commit)
    card = _card(card_path)
    sources, source_manifest = _load_live_sources(root)
    original_token_closures = _load_original_token_closures(root, source_manifest)
    manifest = load_strict_json(root / ("formal/%s_shard_manifest.json" % kind))
    if manifest.get("kind") != kind or shard_id < 0 or shard_id >= manifest["shard_count"]:
        raise P4BError("formal shard selection differs from frozen manifest")
    shard = manifest["shards"][shard_id]
    output = resolve_run_child(root, shard["output_relative"])
    if output.exists():
        raise FileExistsError("formal shard attempt already exists")
    rows = _records_for_kind(root, kind)
    row_map = {row["canonical_identity"]: row for row in rows}
    source_map = {row["canonical_identity"]: row for row in sources}
    renderer = _load_renderer()
    demos = validation_demos_by_category(_load_dataset(card, "validation"))
    runtime = load_native_prefix_runtime(card)
    records = []
    facts = []
    for member in shard["records"]:
        row = row_map[member["canonical_identity"]]
        prefix = render_exact_prefix(_safe_target_from_record(row), demos[row["category"]], renderer)
        expected_tokenization = _token_closure(row, original_token_closures)
        record, fact = acquire_prefix_scalar_record(
            runtime,
            prefix=prefix,
            canonical_identity=row["canonical_identity"],
            category=row["category"],
            expected_tokenization=expected_tokenization,
            producer_provenance=_producer_provenance(
                source=row,
                source_manifest=source_manifest,
                card=card,
                expected_commit=expected_commit,
                mode="formal_%s" % kind,
                expected_tokenization=expected_tokenization,
            ),
            d36_tolerance=card["numeric_contract"]["d36_abs_tolerance"],
        )
        records.append(record)
        facts.append(fact)
    # Do not create a shard directory until the complete in-memory shard has
    # passed.  Pre-artifact model/infrastructure failures remain visible in the
    # Slurm log without leaving a misleading partial completed attempt.
    output.mkdir(parents=True, exist_ok=False)
    record_path = output / "records.jsonl"
    record_hash = write_new_jsonl(record_path, records)
    receipt = {
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "status": "COMPLETED",
        "kind": kind,
        "shard_id": shard_id,
        "membership_sha256": shard["membership_sha256"],
        "record_count": len(records),
        "record_file_sha256": record_hash,
        "ordered_record_sha256": semantic_sha256(records),
        "completed_at_utc": _utc_now(),
        "git_commit": expected_commit,
        "card_sha256": CARD_BYTE_SHA256,
        "target_generation_count": 0,
        "runtime": _runtime_evidence(runtime, facts),
        "argv": list(argv),
    }
    scan_forbidden_fields(receipt, allow_options=False)
    receipt["manifest_sha256"] = semantic_sha256(receipt)
    write_new_json(output / "receipt.json", receipt)
    return {"kind": kind, "shard_id": shard_id, "record_count": len(records)}


def merge_kind(
    *, kind: str, card_path: Path, run_root: Path, expected_commit: str
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = assert_authorized_run_root(run_root, must_exist=True)
    _git_provenance(expected_commit)
    card = _card(card_path)
    rows = _records_for_kind(root, kind)
    membership = [
        {"canonical_identity": row["canonical_identity"], "category": row["category"]}
        for row in rows
    ]
    source_records = rows if kind == "original" else None
    return merge_completed_shards(
        run_root=root,
        shard_manifest=load_strict_json(root / ("formal/%s_shard_manifest.json" % kind)),
        expected_records=membership,
        merged_jsonl_relative=(
            "formal/phase4_shared12032_no_loop_trajectories.jsonl"
            if kind == "original"
            else "controls/%s/merged.jsonl" % kind
        ),
        merged_manifest_relative=(
            "formal/phase4_shared12032_trajectory_manifest.json"
            if kind == "original"
            else "controls/%s/merged_manifest.json" % kind
        ),
        source_records=source_records,
        require_full_population=kind == "original",
        d36_tolerance=card["numeric_contract"]["d36_abs_tolerance"],
        expected_run_root=AUTHORIZED_RUN_ROOT,
    )


def _curves(records: Sequence[Mapping[str, Any]], weights: Optional[Sequence[float]] = None) -> Tuple[Any, Any]:
    import numpy as np

    entropy = np.asarray(
        [[row["full_vocabulary_entropy"] for row in record["boundaries"]] for record in records],
        dtype=np.float64,
    )
    kl = np.asarray(
        [[row["kl_to_final"] for row in record["boundaries"]] for record in records],
        dtype=np.float64,
    )
    if weights is None:
        return entropy.mean(axis=0), kl.mean(axis=0)
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= normalized.sum()
    return normalized @ entropy, normalized @ kl


def _diagnostic_ranking(entropy: Any, kl: Any, keys: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    rows = []
    for width, start in keys:
        e_rate = float((entropy[start] - entropy[start + width]) / width)
        k_rate = float((kl[start + width] - kl[start]) / width)
        rows.append(
            {
                "width": width,
                "start": start,
                "window": "%d:%d" % (start, start + width - 1),
                "E_rate": e_rate,
                "K_rate": k_rate,
                "diagnostic_point_score": min(e_rate, k_rate),
            }
        )
    rows.sort(key=lambda row: (-row["diagnostic_point_score"], row["width"], row["start"]))
    for rank, row in enumerate(rows, 1):
        row["diagnostic_rank"] = rank
    return rows


def _rank_correlation(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> float:
    import numpy as np

    left_rank = {(row["width"], row["start"]): row["diagnostic_rank"] for row in left}
    right_rank = {(row["width"], row["start"]): row["diagnostic_rank"] for row in right}
    keys = sorted(left_rank)
    if keys != sorted(right_rank):
        raise P4BError("diagnostic ranking key closure differs")
    return float(np.corrcoef([left_rank[key] for key in keys], [right_rank[key] for key in keys])[0, 1])


def _control_summary(
    original_records: Sequence[Mapping[str, Any]],
    control_records: Sequence[Mapping[str, Any]],
    *,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    import numpy as np

    original_entropy, original_kl = _curves(original_records, weights=weights)
    control_entropy, control_kl = _curves(control_records, weights=weights)
    if np.array_equal(original_entropy, control_entropy) and np.array_equal(original_kl, control_kl):
        raise P4BError("control trajectories are degenerate identical to original")
    original_strict = _diagnostic_ranking(original_entropy, original_kl, STRICT_KEYS)
    control_strict = _diagnostic_ranking(control_entropy, control_kl, STRICT_KEYS)
    original_edge = _diagnostic_ranking(original_entropy, original_kl, EDGE_KEYS)
    control_edge = _diagnostic_ranking(control_entropy, control_kl, EDGE_KEYS)
    return {
        "record_count": len(control_records),
        "diagnostic_only_never_enters_selector_score": True,
        "boundary_entropy_curve_correlation": float(np.corrcoef(original_entropy, control_entropy)[0, 1]),
        "boundary_kl_curve_correlation": float(np.corrcoef(original_kl, control_kl)[0, 1]),
        "boundary_entropy_max_abs_difference": float(np.max(np.abs(original_entropy - control_entropy))),
        "boundary_kl_max_abs_difference": float(np.max(np.abs(original_kl - control_kl))),
        "strict154_rank_correlation": _rank_correlation(original_strict, control_strict),
        "edge224_rank_correlation": _rank_correlation(original_edge, control_edge),
        "width4_primary_diagnostic_original": [row for row in original_strict if row["width"] == 4],
        "width4_primary_diagnostic_control": [row for row in control_strict if row["width"] == 4],
        "strict154_diagnostic_control": control_strict,
        "edge224_diagnostic_control": control_edge,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    if path.exists():
        raise FileExistsError("refusing to overwrite CSV artifact")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def freeze_selector(
    *, card_path: Path, run_root: Path, expected_commit: str, argv: Sequence[str]
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = assert_authorized_run_root(run_root, must_exist=True)
    git = _git_provenance(expected_commit)
    card = _card(card_path)
    freeze_dir = root / "freeze"
    expected_outputs = [
        freeze_dir / "phase4_selector_report.json",
        freeze_dir / "phase4_selector_freeze.json",
        freeze_dir / "phase4_outcome_panel_manifest.json",
    ]
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError("selector analyzer/freeze is irreversible and already started")
    originals = load_strict_jsonl(
        root / "formal/phase4_shared12032_no_loop_trajectories.jsonl"
    )
    options = load_strict_jsonl(root / "controls/option/merged.jsonl")
    templates = load_strict_jsonl(root / "controls/template/merged.jsonl")
    sources, source_manifest = _load_live_sources(root)
    if len(originals) != 12032 or len(options) != 512 or len(templates) != 14:
        raise P4BError("formal merged trajectory counts differ")

    # This is the sole scientific analyzer call in P4-B.
    report = analyze_selector(
        originals,
        replicates=card["bootstrap"]["selector_replicates"],
        seed=card["bootstrap"]["selector_seed"],
        selection_frequency_threshold=card["panel"]["selection_frequency_threshold"],
        d36_tolerance=card["numeric_contract"]["d36_abs_tolerance"],
    )
    verify_selector_report(report)
    require_non_degenerate_width4_scores(report)
    option_identities = [record["canonical_identity"] for record in options]
    original_map = {record["canonical_identity"]: record for record in originals}
    option_originals = [original_map[identity] for identity in option_identities]
    option_summary = _control_summary(option_originals, options)
    source_counts = Counter(row["category"] for row in sources)
    original_category_aggregate = []
    for category in CATEGORIES:
        category_rows = [record for record in originals if record["category"] == category]
        entropy, kl = _curves(category_rows)
        synthetic = {
            "boundaries": [
                {"full_vocabulary_entropy": float(entropy[index]), "kl_to_final": float(kl[index])}
                for index in range(37)
            ]
        }
        original_category_aggregate.append(synthetic)
    template_by_category = {record["category"]: record for record in templates}
    ordered_templates = [template_by_category[category] for category in CATEGORIES]
    weights = [source_counts[category] for category in CATEGORIES]
    template_summary = _control_summary(
        original_category_aggregate, ordered_templates, weights=weights
    )
    option_summary.update(
        {
            "schema_version": "loopscope.phase4.p4b-option-control-summary.v1",
            "status": "PASS",
            "git_commit": expected_commit,
            "card_sha256": CARD_BYTE_SHA256,
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "control_input_sha256": file_sha256(root / "controls/option/merged.jsonl"),
            "matched_original_identity_sha256": semantic_sha256(option_identities),
        }
    )
    template_summary.update(
        {
            "schema_version": "loopscope.phase4.p4b-template-control-summary.v1",
            "status": "PASS",
            "git_commit": expected_commit,
            "card_sha256": CARD_BYTE_SHA256,
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "control_input_sha256": file_sha256(root / "controls/template/merged.jsonl"),
            "category_weights": dict(source_counts),
        }
    )
    scan_forbidden_fields(option_summary, allow_options=False)
    scan_forbidden_fields(template_summary, allow_options=False)

    known = extract_known_outcome_windows(card)
    panel = build_outcome_panel(report, known)
    report_hash = write_new_json(freeze_dir / "phase4_selector_report.json", report)
    write_new_json(freeze_dir / "phase4_width4_primary_ranking.json", report["width4_primary"])
    write_new_json(freeze_dir / "phase4_variable_width_strict154.json", report["variable_width_strict"])
    write_new_json(freeze_dir / "phase4_variable_width_edge224.json", report["variable_width_edge"])
    _write_csv(
        freeze_dir / "phase4_width4_primary_ranking.csv",
        report["width4_primary"]["rows"],
        ("primary_rank", "start", "window", "E", "K", "score", "eligible", "selection_frequency"),
    )
    _write_csv(
        freeze_dir / "phase4_variable_width_strict154.csv",
        report["variable_width_strict"]["rows"],
        ("cross_width_rank", "within_width_rank", "width", "start", "window", "E_rate", "K_rate", "score"),
    )
    _write_csv(
        freeze_dir / "phase4_variable_width_edge224.csv",
        report["variable_width_edge"]["rows"],
        ("cross_width_rank", "within_width_rank", "width", "start", "window", "E_rate", "K_rate", "score"),
    )
    option_hash = write_new_json(root / "controls/phase4_option_permutation_summary.json", option_summary)
    template_hash = write_new_json(root / "controls/phase4_template_control_summary.json", template_summary)
    implementation_relatives = (
        "src/tflt/loopscope/phase4_schema.py",
        "src/tflt/loopscope/phase4_scalars.py",
        "src/tflt/loopscope/phase4_selector.py",
        "src/tflt/loopscope/phase4_acquisition.py",
        "src/tflt/loopscope/phase4_runtime.py",
        "scripts/loopscope/run_qwen4_phase4_p4b.py",
    )
    implementation_hashes = {
        relative: file_sha256(_repo_root() / relative) for relative in implementation_relatives
    }
    freeze = {
        "schema_version": "loopscope.phase4.selector-freeze.v1",
        "status": "BYTE_FROZEN_OUTCOME_BLIND",
        "frozen_at_utc": _utc_now(),
        "gate": "P4-B",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "git": git,
        "card_sha256": CARD_BYTE_SHA256,
        "control_sha256": CONTROL_BYTE_SHA256,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "input_trajectory_sha256": file_sha256(
            root / "formal/phase4_shared12032_no_loop_trajectories.jsonl"
        ),
        "selector_report_sha256": report_hash,
        "option_summary_sha256": option_hash,
        "template_summary_sha256": template_hash,
        "implementation_hashes": implementation_hashes,
        "width4_candidate_count": 33,
        "width4_consensus_count": 25,
        "variable_width_strict_count": 154,
        "variable_width_edge_count": 224,
        "selected_window": report["width4_primary"]["selected_window"],
        "abstain": report["width4_primary"]["abstain"],
        "known_registry": known,
        "analyzer_invocation_count": 1,
        "outcome_access_before_freeze": False,
        "argv": list(argv),
    }
    freeze["freeze_sha256"] = semantic_sha256(freeze)
    freeze_file_hash = write_new_json(freeze_dir / "phase4_selector_freeze.json", freeze)
    panel_manifest = wrap_outcome_panel(
        panel,
        card=card,
        selector_freeze_sha256=freeze_file_hash,
        bindings={
            "git_commit": expected_commit,
            "card_sha256": CARD_BYTE_SHA256,
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "selector_report_sha256": report_hash,
            "fresh_baseline_required": True,
            "fresh_15_18_required": True,
        },
    )
    panel_hash = write_new_json(
        freeze_dir / "phase4_outcome_panel_manifest.json", panel_manifest
    )
    verifier = {
        "schema_version": "loopscope.phase4.selector-verifier-receipt.v1",
        "status": "PASS",
        "verified_at_utc": _utc_now(),
        "selector_report_sha256": report_hash,
        "selector_freeze_sha256": freeze_file_hash,
        "panel_manifest_sha256": panel_hash,
        "record_count": report["record_count"],
        "width4_candidate_count": report["width4_primary"]["candidate_start_count"],
        "width4_consensus_count": report["width4_primary"]["consensus_start_count"],
        "variable_width_strict_count": report["variable_width_strict"]["row_count"],
        "variable_width_edge_count": report["variable_width_edge"]["row_count"],
        "known_registry_excluded_from_blind": all(
            window not in panel["blind_high3"] + panel["blind_low3"] for window in known
        ),
        "fresh_baseline": True,
        "fresh_15_18": True,
        "variable_width_outcomes_authorized": False,
    }
    verifier["manifest_sha256"] = semantic_sha256(verifier)
    write_new_json(freeze_dir / "phase4_selector_verifier_receipt.json", verifier)
    return {
        "status": "PASS",
        "selected_window": report["width4_primary"]["selected_window"],
        "abstain": report["width4_primary"]["abstain"],
        "selector_freeze_sha256": freeze_file_hash,
        "panel_manifest_sha256": panel_hash,
    }


def write_launcher(
    *,
    stage: str,
    kind: Optional[str],
    card_path: Path,
    run_root: Path,
    expected_commit: str,
    partition: str,
    qos: Optional[str],
    account: Optional[str],
    gres: str,
    time_limit: str,
    cpus: int,
    memory: str,
    array_throttle: Optional[int],
    rationale: str,
) -> Dict[str, Any]:
    root = assert_authorized_run_root(run_root, must_exist=True)
    _git_provenance(expected_commit)
    if stage not in {"smoke", "formal"} or (stage == "formal" and kind not in {"original", "option", "template"}):
        raise P4BError("launcher stage/kind is invalid")
    if any("\n" in str(value) or "\r" in str(value) for value in (partition, qos, account, gres, time_limit, memory, rationale)):
        raise P4BError("launcher resource field contains a newline")
    if cpus < 1 or not rationale.strip():
        raise P4BError("launcher CPU/rationale field is invalid")
    label = "smoke" if stage == "smoke" else str(kind)
    script_path = root / ("slurm/p4b-%s.sbatch" % label)
    if script_path.exists():
        raise FileExistsError("launcher already frozen")
    directives = [
        "#SBATCH --job-name=p4b-%s" % label,
        "#SBATCH --partition=%s" % partition,
        "#SBATCH --gres=%s" % gres,
        "#SBATCH --time=%s" % time_limit,
        "#SBATCH --cpus-per-task=%d" % cpus,
        "#SBATCH --mem=%s" % memory,
        "#SBATCH --output=%s/slurm/p4b-%s-%%A_%%a.out" % (root, label),
        "#SBATCH --error=%s/slurm/p4b-%s-%%A_%%a.err" % (root, label),
    ]
    if qos:
        directives.append("#SBATCH --qos=%s" % qos)
    if account:
        directives.append("#SBATCH --account=%s" % account)
    if stage == "formal":
        manifest = load_strict_json(root / ("formal/%s_shard_manifest.json" % kind))
        throttle = manifest["shard_count"] if array_throttle is None else array_throttle
        if throttle < 1 or throttle > manifest["shard_count"]:
            raise P4BError("array throttle is outside the frozen shard count")
        directives.append(
            "#SBATCH --array=0-%d%%%d" % (manifest["shard_count"] - 1, throttle)
        )
    common = [
        "#!/usr/bin/env bash",
        *directives,
        "set -euo pipefail",
        "if [[ -r /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi",
        "module load anaconda3 cuda/12.4",
        ". %s/bin/activate" % shlex.quote(str(AUDITED_VENV)),
        "cd %s" % shlex.quote(str(REMOTE_REPO)),
        "export PYTHONDONTWRITEBYTECODE=1",
        "export PYTHONPATH=%s" % shlex.quote(str(REMOTE_REPO / "src")),
        "export HF_HOME=%s" % shlex.quote(str(HF_HOME)),
        "export TRANSFORMERS_CACHE=%s" % shlex.quote(str(HF_HOME / "hub")),
        "export HF_DATASETS_CACHE=%s" % shlex.quote(str(HF_DATASETS_CACHE)),
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1",
        "unset HF_ENDPOINT",
    ]
    cli = REMOTE_REPO / "scripts/loopscope/run_qwen4_phase4_p4b.py"
    card_remote = REMOTE_REPO / "configs/loopscope/phase4_pv_ek_trs_card.json"
    if stage == "smoke":
        command = "%s %s acquire-smoke --card %s --run-root %s --expected-commit %s" % (
            shlex.quote(str(AUDITED_VENV / "bin/python")),
            shlex.quote(str(cli)),
            shlex.quote(str(card_remote)),
            shlex.quote(str(root)),
            shlex.quote(expected_commit),
        )
    else:
        command = "%s %s acquire-shard --kind %s --shard-id \"${SLURM_ARRAY_TASK_ID}\" --card %s --run-root %s --expected-commit %s" % (
            shlex.quote(str(AUDITED_VENV / "bin/python")),
            shlex.quote(str(cli)),
            shlex.quote(str(kind)),
            shlex.quote(str(card_remote)),
            shlex.quote(str(root)),
            shlex.quote(expected_commit),
        )
    text = "\n".join(common + [command, ""])
    with script_path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    config = {
        "schema_version": "loopscope.phase4.p4b-launcher-config.v1",
        "stage": stage,
        "kind": kind,
        "partition": partition,
        "qos": qos,
        "account": account,
        "gres": gres,
        "time_limit": time_limit,
        "cpus_per_task": cpus,
        "memory": memory,
        "array_throttle": array_throttle,
        "operational_rationale": rationale,
        "script": str(script_path.relative_to(root)),
        "script_sha256": file_sha256(script_path),
        "git_commit": expected_commit,
    }
    config["manifest_sha256"] = semantic_sha256(config)
    write_new_json(root / ("slurm/p4b-%s-launcher.json" % label), config)
    return config


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Outcome-blind LoopScope Phase 4 P4-B only")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-source", "acquire-smoke", "freeze-selector"):
        child = sub.add_parser(name)
        _common(child)
    shards = sub.add_parser("freeze-shards")
    _common(shards)
    shards.add_argument("--original-shards", required=True, type=int)
    shards.add_argument("--option-shards", required=True, type=int)
    shards.add_argument("--template-shards", required=True, type=int)
    acquire = sub.add_parser("acquire-shard")
    _common(acquire)
    acquire.add_argument("--kind", required=True, choices=("original", "option", "template"))
    acquire.add_argument("--shard-id", required=True, type=int)
    merge = sub.add_parser("merge")
    _common(merge)
    merge.add_argument("--kind", required=True, choices=("original", "option", "template"))
    launcher = sub.add_parser("write-launcher")
    _common(launcher)
    launcher.add_argument("--stage", required=True, choices=("smoke", "formal"))
    launcher.add_argument("--kind", choices=("original", "option", "template"))
    launcher.add_argument("--partition", required=True)
    launcher.add_argument("--qos")
    launcher.add_argument("--account")
    launcher.add_argument("--gres", required=True)
    launcher.add_argument("--time-limit", required=True)
    launcher.add_argument("--cpus", required=True, type=int)
    launcher.add_argument("--memory", required=True)
    launcher.add_argument("--array-throttle", type=int)
    launcher.add_argument("--rationale", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    provenance_argv = [str(Path(__file__).resolve())] + (list(sys.argv[1:]) if argv is None else list(argv))
    common = {
        "card_path": args.card,
        "run_root": args.run_root,
        "expected_commit": args.expected_commit,
    }
    if args.command == "prepare-source":
        result = prepare_source(argv=provenance_argv, **common)
    elif args.command == "acquire-smoke":
        result = acquire_smoke(argv=provenance_argv, **common)
    elif args.command == "freeze-shards":
        result = freeze_shards(
            original_shards=args.original_shards,
            option_shards=args.option_shards,
            template_shards=args.template_shards,
            argv=provenance_argv,
            **common,
        )
    elif args.command == "acquire-shard":
        result = acquire_shard(kind=args.kind, shard_id=args.shard_id, argv=provenance_argv, **common)
    elif args.command == "merge":
        result = merge_kind(kind=args.kind, **common)
    elif args.command == "freeze-selector":
        result = freeze_selector(argv=provenance_argv, **common)
    elif args.command == "write-launcher":
        result = write_launcher(
            stage=args.stage,
            kind=args.kind,
            partition=args.partition,
            qos=args.qos,
            account=args.account,
            gres=args.gres,
            time_limit=args.time_limit,
            cpus=args.cpus,
            memory=args.memory,
            array_throttle=args.array_throttle,
            rationale=args.rationale,
            **common,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
