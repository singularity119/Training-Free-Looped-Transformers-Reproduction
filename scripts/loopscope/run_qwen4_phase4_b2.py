#!/usr/bin/env python3
"""Execute isolated LoopScope Phase 4 Gate B-2 hidden-geometry diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _install_isolated_tflt_namespace() -> None:
    if "tflt" in sys.modules:
        raise RuntimeError("B-2 producer requires a fresh isolated Python process")
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "src/tflt"
    package = types.ModuleType("tflt")
    package.__package__ = "tflt"
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    package.__file__ = str(package_root)
    sys.modules["tflt"] = package


_install_isolated_tflt_namespace()

from tflt.loopscope.phase4_acquisition import (
    load_strict_json,
    load_strict_jsonl,
    write_new_json,
    write_new_jsonl,
)
from tflt.loopscope.phase4_hidden_geometry import (
    COSINE_KEY,
    L2_KEY,
    analyze_hidden_geometry_records,
    strip_geometry_to_p4b,
    validate_hidden_geometry_record,
)
from tflt.loopscope.phase4_runtime import (
    acquire_prefix_scalar_record,
    load_native_prefix_runtime,
    render_exact_prefix,
    tokenization_closure,
    validation_demos_by_category,
)
from tflt.loopscope.phase4_schema import (
    file_sha256,
    load_phase4_card,
    scan_forbidden_fields,
    semantic_sha256,
    validate_shared_identity_contract,
    validate_trajectory_record,
)


EXECUTOR_THREAD_ID = "019f6ed2-1bbb-74d2-80bf-f20846137d49"
PLANNING_THREAD_ID = "019f6bf0-d1ec-7d53-ba06-391e9db2bd88"
BRANCH = "loopscope-phase4-b2"
BASE_COMMIT = "a210da31d734d539b198d939118136e3e035a36f"
CONTRACT_BYTE_SHA256 = "b9591df0136d94d8b3cb7d045eca4f07aa3ac03c1f4fee161c6a3d7e6d4260fc"
CARD_BYTE_SHA256 = "980955386907a1865699808219da1379031c1395d9cdb77029585cf266f160f8"
RENDERER_SHA256 = "74ab409c4e4c96e4351fbe6122d519f64dd3e11381a676957631e858923cc9fc"
DATASET_FINGERPRINT_COMPONENTS = {
    "dataset_info_sha256": "ce81f7ec00c6701dc737889a3171a9dd32d2521914bc1e5029a3f8bb28df9aab",
    "test_arrow_sha256": "4c741944e3b53719b433be6e7916e4ab9e8ff33f70e9c365614b4e317c0476bb",
    "validation_arrow_sha256": "5730c2a9cd07a1ee5f70f4cd1940e43da1df7602c21977b7dc7ef51d496bdf00",
}
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope_phase4_b2"
)
AUDITED_VENV = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope/.venv-loopscope-cu121-20260711"
)
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")
DATASET_CACHE_SNAPSHOT = (
    HF_DATASETS_CACHE
    / "TIGER-Lab___mmlu-pro/default/0.0.0/b189ec765aa7ed75c8acfea42df31fdae71f97be"
)
AUTHORIZED_RUN_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-b2-hidden-geometry-20260717T064424Z"
)
P4B_RUN_ROOT = Path(
    "/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/runs/phase4-p4b-20260716T204154Z"
)
EXPECTED_P4B_FILES = {
    "source/phase4_shared12032_source_manifest.json": "173aeb1f7a8663652975ec4d0faf0998d9a017e9fb4ee30e149f23dc1a2e944f",
    "formal/phase4_shared12032_trajectory_manifest.json": "522f014ba31a5f69882a9853fe535d0758cc8e5454817c7ea22cd4c1decf8d34",
    "formal/phase4_shared12032_no_loop_trajectories.jsonl": "879d63c271bc828cec77f3dafff1661dd36218ffb50decb945ab33dd5ac4332d",
    "formal/original_shard_manifest.json": "201dc5a9bc175fa81d7ec4fa38abf0bc35dc11cc62029aea8fda2ec3119cc297",
    "smoke/membership.json": "30707641105bdc42a4c8924f9b39e10b5b900e6851c556dd26d70aca48edfe54",
}
SMOKE_IDENTITIES = (
    "70:business:ori_mmlu-business_ethics:7967613730e252a49314cfb3dfb30134f09b9e55c0b8404c6a3f6a51d51ac5f7",
    "866:law:ori_mmlu-professional_law:4ef6e81c98e41ee505138e6df2df7a15b1014209cac8fcc126bf1370fd90db88",
    "1986:psychology:ori_mmlu-professional_psychology:2478839201a23311fb6076ddb3e6b47d8e8391fd636f2ef83e8defcdb6da4e0c",
    "2804:biology:ori_mmlu-high_school_biology:1f07729da6ee685072170f005de3debca8aa86842b2927df6fb8d935cdca4bf8",
)
_DATASET_CLOSURE_VERIFIED = False


class B2Error(ValueError):
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
        raise B2Error("Git provenance command failed: %s" % command[0])
    return completed.stdout.strip()


def _git_provenance(expected_commit: str) -> Dict[str, Any]:
    root = _repo_root()
    head = _run(("git", "rev-parse", "HEAD"), cwd=root)
    branch = _run(("git", "symbolic-ref", "--short", "HEAD"), cwd=root)
    origin = _run(("git", "rev-parse", "origin/%s" % BRANCH), cwd=root)
    origin_url = _run(("git", "remote", "get-url", "origin"), cwd=root)
    dirty = bool(_run(("git", "status", "--porcelain"), cwd=root))
    if head != expected_commit or origin != expected_commit or branch != BRANCH or dirty:
        raise B2Error("Git provenance is not clean pushed B-2 at the expected commit")
    if origin_url != "git@github.com:singularity119/Training-Free-Looped-Transformers-Reproduction.git":
        raise B2Error("B-2 origin identity differs")
    return {
        "branch": branch,
        "commit": head,
        "origin_branch": origin,
        "origin_url": origin_url,
        "dirty": False,
    }


def _contract() -> Dict[str, Any]:
    path = _repo_root() / "configs/loopscope/phase4_b2_hidden_geometry_contract.json"
    if file_sha256(path) != CONTRACT_BYTE_SHA256:
        raise B2Error("B-2 contract byte hash differs")
    value = load_strict_json(path)
    if (
        value.get("gate") != "B-2"
        or value.get("executor_thread_id") != EXECUTOR_THREAD_ID
        or value.get("control_sha256")
        != "381fb0bf84cc5b25fc6eaee3a1fd48bc75c1bc8b97a7f324ad748ac5cc9f8288"
    ):
        raise B2Error("B-2 contract authority binding differs")
    return value


def _card(card_path: Path) -> Dict[str, Any]:
    if file_sha256(card_path.resolve()) != CARD_BYTE_SHA256:
        raise B2Error("frozen Phase 4 card byte hash differs")
    return load_phase4_card(card_path.resolve())


def _assert_run_root(run_root: Path, *, must_exist: bool) -> Path:
    value = run_root.resolve(strict=False)
    if value != AUTHORIZED_RUN_ROOT:
        raise B2Error("run root differs from exact B-2 authorization")
    if must_exist and not value.is_dir():
        raise B2Error("authorized B-2 run root does not exist")
    if not must_exist and value.exists():
        raise FileExistsError("authorized B-2 run root already exists")
    return value


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
            raise B2Error("audited offline environment variable differs: %s" % key)
    if os.environ.get("HF_ENDPOINT"):
        raise B2Error("HF_ENDPOINT substitution is forbidden")


def _load_renderer() -> Any:
    from lm_eval.tasks.mmlu_pro import utils

    if file_sha256(Path(utils.__file__).resolve()) != RENDERER_SHA256:
        raise B2Error("installed MMLU-Pro renderer source hash differs")
    return utils


def _load_dataset(card: Mapping[str, Any], split: str) -> Any:
    global _DATASET_CLOSURE_VERIFIED

    from datasets import DownloadMode, load_dataset

    if not _DATASET_CLOSURE_VERIFIED:
        observed = {
            "dataset_info_sha256": file_sha256(DATASET_CACHE_SNAPSHOT / "dataset_info.json"),
            "test_arrow_sha256": file_sha256(DATASET_CACHE_SNAPSHOT / "mmlu-pro-test.arrow"),
            "validation_arrow_sha256": file_sha256(
                DATASET_CACHE_SNAPSHOT / "mmlu-pro-validation.arrow"
            ),
        }
        if observed != DATASET_FINGERPRINT_COMPONENTS:
            raise B2Error("cached dataset snapshot component hash differs")
        _DATASET_CLOSURE_VERIFIED = True
    value = load_dataset(
        card["task"]["dataset"],
        revision=card["task"]["dataset_revision"],
        split=split,
        cache_dir=str(HF_DATASETS_CACHE),
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    expected = 70 if split == "validation" else 12032
    if len(value) != expected:
        raise B2Error("cached dataset split count differs")
    return value


def _copy_new(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    return file_sha256(target)


def _write_text_new(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def _write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise B2Error("refusing to write an empty diagnostic table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def prepare_run(
    *, card_path: Path, run_root: Path, expected_commit: str, argv: Sequence[str]
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = _assert_run_root(run_root, must_exist=False)
    git = _git_provenance(expected_commit)
    card = _card(card_path)
    contract = _contract()
    for relative, expected in EXPECTED_P4B_FILES.items():
        if file_sha256(P4B_RUN_ROOT / relative) != expected:
            raise B2Error("P4-B admission file hash differs: %s" % relative)
    source_manifest = load_strict_json(
        P4B_RUN_ROOT / "source/phase4_shared12032_source_manifest.json"
    )
    sources = load_strict_jsonl(P4B_RUN_ROOT / "source/shared12032.jsonl")
    validate_shared_identity_contract(sources, sources, require_full_population=True)
    if file_sha256(P4B_RUN_ROOT / "source/shared12032.jsonl") != source_manifest["record_file_sha256"]:
        raise B2Error("P4-B sanitized source bytes differ from its manifest")
    token_path = P4B_RUN_ROOT / str(source_manifest["token_closure_file"])
    if file_sha256(token_path) != source_manifest["token_closure_file_sha256"]:
        raise B2Error("P4-B token-closure bytes differ from its manifest")
    token_rows = load_strict_jsonl(token_path)
    if len(token_rows) != 12032 or len({row["canonical_identity"] for row in token_rows}) != 12032:
        raise B2Error("P4-B token-closure membership differs")
    smoke = load_strict_json(P4B_RUN_ROOT / "smoke/membership.json")
    if tuple(smoke.get("original_identities", ())) != SMOKE_IDENTITIES:
        raise B2Error("P4-B smoke original identities differ")
    upstream_shards = load_strict_json(P4B_RUN_ROOT / "formal/original_shard_manifest.json")
    if upstream_shards.get("shard_count") != 24 or upstream_shards.get("record_count") != 12032:
        raise B2Error("P4-B formal shard partition differs")
    derived_shards = {
        "schema_version": "loopscope.phase4.b2-shard-manifest.v1",
        "gate": "B-2",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "run_root": str(root),
        "record_count": 12032,
        "shard_count": 24,
        "distribution": upstream_shards["distribution"],
        "ordered_identity_sha256": upstream_shards["ordered_identity_sha256"],
        "upstream_file_sha256": EXPECTED_P4B_FILES["formal/original_shard_manifest.json"],
        "shards": [
            {
                "shard_id": shard["shard_id"],
                "record_count": shard["record_count"],
                "membership_sha256": shard["membership_sha256"],
                "records": shard["records"],
                "output_relative": "formal/shards/shard-%04d" % shard["shard_id"],
            }
            for shard in upstream_shards["shards"]
        ],
    }
    derived_shards["manifest_sha256"] = semantic_sha256(derived_shards)

    root.mkdir(parents=True, exist_ok=False)
    for relative in ("source", "smoke", "formal", "analysis", "slurm"):
        (root / relative).mkdir()
    copies = {
        "source_manifest": _copy_new(
            P4B_RUN_ROOT / "source/phase4_shared12032_source_manifest.json",
            root / "source/p4b_source_manifest.json",
        ),
        "shared12032": _copy_new(
            P4B_RUN_ROOT / "source/shared12032.jsonl", root / "source/shared12032.jsonl"
        ),
        "token_closure": _copy_new(token_path, root / "source/shared12032_token_closure.jsonl"),
        "smoke_membership": _copy_new(
            P4B_RUN_ROOT / "smoke/membership.json", root / "source/p4b_smoke_membership.json"
        ),
        "upstream_shard_manifest": _copy_new(
            P4B_RUN_ROOT / "formal/original_shard_manifest.json",
            root / "source/p4b_original_shard_manifest.json",
        ),
    }
    write_new_json(root / "formal/phase4_b2_shard_manifest.json", derived_shards)
    receipt = {
        "schema_version": "loopscope.phase4.b2-admission.v1",
        "status": "ADMITTED",
        "created_at_utc": _utc_now(),
        "gate": "B-2",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "card_sha256": CARD_BYTE_SHA256,
        "contract_sha256": CONTRACT_BYTE_SHA256,
        "control_sha256": contract["control_sha256"],
        "p4b_file_hashes": dict(EXPECTED_P4B_FILES),
        "copied_source_hashes": copies,
        "source_manifest_semantic_sha256": source_manifest["manifest_sha256"],
        "ordered_identity_sha256": source_manifest["ordered_identity_sha256"],
        "derived_shard_manifest_sha256": derived_shards["manifest_sha256"],
        "record_count": len(sources),
        "smoke_identities": list(SMOKE_IDENTITIES),
        "model_revision": card["model"]["revision"],
        "dataset_revision": card["task"]["dataset_revision"],
        "argv": list(argv),
    }
    receipt["manifest_sha256"] = semantic_sha256(receipt)
    write_new_json(root / "phase4_b2_admission.json", receipt)
    return {"status": "ADMITTED", "receipt_sha256": receipt["manifest_sha256"]}


def _load_sources(root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    admission = load_strict_json(root / "phase4_b2_admission.json")
    if admission.get("status") != "ADMITTED" or admission.get("contract_sha256") != CONTRACT_BYTE_SHA256:
        raise B2Error("B-2 admission receipt differs")
    source_manifest = load_strict_json(root / "source/p4b_source_manifest.json")
    sources = load_strict_jsonl(root / "source/shared12032.jsonl")
    validate_shared_identity_contract(sources, sources, require_full_population=True)
    if file_sha256(root / "source/shared12032.jsonl") != source_manifest["record_file_sha256"]:
        raise B2Error("copied sanitized source hash differs")
    token_rows = load_strict_jsonl(root / "source/shared12032_token_closure.jsonl")
    if file_sha256(root / "source/shared12032_token_closure.jsonl") != source_manifest["token_closure_file_sha256"]:
        raise B2Error("copied token-closure hash differs")
    closures = {row["canonical_identity"]: row for row in token_rows}
    if len(closures) != 12032:
        raise B2Error("copied token-closure membership differs")
    return sources, source_manifest, closures


def _safe_target(source: Mapping[str, Any]) -> Dict[str, Any]:
    value = {
        key: source[key]
        for key in ("question_id", "category", "src", "question", "ordered_options")
    }
    scan_forbidden_fields(value)
    return value


def _expected_tokenization(
    source: Mapping[str, Any], closures: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    value = closures.get(str(source["canonical_identity"]))
    if value is None:
        raise B2Error("source token-closure identity is missing")
    for key in (
        "rendered_prefix_sha256",
        "rendered_token_ids_sha256",
        "attention_mask_sha256",
        "last_effective_prefix_token_index",
    ):
        if value.get(key) != source.get(key):
            raise B2Error("source/token-closure drift at %s" % key)
    return tokenization_closure(value)


def _producer_provenance(
    *,
    source: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    expected_commit: str,
    mode: str,
    expected_tokenization: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = {
        "gate": "B-2",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git_commit": expected_commit,
        "card_sha256": CARD_BYTE_SHA256,
        "contract_sha256": CONTRACT_BYTE_SHA256,
        "control_sha256": _contract()["control_sha256"],
        "source_manifest_file_sha256": EXPECTED_P4B_FILES[
            "source/phase4_shared12032_source_manifest.json"
        ],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "dataset_revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "mode": mode,
        "forward_type": "one_native_no_loop_prefix_forward",
        "use_cache": False,
        "projection_dtype": "float32",
        "hidden_geometry_reduction_dtype": "float32",
        "final_norm_application": "exactly_once_per_boundary",
        "target_generation_count": 0,
        "source_closure": {
            key: source[key]
            for key in (
                "canonical_identity",
                "sanitized_content_sha256",
                "rendered_prefix_sha256",
                "rendered_token_ids_sha256",
                "attention_mask_sha256",
            )
        },
        "input_closure": {
            "canonical_identity": source["canonical_identity"],
            **dict(expected_tokenization),
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
        "geometry_reduction_dtype": "float32",
        "device_name": str(properties.name),
        "device_memory_bytes": int(properties.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_qos": os.environ.get("SLURM_JOB_QOS"),
        "revision_closure": dict(runtime.revision_closure),
        "forward_count": len(facts),
        "target_generation_count": 0,
        "maximum_final_projection_difference": max(
            float(row["final_projection_max_abs_difference"]) for row in facts
        ),
    }


def _acquire_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    closures: Mapping[str, Mapping[str, Any]],
    card: Mapping[str, Any],
    expected_commit: str,
    mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    renderer = _load_renderer()
    demos = validation_demos_by_category(_load_dataset(card, "validation"))
    runtime = load_native_prefix_runtime(card)
    records: List[Dict[str, Any]] = []
    facts: List[Dict[str, Any]] = []
    for source in rows:
        prefix = render_exact_prefix(
            _safe_target(source), demos[source["category"]], renderer
        )
        expected = _expected_tokenization(source, closures)
        record, fact = acquire_prefix_scalar_record(
            runtime,
            prefix=prefix,
            canonical_identity=source["canonical_identity"],
            category=source["category"],
            expected_tokenization=expected,
            producer_provenance=_producer_provenance(
                source=source,
                source_manifest=source_manifest,
                expected_commit=expected_commit,
                mode=mode,
                expected_tokenization=expected,
            ),
            d36_tolerance=card["numeric_contract"]["d36_abs_tolerance"],
            include_hidden_geometry=True,
        )
        validate_hidden_geometry_record(record)
        records.append(record)
        facts.append(fact)
    return records, facts, _runtime_evidence(runtime, facts)


def acquire_smoke(
    *,
    pass_id: int,
    card_path: Path,
    run_root: Path,
    expected_commit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = _assert_run_root(run_root, must_exist=True)
    _git_provenance(expected_commit)
    card = _card(card_path)
    sources, source_manifest, closures = _load_sources(root)
    if pass_id not in (1, 2):
        raise B2Error("smoke pass id must be 1 or 2")
    output = root / ("smoke/pass-%d" % pass_id)
    if output.exists():
        raise FileExistsError("B-2 deterministic smoke pass already exists")
    source_map = {row["canonical_identity"]: row for row in sources}
    rows = [source_map[value] for value in SMOKE_IDENTITIES]
    records, facts, runtime = _acquire_rows(
        rows=rows,
        source_manifest=source_manifest,
        closures=closures,
        card=card,
        expected_commit=expected_commit,
        mode="smoke_pass_%d" % pass_id,
    )
    output.mkdir(parents=True, exist_ok=False)
    records_hash = write_new_jsonl(output / "records.jsonl", records)
    receipt = {
        "schema_version": "loopscope.phase4.b2-smoke-pass.v1",
        "status": "COMPLETED",
        "pass_id": pass_id,
        "completed_at_utc": _utc_now(),
        "record_count": 4,
        "identities": list(SMOKE_IDENTITIES),
        "record_file_sha256": records_hash,
        "git_commit": expected_commit,
        "target_generation_count": 0,
        "runtime": runtime,
        "argv": list(argv),
    }
    receipt["manifest_sha256"] = semantic_sha256(receipt)
    write_new_json(output / "receipt.json", receipt)
    return {"status": "COMPLETED", "pass_id": pass_id, "record_file_sha256": records_hash}


def verify_smoke(
    *, card_path: Path, run_root: Path, expected_commit: str, argv: Sequence[str]
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = _assert_run_root(run_root, must_exist=True)
    _git_provenance(expected_commit)
    card = _card(card_path)
    report_path = root / "smoke/phase4_b2_smoke_report.json"
    if report_path.exists():
        raise FileExistsError("B-2 smoke report already exists")
    passes = []
    for pass_id in (1, 2):
        output = root / ("smoke/pass-%d" % pass_id)
        receipt = load_strict_json(output / "receipt.json")
        records = load_strict_jsonl(output / "records.jsonl")
        if file_sha256(output / "records.jsonl") != receipt["record_file_sha256"]:
            raise B2Error("B-2 smoke record file hash differs")
        if [row["canonical_identity"] for row in records] != list(SMOKE_IDENTITIES):
            raise B2Error("B-2 smoke identity/order differs")
        for record in records:
            validate_hidden_geometry_record(
                record,
                d36_tolerance=card["numeric_contract"]["d36_abs_tolerance"],
            )
            validate_trajectory_record(
                strip_geometry_to_p4b(record),
                d36_tolerance=card["numeric_contract"]["d36_abs_tolerance"],
            )
        passes.append((receipt, records))
    maximum_abs = 0.0
    maximum_relative = 0.0
    for first, second in zip(passes[0][1], passes[1][1]):
        for left, right in zip(first["boundaries"], second["boundaries"]):
            for key in (L2_KEY, COSINE_KEY):
                absolute = abs(float(left[key]) - float(right[key]))
                scale = max(abs(float(left[key])), abs(float(right[key])))
                relative = 0.0 if scale == 0.0 else absolute / scale
                maximum_abs = max(maximum_abs, absolute)
                maximum_relative = max(maximum_relative, relative)
                if absolute > 1e-6 + 1e-6 * abs(float(right[key])):
                    raise B2Error("B-2 repeated geometry scalar exceeds deterministic tolerance")
    report = {
        "schema_version": "loopscope.phase4.b2-smoke-report.v1",
        "status": "PASS",
        "gate": "B-2",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "completed_at_utc": _utc_now(),
        "git_commit": expected_commit,
        "record_count_per_pass": 4,
        "boundary_count_per_record": 37,
        "identities": list(SMOKE_IDENTITIES),
        "pass_record_file_sha256": [value[0]["record_file_sha256"] for value in passes],
        "target_generation_count": 0,
        "use_cache": False,
        "endpoint_l2_abs_tolerance": 1e-7,
        "endpoint_cosine_abs_from_one_tolerance": 1e-5,
        "determinism_atol": 1e-6,
        "determinism_rtol": 1e-6,
        "maximum_geometry_abs_difference": maximum_abs,
        "maximum_geometry_relative_difference": maximum_relative,
        "p4b_scalar_projection_contract_pass": True,
        "cosine_distance_persisted": False,
        "forbidden_payloads_persisted": False,
        "runtime": [value[0]["runtime"] for value in passes],
        "argv": list(argv),
    }
    scan_forbidden_fields(report, allow_options=False)
    report["manifest_sha256"] = semantic_sha256(report)
    write_new_json(report_path, report)
    return {"status": "PASS", "report_sha256": report["manifest_sha256"]}


def _derived_shard_manifest(root: Path) -> Dict[str, Any]:
    manifest = load_strict_json(root / "formal/phase4_b2_shard_manifest.json")
    expected_sha = semantic_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != expected_sha:
        raise B2Error("B-2 shard manifest semantic hash differs")
    if manifest.get("shard_count") != 24 or manifest.get("record_count") != 12032:
        raise B2Error("B-2 shard manifest shape differs")
    return manifest


def acquire_shard(
    *,
    shard_id: int,
    card_path: Path,
    run_root: Path,
    expected_commit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = _assert_run_root(run_root, must_exist=True)
    _git_provenance(expected_commit)
    card = _card(card_path)
    smoke = load_strict_json(root / "smoke/phase4_b2_smoke_report.json")
    if smoke.get("status") != "PASS":
        raise B2Error("formal B-2 acquisition requires passing deterministic smoke")
    sources, source_manifest, closures = _load_sources(root)
    manifest = _derived_shard_manifest(root)
    if shard_id < 0 or shard_id >= manifest["shard_count"]:
        raise B2Error("formal B-2 shard id is outside the frozen partition")
    shard = manifest["shards"][shard_id]
    output = root / shard["output_relative"]
    if output.exists():
        raise FileExistsError("formal B-2 shard attempt already exists")
    source_map = {row["canonical_identity"]: row for row in sources}
    rows = [source_map[member["canonical_identity"]] for member in shard["records"]]
    if semantic_sha256(
        [
            {
                "ordinal": member["ordinal"],
                "canonical_identity": member["canonical_identity"],
                "source_record_sha256": member["source_record_sha256"],
            }
            for member in shard["records"]
        ]
    ) != shard["membership_sha256"]:
        raise B2Error("formal B-2 shard membership differs from P4-B")
    records, facts, runtime = _acquire_rows(
        rows=rows,
        source_manifest=source_manifest,
        closures=closures,
        card=card,
        expected_commit=expected_commit,
        mode="formal_original",
    )
    output.mkdir(parents=True, exist_ok=False)
    records_hash = write_new_jsonl(output / "records.jsonl", records)
    receipt = {
        "schema_version": "loopscope.phase4.b2-shard-receipt.v1",
        "status": "COMPLETED",
        "shard_id": shard_id,
        "membership_sha256": shard["membership_sha256"],
        "record_count": len(records),
        "record_file_sha256": records_hash,
        "ordered_record_sha256": semantic_sha256(records),
        "completed_at_utc": _utc_now(),
        "git_commit": expected_commit,
        "target_generation_count": 0,
        "runtime": runtime,
        "argv": list(argv),
    }
    scan_forbidden_fields(receipt, allow_options=False)
    receipt["manifest_sha256"] = semantic_sha256(receipt)
    write_new_json(output / "receipt.json", receipt)
    return {"status": "COMPLETED", "shard_id": shard_id, "record_count": len(records)}


def merge_formal(
    *, card_path: Path, run_root: Path, expected_commit: str, argv: Sequence[str]
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = _assert_run_root(run_root, must_exist=True)
    _git_provenance(expected_commit)
    card = _card(card_path)
    output = root / "formal/phase4_b2_hidden_geometry_trajectories.jsonl"
    manifest_path = root / "formal/phase4_b2_hidden_geometry_trajectory_manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("formal B-2 merge already exists")
    sources, source_manifest, _ = _load_sources(root)
    shards = _derived_shard_manifest(root)
    by_identity: Dict[str, Dict[str, Any]] = {}
    shard_evidence = []
    for shard in shards["shards"]:
        directory = root / shard["output_relative"]
        receipt = load_strict_json(directory / "receipt.json")
        records = load_strict_jsonl(directory / "records.jsonl")
        if (
            receipt.get("status") != "COMPLETED"
            or receipt.get("membership_sha256") != shard["membership_sha256"]
            or receipt.get("record_count") != shard["record_count"]
            or file_sha256(directory / "records.jsonl") != receipt.get("record_file_sha256")
        ):
            raise B2Error("completed B-2 shard evidence differs")
        for record, member in zip(records, shard["records"]):
            validate_hidden_geometry_record(
                record,
                d36_tolerance=card["numeric_contract"]["d36_abs_tolerance"],
            )
            identity = record["canonical_identity"]
            if identity != member["canonical_identity"] or identity in by_identity:
                raise B2Error("formal B-2 shard identity/order/uniqueness differs")
            by_identity[identity] = record
        shard_evidence.append(
            {
                "shard_id": shard["shard_id"],
                "record_count": len(records),
                "record_file_sha256": receipt["record_file_sha256"],
                "receipt_file_sha256": file_sha256(directory / "receipt.json"),
                "slurm_job_id": receipt["runtime"].get("slurm_job_id"),
                "device_name": receipt["runtime"].get("device_name"),
            }
        )
    expected = [source["canonical_identity"] for source in sources]
    if set(by_identity) != set(expected) or len(by_identity) != 12032:
        raise B2Error("formal B-2 merge membership is incomplete")
    ordered = [by_identity[identity] for identity in expected]
    output_hash = write_new_jsonl(output, ordered)
    merged = {
        "schema_version": "loopscope.phase4.b2-merged-trajectory-manifest.v1",
        "status": "COMPLETE",
        "gate": "B-2",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "run_root": str(root),
        "git_commit": expected_commit,
        "record_count": len(ordered),
        "shard_count": 24,
        "ordered_identity_sha256": semantic_sha256(expected),
        "ordered_record_sha256": semantic_sha256(ordered),
        "record_file": str(output.relative_to(root)),
        "record_file_sha256": output_hash,
        "source_manifest_file_sha256": EXPECTED_P4B_FILES[
            "source/phase4_shared12032_source_manifest.json"
        ],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "shard_manifest_sha256": shards["manifest_sha256"],
        "shards": shard_evidence,
        "target_generation_count": 0,
        "argv": list(argv),
    }
    merged["manifest_sha256"] = semantic_sha256(merged)
    write_new_json(manifest_path, merged)
    return {"status": "COMPLETE", "record_count": len(ordered), "record_file_sha256": output_hash}


def analyze_formal(
    *, card_path: Path, run_root: Path, expected_commit: str, argv: Sequence[str]
) -> Dict[str, Any]:
    _assert_offline_environment()
    root = _assert_run_root(run_root, must_exist=True)
    git = _git_provenance(expected_commit)
    _card(card_path)
    analysis_path = root / "analysis/phase4_b2_hidden_geometry_analysis.json"
    receipt_path = root / "phase4_b2_gate_receipt.json"
    if analysis_path.exists() or receipt_path.exists():
        raise FileExistsError("B-2 analyzer/receipt is write-once and already started")
    merged = load_strict_json(root / "formal/phase4_b2_hidden_geometry_trajectory_manifest.json")
    record_path = root / str(merged["record_file"])
    if file_sha256(record_path) != merged.get("record_file_sha256") or merged.get("record_count") != 12032:
        raise B2Error("B-2 merged trajectory closure differs")
    records = load_strict_jsonl(record_path)
    report = analyze_hidden_geometry_records(records)
    identities = report.pop("ordered_identities")
    report.update(
        {
            "created_at_utc": _utc_now(),
            "gate": "B-2",
            "executor_thread_id": EXECUTOR_THREAD_ID,
            "planning_thread_id": PLANNING_THREAD_ID,
            "git_commit": expected_commit,
            "card_sha256": CARD_BYTE_SHA256,
            "contract_sha256": CONTRACT_BYTE_SHA256,
            "control_sha256": _contract()["control_sha256"],
            "input_record_file_sha256": merged["record_file_sha256"],
            "ordered_identity_sha256": semantic_sha256(identities),
            "analyzer_invocation_count": 1,
            "argv": list(argv),
        }
    )
    report["manifest_sha256"] = semantic_sha256(report)
    write_new_json(analysis_path, report)
    boundary_hash = _write_csv_new(
        root / "analysis/phase4_b2_boundary_means.csv",
        report["boundary_means_and_deltas"],
    )
    width4_hash = _write_csv_new(
        root / "analysis/phase4_b2_width4_deltas.csv",
        report["width4_exit_minus_entry"],
    )
    widths_hash = _write_csv_new(
        root / "analysis/phase4_b2_width2_8_deltas.csv",
        report["width2_8_exit_minus_entry"],
    )
    correlation = {
        tuple(row["metrics"]): row for row in report["mean_curve_correlations"]
    }
    interpretation = (
        "# LoopScope Phase 4 Gate B-2 hidden-geometry interpretation\n\n"
        "This artifact is outcome-blind and diagnostic only. It evaluates internal geometry/output-geometry agreement and makes no claim about loop gain prediction.\n\n"
        "- KL vs final-normalized RMS-L2: Pearson %.6f; Spearman %.6f.\n"
        "- KL vs cosine distance: Pearson %.6f; Spearman %.6f.\n"
        "- RMS-L2 vs cosine distance: Pearson %.6f; Spearman %.6f.\n"
        "- Exit-minus-entry additivity maximum absolute error: %.3e.\n"
        "- Delta cosine-distance plus delta raw-cosine maximum absolute error: %.3e.\n"
        "\nThese associations describe whether the native no-loop hidden trajectory approaches its final boundary in ways aligned with output-distribution KL. They do not evaluate answers, correctness, generated text, P4-C outcomes, or any selector/panel change.\n"
        % (
            correlation[("kl_to_final", L2_KEY)]["pearson"],
            correlation[("kl_to_final", L2_KEY)]["spearman"],
            correlation[("kl_to_final", "hidden_cosine_distance_to_final")]["pearson"],
            correlation[("kl_to_final", "hidden_cosine_distance_to_final")]["spearman"],
            correlation[(L2_KEY, "hidden_cosine_distance_to_final")]["pearson"],
            correlation[(L2_KEY, "hidden_cosine_distance_to_final")]["spearman"],
            report["checks"]["maximum_additivity_abs_error"],
            report["checks"]["maximum_delta_cosine_distance_plus_delta_cosine_abs_error"],
        )
    )
    interpretation_hash = _write_text_new(
        root / "analysis/phase4_b2_interpretation.md", interpretation
    )
    smoke = load_strict_json(root / "smoke/phase4_b2_smoke_report.json")
    artifact_hashes = {
        "analysis_json": file_sha256(analysis_path),
        "boundary_means_csv": boundary_hash,
        "width4_deltas_csv": width4_hash,
        "width2_8_deltas_csv": widths_hash,
        "interpretation_md": interpretation_hash,
        "merged_trajectory_jsonl": merged["record_file_sha256"],
        "merged_manifest_json": file_sha256(
            root / "formal/phase4_b2_hidden_geometry_trajectory_manifest.json"
        ),
        "smoke_report_json": file_sha256(root / "smoke/phase4_b2_smoke_report.json"),
    }
    jobs = sorted(
        {
            str(row.get("slurm_job_id"))
            for row in merged["shards"]
            if row.get("slurm_job_id")
        }
    )
    devices = sorted(
        {str(row.get("device_name")) for row in merged["shards"] if row.get("device_name")}
    )
    nodes = sorted(
        {
            str(row.get("slurm_node"))
            for row in (item["runtime"] for item in (
                load_strict_json(root / ("formal/shards/shard-%04d/receipt.json" % shard_id))
                for shard_id in range(24)
            ))
            if row.get("slurm_node")
        }
    )
    launchers = {}
    for stage in ("smoke", "formal"):
        launcher_path = root / ("slurm/b2-%s-launcher.json" % stage)
        launcher = load_strict_json(launcher_path)
        launchers[stage] = {
            "file_sha256": file_sha256(launcher_path),
            "partition": launcher["partition"],
            "qos": launcher["qos"],
            "account": launcher["account"],
            "gres": launcher["gres"],
            "time_limit": launcher["time_limit"],
            "cpus_per_task": launcher["cpus_per_task"],
            "memory": launcher["memory"],
            "array_throttle": launcher["array_throttle"],
            "script_sha256": launcher["script_sha256"],
        }
    smoke_pass_receipts = [
        load_strict_json(root / ("smoke/pass-%d/receipt.json" % pass_id))
        for pass_id in (1, 2)
    ]
    jobs = sorted(
        set(jobs)
        | {
            str(item["runtime"].get("slurm_job_id"))
            for item in smoke_pass_receipts
            if item["runtime"].get("slurm_job_id")
        }
    )
    receipt = {
        "schema_version": "loopscope.phase4.b2-gate-receipt.v1",
        "status": "READY_FOR_PLANNING_AUDIT",
        "created_at_utc": _utc_now(),
        "gate": "B-2",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "git": git,
        "run_root": str(root),
        "card_sha256": CARD_BYTE_SHA256,
        "contract_sha256": CONTRACT_BYTE_SHA256,
        "control_sha256": _contract()["control_sha256"],
        "upstream_p4b_file_hashes": dict(EXPECTED_P4B_FILES),
        "source_manifest_sha256": merged["source_manifest_sha256"],
        "shard_manifest_sha256": merged["shard_manifest_sha256"],
        "smoke": {
            "status": smoke["status"],
            "record_count_per_pass": smoke["record_count_per_pass"],
            "maximum_geometry_abs_difference": smoke["maximum_geometry_abs_difference"],
            "target_generation_count": smoke["target_generation_count"],
        },
        "formal": {
            "record_count": merged["record_count"],
            "shard_count": merged["shard_count"],
            "ordered_identity_sha256": merged["ordered_identity_sha256"],
            "target_generation_count": merged["target_generation_count"],
        },
        "jobs": jobs,
        "devices": devices,
        "nodes": nodes,
        "resources": launchers,
        "command_evidence": {
            "admission_argv_sha256": semantic_sha256(
                load_strict_json(root / "phase4_b2_admission.json")["argv"]
            ),
            "smoke_pass_argv_sha256": [semantic_sha256(item["argv"]) for item in smoke_pass_receipts],
            "smoke_verify_argv_sha256": semantic_sha256(smoke["argv"]),
            "formal_shard_argv_sha256": sorted(
                {
                    semantic_sha256(
                        load_strict_json(
                            root / ("formal/shards/shard-%04d/receipt.json" % shard_id)
                        )["argv"]
                    )
                    for shard_id in range(24)
                }
            ),
            "merge_argv_sha256": semantic_sha256(merged["argv"]),
            "analyze_argv_sha256": semantic_sha256(list(argv)),
        },
        "artifact_hashes": artifact_hashes,
        "analysis_scope": "diagnostic_only_no_outcome_no_gain_claim",
        "selector_or_panel_changed": False,
        "p4c_accessed": False,
        "argv": list(argv),
    }
    receipt["manifest_sha256"] = semantic_sha256(receipt)
    write_new_json(receipt_path, receipt)
    return {
        "status": "READY_FOR_PLANNING_AUDIT",
        "record_count": 12032,
        "receipt_sha256": receipt["manifest_sha256"],
    }


def write_launcher(
    *,
    stage: str,
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
    root = _assert_run_root(run_root, must_exist=True)
    _git_provenance(expected_commit)
    _card(card_path)
    if stage not in {"smoke", "formal"}:
        raise B2Error("B-2 launcher stage is invalid")
    if any("\n" in str(value) or "\r" in str(value) for value in (partition, qos, account, gres, time_limit, memory, rationale)):
        raise B2Error("B-2 launcher resource field contains a newline")
    if cpus < 1 or not rationale.strip():
        raise B2Error("B-2 launcher CPU/rationale field is invalid")
    script_path = root / ("slurm/b2-%s.sbatch" % stage)
    if script_path.exists():
        raise FileExistsError("B-2 launcher already frozen")
    directives = [
        "#SBATCH --job-name=b2-%s" % stage,
        "#SBATCH --partition=%s" % partition,
        "#SBATCH --gres=%s" % gres,
        "#SBATCH --time=%s" % time_limit,
        "#SBATCH --cpus-per-task=%d" % cpus,
        "#SBATCH --mem=%s" % memory,
        "#SBATCH --output=%s/slurm/b2-%s-%%A_%%a.out" % (root, stage),
        "#SBATCH --error=%s/slurm/b2-%s-%%A_%%a.err" % (root, stage),
    ]
    if qos:
        directives.append("#SBATCH --qos=%s" % qos)
    if account:
        directives.append("#SBATCH --account=%s" % account)
    if stage == "formal":
        manifest = _derived_shard_manifest(root)
        throttle = manifest["shard_count"] if array_throttle is None else array_throttle
        if throttle < 1 or throttle > manifest["shard_count"]:
            raise B2Error("B-2 formal array throttle is invalid")
        directives.append("#SBATCH --array=0-23%%%d" % throttle)
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
    python = shlex.quote(str(AUDITED_VENV / "bin/python"))
    cli = shlex.quote(str(REMOTE_REPO / "scripts/loopscope/run_qwen4_phase4_b2.py"))
    card_remote = shlex.quote(str(REMOTE_REPO / "configs/loopscope/phase4_pv_ek_trs_card.json"))
    common_args = "--card %s --run-root %s --expected-commit %s" % (
        card_remote,
        shlex.quote(str(root)),
        shlex.quote(expected_commit),
    )
    if stage == "smoke":
        commands = [
            "%s %s acquire-smoke --pass-id 1 %s" % (python, cli, common_args),
            "%s %s acquire-smoke --pass-id 2 %s" % (python, cli, common_args),
            "%s %s verify-smoke %s" % (python, cli, common_args),
        ]
    else:
        commands = [
            "%s %s acquire-shard --shard-id \"${SLURM_ARRAY_TASK_ID}\" %s"
            % (python, cli, common_args)
        ]
    script_hash = _write_text_new(script_path, "\n".join(common + commands + [""]))
    config = {
        "schema_version": "loopscope.phase4.b2-launcher.v1",
        "stage": stage,
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
        "script_sha256": script_hash,
        "git_commit": expected_commit,
    }
    config["manifest_sha256"] = semantic_sha256(config)
    write_new_json(root / ("slurm/b2-%s-launcher.json" % stage), config)
    return config


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outcome-blind LoopScope Phase 4 Gate B-2 hidden geometry only"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-run", "verify-smoke", "merge", "analyze"):
        child = sub.add_parser(name)
        _common(child)
    smoke = sub.add_parser("acquire-smoke")
    _common(smoke)
    smoke.add_argument("--pass-id", required=True, type=int, choices=(1, 2))
    shard = sub.add_parser("acquire-shard")
    _common(shard)
    shard.add_argument("--shard-id", required=True, type=int)
    launcher = sub.add_parser("write-launcher")
    _common(launcher)
    launcher.add_argument("--stage", required=True, choices=("smoke", "formal"))
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
    provenance_argv = [str(Path(__file__).resolve())] + (
        list(sys.argv[1:]) if argv is None else list(argv)
    )
    common = {
        "card_path": args.card,
        "run_root": args.run_root,
        "expected_commit": args.expected_commit,
    }
    if args.command == "prepare-run":
        result = prepare_run(argv=provenance_argv, **common)
    elif args.command == "acquire-smoke":
        result = acquire_smoke(pass_id=args.pass_id, argv=provenance_argv, **common)
    elif args.command == "verify-smoke":
        result = verify_smoke(argv=provenance_argv, **common)
    elif args.command == "acquire-shard":
        result = acquire_shard(shard_id=args.shard_id, argv=provenance_argv, **common)
    elif args.command == "merge":
        result = merge_formal(argv=provenance_argv, **common)
    elif args.command == "analyze":
        result = analyze_formal(argv=provenance_argv, **common)
    elif args.command == "write-launcher":
        result = write_launcher(
            stage=args.stage,
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
