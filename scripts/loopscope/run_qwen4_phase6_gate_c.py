#!/usr/bin/env python3
"""Run and independently verify the four-identity LoopScope Phase 6 Gate C smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tflt.loopscope.phase4_runtime import (  # noqa: E402
    project_target_row,
    project_test_dataset,
    render_exact_prefix,
    validation_demos_by_category,
)
from tflt.loopscope.phase4_schema import (  # noqa: E402
    canonical_identity,
    file_sha256,
    sanitized_content_sha256,
)
from tflt.loopscope.phase6_runtime import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    GENERATION_MAX_NEW_TOKENS,
    GENERATION_STOP_STRING,
    MODEL_REPO,
    MODEL_REVISION,
    PRODUCER_VERSION,
    acquire_two_pass_record,
    canonical_json_bytes,
    load_gate_c_runtime,
    semantic_sha256,
)
from tflt.loopscope.phase6_schema import (  # noqa: E402
    load_json,
    scan_forbidden_fields,
    validate_card,
    validate_trajectory_record,
)


EXECUTOR_THREAD_ID = "019fb452-e2af-7bf2-a411-612c81971245"
PLANNING_THREAD_ID = "019fb3de-2298-75f2-a083-0dca453ea79c"
CARD_PATH = REPO_ROOT / "configs/loopscope/phase6_pre_answer_v2_card.json"
CARD_SHA256 = "feef7250b72fa8ecd154d58835ad42eacc1fc96509ed984acdafbfa4c20432d4"
GATE_B_MANIFEST_SHA256 = (
    "ccb148bd61971d24ad81b240cbe1b5f0810e06e620fb568473a61b2147b44a1d"
)
ORDERED_IDENTITY_SHA256 = (
    "ac52d6e43c693bd0b47b567c2955dae0c6be895ce3230e854d4e1538f05503a5"
)
ORDERED_SAFE_TOKEN_CLOSURE_SHA256 = (
    "0d7e9ef3139f687fd3f15d825e883e52d1f84a504cae134ad4b17ebd6ad82108"
)
REMOTE_REPO = Path(
    "/hpc2hdd/home/xhuang225/projects/training_free_looped_transformers_loopscope"
)
AUDITED_VENV = REMOTE_REPO / ".venv-loopscope-cu121-20260711"
HF_HOME = Path("/hpc2hdd/home/xhuang225/shared/hf_home")
HF_DATASETS_CACHE = Path("/hpc2hdd/home/xhuang225/shared/datasets")
RECORDS_NAME = "gate_c_sanitized_records.json"
DUPLICATE_NAME = "gate_c_duplicate_closure.json"
MANIFEST_NAME = "gate_c_manifest.json"
VERIFIER_NAME = "gate_c_verifier_receipt.json"


class GateCError(RuntimeError):
    """Gate C smoke or independent verification failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateCError(message)


def _run(command: Sequence[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        list(command), cwd=str(cwd), check=True, text=True, capture_output=True
    )
    return completed.stdout.strip()


def _write_new_json(path: Path, payload: Any) -> str:
    data = json.dumps(
        payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def _load_json_any(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            GateCError("non-finite JSON constant: %s" % value)
        ))


def _git_provenance(expected_commit: str) -> Dict[str, Any]:
    root = Path(_run(("git", "rev-parse", "--show-toplevel"), cwd=REPO_ROOT))
    branch = _run(("git", "symbolic-ref", "--short", "HEAD"), cwd=REPO_ROOT)
    commit = _run(("git", "rev-parse", "HEAD"), cwd=REPO_ROOT)
    origin = _run(("git", "rev-parse", "origin/loopscope"), cwd=REPO_ROOT)
    dirty = _run(("git", "status", "--porcelain"), cwd=REPO_ROOT)
    _require(root == REPO_ROOT, "not running in the dedicated LoopScope clone")
    _require(branch == "loopscope", "remote branch differs from loopscope")
    _require(commit == expected_commit, "remote commit differs from expected commit")
    _require(origin == expected_commit, "remote origin/loopscope differs")
    _require(dirty == "", "remote dedicated clone is dirty")
    return {
        "repo": str(root),
        "branch": branch,
        "commit": commit,
        "origin_loopscope": origin,
        "dirty": False,
    }


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
        _require(os.environ.get(key) == expected, "offline environment differs: %s" % key)
    _require(not os.environ.get("HF_ENDPOINT"), "HF_ENDPOINT substitution is forbidden")


def _load_card() -> Dict[str, Any]:
    _require(file_sha256(CARD_PATH) == CARD_SHA256, "frozen Phase 6 card hash differs")
    card = load_json(CARD_PATH)
    validate_card(card)
    return card


def _load_gate_b_manifest(path: Path) -> Dict[str, Any]:
    _require(path.is_file(), "Gate B manifest is missing")
    _require(file_sha256(path) == GATE_B_MANIFEST_SHA256, "Gate B manifest hash differs")
    manifest = load_json(path)
    identity = manifest.get("identity")
    _require(isinstance(identity, Mapping), "Gate B identity closure is missing")
    _require(
        identity.get("ordered_identity_sha256") == ORDERED_IDENTITY_SHA256,
        "Gate B ordered identity hash differs",
    )
    _require(
        identity.get("ordered_safe_token_closure_sha256")
        == ORDERED_SAFE_TOKEN_CLOSURE_SHA256,
        "Gate B ordered safe token closure hash differs",
    )
    smoke = identity.get("smoke_identities")
    _require(isinstance(smoke, list) and len(smoke) == 4, "Gate B smoke membership differs")
    _require(len({row["canonical_identity"] for row in smoke}) == 4, "smoke identities duplicate")
    _require(
        not manifest.get("target_labels_accessed")
        and not manifest.get("outcomes_read")
        and not manifest.get("generation_or_replay_executed"),
        "Gate B information barrier/provenance differs",
    )
    return manifest


def _load_renderer() -> Any:
    from lm_eval.tasks.mmlu_pro import utils

    return utils


def _load_dataset(split: str) -> Any:
    from datasets import DownloadMode, load_dataset

    value = load_dataset(
        DATASET_REPO,
        revision=DATASET_REVISION,
        split=split,
        cache_dir=str(HF_DATASETS_CACHE),
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    _require(
        len(value) == (12032 if split == "test" else 70),
        "cached %s split count differs" % split,
    )
    return value


def _safe_smoke_prefixes(
    gate_b_manifest: Mapping[str, Any], tokenizer: Any
) -> list[tuple[Mapping[str, Any], str]]:
    renderer = _load_renderer()
    validation = _load_dataset("validation")
    demos = validation_demos_by_category(validation)
    projected = project_test_dataset(_load_dataset("test"))
    result = []
    for frozen in gate_b_manifest["identity"]["smoke_identities"]:
        ordinal = int(frozen["ordinal"])
        target = project_target_row(projected[ordinal])
        _require(
            canonical_identity(target) == frozen["canonical_identity"],
            "live safe identity differs at ordinal %d" % ordinal,
        )
        _require(
            sanitized_content_sha256(target["question"], target["ordered_options"])
            == frozen["safe_content_sha256"],
            "live safe content hash differs at ordinal %d" % ordinal,
        )
        prefix = render_exact_prefix(target, demos[target["category"]], renderer)
        encoded = tokenizer.encode(prefix)
        _require(
            isinstance(encoded, list) and len(encoded) == int(frozen["sequence_length"]),
            "live rendered sequence length differs at ordinal %d" % ordinal,
        )
        result.append((frozen, prefix))
    return result


def _resource_evidence(runtime: Any) -> Dict[str, Any]:
    torch = runtime.torch
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_qos": os.environ.get("SLURM_JOB_QOS"),
        "slurm_job_gres": os.environ.get("SLURM_JOB_GPUS"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_total_memory_bytes": int(properties.total_memory),
        "cpu_count": os.cpu_count(),
    }


def _duplicate_closure(
    records: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    by_identity: Dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for record_entry, evidence_entry in zip(records, evidence):
        identity = str(record_entry["record"]["canonical_identity"])
        by_identity.setdefault(identity, []).append((record_entry, evidence_entry))
    _require(len(by_identity) == 4, "duplicate closure does not contain four identities")
    rows = []
    compared_fields = (
        "prompt_sha256",
        "generated_completion_sha256",
        "generation_ids_sha256",
        "replay_ids_sha256",
        "generation_length",
        "replay_length",
        "anchor_token_index",
        "answer_span_start_offset",
        "answer_span_end_offset",
        "answer_match_count",
        "selected_match_ordinal",
        "answer_first_token_index",
        "record_semantic_sha256",
    )
    for identity, pair in sorted(by_identity.items()):
        _require(len(pair) == 2, "identity does not contain exactly two replicates")
        (record1, closure1), (record2, closure2) = pair
        _require(
            [record1["replicate"], record2["replicate"]] == [1, 2],
            "duplicate replicate indices differ",
        )
        mismatched = [
            key for key in compared_fields if closure1.get(key) != closure2.get(key)
        ]
        _require(not mismatched, "duplicate deterministic closure mismatch")
        _require(record1["record"] == record2["record"], "sanitized records differ")
        for closure in (closure1, closure2):
            _require(
                closure.get("generation_count") == 1
                and closure.get("replay_count") == 1
                and closure.get("loop_insertions") == 0,
                "runtime generation/replay/loop count closure differs",
            )
            _require(
                closure.get("anchor_resolved") is True
                and closure.get("unique_token_mapping") is True,
                "runtime anchor closure differs",
            )
            _require(
                int(closure.get("answer_match_count", 0)) >= 1
                and closure.get("selected_match_ordinal") == 0,
                "runtime first-match closure differs",
            )
            _require(
                closure.get("final_norm_pre_hook_count") == 1
                and closure.get("raw_boundary_count") == 37,
                "runtime raw boundary hook closure differs",
            )
            _require(
                closure.get("final_norm_postnorm_allclose") is True
                and closure.get("replay_next_token_closure") is True,
                "runtime replay/final-norm closure differs",
            )
        rows.append(
            {
                "canonical_identity": identity,
                "replicates": 2,
                "record_semantic_sha256": closure1["record_semantic_sha256"],
                "generation_replay_hashes_equal": True,
                "anchor_indices_equal": True,
                "answer_match_count": int(closure1["answer_match_count"]),
                "selected_match_ordinal": 0,
                "sanitized_scalars_equal": True,
                "replay_next_token_closure": True,
                "final_norm_pre_hook_count_each": 1,
                "raw_boundary_count_each": 37,
                "generation_count_each": 1,
                "replay_count_each": 1,
                "loop_insertions_each": 0,
                "max_final_norm_postnorm_abs": max(
                    float(closure1["final_norm_postnorm_max_abs"]),
                    float(closure2["final_norm_postnorm_max_abs"]),
                ),
            }
        )
    return {
        "schema_version": "loopscope.phase6.gate-c-duplicate-closure.v2",
        "identity_count": 4,
        "replicates_per_identity": 2,
        "deterministic_duplicate_pass": True,
        "rows": rows,
    }


def run_smoke(
    *,
    run_root: Path,
    gate_b_manifest_path: Path,
    expected_commit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    _assert_offline_environment()
    _require(run_root.is_dir(), "fresh Gate C run root must already exist")
    for name in (RECORDS_NAME, DUPLICATE_NAME, MANIFEST_NAME, VERIFIER_NAME):
        _require(not (run_root / name).exists(), "Gate C artifact already exists: %s" % name)
    git = _git_provenance(expected_commit)
    card = _load_card()
    gate_b = _load_gate_b_manifest(gate_b_manifest_path)
    runtime = load_gate_c_runtime()
    prefixes = _safe_smoke_prefixes(gate_b, runtime.tokenizer)
    record_entries = []
    evidence_entries = []
    for frozen, prefix in prefixes:
        for replicate in (1, 2):
            record, evidence = acquire_two_pass_record(
                runtime,
                prefix=prefix,
                identity=frozen,
                gate_b_manifest_sha256=GATE_B_MANIFEST_SHA256,
                card_sha256=CARD_SHA256,
            )
            validate_trajectory_record(record)
            scan_forbidden_fields(record)
            record_entries.append({"replicate": replicate, "record": record})
            evidence_entries.append({"replicate": replicate, **evidence})
            runtime.torch.cuda.empty_cache()
    duplicate = _duplicate_closure(record_entries, evidence_entries)
    records_payload = {
        "schema_version": "loopscope.phase6.gate-c-sanitized-smoke.v2",
        "identity_count": 4,
        "replicates_per_identity": 2,
        "records": record_entries,
    }
    records_sha = _write_new_json(run_root / RECORDS_NAME, records_payload)
    duplicate["records_file_sha256"] = records_sha
    duplicate_sha = _write_new_json(run_root / DUPLICATE_NAME, duplicate)
    manifest = {
        "schema_version": "loopscope.phase6.gate-c-manifest.v2",
        "gate": "C",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "created_at_utc": _utc_now(),
        "argv": list(argv),
        "run_root": str(run_root),
        "git": git,
        "card_sha256": CARD_SHA256,
        "gate_b_manifest_path": str(gate_b_manifest_path),
        "gate_b_manifest_sha256": GATE_B_MANIFEST_SHA256,
        "ordered_identity_sha256": ORDERED_IDENTITY_SHA256,
        "ordered_safe_token_closure_sha256": ORDERED_SAFE_TOKEN_CLOSURE_SHA256,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "producer_version": PRODUCER_VERSION,
        "generation_contract": {
            "mode": "greedy",
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
            "stop_string": GENERATION_STOP_STRING,
        },
        "membership": [
            {
                "ordinal": int(row["ordinal"]),
                "canonical_identity": row["canonical_identity"],
                "category": row["category"],
                "question_id": int(row["question_id"]),
                "safe_content_sha256": row["safe_content_sha256"],
                "rendered_prefix_sha256": row["rendered_prefix_sha256"],
                "rendered_token_ids_sha256": row["rendered_token_ids_sha256"],
                "sequence_length": int(row["sequence_length"]),
            }
            for row in gate_b["identity"]["smoke_identities"]
        ],
        "identity_count": 4,
        "replicates_per_identity": 2,
        "resource": _resource_evidence(runtime),
        "environment": {
            "python": platform.python_version(),
            "venv": str(AUDITED_VENV),
            "offline": True,
        },
        "runtime_paths": {
            "final_norm_path": runtime.final_norm_path,
            "lm_head_path": runtime.lm_head_path,
        },
        "artifacts": {
            RECORDS_NAME: records_sha,
            DUPLICATE_NAME: duplicate_sha,
        },
        "closure_summary": {
            "selected_anchor_count": 8,
            "unique_token_mapping_count": 8,
            "generation_count_each": 1,
            "replay_count_each": 1,
            "loop_insertions_each": 0,
            "raw_boundary_count_each": 37,
            "angular_transition_count_each": 36,
            "final_norm_pre_hook_count_each": 1,
            "replay_next_token_closure_count": 8,
            "duplicate_determinism": "PASS",
        },
        "information_barrier": {
            "protected_target_fields_accessed": False,
            "generated_payload_persisted": False,
            "raw_tensor_payload_persisted": False,
            "full_vocabulary_distribution_payload_persisted": False,
            "selector_executed": False,
            "later_gate_entered": False,
        },
        "status": "READY_FOR_FRESH_PROCESS_VERIFICATION",
    }
    manifest_sha = _write_new_json(run_root / MANIFEST_NAME, manifest)
    return {
        "status": "READY_FOR_FRESH_PROCESS_VERIFICATION",
        "manifest_sha256": manifest_sha,
        "records_sha256": records_sha,
        "duplicate_sha256": duplicate_sha,
    }


def verify_smoke(
    *,
    run_root: Path,
    gate_b_manifest_path: Path,
    expected_commit: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    _assert_offline_environment()
    git = _git_provenance(expected_commit)
    _load_card()
    gate_b = _load_gate_b_manifest(gate_b_manifest_path)
    manifest_path = run_root / MANIFEST_NAME
    records_path = run_root / RECORDS_NAME
    duplicate_path = run_root / DUPLICATE_NAME
    _require(manifest_path.is_file(), "Gate C manifest is missing")
    _require(records_path.is_file(), "Gate C sanitized records are missing")
    _require(duplicate_path.is_file(), "Gate C duplicate closure is missing")
    manifest = load_json(manifest_path)
    records_payload = _load_json_any(records_path)
    duplicate = load_json(duplicate_path)
    _require(manifest["git"] == git, "Gate C manifest Git provenance differs")
    _require(
        manifest["gate_b_manifest_sha256"] == GATE_B_MANIFEST_SHA256,
        "Gate C manifest Gate B binding differs",
    )
    expected_members = [
        row["canonical_identity"] for row in gate_b["identity"]["smoke_identities"]
    ]
    observed_members = [row["canonical_identity"] for row in manifest["membership"]]
    _require(observed_members == expected_members, "Gate C membership/order differs")
    _require(
        manifest["artifacts"][RECORDS_NAME] == file_sha256(records_path),
        "sanitized records byte hash differs",
    )
    _require(
        manifest["artifacts"][DUPLICATE_NAME] == file_sha256(duplicate_path),
        "duplicate closure byte hash differs",
    )
    entries = records_payload.get("records")
    _require(isinstance(entries, list) and len(entries) == 8, "record membership is not 4x2")
    counts: Dict[str, int] = {}
    record_hashes: Dict[str, str] = {}
    for entry in entries:
        _require(set(entry) == {"replicate", "record"}, "record wrapper differs")
        _require(entry["replicate"] in (1, 2), "replicate index differs")
        record = entry["record"]
        validate_trajectory_record(record)
        scan_forbidden_fields(record)
        identity = record["canonical_identity"]
        counts[identity] = counts.get(identity, 0) + 1
        _require(len(record["H"]) == 37 and len(record["D"]) == 37, "H/D length differs")
        _require(
            len(record["hidden_rms_l2_to_final"]) == 37
            and len(record["hidden_cosine_to_final"]) == 37
            and len(record["hidden_cosine_distance_to_final"]) == 37,
            "hidden diagnostic length differs",
        )
        _require(
            len(record["adjacent_angular_distance"]) == 36,
            "angular trajectory length differs",
        )
        _require(record["D"][-1] <= 1e-6, "D36 endpoint differs")
        _require(
            record["generation_count"] == 1
            and record["replay_count"] == 1
            and record["loop_insertions"] == 0,
            "generation/replay/loop counts differ",
        )
        _require(
            record["answer_match_count"] >= 1
            and record["selected_match_ordinal"] == 0,
            "first-match anchor metadata differs",
        )
        _require(
            all(
                left == 1.0 - right
                for left, right in zip(
                    record["hidden_cosine_distance_to_final"],
                    record["hidden_cosine_to_final"],
                )
            ),
            "cosine distance is not exactly derived as 1-cosine",
        )
        current_hash = semantic_sha256(record)
        if identity in record_hashes:
            _require(
                record_hashes[identity] == current_hash,
                "fresh-process duplicate record hash differs",
            )
        record_hashes[identity] = current_hash
    _require(
        counts == {identity: 2 for identity in expected_members},
        "exact duplicate identity membership differs",
    )
    _require(duplicate["deterministic_duplicate_pass"], "duplicate receipt is not PASS")
    duplicate_rows = {
        row["canonical_identity"]: row for row in duplicate.get("rows", [])
    }
    _require(set(duplicate_rows) == set(expected_members), "duplicate receipt membership differs")
    for identity in expected_members:
        row = duplicate_rows[identity]
        _require(
            row["record_semantic_sha256"] == record_hashes[identity]
            and row["replicates"] == 2
            and row["generation_replay_hashes_equal"] is True
            and row["anchor_indices_equal"] is True
            and row["answer_match_count"] >= 1
            and row["selected_match_ordinal"] == 0
            and row["sanitized_scalars_equal"] is True
            and row["replay_next_token_closure"] is True
            and row["final_norm_pre_hook_count_each"] == 1
            and row["raw_boundary_count_each"] == 37
            and row["generation_count_each"] == 1
            and row["replay_count_each"] == 1
            and row["loop_insertions_each"] == 0,
            "duplicate runtime closure receipt differs",
        )
        _require(
            math.isfinite(float(row["max_final_norm_postnorm_abs"])),
            "duplicate final-norm closure is non-finite",
        )
    barrier = manifest["information_barrier"]
    _require(not any(bool(value) for value in barrier.values()), "information barrier differs")
    receipt = {
        "schema_version": "loopscope.phase6.gate-c-verifier-receipt.v2",
        "gate": "C",
        "executor_thread_id": EXECUTOR_THREAD_ID,
        "planning_thread_id": PLANNING_THREAD_ID,
        "verified_at_utc": _utc_now(),
        "argv": list(argv),
        "run_root": str(run_root),
        "git": git,
        "manifest_file_sha256": file_sha256(manifest_path),
        "records_file_sha256": file_sha256(records_path),
        "duplicate_file_sha256": file_sha256(duplicate_path),
        "identity_count": 4,
        "replicates_per_identity": 2,
        "selected_anchor_count": 8,
        "replay_next_token_closure_count": 8,
        "raw_boundary_count_each": 37,
        "angular_transition_count_each": 36,
        "final_norm_pre_hook_count_each": 1,
        "forbidden_persisted_payloads": False,
        "status": "PASS",
    }
    receipt_sha = _write_new_json(run_root / VERIFIER_NAME, receipt)
    return {"status": "PASS", "verifier_receipt_sha256": receipt_sha}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run", help="validate the frozen local Gate C plan")
    for command in ("smoke", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--run-root", type=Path, required=True)
        subparser.add_argument("--gate-b-manifest", type=Path, required=True)
        subparser.add_argument("--expected-commit", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    if args.command == "dry-run":
        card = _load_card()
        print(
            json.dumps(
                {
                    "status": "GATE_C_DRY_RUN_PASS",
                    "card": card["card"],
                    "identity_count": 4,
                    "replicates_per_identity": 2,
                    "generation_count_each": 1,
                    "replay_count_each": 1,
                    "loop_insertions": 0,
                    "model_or_data_loaded": False,
                    "gpu_or_slurm_used": False,
                    "gate_d_entered": False,
                },
                sort_keys=True,
            )
        )
        return 0
    kwargs = {
        "run_root": args.run_root.resolve(),
        "gate_b_manifest_path": args.gate_b_manifest.resolve(),
        "expected_commit": args.expected_commit,
        "argv": [str(Path(__file__).relative_to(REPO_ROOT)), *arguments],
    }
    result = run_smoke(**kwargs) if args.command == "smoke" else verify_smoke(**kwargs)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
