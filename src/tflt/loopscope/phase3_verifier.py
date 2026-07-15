"""Independent deterministic verifier for the byte-frozen Phase 3 contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tflt.loopscope.schema import (
    SchemaError,
    attach_manifest_sha256,
    canonical_json_bytes,
    verify_manifest_sha256,
    write_new_json,
)
from tflt.loopscope.phase3_analysis import (
    analyze_window_signals,
    trajectory_record_from_logits,
    verify_analysis_report,
)
from tflt.loopscope.phase3_pool import (
    make_pool_manifest,
    make_source_manifest,
    validate_source_contract,
)
from tflt.loopscope.phase3_schema import (
    BOUNDARY_IDS,
    PHASE3_CARD_BYTE_SHA256,
    PHASE3_CARD_SEMANTIC_SHA256,
    PHASE3_VERIFIER_RECEIPT_SCHEMA_VERSION,
    canonical_json_bytes as phase3_canonical_json_bytes,
    file_sha256,
    load_phase3_card,
    pool_record_json_schema,
    sanitized_content_sha256,
    trajectory_record_json_schema,
    validate_trajectory_record,
)


SOURCE_CONTRACT_RELATIVE = "configs/loopscope/phase3_source_manifest.json"
POOL_SCHEMA_RELATIVE = "configs/loopscope/phase3_pool_schema.json"
TRAJECTORY_SCHEMA_RELATIVE = "configs/loopscope/phase3_trajectory_schema.json"
IMPLEMENTATION_RELATIVES = (
    "src/tflt/loopscope/phase3_schema.py",
    "src/tflt/loopscope/phase3_pool.py",
    "src/tflt/loopscope/phase3_analysis.py",
    "src/tflt/loopscope/phase3_verifier.py",
    "scripts/loopscope/prepare_qwen17_phase3.py",
)
SYNTHETIC_FIXTURE_VERSION = "loopscope.phase3.p3a-deterministic-fixture.v1"


def build_verifier_receipt(card_path: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Recompute every stable P3-A receipt field from source artifacts."""

    card_path = Path(card_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else card_path.parents[2]
    card = load_phase3_card(card_path)
    source_contract_path = root / SOURCE_CONTRACT_RELATIVE
    pool_schema_path = root / POOL_SCHEMA_RELATIVE
    trajectory_schema_path = root / TRAJECTORY_SCHEMA_RELATIVE

    source_contract = _load_json(source_contract_path)
    validate_source_contract(source_contract, card)
    pool_schema = _load_json(pool_schema_path)
    trajectory_schema = _load_json(trajectory_schema_path)
    if pool_schema != pool_record_json_schema():
        raise SchemaError("committed Phase 3 pool JSON Schema differs from production code")
    if trajectory_schema != trajectory_record_json_schema():
        raise SchemaError("committed Phase 3 trajectory JSON Schema differs from production code")

    pool_records = synthetic_pool_records(card)
    source_manifest = make_source_manifest(pool_records, card)
    pool_manifest = make_pool_manifest(pool_records, source_manifest, card)
    raw_trajectory = synthetic_raw_trajectory(
        pool_records[0], source_manifest, pool_manifest, card
    )
    trajectory_hash = hashlib.sha256(canonical_json_bytes(raw_trajectory)).hexdigest()
    boundary_tamper = copy.deepcopy(raw_trajectory)
    boundary_tamper["boundaries"][12]["choice_entropy"] += 0.01
    boundary_tamper_rejected = _rejects(
        lambda: validate_trajectory_record(boundary_tamper, card)
    )
    if not boundary_tamper_rejected:
        raise SchemaError("decision-critical boundary metric tamper was not rejected")

    analysis_input = synthetic_signal_records()
    analysis_report = analyze_window_signals(
        analysis_input,
        card,
        replicates=31,
        seed=20260716,
        enforce_population=False,
    )
    verify_analysis_report(
        analysis_input,
        analysis_report,
        card,
        replicates=31,
        seed=20260716,
        enforce_population=False,
    )
    analysis_tamper = copy.deepcopy(analysis_report)
    analysis_tamper["variants"]["SHIFT"]["windows"][0]["score"] += 1.0
    analysis_tamper_rejected = _rejects(
        lambda: verify_analysis_report(
            analysis_input,
            analysis_tamper,
            card,
            replicates=31,
            seed=20260716,
            enforce_population=False,
        )
    )
    if not analysis_tamper_rejected:
        raise SchemaError("decision-critical analysis score tamper was not rejected")

    implementations = {}
    for relative in IMPLEMENTATION_RELATIVES:
        path = root / relative
        if not path.is_file():
            raise SchemaError("Phase 3 implementation file is missing: %s" % relative)
        implementations[relative] = file_sha256(path)

    payload: Dict[str, Any] = {
        "schema_version": PHASE3_VERIFIER_RECEIPT_SCHEMA_VERSION,
        "artifact_role": "p3a_card_and_producer_contract_verifier_receipt",
        "card": {
            "relative_path": "configs/loopscope/phase3_card.json",
            "byte_sha256": PHASE3_CARD_BYTE_SHA256,
            "semantic_sha256": PHASE3_CARD_SEMANTIC_SHA256,
        },
        "contract_artifacts": {
            SOURCE_CONTRACT_RELATIVE: {
                "file_sha256": file_sha256(source_contract_path),
                "manifest_sha256": source_contract["manifest_sha256"],
            },
            POOL_SCHEMA_RELATIVE: {"file_sha256": file_sha256(pool_schema_path)},
            TRAJECTORY_SCHEMA_RELATIVE: {
                "file_sha256": file_sha256(trajectory_schema_path)
            },
        },
        "implementation_sha256": implementations,
        "synthetic_fixture": {
            "version": SYNTHETIC_FIXTURE_VERSION,
            "contains_real_data": False,
            "record_count": len(pool_records),
            "subject_count": len({record["subject"] for record in pool_records}),
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "pool_manifest_sha256": pool_manifest["manifest_sha256"],
            "raw_trajectory_sha256": trajectory_hash,
            "raw_boundary_recomputation": True,
            "boundary_metric_tamper_rejected": boundary_tamper_rejected,
            "analysis_input_sha256": hashlib.sha256(
                canonical_json_bytes(analysis_input)
            ).hexdigest(),
            "analysis_report_sha256": hashlib.sha256(
                canonical_json_bytes(analysis_report)
            ).hexdigest(),
            "analysis_fixture_replicates": 31,
            "analysis_fixture_seed": 20260716,
            "analysis_score_tamper_rejected": analysis_tamper_rejected,
            "analysis_fixture_only_not_scientific_output": True,
        },
        "checks": {
            "card_byte_frozen": True,
            "source_contract_closed_world": True,
            "synthetic_1531_57_source_pool_closure": True,
            "raw_logits_removed_from_selector_artifact": True,
            "b0_through_b28_recomputed": True,
            "h_kl_ce_identity_recomputed": True,
            "joint_bootstrap_analysis_recomputed": True,
            "decision_critical_tampers_rejected": True,
            "torch_transformers_lm_eval_imports_required": False,
        },
        "external_actions_performed": [],
    }
    attach_manifest_sha256(payload)
    return payload


def verify_committed_receipt(card_path: Path, receipt_path: Path) -> Dict[str, Any]:
    observed = _load_json(Path(receipt_path))
    verify_manifest_sha256(observed)
    expected = build_verifier_receipt(Path(card_path))
    if phase3_canonical_json_bytes(observed) != phase3_canonical_json_bytes(expected):
        raise SchemaError("committed Phase 3 verifier receipt differs from recomputation")
    return dict(observed)


def write_new_verifier_receipt(card_path: Path, receipt_path: Path) -> Dict[str, Any]:
    receipt = build_verifier_receipt(Path(card_path))
    write_new_json(Path(receipt_path), receipt)
    return receipt


def synthetic_pool_records(card: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return a deterministic 1,531/57 synthetic source population."""

    records: List[Dict[str, Any]] = []
    for subject_index in range(57):
        subject = "subject_%02d" % subject_index
        task = "mmlu_%s" % subject
        count = 27 if subject_index < 49 else 26
        for doc_index in range(count):
            doc_id = "%04d" % doc_index
            doc_hash = _digest("doc\0%s\0%s" % (subject, doc_id))
            question = "Synthetic question %s %s?" % (subject, doc_id)
            choices = [
                "Synthetic choice %s %s %s" % (label, subject, doc_id)
                for label in ("A", "B", "C", "D")
            ]
            rendered_prompt = "Synthetic five-shot prompt\n%s\n%s" % (
                question,
                "\n".join("%s. %s" % (label, choice) for label, choice in zip("ABCD", choices)),
            )
            prompt_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
            records.append(
                {
                    "schema_version": "loopscope.phase3.pool-record.v1",
                    "identity": {"task": task, "doc_id": doc_id, "doc_hash": doc_hash},
                    "subject": subject,
                    "split": "validation",
                    "question": question,
                    "ordered_choices": choices,
                    "rendered_prompt": rendered_prompt,
                    "prompt_sha256": prompt_hash,
                    "sanitized_content_sha256": sanitized_content_sha256(question, choices),
                    "renderer_provenance": {
                        "dataset_repo": card["task"]["dataset"],
                        "dataset_revision": card["task"]["dataset_revision"],
                        "task_group": card["task"]["task_group"],
                        "fewshot_split": card["task"]["fewshot_split"],
                        "num_fewshot": card["task"]["num_fewshot"],
                        "renderer_id": "p3a_synthetic_no_live_renderer",
                        "renderer_source_sha256": _digest("synthetic-renderer-source"),
                        "render_contract_sha256": _digest("synthetic-render-contract"),
                        "source_projection_sha256": _digest("synthetic-source-projection"),
                        "render_sha256": prompt_hash,
                    },
                }
            )
    if len(records) != 1531 or len({record["subject"] for record in records}) != 57:
        raise AssertionError("internal Phase 3 synthetic population shape drift")
    return records


def synthetic_raw_trajectory(
    pool_record: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    pool_manifest: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    logits = {}
    for index, boundary_id in enumerate(BOUNDARY_IDS):
        logits[boundary_id] = [
            0.015 * index,
            -0.2 + 0.004 * index,
            -0.5 - 0.003 * index,
            -0.8 + 0.001 * index,
        ]
    renderer = {
        "dataset_repo": card["task"]["dataset"],
        "dataset_revision": card["task"]["dataset_revision"],
        "model_repo": card["model"]["repo"],
        "model_revision": card["model"]["revision"],
        "tokenizer_revision": card["model"]["revision"],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "pool_manifest_sha256": pool_manifest["manifest_sha256"],
        "renderer_manifest_sha256": _digest("synthetic-renderer-manifest"),
        "forward_type": card["trajectory"]["forward_type"],
        "formal_forward_count_per_identity": 1,
        "loop_insertions": 0,
        "answer_position": card["trajectory"]["answer_position"],
        "projection": card["trajectory"]["projection"],
        "choice_order": list(card["trajectory"]["choice_order"]),
    }
    metadata = {
        "identity": pool_record["identity"],
        "subject": pool_record["subject"],
        "split": pool_record["split"],
        "prompt_sha256": pool_record["prompt_sha256"],
    }
    return trajectory_record_from_logits(metadata, logits, renderer, card)


def synthetic_signal_records() -> List[Dict[str, Any]]:
    records = []
    for index in range(8):
        subject = "subject_%02d" % (index // 4)
        entropy = [-0.004] * 25
        divergence = [-0.003] * 25
        jitter = (index - 3.5) * 0.0002
        entropy[12] = 0.030 + jitter
        divergence[12] = 0.025 + 0.5 * jitter
        for comparison in (8, 11, 13, 16):
            entropy[comparison] = -0.020 + 0.1 * jitter
            divergence[comparison] = -0.018 + 0.1 * jitter
        windows = [
            {
                "start": start,
                "window": "%d:%d" % (start, start + 3),
                "entropy_drop": entropy[start],
                "kl_rise": divergence[start],
                "ce_drop": entropy[start] - divergence[start],
            }
            for start in range(25)
        ]
        records.append(
            {
                "identity": {
                    "task": "mmlu_%s" % subject,
                    "doc_id": "%02d" % index,
                    "doc_hash": _digest("analysis-signal-%d" % index),
                },
                "subject": subject,
                "windows": windows,
            }
        )
    return records


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_bad_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SchemaError("cannot load strict JSON artifact: %s" % path) from exc
    if not isinstance(payload, dict):
        raise SchemaError("JSON artifact must contain an object: %s" % path)
    return payload


def _rejects(callback: Any) -> bool:
    try:
        callback()
    except (SchemaError, ValueError, TypeError):
        return True
    return False


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bad_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant %s" % value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute and verify the committed LoopScope Phase 3 P3-A receipt."
    )
    parser.add_argument("--card", required=True)
    parser.add_argument("--receipt", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = verify_committed_receipt(Path(args.card), Path(args.receipt))
    print(receipt["manifest_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_verifier_receipt",
    "synthetic_pool_records",
    "synthetic_raw_trajectory",
    "synthetic_signal_records",
    "verify_committed_receipt",
    "write_new_verifier_receipt",
]
