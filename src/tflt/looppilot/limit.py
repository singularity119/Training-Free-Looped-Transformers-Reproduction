"""Strict, no-site Gate D configuration and doc-key join helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


GATE_D_TASKS: Tuple[str, ...] = (
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_astronomy",
    "mmlu_business_ethics",
)
GATE_D_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
GATE_D_DOC_INDICES: Tuple[int, ...] = (0, 1, 2, 3, 4)
GATE_D_CONTROLLER_INPUT_FIELDS: Tuple[str, ...] = (
    "r0_median",
    "c0",
    "n0_median",
)

_EXPECTED_CONFIG: Dict[str, Any] = {
    "schema_version": 1,
    "gate": "D",
    "model": "Qwen/Qwen3-1.7B-Base",
    "model_revision": GATE_D_REVISION,
    "tokenizer_revision": GATE_D_REVISION,
    "dtype": "float16",
    "task": "mmlu",
    "tasks": list(GATE_D_TASKS),
    "split": "test",
    "doc_indices": {task: list(GATE_D_DOC_INDICES) for task in GATE_D_TASKS},
    "limit": 5,
    "total_docs": 20,
    "num_fewshot": 5,
    "fewshot_seed": 1234,
    "batch_size": 1,
    "window": [12, 15],
    "k": 2,
    "iteration_mode": "block",
    "strategy": "damped_euler",
    "alpha": 1.0,
    "beta": 0.0,
    "cache_strategy": "last",
    "decode_mode": "bypass",
    "controller": "always_loop",
    "epsilon": 1e-12,
    "lm_eval_version": "0.4.11",
    "apply_chat_template": False,
    "fewshot_as_multiturn": False,
    "threshold_selected": False,
    "ranking_computed": False,
    "limit_for_scientific_claim": False,
    "gate_c_results_manifest_sha256": (
        "deb16bc4f0f8968a735485e3db33eb96dc798f136f39c3722f57dec6e86edede"
    ),
}

_FORBIDDEN_SIGNAL_FIELDS = {
    "answer",
    "gold",
    "target",
    "correctness",
    "is_correct",
    "prediction",
    "margin",
    "logits",
    "full_logits",
    "raw_hidden",
    "raw_hidden_state",
    "hidden_states",
    "continuation",
    "choice_continuation",
}
_SIGNAL_FIELDS = (
    "r0_median",
    "r0_p90",
    "r0_max",
    "c0",
    "n0_median",
    "n0_p90",
    "n0_max",
    "q1",
    "residual_cosine",
)
_PAIR_IDENTITY_FIELDS = (
    "task",
    "task_version",
    "renderer_hash",
    "doc_hash",
    "occurrence_id",
    "revision_closure_id",
    "subject",
    "gold",
    "choices",
)


def validate_gate_d_config(payload: Mapping[str, Any]) -> None:
    """Reject any attempt to broaden or drift the one authorized limit run."""

    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in _EXPECTED_CONFIG.items()
        if payload.get(key) != expected
    }
    extras = sorted(set(payload).difference(_EXPECTED_CONFIG))
    if extras:
        mismatches["unexpected_fields"] = extras
    if mismatches:
        raise ValueError("Gate D config mismatch: %s" % json.dumps(mismatches, sort_keys=True))


def build_decision_id(doc_key: str, action: str, reason: str) -> str:
    if not all(isinstance(value, str) and value.strip() for value in (doc_key, action, reason)):
        raise ValueError("decision id inputs must be non-empty strings")
    payload = json.dumps(
        {"doc_key": doc_key, "action": action, "reason": reason},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_gate_d_signal_payload_safe(payload: Any, path: str = "$") -> None:
    """Reject labels, continuations, outputs, raw tensors, and non-finite scalars."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_SIGNAL_FIELDS:
                raise ValueError("forbidden Gate D signal field at %s.%s" % (path, key))
            assert_gate_d_signal_payload_safe(value, "%s.%s" % (path, key))
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_gate_d_signal_payload_safe(value, "%s[%d]" % (path, index))
        return
    if isinstance(payload, float) and not math.isfinite(payload):
        raise ValueError("non-finite Gate D signal value at %s" % path)


def verify_gate_d_records(
    baseline_rows: Iterable[Mapping[str, Any]],
    loop_rows: Iterable[Mapping[str, Any]],
    signal_rows: Iterable[Mapping[str, Any]],
    decision_rows: Iterable[Mapping[str, Any]],
    expected_doc_count: int = 20,
    expected_signal_count: int = 80,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Strictly join all Gate D terminal records by authoritative canonical key."""

    baseline = _unique_index(baseline_rows, "baseline")
    loop = _unique_index(loop_rows, "loop")
    decisions = _unique_index(decision_rows, "decision")
    signals = list(signal_rows)
    if len(baseline) != expected_doc_count or len(loop) != expected_doc_count:
        raise ValueError("Gate D baseline/loop doc count mismatch")
    if len(decisions) != expected_doc_count or len(signals) != expected_signal_count:
        raise ValueError("Gate D decision/signal count mismatch")

    signal_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    seen_requests = set()
    for row in signals:
        assert_gate_d_signal_payload_safe(row)
        canonical = _doc_key(row)
        choice_index = _choice_index(row)
        request_key = (canonical, choice_index)
        if request_key in seen_requests:
            raise ValueError("duplicate Gate D request key: %s choice %d" % request_key)
        seen_requests.add(request_key)
        signal_groups[canonical].append(row)

    key_sets = {
        "baseline": set(baseline),
        "loop": set(loop),
        "signals": set(signal_groups),
        "decisions": set(decisions),
    }
    first = key_sets["baseline"]
    if any(keys != first for keys in key_sets.values()):
        raise ValueError(
            "Gate D join is not closed: %s"
            % {name: sorted(keys) for name, keys in key_sets.items()}
        )

    for canonical in sorted(first):
        left = baseline[canonical]
        right = loop[canonical]
        if left.get("arm") != "baseline" or right.get("arm") != "always_loop":
            raise ValueError("Gate D arm label mismatch for %s" % canonical)
        for field in _PAIR_IDENTITY_FIELDS:
            if left.get(field) != right.get(field):
                raise ValueError("paired %s mismatch for %s" % (field, canonical))
        _validate_key_fields(left, canonical)
        _validate_key_fields(right, canonical)

        group = signal_groups[canonical]
        if {_choice_index(row) for row in group} != {0, 1, 2, 3}:
            raise ValueError("doc %s must contain choice indices 0,1,2,3" % canonical)
        for row in group:
            _validate_key_fields(row, canonical)
            _validate_signal_row(row)
        decision_ids = {row.get("decision_id") for row in group}
        actions = {row.get("action") for row in group}
        reasons = {row.get("decision_reason") for row in group}
        if len(decision_ids) != 1 or actions != {"LOOP_K2"} or reasons != {
            "always_loop_engineering_control"
        }:
            raise ValueError("doc %s has inconsistent four-choice decisions" % canonical)

        decision = decisions[canonical]
        assert_gate_d_signal_payload_safe(decision)
        _validate_key_fields(decision, canonical)
        expected_decision_id = build_decision_id(
            canonical, "LOOP_K2", "always_loop_engineering_control"
        )
        if decision.get("decision_id") != expected_decision_id or decision_ids != {
            expected_decision_id
        }:
            raise ValueError("doc %s decision id is not stable" % canonical)
        if decision.get("action") != "LOOP_K2" or decision.get(
            "decision_reason"
        ) != "always_loop_engineering_control":
            raise ValueError("doc %s decision record mismatch" % canonical)
        if decision.get("choice_indices") != [0, 1, 2, 3]:
            raise ValueError("doc %s decision must aggregate four choices" % canonical)
        expected_inputs = list(GATE_D_CONTROLLER_INPUT_FIELDS)
        if decision.get("controller_input_fields") != expected_inputs:
            raise ValueError("doc %s controller input contract mismatch" % canonical)
        if any(row.get("controller_input_fields") != expected_inputs for row in group):
            raise ValueError("doc %s request controller input contract mismatch" % canonical)

        reference = group[0]
        for field in _SIGNAL_FIELDS:
            if any(row.get(field) != reference.get(field) for row in group[1:]):
                raise ValueError("doc %s choice requests disagree on %s" % (canonical, field))
        for field in GATE_D_CONTROLLER_INPUT_FIELDS:
            if decision.get(field) != reference.get(field):
                raise ValueError("doc %s decision aggregation mismatch for %s" % (canonical, field))

    report = {
        "schema_version": 1,
        "join_key": "task|task_version|renderer_hash|doc_hash|occurrence_id",
        "baseline_doc_count": len(baseline),
        "loop_doc_count": len(loop),
        "signal_request_count": len(signals),
        "decision_doc_count": len(decisions),
        "joined_doc_count": len(first),
        "coverage": 1.0,
        "missing_keys": [],
        "duplicate_keys": [],
        "extra_keys": [],
        "four_choice_same_decision": True,
        "paired_revision_and_metadata_closed": True,
        "joined_by_line_number": False,
    }
    summary = {
        "schema_version": 1,
        "gate": "D",
        "status": "engineering_semantics_closed",
        "doc_count": len(first),
        "signal_request_count": len(signals),
        "join_coverage": 1.0,
        "body_call_closed": True,
        "signals_finite_and_in_range": True,
        "wrapper_restore_closed": True,
        "forbidden_fields_absent": True,
        "controller_input_fields": list(GATE_D_CONTROLLER_INPUT_FIELDS),
        "always_loop_legacy_equivalence_basis": "accepted_gate_c_manifest",
        "threshold_selected": False,
        "ranking_computed": False,
        "limit_for_scientific_claim": False,
        "compute_saving_claimed": False,
    }
    return report, summary


def _unique_index(
    rows: Iterable[Mapping[str, Any]], label: str
) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        canonical = _doc_key(row)
        if canonical in result:
            raise ValueError("duplicate %s doc key: %s" % (label, canonical))
        result[canonical] = row
    return result


def _doc_key(row: Mapping[str, Any]) -> str:
    value = row.get("doc_key")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Gate D record is missing doc_key")
    return value


def _choice_index(row: Mapping[str, Any]) -> int:
    value = row.get("choice_index")
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(4):
        raise ValueError("Gate D choice_index must be 0..3")
    return value


def _validate_key_fields(row: Mapping[str, Any], canonical: str) -> None:
    expected = "|".join(
        str(row.get(field, ""))
        for field in ("task", "task_version", "renderer_hash", "doc_hash", "occurrence_id")
    )
    if expected != canonical:
        raise ValueError("Gate D canonical key fields mismatch for %s" % canonical)


def _validate_signal_row(row: Mapping[str, Any]) -> None:
    if row.get("operator_body_calls") != 2 or row.get("k_used") != 2:
        raise ValueError("Gate D body-call closure failed")
    if row.get("wrapper_restored") is not True:
        raise ValueError("Gate D wrapper restore flag is not closed")
    if not isinstance(row.get("question_token_count"), int) or row["question_token_count"] < 1:
        raise ValueError("Gate D question token count must be positive")
    for field in _SIGNAL_FIELDS:
        value = row.get(field)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Gate D signal %s must be numeric" % field)
        if not math.isfinite(float(value)):
            raise ValueError("Gate D signal %s must be finite" % field)
    for field in ("r0_median", "r0_p90", "r0_max", "n0_median", "n0_p90", "n0_max", "q1"):
        if float(row[field]) < 0:
            raise ValueError("Gate D signal %s is below Gate C range" % field)
    if not 0.0 <= float(row["c0"]) <= 1.0 + 1e-6:
        raise ValueError("Gate D c0 is outside Gate C range")
    if not -1.0 - 1e-6 <= float(row["residual_cosine"]) <= 1.0 + 1e-6:
        raise ValueError("Gate D residual cosine is outside Gate C range")
