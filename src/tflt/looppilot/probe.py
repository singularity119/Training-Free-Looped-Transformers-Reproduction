"""Pure helpers for the frozen, four-document Gate C probe."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tflt.looppilot.schema import DocKey


GATE_C_TASKS: Tuple[str, ...] = (
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_astronomy",
    "mmlu_business_ethics",
)
GATE_C_SPLIT = "test"
GATE_C_DOC_INDEX = 0
GATE_C_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"

_EXPECTED = {
    "schema_version": 1,
    "model": "Qwen/Qwen3-1.7B-Base",
    "model_revision": GATE_C_REVISION,
    "tokenizer_revision": GATE_C_REVISION,
    "dtype": "float16",
    "task": "mmlu",
    "num_fewshot": 5,
    "batch_size": 1,
    "window": [12, 15],
    "k": 2,
    "iteration_mode": "block",
    "strategy": "damped_euler",
    "alpha": 1.0,
    "beta": 0.0,
    "cache_strategy": "last",
    "decode_mode": "bypass",
    "tasks": list(GATE_C_TASKS),
    "fewshot_seed": 1234,
    "split": GATE_C_SPLIT,
    "doc_index": GATE_C_DOC_INDEX,
    "epsilon": 1e-12,
    "use_cache": True,
    "endpoint_allclose": {"rtol": 1e-3, "atol": 1e-3},
    "restore_allclose": {"rtol": 1e-3, "atol": 1e-3},
}
_FORBIDDEN_FIELDS = {"answer", "gold", "correctness", "is_correct", "target"}


def validate_gate_c_config(payload: Mapping[str, Any]) -> None:
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in _EXPECTED.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError("Gate C config mismatch: %s" % json.dumps(mismatches, sort_keys=True))


def build_question_mask(
    context: str,
    current_doc_text: str,
    offsets: Sequence[Sequence[int]],
    attention_mask: Sequence[int],
) -> List[bool]:
    if not current_doc_text or not context.endswith(current_doc_text):
        raise ValueError("rendered context must end with the current unlabeled document text")
    if len(offsets) != len(attention_mask):
        raise ValueError("offsets and attention mask length mismatch")
    start = len(context) - len(current_doc_text)
    result = []
    for offset, attended in zip(offsets, attention_mask):
        if len(offset) != 2:
            raise ValueError("token offset must contain start and end")
        left, right = int(offset[0]), int(offset[1])
        include = bool(attended) and right > left and right > start and left < len(context)
        result.append(include)
    if not any(result):
        raise ValueError("question mask selected no current-document tokens")
    return result


def build_doc_identity(
    task: str,
    task_version: str,
    renderer_hash: str,
    occurrence_id: str,
    doc: Mapping[str, Any],
) -> Dict[str, Any]:
    question = doc.get("question")
    choices = doc.get("choices")
    subject = doc.get("subject")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("MMLU document is missing question text")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise ValueError("MMLU document is missing choices")
    if not all(isinstance(choice, str) for choice in choices):
        raise ValueError("MMLU choices must be strings")
    manifest_doc: Dict[str, Any] = {"question": question, "choices": list(choices)}
    if isinstance(subject, str) and subject.strip():
        manifest_doc["subject"] = subject
    manifest_doc_json = json.dumps(
        manifest_doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    doc_hash = hashlib.sha256(manifest_doc_json.encode("utf-8")).hexdigest()
    key = DocKey(task, str(task_version), renderer_hash, doc_hash, occurrence_id)
    return {
        "doc_key": key.canonical(),
        "doc_hash": doc_hash,
        "manifest_doc": manifest_doc,
        "manifest_doc_json": manifest_doc_json,
        "manifest_doc_hash": doc_hash,
    }


def assert_no_forbidden_fields(payload: Any, path: str = "$") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_FIELDS:
                raise ValueError("forbidden labeled field at %s.%s" % (path, key))
            assert_no_forbidden_fields(value, "%s.%s" % (path, key))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_no_forbidden_fields(value, "%s[%d]" % (path, index))


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def body_call_closed(action_rows: Iterable[Mapping[str, Any]]) -> bool:
    rows = list(action_rows)
    return sum(int(row["operator_body_calls"]) for row in rows) == sum(
        int(row["k_used"]) for row in rows
    )


def validate_action_record(row: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "doc_key",
        "task",
        "action",
        "decision_reason",
        "k_used",
        "operator_body_calls",
        "endpoint",
        "audit",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError("Gate C action record is missing fields: %s" % missing)
    if row["action"] not in {"legacy_k2", "always_loop", "never_loop"}:
        raise ValueError("unsupported Gate C action record")
    expected_calls = {"legacy_k2": 2, "always_loop": 2, "never_loop": 1}[row["action"]]
    if int(row["k_used"]) != expected_calls or int(row["operator_body_calls"]) != expected_calls:
        raise ValueError("action body-call closure failed")
    assert_no_forbidden_fields(row)
