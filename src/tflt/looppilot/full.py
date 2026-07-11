"""Frozen Gate E manifests, joins, candidate selection, and terminal routing."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tflt.looppilot.limit import assert_gate_d_signal_payload_safe


GATE_E_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
GATE_E_SPLIT_SALT = "looppilot-phase1-gate-e-v1"
GATE_E_CANDIDATE_FIELDS: Tuple[str, ...] = ("r0_median", "c0", "n0_median")
GATE_E_DIRECTIONS: Tuple[str, ...] = ("BASELINE_IF_HIGH", "BASELINE_IF_LOW")
GATE_E_COVERAGES: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)

_EXPECTED_CONFIG: Dict[str, Any] = {
    "schema_version": 1,
    "gate": "E",
    "model": "Qwen/Qwen3-1.7B-Base",
    "model_revision": GATE_E_REVISION,
    "tokenizer_revision": GATE_E_REVISION,
    "lm_eval_version": "0.4.11",
    "dtype": "float16",
    "task_group": "mmlu",
    "leaf_task_count": 57,
    "split": "test",
    "all_docs": True,
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
    "arms": ["baseline_a", "baseline_b", "always_loop"],
    "shard_count": 8,
    "split_salt": GATE_E_SPLIT_SALT,
    "candidate_signals": list(GATE_E_CANDIDATE_FIELDS),
    "candidate_directions": list(GATE_E_DIRECTIONS),
    "baseline_coverages": list(GATE_E_COVERAGES),
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260712,
    "random_null_replicates": 10000,
    "random_null_seed": 20260711,
    "gate_c_results_manifest_sha256": "deb16bc4f0f8968a735485e3db33eb96dc798f136f39c3722f57dec6e86edede",
    "gate_d_results_manifest_sha256": "1c5d84c3abeff8b36e4604fb539077553a7f16f3a4455bc51f6f63080012a7c9",
    "gate_d_canonical_job": "9965452",
    "gate_d_rejected_duplicate_job": "9965453",
}


def validate_gate_e_config(payload: Mapping[str, Any]) -> None:
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in _EXPECTED_CONFIG.items()
        if payload.get(key) != expected
    }
    extras = sorted(set(payload).difference(_EXPECTED_CONFIG))
    if extras:
        mismatches["unexpected_fields"] = extras
    if mismatches:
        raise ValueError("Gate E config mismatch: %s" % json.dumps(mismatches, sort_keys=True))


def validate_leaf_tasks(tasks: Iterable[str]) -> Tuple[str, ...]:
    values = list(tasks)
    if len(values) != 57 or len(set(values)) != 57:
        raise ValueError("authoritative mmlu expansion must contain 57 unique leaf tasks")
    if any(not isinstance(name, str) or not name.startswith("mmlu_") or name == "mmlu" for name in values):
        raise ValueError("authoritative mmlu expansion contains a non-leaf task")
    return tuple(sorted(values))


def split_bucket(doc_key: str, salt: str = GATE_E_SPLIT_SALT) -> int:
    if not isinstance(doc_key, str) or not doc_key:
        raise ValueError("doc_key must be non-empty")
    return int(hashlib.sha256((salt + "|" + doc_key).encode()).hexdigest(), 16) % 2


def build_split_rows(documents: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    counts: Dict[str, Counter] = defaultdict(Counter)
    for document in documents:
        doc_key = str(document.get("doc_key", ""))
        task = str(document.get("task", ""))
        if not doc_key or not task or doc_key in seen:
            raise ValueError("split manifest requires unique doc_key and task")
        seen.add(doc_key)
        bucket = split_bucket(doc_key)
        split = "development" if bucket == 0 else "holdout"
        counts[task][split] += 1
        rows.append({"doc_key": doc_key, "task": task, "bucket": bucket, "split": split})
    empty = [task for task, count in counts.items() if not count["development"] or not count["holdout"]]
    if empty:
        raise ValueError("Gate E subject has empty development/holdout side: %s" % sorted(empty))
    return sorted(rows, key=lambda row: row["doc_key"])


def assign_task_shards(task_doc_counts: Mapping[str, int], shard_count: int = 8) -> List[Dict[str, Any]]:
    if shard_count != 8 or len(task_doc_counts) != 57:
        raise ValueError("Gate E requires 57 tasks and exactly eight shards")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in task_doc_counts.values()):
        raise ValueError("task doc counts must be positive integers")
    shards = [{"shard_id": index, "doc_count": 0, "tasks": []} for index in range(shard_count)]
    for task, count in sorted(task_doc_counts.items(), key=lambda item: (-item[1], item[0])):
        target = min(shards, key=lambda row: (row["doc_count"], row["shard_id"]))
        target["tasks"].append(task)
        target["doc_count"] += count
    for row in shards:
        row["tasks"].sort()
    return shards


def verify_three_arm_rows(
    baseline_a_rows: Iterable[Mapping[str, Any]],
    baseline_b_rows: Iterable[Mapping[str, Any]],
    loop_rows: Iterable[Mapping[str, Any]],
    signal_rows: Iterable[Mapping[str, Any]],
    decision_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    arms = {
        "baseline_a": _index(baseline_a_rows, "baseline_a"),
        "baseline_b": _index(baseline_b_rows, "baseline_b"),
        "always_loop": _index(loop_rows, "always_loop"),
    }
    decisions = _index(decision_rows, "decision")
    key_set = set(arms["baseline_a"])
    if not key_set or any(set(values) != key_set for values in arms.values()) or set(decisions) != key_set:
        raise ValueError("Gate E three-arm join is not closed")
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in signal_rows:
        assert_gate_d_signal_payload_safe(row)
        grouped[_key(row)].append(row)
    if set(grouped) != key_set:
        raise ValueError("Gate E signal join is not closed")
    self_flips = 0
    loop_flips = 0
    outcomes = Counter()
    for key in sorted(key_set):
        a, b, loop = (arms[name][key] for name in ("baseline_a", "baseline_b", "always_loop"))
        for row, expected in ((a, "baseline_a"), (b, "baseline_b"), (loop, "always_loop")):
            if row.get("arm") != expected:
                raise ValueError("Gate E arm mismatch")
        identity = ("task", "task_version", "renderer_hash", "doc_hash", "occurrence_id", "revision_closure_id", "subject", "gold")
        if any(a.get(field) != b.get(field) or a.get(field) != loop.get(field) for field in identity):
            raise ValueError("Gate E paired metadata/revision mismatch for %s" % key)
        self_flips += int(a.get("prediction") != b.get("prediction"))
        loop_flips += int(a.get("prediction") != loop.get("prediction"))
        ac = a.get("prediction") == a.get("gold")
        lc = loop.get("prediction") == loop.get("gold")
        outcomes["both_correct" if ac and lc else "both_wrong" if not ac and not lc else "loop_helps" if lc else "loop_hurts"] += 1
        group = sorted(grouped[key], key=lambda row: row.get("choice_index", -1))
        if [row.get("choice_index") for row in group] != [0, 1, 2, 3]:
            raise ValueError("Gate E requires four unique choice signal rows per doc")
        for row in group:
            if row.get("operator_body_calls") != 2 or row.get("k_used") != 2 or row.get("wrapper_restored") is not True:
                raise ValueError("Gate E signal body-call/restore closure failed")
        decision = decisions[key]
        assert_gate_d_signal_payload_safe(decision)
        if decision.get("choice_indices") != [0, 1, 2, 3] or decision.get("controller_input_fields") != list(GATE_E_CANDIDATE_FIELDS):
            raise ValueError("Gate E decision contract mismatch")
        for field in GATE_E_CANDIDATE_FIELDS:
            if any(row.get(field) != group[0].get(field) for row in group[1:]) or decision.get(field) != group[0].get(field):
                raise ValueError("Gate E four-choice signal aggregation mismatch")
    return {
        "joined_doc_count": len(key_set),
        "signal_request_count": sum(len(rows) for rows in grouped.values()),
        "join_coverage": 1.0,
        "self_flip_count": self_flips,
        "loop_flip_count": loop_flips,
        "outcomes": dict(outcomes),
        "body_call_closed": True,
        "restore_closed": True,
        "forbidden_fields_absent": True,
    }


def select_candidate(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    development = [row for row in rows if row.get("split") == "development"]
    if not development or any(row.get("split") != "development" for row in development):
        raise ValueError("candidate selection requires development rows")
    grid: List[Dict[str, Any]] = []
    for signal in GATE_E_CANDIDATE_FIELDS:
        values = sorted(float(row[signal]) for row in development)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("candidate signal must be finite")
        for direction in GATE_E_DIRECTIONS:
            for coverage in GATE_E_COVERAGES:
                quantile = 1.0 - coverage if direction == "BASELINE_IF_HIGH" else coverage
                threshold = _quantile(values, quantile)
                correct = 0
                baseline_count = 0
                for row in development:
                    baseline = float(row[signal]) >= threshold if direction == "BASELINE_IF_HIGH" else float(row[signal]) <= threshold
                    baseline_count += int(baseline)
                    correct += int(bool(row["baseline_correct"] if baseline else row["loop_correct"]))
                grid.append({
                    "signal": signal,
                    "direction": direction,
                    "target_baseline_coverage": coverage,
                    "threshold": threshold,
                    "development_accuracy": correct / len(development),
                    "development_baseline_count": baseline_count,
                    "development_count": len(development),
                })
    signal_rank = {name: index for index, name in enumerate(GATE_E_CANDIDATE_FIELDS)}
    direction_rank = {name: index for index, name in enumerate(GATE_E_DIRECTIONS)}
    selected = min(
        grid,
        key=lambda row: (
            -row["development_accuracy"],
            signal_rank[row["signal"]],
            direction_rank[row["direction"]],
            row["target_baseline_coverage"],
        ),
    )
    return dict(selected), grid


def terminal_recommendation(flags: Mapping[str, Any]) -> str:
    if flags.get("technical_closed") is not True:
        return "BLOCK"
    if flags.get("assessable") is not True:
        return "PIVOT_TO_PHASE3"
    if flags.get("pass_to_online") is True:
        return "PASS_TO_ONLINE"
    if flags.get("subject_capture_ge_80pct") is True and flags.get("hidden_capture_le_subject") is True:
        return "PIVOT_TO_COARSE"
    return "PIVOT_TO_PHASE3"


def _index(rows: Iterable[Mapping[str, Any]], label: str) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError("duplicate %s doc key: %s" % (label, key))
        result[key] = row
    return result


def _key(row: Mapping[str, Any]) -> str:
    key = row.get("doc_key")
    if not isinstance(key, str) or not key:
        raise ValueError("record is missing doc_key")
    return key


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
