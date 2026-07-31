"""Independent local verifier for LoopScope Phase 6 Gate A contracts."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, Mapping

from tflt.loopscope import phase6_anchor
from tflt.loopscope.phase6_schema import (
    Phase6ContractError,
    load_json,
    selector_freeze_json_schema,
    trajectory_json_schema,
    validate_card,
    validate_schema_document,
)


class Phase6VerificationError(ValueError):
    """Independent Gate A verifier failure."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise Phase6VerificationError(message)


def _independent_card_checks(card: Mapping[str, Any]) -> None:
    """Recompute decision-critical invariants without trusting card prose."""

    anchor = card["anchor"]
    _check(anchor["lm_eval_version"] == phase6_anchor.LM_EVAL_VERSION, "lm-eval version")
    _check(
        anchor["renderer_source_sha256"] == phase6_anchor.LM_EVAL_UTILS_SHA256,
        "renderer source hash",
    )
    _check(
        anchor["task_group_sha256"] == phase6_anchor.MMLU_PRO_YAML_SHA256,
        "task-group source hash",
    )
    _check(
        anchor["filter_source_sha256"]
        == phase6_anchor.DEFAULT_TEMPLATE_YAML_SHA256,
        "filter source hash",
    )
    _check(
        anchor["regex_pattern"] == phase6_anchor.ANSWER_REGEX_PATTERN,
        "extractor regex",
    )
    _check(re.compile(anchor["regex_pattern"]).flags & re.IGNORECASE == 0, "regex case")
    _check(
        anchor["match_rule"] == "collect_all_capture_spans_select_ordinal_0",
        "extractor match rule",
    )
    _check(anchor["outcome_take_first_used_for_anchor"] is True, "take_first anchor")

    width_starts = card["candidate_domain"]["width_starts"]
    expected = {
        3: list(range(11, 23)),
        4: list(range(11, 22)),
        5: list(range(11, 21)),
        6: list(range(11, 20)),
    }
    observed_count = 0
    observed_windows = set()
    for width, starts in expected.items():
        lower, upper = width_starts[str(width)]
        observed = list(range(lower, upper + 1))
        _check(observed == starts, "candidate starts for width %d" % width)
        observed_count += len(observed)
        observed_windows.update("%d:%d" % (start, start + width) for start in observed)
    _check(observed_count == 42 and len(observed_windows) == 42, "candidate closure")

    selector_allowed = set(card["information_barrier"]["selector_allowed_fields"])
    diagnostics = set(card["information_barrier"]["diagnostic_only_fields"])
    _check(selector_allowed == {"canonical_identity", "category", "H", "D"}, "selector fields")
    _check(selector_allowed.isdisjoint(diagnostics), "diagnostic isolation")
    registry = set(card["known_outcome_registry"])
    _check({"no-loop", "15:18"}.issubset(registry), "known registry closure")


def verify_gate_a_contracts(
    card_path: Path,
    trajectory_schema_path: Path,
    selector_schema_path: Path,
) -> Dict[str, Any]:
    """Validate the card and both checked-in closed-world schemas."""

    try:
        card = load_json(card_path)
        trajectory_schema = load_json(trajectory_schema_path)
        selector_schema = load_json(selector_schema_path)
        validate_card(card)
        validate_schema_document(trajectory_schema, trajectory_json_schema())
        validate_schema_document(selector_schema, selector_freeze_json_schema())
        _independent_card_checks(card)
    except (OSError, KeyError, TypeError, ValueError, Phase6ContractError) as exc:
        raise Phase6VerificationError(str(exc)) from exc
    return {
        "status": "PASS",
        "claim_boundary": card["claim_boundary"],
        "candidate_count": 42,
        "trajectory_schema_closed": True,
        "selector_schema_closed": True,
        "model_or_data_loaded": False,
        "selector_executed": False,
        "outcomes_read": False,
    }


def run_self_test(
    card_path: Path,
    trajectory_schema_path: Path,
    selector_schema_path: Path,
) -> Dict[str, Any]:
    """Exercise one valid and several decision-critical invalid fixtures."""

    result = verify_gate_a_contracts(
        card_path, trajectory_schema_path, selector_schema_path
    )
    card = load_json(card_path)
    rejected = []
    mutations = {
        "regex": ("anchor", "regex_pattern", r"Answer is ([A-J])"),
        "candidate_count": ("candidate_domain", "candidate_count", 41),
        "selector_threshold": ("selector", "selection_frequency_threshold", 0.79),
    }
    for label, (section, field, replacement) in mutations.items():
        invalid = copy.deepcopy(card)
        invalid[section][field] = replacement
        try:
            validate_card(invalid)
        except Phase6ContractError:
            rejected.append(label)
        else:
            raise Phase6VerificationError("self-test accepted invalid %s card" % label)

    invalid_schema = copy.deepcopy(load_json(trajectory_schema_path))
    invalid_schema["additionalProperties"] = True
    try:
        validate_schema_document(invalid_schema, trajectory_json_schema())
    except Phase6ContractError:
        rejected.append("open_schema")
    else:
        raise Phase6VerificationError("self-test accepted an open schema")

    result["self_test"] = "PASS"
    result["invalid_fixtures_rejected"] = rejected
    return result


__all__ = [
    "Phase6VerificationError",
    "run_self_test",
    "verify_gate_a_contracts",
]
