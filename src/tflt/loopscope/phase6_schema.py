"""Closed-world contracts for LoopScope Phase 6 Gate A."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


CARD_SCHEMA_VERSION = "loopscope.phase6.pre-answer-rbr-v2-card.v2"
TRAJECTORY_SCHEMA_VERSION = "loopscope.phase6.pre-answer-trajectory.v2"
SELECTOR_SCHEMA_VERSION = "loopscope.phase6.selector-freeze.v1"
METHOD = "RELATIVE_BIPHASIC_REVERSAL_V2_ABSOLUTE_RATE"
EXPECTED_POPULATION = 12032
BOUNDARY_COUNT = 37
TRANSITION_COUNT = 36
D36_TOLERANCE = 1e-6
ENDPOINT_TOLERANCE = 1e-6
HEX64 = re.compile(r"^[0-9a-f]{64}$")

SCIENTIFIC_STATES = (
    "SELECTED_WINDOW",
    "ABSTAIN_NO_RATE_STABLE_ELIGIBLE",
    "ABSTAIN_NO_UNIQUE_TOP1",
    "ABSTAIN_RATE_RANK_UNSTABLE",
)
ENGINEERING_BLOCK_STATES = (
    "BLOCK_PROVENANCE_MISMATCH",
    "BLOCK_ANCHOR_COVERAGE_NOT_EXACT",
    "BLOCK_GENERATION_REPLAY_MISMATCH",
    "BLOCK_NUMERIC_OR_ENDPOINT_INVALID",
    "BLOCK_POPULATION_OR_HASH_MISMATCH",
    "BLOCK_INFORMATION_BARRIER_VIOLATION",
    "BLOCK_V1_COMPATIBILITY_MISMATCH",
    "BLOCK_DIAGNOSTIC_PANEL_UNDERPOPULATED",
    "BLOCK_VERIFIER_MISMATCH",
    "BLOCK_OUTCOME_CELL_INCOMPLETE",
)

CARD_KEYS = {
    "schema_version",
    "card",
    "claim_boundary",
    "cell",
    "generation",
    "loop",
    "anchor",
    "trajectory",
    "candidate_domain",
    "selector",
    "bootstrap",
    "known_outcome_registry",
    "panel",
    "information_barrier",
    "scientific_states",
    "engineering_block_states",
}
TRAJECTORY_KEYS = {
    "schema_version",
    "model_repo",
    "model_revision",
    "dataset_repo",
    "dataset_revision",
    "split",
    "canonical_identity",
    "category",
    "prompt_sha256",
    "generated_completion_sha256",
    "generation_length",
    "replay_length",
    "anchor_token_index",
    "answer_span_start_offset",
    "answer_span_end_offset",
    "answer_match_count",
    "selected_match_ordinal",
    "answer_first_token_index",
    "answer_span_extractor_sha256",
    "generated_id_text_aligner_sha256",
    "generation_count",
    "replay_count",
    "loop_insertions",
    "H",
    "D",
    "hidden_rms_l2_to_final",
    "hidden_cosine_to_final",
    "hidden_cosine_distance_to_final",
    "adjacent_angular_distance",
    "provenance",
}
PROVENANCE_KEYS = {
    "producer_version",
    "card_sha256",
    "renderer_manifest_sha256",
    "tokenizer_manifest_sha256",
    "generation_ids_sha256",
    "replay_ids_sha256",
    "boundary_capture",
    "final_norm_path",
    "lm_head_path",
}
CANDIDATE_KEYS = {
    "width",
    "start",
    "end",
    "window",
    "boundary_entry",
    "boundary_exit",
    "Scorable",
    "NetPositive",
    "SoftRelativeStable",
    "BiphasicStable",
    "NewEligible",
    "G_H",
    "G_K",
    "S_RATE",
    "RateStable",
    "rate_SE_H",
    "rate_SE_K",
    "rate_c95",
    "rate_LCB_H",
    "rate_LCB_K",
    "ranking_candidate",
    "point_top_tie",
    "selected",
    "display_rank",
    "selection_frequency",
    "best_q",
    "best_tau",
    "tau_pair_frequency",
    "v1_failure_reasons",
    "v2_failure_reasons",
}
SELECTOR_KEYS = {
    "schema_version",
    "card_sha256",
    "input_manifest_sha256",
    "method",
    "population",
    "category_count",
    "candidate_count",
    "candidate_counts_by_width",
    "replicates",
    "selector_seed",
    "decision",
    "selected_window",
    "point_top_window",
    "selection_frequency",
    "selected_known_outcome_status",
    "hidden_fields_consumed",
    "outcome_fields_consumed",
    "known_outcome_registry",
    "high3",
    "low3",
    "panel_cells",
    "candidates",
}

PERSISTENCE_SAFE_KEYS = {
    "answer_span_start_offset",
    "answer_span_end_offset",
    "answer_match_count",
    "selected_match_ordinal",
    "answer_first_token_index",
    "answer_span_extractor_sha256",
    "known_outcome_registry",
    "selected_known_outcome_status",
    "outcome_fields_consumed",
}
FORBIDDEN_KEY_TOKENS = (
    "prompt_text",
    "generated_text",
    "token_ids",
    "input_ids",
    "answer_content",
    "answer_span_hash",
    "prediction",
    "gold",
    "label",
    "correctness",
    "accuracy",
    "gain",
    "flip",
    "outcome",
    "logits",
    "probabilities",
    "log_probabilities",
    "hidden_tensor",
    "residual_tensor",
)


class Phase6ContractError(ValueError):
    """Fail-closed Phase 6 contract error."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise Phase6ContractError("JSON root must be an object")
    return value


def require_exact_keys(value: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise Phase6ContractError("%s must be an object" % label)
    expected = set(keys)
    missing, extra = expected - set(value), set(value) - expected
    if missing or extra:
        raise Phase6ContractError(
            "%s is not closed-world: missing=%s extra=%s"
            % (label, sorted(missing), sorted(extra))
        )


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise Phase6ContractError("%s must be a lowercase SHA256" % label)
    return value


def require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase6ContractError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise Phase6ContractError("%s must be finite" % label)
    return result


def require_integer(value: Any, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise Phase6ContractError("%s must be an integer >= %d" % (label, minimum))
    return value


def scan_forbidden_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise Phase6ContractError("non-string field name at %s" % path)
            lowered = raw_key.lower()
            if raw_key not in PERSISTENCE_SAFE_KEYS and any(
                token in lowered for token in FORBIDDEN_KEY_TOKENS
            ):
                raise Phase6ContractError(
                    "BLOCK_INFORMATION_BARRIER_VIOLATION: %s.%s" % (path, raw_key)
                )
            scan_forbidden_fields(child, "%s.%s" % (path, raw_key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_forbidden_fields(child, "%s[%d]" % (path, index))
    elif isinstance(value, float) and not math.isfinite(value):
        raise Phase6ContractError("non-finite value at %s" % path)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise Phase6ContractError("non-JSON value at %s" % path)


def validate_card(card: Mapping[str, Any]) -> None:
    require_exact_keys(card, CARD_KEYS, "card")
    if card["schema_version"] != CARD_SCHEMA_VERSION:
        raise Phase6ContractError("card schema_version differs")
    if card["card"] != "H6_PRE_ANSWER_FULLVOCAB_RBR_V2_WINDOW_SELECTION":
        raise Phase6ContractError("card identity differs")
    if card["claim_boundary"] != (
        "local_contract_only_no_model_data_generation_replay_selector_or_outcome"
    ):
        raise Phase6ContractError("claim boundary differs")
    expected_cell = {
        "model_repo": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "tokenizer_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "decoder_layers": 36,
        "dtype": "bfloat16",
        "dataset_repo": "TIGER-Lab/MMLU-Pro",
        "dataset_revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "split": "test",
        "population": EXPECTED_POPULATION,
        "fewshot": 5,
        "fewshot_split": "validation",
    }
    require_exact_keys(card["cell"], expected_cell, "card.cell")
    if dict(card["cell"]) != expected_cell:
        raise Phase6ContractError("frozen cell differs")
    generation = card["generation"]
    expected_generation = {
        "mode": "greedy",
        "do_sample": False,
        "temperature": 0.0,
        "max_gen_toks": 2048,
        "until": "Question:",
        "generation_count": 1,
    }
    require_exact_keys(generation, expected_generation, "card.generation")
    if dict(generation) != expected_generation:
        raise Phase6ContractError("generation contract differs")
    loop = card["loop"]
    expected_loop = {
        "k": 3,
        "iteration_mode": "block",
        "strategy": "euler",
        "step_size": "1/3",
        "total_horizon": 1,
        "cache_strategy": "first",
        "decode_mode": "full",
    }
    require_exact_keys(loop, expected_loop, "card.loop")
    if dict(loop) != expected_loop:
        raise Phase6ContractError("loop contract differs")
    anchor = card["anchor"]
    required_anchor = {
        "lm_eval_version",
        "renderer_source_sha256",
        "task_group_sha256",
        "filter_source_sha256",
        "filter_name",
        "regex_pattern",
        "capture_group",
        "case_sensitive",
        "match_rule",
        "outcome_take_first_used_for_anchor",
        "offset_unit",
        "token_intervals",
        "boundary_owner",
        "probe_rule",
        "fallback",
    }
    require_exact_keys(anchor, required_anchor, "card.anchor")
    expected_anchor = {
        "lm_eval_version": "0.4.11",
        "renderer_source_sha256": "74ab409c4e4c96e4351fbe6122d519f64dd3e11381a676957631e858923cc9fc",
        "task_group_sha256": "0271e3fdbbb0e8df5b6909786600c1292da29842086b521a1181954941156f94",
        "filter_source_sha256": "356e937a958288fafd8f07b03da7fd4825977e9bb3d62136931c9f649b600647",
        "filter_name": "custom-extract",
        "regex_pattern": r"answer is \(?([ABCDEFGHIJ])\)?",
        "capture_group": 1,
        "case_sensitive": True,
        "match_rule": "collect_all_capture_spans_select_ordinal_0",
        "outcome_take_first_used_for_anchor": True,
        "offset_unit": "utf8_byte",
        "token_intervals": "half_open",
        "boundary_owner": "right_token",
        "probe_rule": "token_strictly_preceding_answer_first_token",
        "fallback": "forbidden",
    }
    if dict(anchor) != expected_anchor:
        raise Phase6ContractError("anchor contract differs")
    trajectory = card["trajectory"]
    if trajectory != {
        "raw_boundaries": "B0...B36_final_norm_pre_hook",
        "selector_fields": ["H", "D"],
        "diagnostic_fields": [
            "hidden_rms_l2_to_final",
            "hidden_cosine_to_final",
            "hidden_cosine_distance_to_final",
            "adjacent_angular_distance",
        ],
        "D36_tolerance": D36_TOLERANCE,
        "hidden_endpoint_tolerance": ENDPOINT_TOLERANCE,
    }:
        raise Phase6ContractError("trajectory contract differs")
    domain = card["candidate_domain"]
    if domain != {
        "central_blocks": [11, 24],
        "width_starts": {
            "3": [11, 22],
            "4": [11, 21],
            "5": [11, 20],
            "6": [11, 19],
        },
        "candidate_count": 42,
    }:
        raise Phase6ContractError("candidate domain differs")
    selector = card["selector"]
    if selector != {
        "method": METHOD,
        "new_eligible": [
            "Scorable",
            "NetPositive",
            "SoftRelativeStable",
            "BiphasicStable",
        ],
        "score": "sqrt(G_H*G_K)",
        "rate_stable": "within_window_HK_one_sided_fixed_SE_max_t_LCB_both_gt_0",
        "tie_rel_tol": 1e-12,
        "tie_abs_tol": 1e-12,
        "selection_frequency_threshold": 0.8,
    }:
        raise Phase6ContractError("selector contract differs")
    if card["bootstrap"] != {
        "method": "category_stratified_joint",
        "replicates": 2000,
        "selector_seed": 20260801,
        "outcome_seed": 20260802,
        "standard_deviation_ddof": 1,
        "epsilon": 1e-12,
        "quantile": "linear_q_times_R_minus_1",
    }:
        raise Phase6ContractError("bootstrap contract differs")
    if card["known_outcome_registry"] != [
        "no-loop",
        "15:18",
        "6:9",
        "10:13",
        "25:28",
        "4:7",
        "5:8",
        "22:25",
    ]:
        raise Phase6ContractError("known registry differs")
    panel = card["panel"]
    expected_panel = {
        "baseline": "no-loop",
        "fixed_comparator": "15:18",
        "diagnostic_universe": (
            "registry_excluded_positive_finite_S_RATE_candidates"
        ),
        "high_count": 3,
        "low_count": 3,
        "selected_if_distinct": True,
        "union": "unique",
        "underpopulation_block": "BLOCK_DIAGNOSTIC_PANEL_UNDERPOPULATED",
    }
    require_exact_keys(panel, expected_panel, "card.panel")
    if dict(panel) != expected_panel:
        raise Phase6ContractError("panel contract differs")
    barrier = card["information_barrier"]
    expected_barrier = {
        "selector_allowed_fields": [
            "canonical_identity",
            "category",
            "H",
            "D",
        ],
        "diagnostic_only_fields": [
            "hidden_rms_l2_to_final",
            "hidden_cosine_to_final",
            "hidden_cosine_distance_to_final",
            "adjacent_angular_distance",
        ],
        "sanitized_metadata_allowed": [
            "span_offsets",
            "token_indices",
            "extractor_aligner_hashes",
            "scalar_arrays",
        ],
        "forbidden_payloads": [
            "prompt_text",
            "generated_text",
            "token_ids",
            "answer_content",
            "answer_span_hash",
            "prediction",
            "gold",
            "label",
            "correctness",
            "accuracy",
            "gain",
            "flip",
            "outcome",
        ],
    }
    require_exact_keys(barrier, expected_barrier, "card.information_barrier")
    if dict(barrier) != expected_barrier:
        raise Phase6ContractError("information barrier differs")
    if card["scientific_states"] != list(SCIENTIFIC_STATES):
        raise Phase6ContractError("scientific states differ")
    if card["engineering_block_states"] != list(ENGINEERING_BLOCK_STATES):
        raise Phase6ContractError("engineering BLOCK states differ")


def _vector(
    value: Any,
    length: int,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Phase6ContractError("%s must be a vector" % label)
    if len(value) != length:
        raise Phase6ContractError("%s length must be %d" % (label, length))
    result = [require_finite(item, label) for item in value]
    if lower is not None and any(item < lower for item in result):
        raise Phase6ContractError("%s is below its lower bound" % label)
    if upper is not None and any(item > upper for item in result):
        raise Phase6ContractError("%s is above its upper bound" % label)
    return result


def validate_trajectory_record(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, TRAJECTORY_KEYS, "trajectory record")
    scan_forbidden_fields(record)
    if record["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise Phase6ContractError("trajectory schema_version differs")
    constants = {
        "model_repo": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "dataset_repo": "TIGER-Lab/MMLU-Pro",
        "dataset_revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "split": "test",
        "generation_count": 1,
        "replay_count": 1,
        "loop_insertions": 0,
    }
    for key, expected in constants.items():
        if record[key] != expected:
            raise Phase6ContractError("trajectory %s differs" % key)
    if not isinstance(record["canonical_identity"], str) or not record["canonical_identity"]:
        raise Phase6ContractError("canonical_identity is missing")
    if not isinstance(record["category"], str) or not record["category"].strip():
        raise Phase6ContractError("category is missing")
    for key in (
        "prompt_sha256",
        "generated_completion_sha256",
        "answer_span_extractor_sha256",
        "generated_id_text_aligner_sha256",
    ):
        require_sha256(record[key], key)
    generation_length = require_integer(record["generation_length"], "generation_length", 1)
    replay_length = require_integer(record["replay_length"], "replay_length", 1)
    answer_first = require_integer(
        record["answer_first_token_index"], "answer_first_token_index", 1
    )
    anchor = require_integer(record["anchor_token_index"], "anchor_token_index")
    if answer_first >= generation_length or anchor != answer_first - 1:
        raise Phase6ContractError("answer/probe token indices differ")
    start = require_integer(record["answer_span_start_offset"], "answer_span_start_offset")
    end = require_integer(record["answer_span_end_offset"], "answer_span_end_offset", 1)
    match_count = require_integer(record["answer_match_count"], "answer_match_count", 1)
    selected_ordinal = require_integer(
        record["selected_match_ordinal"], "selected_match_ordinal"
    )
    if selected_ordinal != 0 or selected_ordinal >= match_count:
        raise Phase6ContractError("selected answer match ordinal differs")
    if end <= start or replay_length <= anchor:
        raise Phase6ContractError("span/replay closure failed")
    _vector(record["H"], BOUNDARY_COUNT, "H", 0.0)
    d_values = _vector(record["D"], BOUNDARY_COUNT, "D", -D36_TOLERANCE)
    rms = _vector(
        record["hidden_rms_l2_to_final"],
        BOUNDARY_COUNT,
        "hidden_rms_l2_to_final",
        0.0,
    )
    cosine = _vector(
        record["hidden_cosine_to_final"],
        BOUNDARY_COUNT,
        "hidden_cosine_to_final",
        -1.0 - ENDPOINT_TOLERANCE,
        1.0 + ENDPOINT_TOLERANCE,
    )
    distance = _vector(
        record["hidden_cosine_distance_to_final"],
        BOUNDARY_COUNT,
        "hidden_cosine_distance_to_final",
        -ENDPOINT_TOLERANCE,
        2.0 + ENDPOINT_TOLERANCE,
    )
    _vector(
        record["adjacent_angular_distance"],
        TRANSITION_COUNT,
        "adjacent_angular_distance",
        0.0,
        1.0,
    )
    for left, right in zip(distance, cosine):
        if not math.isclose(left, 1.0 - right, rel_tol=1e-10, abs_tol=1e-10):
            raise Phase6ContractError("cosine distance must equal 1-cosine")
    if abs(d_values[-1]) > D36_TOLERANCE:
        raise Phase6ContractError("D36 exceeds tolerance")
    if (
        abs(rms[-1]) > ENDPOINT_TOLERANCE
        or abs(cosine[-1] - 1.0) > ENDPOINT_TOLERANCE
        or abs(distance[-1]) > ENDPOINT_TOLERANCE
    ):
        raise Phase6ContractError("hidden final endpoints do not close")
    require_exact_keys(record["provenance"], PROVENANCE_KEYS, "trajectory provenance")
    for key in PROVENANCE_KEYS - {
        "producer_version",
        "boundary_capture",
        "final_norm_path",
        "lm_head_path",
    }:
        require_sha256(record["provenance"][key], "provenance.%s" % key)
    if record["provenance"]["boundary_capture"] != "raw_B0_through_B36_final_norm_pre_hook":
        raise Phase6ContractError("raw boundary capture differs")


def selector_sample(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a full record to the selector's only authorized inputs."""

    validate_trajectory_record(record)
    return {
        "identity": record["canonical_identity"],
        "category": record["category"],
        "H": list(record["H"]),
        "D": list(record["D"]),
    }


def validate_selector_freeze(payload: Mapping[str, Any]) -> None:
    require_exact_keys(payload, SELECTOR_KEYS, "selector freeze")
    scan_forbidden_fields(payload)
    if payload["schema_version"] != SELECTOR_SCHEMA_VERSION:
        raise Phase6ContractError("selector schema_version differs")
    if payload["method"] != METHOD or payload["decision"] not in SCIENTIFIC_STATES:
        raise Phase6ContractError("selector method/decision differs")
    if payload["candidate_count"] != 42 or payload["candidate_counts_by_width"] != {
        "3": 12,
        "4": 11,
        "5": 10,
        "6": 9,
    }:
        raise Phase6ContractError("candidate closure differs")
    if payload["selector_seed"] != 20260801:
        raise Phase6ContractError("selector seed differs")
    if payload["replicates"] != 2000:
        raise Phase6ContractError("selector replicate count differs")
    if payload["population"] != EXPECTED_POPULATION:
        raise Phase6ContractError("selector population differs")
    require_integer(payload["category_count"], "category_count", 1)
    if payload["hidden_fields_consumed"] is not False:
        raise Phase6ContractError("hidden diagnostics entered selector")
    if payload["outcome_fields_consumed"] is not False:
        raise Phase6ContractError("outcome fields entered selector")
    require_sha256(payload["card_sha256"], "card_sha256")
    require_sha256(payload["input_manifest_sha256"], "input_manifest_sha256")
    expected_registry = [
        "no-loop",
        "15:18",
        "6:9",
        "10:13",
        "25:28",
        "4:7",
        "5:8",
        "22:25",
    ]
    if payload["known_outcome_registry"] != expected_registry:
        raise Phase6ContractError("selector known registry differs")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 42:
        raise Phase6ContractError("selector must contain 42 candidate rows")
    selected = []
    observed_keys = set()
    positive_scores: Dict[str, tuple[float, int, int]] = {}
    for row in candidates:
        require_exact_keys(row, CANDIDATE_KEYS, "selector candidate")
        width = require_integer(row["width"], "candidate.width", 3)
        start = require_integer(row["start"], "candidate.start")
        if width not in (3, 4, 5, 6) or row["end"] != start + width - 1:
            raise Phase6ContractError("candidate identity differs")
        if row["window"] != "%d:%d" % (start, start + width):
            raise Phase6ContractError("candidate window label differs")
        if (
            row["boundary_entry"] != "B_%d" % start
            or row["boundary_exit"] != "B_%d" % (start + width)
        ):
            raise Phase6ContractError("candidate boundary labels differ")
        observed_keys.add((width, start))
        formula_fields = (
            "Scorable",
            "NetPositive",
            "SoftRelativeStable",
            "BiphasicStable",
        )
        if any(not isinstance(row[key], bool) for key in formula_fields):
            raise Phase6ContractError("candidate admission fields must be booleans")
        if row["NewEligible"] is not all(row[key] for key in formula_fields):
            raise Phase6ContractError("NewEligible formula differs")
        if not isinstance(row["RateStable"], bool):
            raise Phase6ContractError("RateStable must be boolean")
        g_h = require_finite(row["G_H"], "candidate.G_H")
        g_k = require_finite(row["G_K"], "candidate.G_K")
        expected_score = math.sqrt(g_h * g_k) if g_h > 0.0 and g_k > 0.0 else None
        if expected_score is None:
            if row["S_RATE"] is not None:
                raise Phase6ContractError("nonpositive candidate S_RATE differs")
        else:
            score = require_finite(row["S_RATE"], "candidate.S_RATE")
            if not math.isclose(score, expected_score, rel_tol=1e-12, abs_tol=1e-12):
                raise Phase6ContractError("candidate S_RATE formula differs")
            positive_scores[row["window"]] = (score, width, start)
        if row["selected"]:
            selected.append(row)
    expected_keys = {
        (width, start)
        for width, upper in ((3, 22), (4, 21), (5, 20), (6, 19))
        for start in range(11, upper + 1)
    }
    if observed_keys != expected_keys:
        raise Phase6ContractError("candidate enumeration differs")
    if payload["decision"] == "SELECTED_WINDOW":
        if len(selected) != 1 or payload["selected_window"] != selected[0]["window"]:
            raise Phase6ContractError("selected-window closure failed")
    elif selected or payload["selected_window"] is not None:
        raise Phase6ContractError("ABSTAIN cannot contain selected window")
    if len(payload["high3"]) != 3 or len(payload["low3"]) != 3:
        raise Phase6ContractError("High3/Low3 must each contain three windows")
    if set(payload["high3"]) & set(payload["low3"]):
        raise Phase6ContractError("High3/Low3 must be disjoint")
    diagnostic = {
        window: values
        for window, values in positive_scores.items()
        if window not in set(expected_registry)
    }
    expected_high = [
        window
        for window, _ in sorted(
            diagnostic.items(),
            key=lambda item: (-item[1][0], item[1][1], item[1][2]),
        )[:3]
    ]
    expected_low = [
        window
        for window, _ in sorted(
            diagnostic.items(),
            key=lambda item: (item[1][0], item[1][1], item[1][2]),
        )[:3]
    ]
    if len(diagnostic) < 6:
        raise Phase6ContractError("BLOCK_DIAGNOSTIC_PANEL_UNDERPOPULATED")
    if payload["high3"] != expected_high or payload["low3"] != expected_low:
        raise Phase6ContractError("diagnostic High3/Low3 differ")
    expected_panel = []
    for cell in [
        "no-loop",
        "15:18",
        *payload["high3"],
        *payload["low3"],
        payload["selected_window"],
    ]:
        if cell is not None and cell not in expected_panel:
            expected_panel.append(cell)
    if payload["panel_cells"] != expected_panel:
        raise Phase6ContractError("panel union differs")
    if len(payload["panel_cells"]) != len(set(payload["panel_cells"])):
        raise Phase6ContractError("panel union must be deduplicated")
    if payload["selected_window"] is None:
        expected_status = "NOT_APPLICABLE"
    elif payload["selected_window"] in expected_registry:
        expected_status = "SELECTED_KNOWN_OUTCOME_WINDOW"
    else:
        expected_status = "SELECTED_NEW_OUTCOME_WINDOW"
    if payload["selected_known_outcome_status"] != expected_status:
        raise Phase6ContractError("selected known-outcome status differs")


def trajectory_json_schema() -> Dict[str, Any]:
    vector37 = {
        "type": "array",
        "minItems": 37,
        "maxItems": 37,
        "items": {"type": "number"},
    }
    vector36 = {
        "type": "array",
        "minItems": 36,
        "maxItems": 36,
        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    properties: Dict[str, Any] = {key: {} for key in sorted(TRAJECTORY_KEYS)}
    properties.update(
        {
            "schema_version": {"const": TRAJECTORY_SCHEMA_VERSION},
            "model_repo": {"const": "Qwen/Qwen3-4B-Instruct-2507"},
            "model_revision": {
                "const": "cdbee75f17c01a7cc42f958dc650907174af0554"
            },
            "dataset_repo": {"const": "TIGER-Lab/MMLU-Pro"},
            "dataset_revision": {
                "const": "b189ec765aa7ed75c8acfea42df31fdae71f97be"
            },
            "split": {"const": "test"},
            "H": vector37,
            "D": vector37,
            "hidden_rms_l2_to_final": vector37,
            "hidden_cosine_to_final": vector37,
            "hidden_cosine_distance_to_final": vector37,
            "adjacent_angular_distance": vector36,
            "generation_count": {"const": 1},
            "replay_count": {"const": 1},
            "loop_insertions": {"const": 0},
            "answer_match_count": {"type": "integer", "minimum": 1},
            "selected_match_ordinal": {"const": 0},
            "provenance": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(PROVENANCE_KEYS),
                "properties": {key: {} for key in sorted(PROVENANCE_KEYS)},
            },
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": TRAJECTORY_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": sorted(TRAJECTORY_KEYS),
        "properties": properties,
    }


def selector_freeze_json_schema() -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SELECTOR_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": sorted(SELECTOR_KEYS),
        "properties": {
            **{key: {} for key in sorted(SELECTOR_KEYS)},
            "schema_version": {"const": SELECTOR_SCHEMA_VERSION},
            "method": {"const": METHOD},
            "candidate_count": {"const": 42},
            "candidates": {
                "type": "array",
                "minItems": 42,
                "maxItems": 42,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(CANDIDATE_KEYS),
                    "properties": {key: {} for key in sorted(CANDIDATE_KEYS)},
                },
            },
            "hidden_fields_consumed": {"const": False},
            "outcome_fields_consumed": {"const": False},
        },
    }


def validate_schema_document(schema: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if schema != expected or schema.get("additionalProperties") is not False:
        raise Phase6ContractError("closed schema document differs from implementation")


def _reject_constant(value: str) -> None:
    raise Phase6ContractError("non-finite JSON constant %s" % value)


__all__ = [
    "BOUNDARY_COUNT",
    "CANDIDATE_KEYS",
    "CARD_SCHEMA_VERSION",
    "ENGINEERING_BLOCK_STATES",
    "EXPECTED_POPULATION",
    "METHOD",
    "Phase6ContractError",
    "SCIENTIFIC_STATES",
    "SELECTOR_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "canonical_json_bytes",
    "file_sha256",
    "load_json",
    "require_exact_keys",
    "require_finite",
    "require_integer",
    "require_sha256",
    "scan_forbidden_fields",
    "selector_freeze_json_schema",
    "selector_sample",
    "semantic_sha256",
    "trajectory_json_schema",
    "validate_card",
    "validate_schema_document",
    "validate_selector_freeze",
    "validate_trajectory_record",
]
