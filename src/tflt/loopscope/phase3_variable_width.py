"""Outcome-blind variable-width rankings for LoopScope Phase 3 Gate F.

This sidecar consumes only the validated P3-B no-loop B_0...B_28 choice
trajectories.  It publishes the frozen strict CONSENSUS ranking and the
separately named edge-aware full-coverage ranking without loading loop
outcomes, labels, or model/data runtimes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.phase3_analysis import (
    bootstrap_summary,
    iter_subject_bootstrap_indices,
)
from tflt.loopscope.phase3_schema import (
    canonical_record_key,
    load_phase3_card,
    validate_trajectory_record,
)
from tflt.loopscope.schema import (
    SchemaError,
    attach_manifest_sha256,
    canonical_json_bytes,
    verify_manifest_sha256,
)


DUAL_CARD_SHA256 = "b99352c415d9aa041f90d6d8c5b7cffd3771c549b348db36f10740998ec23612"
PARENT_CARD_SHA256 = "5b5cb333d0bb191ab8da1db23b7d4b36f9f4e9ccc97c16dc46c5e7aeb2ca4825"
SELECTOR_FREEZE_SHA256 = "8e0737d68cdf3c4cdb59a6f28df1e25932a156d0c944c2caa2e33ce63f8b2eef"
EXPECTED_STREAM_SHA256 = "4563c9c8f7ef4d8abfcefaf7eedf23de2ccd9b44eb850bb95214ae8ea04653b8"

STRICT_JSON = "strict_consensus_98_table.json"
STRICT_CSV = "strict_consensus_98_table.csv"
EDGE_JSON = "edge_aware_168_table.json"
EDGE_CSV = "edge_aware_168_table.csv"
RANKINGS_JSON = "dual_variable_width_rankings.json"
VERIFIER_JSON = "dual_variable_width_verifier_receipt.json"

STRICT_NAMES = ("O_H", "O_K", "F_H", "F_K")
EDGE_NAMES = ("A_H", "A_K")
COMMON_CSV_FIELDS = (
    "width",
    "start",
    "window",
    "E_raw",
    "K_raw",
    "CEdrop_raw",
    "E_rate",
    "K_rate",
    "CEdrop_rate",
    "score",
    "rank",
    "point_eligible",
    "eligibility_failures",
)
STRICT_CSV_FIELDS = COMMON_CSV_FIELDS + (
    "comparison_starts",
    "O_H",
    "O_K",
    "F_H",
    "F_K",
    "z_O_H",
    "z_O_K",
    "z_F_H",
    "z_F_K",
    "O_H_se",
    "O_K_se",
    "F_H_se",
    "F_K_se",
    "O_H_ci95",
    "O_K_ci95",
    "F_H_ci95",
    "F_K_ci95",
)
EDGE_CSV_FIELDS = COMMON_CSV_FIELDS + (
    "reference_starts",
    "reference_count",
    "A_H",
    "A_K",
    "z_A_H",
    "z_A_K",
    "A_H_se",
    "A_K_se",
    "A_H_ci95",
    "A_K_ci95",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strict_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SchemaError("cannot load strict JSON: %s" % path) from exc
    if not isinstance(value, dict):
        raise SchemaError("JSON artifact must be an object: %s" % path)
    return value


def load_strict_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise SchemaError("blank JSONL line at %s:%d" % (path, line_number))
                value = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(value, dict):
                    raise SchemaError("JSONL row is not an object at %s:%d" % (path, line_number))
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, SchemaError):
            raise
        raise SchemaError("cannot load strict JSONL: %s" % path) from exc
    if not records:
        raise SchemaError("trajectory JSONL is empty")
    return records


def load_dual_card(path: Path) -> Dict[str, Any]:
    if file_sha256(path) != DUAL_CARD_SHA256:
        raise SchemaError("variable-width dual card SHA256 mismatch")
    card = load_strict_json(path)
    if card.get("schema_version") != "loopscope.phase3.variable_width_dual_card.v1":
        raise SchemaError("unsupported variable-width dual card")
    if card.get("status") != "PLANNING_FROZEN":
        raise SchemaError("variable-width dual card is not planning-frozen")
    if card.get("parent", {}).get("phase3_card_sha256") != PARENT_CARD_SHA256:
        raise SchemaError("parent Phase 3 card binding differs")
    if card.get("parent", {}).get("width4_selector_freeze_sha256") != SELECTOR_FREEZE_SHA256:
        raise SchemaError("selector-freeze binding differs")
    if card.get("universe", {}).get("widths") != [2, 3, 4, 5, 6, 7, 8]:
        raise SchemaError("variable-width universe differs")
    if card.get("universe", {}).get("candidate_count") != 168:
        raise SchemaError("variable-width candidate count differs")
    if card.get("strict_consensus_98", {}).get("scored_count") != 98:
        raise SchemaError("strict CONSENSUS count differs")
    if card.get("bootstrap", {}).get("expected_stream_sha256") != EXPECTED_STREAM_SHA256:
        raise SchemaError("bootstrap stream binding differs")
    if card.get("information_boundary", {}).get("outcome_values_consumed") is not False:
        raise SchemaError("card does not preserve the outcome boundary")
    return card


def extract_boundary_metric_records(
    trajectory_records: Sequence[Mapping[str, Any]],
    parent_card: Mapping[str, Any],
    *,
    enforce_population: bool = True,
) -> List[Dict[str, Any]]:
    """Validate raw trajectories and retain only selector-safe H/D boundaries."""

    normalized = [validate_trajectory_record(record, parent_card) for record in trajectory_records]
    normalized.sort(key=canonical_record_key)
    identities = [
        (row["identity"]["task"], row["identity"]["doc_id"], row["identity"]["doc_hash"])
        for row in normalized
    ]
    if len(set(identities)) != len(identities):
        raise SchemaError("trajectory records contain duplicate identities")
    subject_count = len({row["subject"] for row in normalized})
    if enforce_population:
        if len(normalized) != 1531:
            raise SchemaError("production variable-width analysis requires 1,531 records")
        if subject_count != 57:
            raise SchemaError("production variable-width analysis requires 57 subjects")
    return [
        {
            "identity": dict(row["identity"]),
            "subject": row["subject"],
            "H": [float(item["choice_entropy"]) for item in row["boundaries"]],
            "D": [float(item["kl_to_final"]) for item in row["boundaries"]],
        }
        for row in normalized
    ]


def analyze_boundary_metric_records(
    records: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    *,
    replicates: Optional[int] = None,
    seed: Optional[int] = None,
    enforce_population: bool = True,
) -> Dict[str, Any]:
    """Compute both frozen rankings from one boundary-first bootstrap stream."""

    normalized = _normalize_boundary_records(records)
    if not normalized:
        raise SchemaError("variable-width analysis requires records")
    if enforce_population:
        if len(normalized) != int(card["trajectory_source"]["sample_count"]):
            raise SchemaError("production variable-width analysis requires 1,531 records")
        if len({row["subject"] for row in normalized}) != int(
            card["trajectory_source"]["subject_count"]
        ):
            raise SchemaError("production variable-width analysis requires 57 subjects")
    configured_replicates = int(card["bootstrap"]["replicates"])
    configured_seed = int(card["bootstrap"]["seed"])
    if enforce_population and replicates not in (None, configured_replicates):
        raise SchemaError("production bootstrap replicate count cannot be overridden")
    if enforce_population and seed not in (None, configured_seed):
        raise SchemaError("production bootstrap seed cannot be overridden")
    selected_replicates = configured_replicates if replicates is None else _positive_int(replicates)
    selected_seed = configured_seed if seed is None else _strict_int(seed)
    if selected_replicates < 2:
        raise SchemaError("bootstrap requires at least two replicates")

    h_rows = [row["H"] for row in normalized]
    d_rows = [row["D"] for row in normalized]
    point_h = _column_means(h_rows)
    point_d = _column_means(d_rows)

    bootstrap_h: List[List[float]] = []
    bootstrap_d: List[List[float]] = []
    index_digest = hashlib.sha256()
    for indices in iter_subject_bootstrap_indices(
        normalized, replicates=selected_replicates, seed=selected_seed
    ):
        _update_index_digest(index_digest, indices)
        replicate_h, replicate_d = _resampled_boundary_means(h_rows, d_rows, indices)
        bootstrap_h.append(replicate_h)
        bootstrap_d.append(replicate_d)
    stream_sha = index_digest.hexdigest()
    if enforce_population and stream_sha != str(card["bootstrap"]["expected_stream_sha256"]):
        raise SchemaError("bootstrap index stream SHA256 differs")

    widths = [int(value) for value in card["universe"]["widths"]]
    points: Dict[Tuple[int, int], Dict[str, float]] = {}
    boot_e: Dict[Tuple[int, int], List[float]] = {}
    boot_k: Dict[Tuple[int, int], List[float]] = {}
    for width in widths:
        for start in range(29 - width):
            key = (width, start)
            e_raw = point_h[start] - point_h[start + width]
            k_raw = point_d[start + width] - point_d[start]
            points[key] = {
                "E_raw": e_raw,
                "K_raw": k_raw,
                "CEdrop_raw": e_raw - k_raw,
                "E_rate": e_raw / width,
                "K_rate": k_raw / width,
                "CEdrop_rate": (e_raw - k_raw) / width,
            }
            boot_e[key] = [
                (row[start] - row[start + width]) / width for row in bootstrap_h
            ]
            boot_k[key] = [
                (row[start + width] - row[start]) / width for row in bootstrap_d
            ]

    strict_rows, strict_boot = _strict_rows(widths, points, boot_e, boot_k)
    edge_rows, edge_boot = _edge_rows(widths, points, boot_e, boot_k)
    _assign_ranks(strict_rows)
    _assign_ranks(edge_rows)
    strict_summary = _selection_summary(strict_rows, strict_boot, STRICT_NAMES, card)
    edge_summary = _selection_summary(edge_rows, edge_boot, EDGE_NAMES, card)
    width4_rows = [row for row in strict_rows if int(row["width"]) == 4]
    width4_boot = {
        key: value for key, value in strict_boot.items() if key[0] == 4
    }
    width4_summary = _selection_summary(width4_rows, width4_boot, STRICT_NAMES, card)
    return {
        "record_count": len(normalized),
        "subject_count": len({row["subject"] for row in normalized}),
        "bootstrap": {
            "method": card["bootstrap"]["method"],
            "replicates": selected_replicates,
            "seed": selected_seed,
            "index_stream_sha256": stream_sha,
            "joint_across_both_rules": True,
            "fixture_override": not (
                selected_replicates == configured_replicates and selected_seed == configured_seed
            ),
        },
        "strict_rows": strict_rows,
        "edge_rows": edge_rows,
        "strict_summary": strict_summary,
        "edge_summary": edge_summary,
        "width4_summary": width4_summary,
        "outcome_values_consumed": False,
    }


def build_artifact_payloads(
    analysis: Mapping[str, Any],
    card: Mapping[str, Any],
    *,
    git_commit: str,
    input_provenance: Mapping[str, Any],
    backward_compatibility: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str, Dict[str, Any], str, Dict[str, Any]]:
    bootstrap = dict(analysis["bootstrap"])
    common = {
        "card_sha256": DUAL_CARD_SHA256,
        "git_commit": _commit(git_commit),
        "record_count": int(analysis["record_count"]),
        "subject_count": int(analysis["subject_count"]),
        "bootstrap": bootstrap,
        "inputs": dict(input_provenance),
        "outcome_values_consumed": False,
    }
    strict = {
        "schema_version": "loopscope.phase3.strict_consensus_98_table.v1",
        "artifact_role": "original_rule_primary_strict_consensus_variable_width",
        **common,
        "row_count": len(analysis["strict_rows"]),
        "rows": list(analysis["strict_rows"]),
    }
    edge = {
        "schema_version": "loopscope.phase3.edge_aware_168_table.v1",
        "artifact_role": "exploratory_edge_aware_full_coverage_variable_width",
        **common,
        "row_count": len(analysis["edge_rows"]),
        "rows": list(analysis["edge_rows"]),
    }
    rankings = {
        "schema_version": "loopscope.phase3.dual_variable_width_rankings.v1",
        "artifact_role": "outcome_blind_dual_variable_width_rankings",
        **common,
        "strict_consensus_98": {
            **dict(analysis["strict_summary"]),
            "published_ranking": _published_ranking(analysis["strict_rows"]),
        },
        "edge_aware_168": {
            **dict(analysis["edge_summary"]),
            "published_ranking": _published_ranking(analysis["edge_rows"]),
        },
        "width4_backward_compatibility": dict(backward_compatibility),
        "cross_rule_score_comparison_performed": False,
    }
    attach_manifest_sha256(strict)
    attach_manifest_sha256(edge)
    attach_manifest_sha256(rankings)
    return (
        strict,
        render_table_csv(strict["rows"], STRICT_CSV_FIELDS),
        edge,
        render_table_csv(edge["rows"], EDGE_CSV_FIELDS),
        rankings,
    )


def verify_width4_backward_compatibility(
    strict_rows: Sequence[Mapping[str, Any]],
    width4_summary: Mapping[str, Any],
    selector_freeze: Mapping[str, Any],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    tolerance = float(
        card["strict_consensus_98"]["backward_compatibility"]["absolute_tolerance"]
    )
    try:
        old_consensus = selector_freeze["analysis"]["variants"]["CONSENSUS"]
        old_rows = old_consensus["windows"]
        old_scope = old_consensus["scopes"]["deployment"]
    except (KeyError, TypeError) as exc:
        raise SchemaError("selector freeze lacks width-4 CONSENSUS evidence") from exc
    current = {int(row["start"]): row for row in strict_rows if int(row["width"]) == 4}
    previous = {int(row["start"]): row for row in old_rows}
    if set(current) != set(range(4, 21)) or set(previous) != set(range(4, 21)):
        raise SchemaError("width-4 strict support differs from the selector freeze")
    max_difference = 0.0
    for start in range(4, 21):
        left = current[start]
        right = previous[start]
        comparisons = (
            (float(left["E_raw"]), float(right["center"]["entropy_drop"])),
            (float(left["K_raw"]), float(right["center"]["kl_rise"])),
            (float(left["CEdrop_raw"]), float(right["center"]["ce_drop"])),
            (float(left["score"]), float(right["score"])),
        )
        for name in STRICT_NAMES:
            comparisons += (
                (float(left["z_" + name]), float(right["score_components"][name])),
            )
        for observed, expected in comparisons:
            max_difference = max(max_difference, abs(observed - expected))
            if abs(observed - expected) > tolerance:
                raise SchemaError("width-4 backward compatibility numeric mismatch")
        if bool(left["point_eligible"]) is not bool(right["point_eligible"]):
            raise SchemaError("width-4 eligibility differs from selector freeze")
    eligible = [row["window"] for row in current.values() if row["point_eligible"]]
    required = card["strict_consensus_98"]["backward_compatibility"][
        "required_width4_point_eligible"
    ]
    if eligible != required:
        raise SchemaError("width-4 eligible set differs")
    score_12 = float(current[12]["score"])
    expected_score = float(
        card["strict_consensus_98"]["backward_compatibility"]["reference_12_15_score"]
    )
    if abs(score_12 - expected_score) > tolerance:
        raise SchemaError("12:15 score differs from the frozen reference")
    frequency = width4_summary.get("selection_frequency")
    old_frequency = old_scope.get("selection_frequency")
    required_frequency = float(
        card["strict_consensus_98"]["backward_compatibility"][
            "required_width4_selection_frequency"
        ]
    )
    if frequency is None or old_frequency is None:
        raise SchemaError("width-4 selection frequency is absent")
    if abs(float(frequency) - required_frequency) > tolerance or abs(
        float(old_frequency) - required_frequency
    ) > tolerance:
        raise SchemaError("width-4 selection frequency differs")
    return {
        "status": "WIDTH4_EXACT_REPRODUCTION_PASS",
        "row_count": 17,
        "max_absolute_difference": max_difference,
        "score_12_15": score_12,
        "point_eligible_windows": eligible,
        "selection_frequency": float(frequency),
        "absolute_tolerance": tolerance,
    }


def verify_recomputed_artifacts(
    analysis: Mapping[str, Any],
    card: Mapping[str, Any],
    selector_freeze: Mapping[str, Any],
    *,
    git_commit: str,
    input_provenance: Mapping[str, Any],
    strict_payload: Mapping[str, Any],
    strict_csv: str,
    edge_payload: Mapping[str, Any],
    edge_csv: str,
    rankings_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    compatibility = verify_width4_backward_compatibility(
        analysis["strict_rows"], analysis["width4_summary"], selector_freeze, card
    )
    expected = build_artifact_payloads(
        analysis,
        card,
        git_commit=git_commit,
        input_provenance=input_provenance,
        backward_compatibility=compatibility,
    )
    observed = (strict_payload, strict_csv, edge_payload, edge_csv, rankings_payload)
    for index, (left, right) in enumerate(zip(observed, expected)):
        if isinstance(left, str):
            matches = left == right
        else:
            try:
                matches = canonical_json_bytes(left) == canonical_json_bytes(right)
            except (TypeError, ValueError) as exc:
                raise SchemaError("artifact contains non-canonical numeric data") from exc
        if not matches:
            raise SchemaError("dual variable-width artifact %d differs from raw recomputation" % index)
    return compatibility


def render_table_csv(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return output.getvalue()


def write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError("refusing to overwrite artifact: %s" % path)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def write_new_text(path: Path, text: str) -> str:
    if path.exists():
        raise FileExistsError("refusing to overwrite artifact: %s" % path)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def _strict_rows(
    widths: Sequence[int],
    points: Mapping[Tuple[int, int], Mapping[str, float]],
    boot_e: Mapping[Tuple[int, int], Sequence[float]],
    boot_k: Mapping[Tuple[int, int], Sequence[float]],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], Dict[str, List[float]]]]:
    rows: List[Dict[str, Any]] = []
    all_boot: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    for width in widths:
        for start in range(width, 29 - 2 * width):
            key = (width, start)
            comparisons = (start - 1, start + 1, start - width, start + width)
            point = {
                "O_H": points[key]["E_rate"]
                - 0.5
                * (points[(width, start - 1)]["E_rate"] + points[(width, start + 1)]["E_rate"]),
                "O_K": points[key]["K_rate"]
                - 0.5
                * (points[(width, start - 1)]["K_rate"] + points[(width, start + 1)]["K_rate"]),
                "F_H": points[key]["E_rate"]
                - 0.5
                * (points[(width, start - width)]["E_rate"] + points[(width, start + width)]["E_rate"]),
                "F_K": points[key]["K_rate"]
                - 0.5
                * (points[(width, start - width)]["K_rate"] + points[(width, start + width)]["K_rate"]),
            }
            count = len(boot_e[key])
            boot = {
                "O_H": [
                    boot_e[key][i]
                    - 0.5
                    * (boot_e[(width, start - 1)][i] + boot_e[(width, start + 1)][i])
                    for i in range(count)
                ],
                "O_K": [
                    boot_k[key][i]
                    - 0.5
                    * (boot_k[(width, start - 1)][i] + boot_k[(width, start + 1)][i])
                    for i in range(count)
                ],
                "F_H": [
                    boot_e[key][i]
                    - 0.5
                    * (boot_e[(width, start - width)][i] + boot_e[(width, start + width)][i])
                    for i in range(count)
                ],
                "F_K": [
                    boot_k[key][i]
                    - 0.5
                    * (boot_k[(width, start - width)][i] + boot_k[(width, start + width)][i])
                    for i in range(count)
                ],
            }
            summaries = {name: bootstrap_summary(point[name], boot[name]) for name in STRICT_NAMES}
            failures = _eligibility_failures(
                key, comparisons, points, summaries, STRICT_NAMES
            )
            row = _base_row(width, start, points[key], failures)
            row["comparison_starts"] = list(comparisons)
            _add_contrasts(row, summaries, STRICT_NAMES)
            row["score"] = min(float(row["z_" + name]) for name in STRICT_NAMES)
            rows.append(row)
            all_boot[key] = boot
    return rows, all_boot


def _edge_rows(
    widths: Sequence[int],
    points: Mapping[Tuple[int, int], Mapping[str, float]],
    boot_e: Mapping[Tuple[int, int], Sequence[float]],
    boot_k: Mapping[Tuple[int, int], Sequence[float]],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, int], Dict[str, List[float]]]]:
    rows: List[Dict[str, Any]] = []
    all_boot: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    for width in widths:
        last = 28 - width
        for start in range(last + 1):
            key = (width, start)
            references = sorted(
                {value for value in (start - 1, start + 1, start - width, start + width) if 0 <= value <= last}
            )
            if not 2 <= len(references) <= 4:
                raise SchemaError("edge-aware reference count lies outside 2..4")
            point = {
                "A_H": points[key]["E_rate"]
                - sum(points[(width, value)]["E_rate"] for value in references) / len(references),
                "A_K": points[key]["K_rate"]
                - sum(points[(width, value)]["K_rate"] for value in references) / len(references),
            }
            count = len(boot_e[key])
            boot = {
                "A_H": [
                    boot_e[key][i]
                    - sum(boot_e[(width, value)][i] for value in references) / len(references)
                    for i in range(count)
                ],
                "A_K": [
                    boot_k[key][i]
                    - sum(boot_k[(width, value)][i] for value in references) / len(references)
                    for i in range(count)
                ],
            }
            summaries = {name: bootstrap_summary(point[name], boot[name]) for name in EDGE_NAMES}
            failures = _eligibility_failures(key, references, points, summaries, EDGE_NAMES)
            row = _base_row(width, start, points[key], failures)
            row["reference_starts"] = references
            row["reference_count"] = len(references)
            _add_contrasts(row, summaries, EDGE_NAMES)
            row["score"] = min(float(row["z_" + name]) for name in EDGE_NAMES)
            rows.append(row)
            all_boot[key] = boot
    return rows, all_boot


def _eligibility_failures(
    key: Tuple[int, int],
    references: Iterable[int],
    points: Mapping[Tuple[int, int], Mapping[str, float]],
    summaries: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
) -> List[str]:
    width, _start = key
    failures: List[str] = []
    if not points[key]["E_rate"] > 0.0:
        failures.append("center_E_rate_not_strictly_positive")
    if not points[key]["K_rate"] > 0.0:
        failures.append("center_K_rate_not_strictly_positive")
    for reference in references:
        if not points[(width, reference)]["E_rate"] < 0.0:
            failures.append("reference_%d_E_rate_not_strictly_negative" % reference)
        if not points[(width, reference)]["K_rate"] < 0.0:
            failures.append("reference_%d_K_rate_not_strictly_negative" % reference)
    for name in names:
        if not float(summaries[name]["percentile_95_ci"][0]) > 0.0:
            failures.append("%s_ci_lower_not_strictly_positive" % name)
    return failures


def _base_row(
    width: int, start: int, metrics: Mapping[str, float], failures: Sequence[str]
) -> Dict[str, Any]:
    return {
        "width": width,
        "start": start,
        "window": "%d:%d" % (start, start + width - 1),
        **{key: float(value) for key, value in metrics.items()},
        "score": None,
        "rank": None,
        "point_eligible": not failures,
        "eligibility_failures": list(failures),
    }


def _add_contrasts(
    row: Dict[str, Any], summaries: Mapping[str, Mapping[str, Any]], names: Sequence[str]
) -> None:
    for name in names:
        point = float(summaries[name]["point"])
        se = float(summaries[name]["standard_error"])
        row[name] = point
        row["z_" + name] = point / max(se, 1e-12)
        row[name + "_se"] = se
        row[name + "_ci95"] = [float(value) for value in summaries[name]["percentile_95_ci"]]


def _assign_ranks(rows: Sequence[Dict[str, Any]]) -> None:
    ranked = sorted(rows, key=lambda row: (-float(row["score"]), int(row["width"]), int(row["start"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank


def _selection_summary(
    rows: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[Tuple[int, int], Mapping[str, Sequence[float]]],
    names: Sequence[str],
    card: Mapping[str, Any],
) -> Dict[str, Any]:
    ranked = sorted(rows, key=lambda row: int(row["rank"]))
    eligible = [row for row in ranked if bool(row["point_eligible"])]
    base: Dict[str, Any] = {
        "row_count": len(rows),
        "score_top1": _rank_entry(ranked[0]) if ranked else None,
        "top10": [_rank_entry(row) for row in ranked[:10]],
        "point_eligible_pairs": [_pair(row) for row in eligible],
        "eligible_top1": None,
        "selection_frequency": None,
        "window_decision": "ABSTAIN_NO_POINT_ELIGIBLE",
    }
    if not eligible:
        return base
    point_ranked = sorted(
        eligible,
        key=lambda row: (-float(row["score"]), int(row["width"]), int(row["start"])),
    )
    top = point_ranked[0]
    if len(point_ranked) > 1 and math.isclose(
        float(top["score"]),
        float(point_ranked[1]["score"]),
        rel_tol=float(card["ranking_outputs"]["point_top_tie_rel_tolerance"]),
        abs_tol=float(card["ranking_outputs"]["point_top_tie_abs_tolerance"]),
    ):
        base["window_decision"] = "ABSTAIN_NO_UNIQUE_TOP1"
        return base
    top_key = (int(top["width"]), int(top["start"]))
    replicate_count = len(next(iter(bootstrap[top_key].values())))
    wins = 0
    for replicate in range(replicate_count):
        replicate_scores: Dict[Tuple[int, int], float] = {}
        for row in eligible:
            key = (int(row["width"]), int(row["start"]))
            replicate_scores[key] = min(
                float(bootstrap[key][name][replicate]) / max(float(row[name + "_se"]), 1e-12)
                for name in names
            )
        winner = sorted(
            replicate_scores,
            key=lambda key: (-replicate_scores[key], key[0], key[1]),
        )[0]
        if winner == top_key:
            wins += 1
    frequency = wins / replicate_count
    threshold = float(card["ranking_outputs"]["selection_frequency_threshold"])
    base.update(
        {
            "eligible_top1": _rank_entry(top),
            "selection_frequency": frequency,
            "window_decision": "SELECTED"
            if frequency >= threshold
            else "ABSTAIN_LOW_SELECTION_FREQUENCY",
        }
    )
    return base


def _published_ranking(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [_rank_entry(row) for row in sorted(rows, key=lambda row: int(row["rank"]))]


def _rank_entry(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "rank": int(row["rank"]),
        "width": int(row["width"]),
        "start": int(row["start"]),
        "window": str(row["window"]),
        "score": float(row["score"]),
        "point_eligible": bool(row["point_eligible"]),
    }


def _pair(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "width": int(row["width"]),
        "start": int(row["start"]),
        "window": str(row["window"]),
        "score": float(row["score"]),
    }


def _normalize_boundary_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for record in records:
        if set(record) != {"identity", "subject", "H", "D"}:
            raise SchemaError("boundary metric record fields differ")
        h_values = _finite_vector(record["H"], "H")
        d_values = _finite_vector(record["D"], "D")
        if len(h_values) != 29 or len(d_values) != 29:
            raise SchemaError("boundary metric records require 29 H/D values")
        normalized.append(
            {
                "identity": dict(record["identity"]),
                "subject": str(record["subject"]),
                "H": h_values,
                "D": d_values,
            }
        )
    normalized.sort(key=canonical_record_key)
    identities = [
        (row["identity"]["task"], row["identity"]["doc_id"], row["identity"]["doc_hash"])
        for row in normalized
    ]
    if len(set(identities)) != len(identities):
        raise SchemaError("boundary metric records contain duplicate identities")
    return normalized


def _column_means(rows: Sequence[Sequence[float]]) -> List[float]:
    return [sum(row[index] for row in rows) / len(rows) for index in range(29)]


def _resampled_boundary_means(
    h_rows: Sequence[Sequence[float]],
    d_rows: Sequence[Sequence[float]],
    indices: Sequence[int],
) -> Tuple[List[float], List[float]]:
    h_sum = [0.0] * 29
    d_sum = [0.0] * 29
    for index in indices:
        h_row = h_rows[index]
        d_row = d_rows[index]
        for boundary in range(29):
            h_sum[boundary] += h_row[boundary]
            d_sum[boundary] += d_row[boundary]
    denominator = len(indices)
    return (
        [value / denominator for value in h_sum],
        [value / denominator for value in d_sum],
    )


def _update_index_digest(digest: Any, indices: Sequence[int]) -> None:
    digest.update(struct.pack(">I", len(indices)))
    for index in indices:
        digest.update(struct.pack(">I", int(index)))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _finite_vector(value: Any, name: str) -> List[float]:
    if not isinstance(value, list):
        raise SchemaError("%s must be a list" % name)
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SchemaError("%s contains a non-number" % name)
        number = float(item)
        if not math.isfinite(number):
            raise SchemaError("%s contains a non-finite number" % name)
        result.append(number)
    return result


def _strict_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError("expected integer")
    return int(value)


def _positive_int(value: Any) -> int:
    result = _strict_int(value)
    if result < 1:
        raise SchemaError("expected positive integer")
    return result


def _commit(value: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise SchemaError("expected exact lowercase 40-hex commit")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


__all__ = [
    "DUAL_CARD_SHA256",
    "EDGE_CSV",
    "EDGE_JSON",
    "RANKINGS_JSON",
    "STRICT_CSV",
    "STRICT_JSON",
    "VERIFIER_JSON",
    "analyze_boundary_metric_records",
    "build_artifact_payloads",
    "extract_boundary_metric_records",
    "file_sha256",
    "load_dual_card",
    "load_strict_json",
    "load_strict_jsonl",
    "render_table_csv",
    "verify_recomputed_artifacts",
    "verify_width4_backward_compatibility",
    "write_new_json",
    "write_new_text",
]
