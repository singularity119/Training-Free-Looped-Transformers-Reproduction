"""Outcome-blind Phase 5 Extension Gate E multi-width scoring.

This sidecar consumes only the already-closed validation-1531 native no-loop
trajectory and the existing width-4 selector report.  It never imports model,
dataset, evaluator, outcome, or GPU code.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from tflt.loopscope.phase5_schema import (
    Phase5ContractError,
    canonical_json_bytes,
    semantic_sha256,
    validate_trajectory_record,
)


WIDTHS = (3, 4, 5, 6)
CANDIDATE_STARTS = {
    3: tuple(range(11, 23)),
    4: tuple(range(11, 22)),
    5: tuple(range(11, 21)),
    6: tuple(range(11, 20)),
}
CONTRASTS = ("OH", "OK", "FH", "FK")
OLD_CONTRAST_NAMES = {"OH": "O_H", "OK": "O_K", "FH": "F_H", "FK": "F_K"}
EXPECTED_RECORD_COUNT = 1531
EXPECTED_SUBJECT_COUNT = 57
EXPECTED_BOUNDARY_COUNT = 37
EXPECTED_CANDIDATE_COUNT = 42
EXPECTED_REPLICATES = 2000
EXPECTED_SEED = 20260722

SCORES_JSON = "phase5_multiwidth_mid40_scores.json"
SCORES_CSV = "phase5_multiwidth_mid40_scores.csv"
SUMMARY_JSON = "phase5_multiwidth_mid40_summary.json"
SUMMARY_ZH = "phase5_multiwidth_mid40_summary_zh.md"
VERIFIER_RECEIPT = "phase5_multiwidth_mid40_verifier_receipt.json"
MANIFEST_RECEIPT = "phase5_multiwidth_mid40_manifest_receipt.json"

CSV_FIELDS = (
    "global_rank",
    "width",
    "width_rank",
    "start",
    "window",
    "boundary_transition",
    "E_raw",
    "K_raw",
    "CEdrop_raw",
    "E_rate",
    "K_rate",
    "CEdrop_rate",
    "O_H",
    "O_K",
    "F_H",
    "F_K",
    "z_OH",
    "z_OK",
    "z_FH",
    "z_FK",
    "score",
    "point_eligible",
    "eligibility_failures",
    "comparison_starts",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strict_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise Phase5ContractError("cannot load strict JSON: %s" % path) from exc
    if not isinstance(value, dict):
        raise Phase5ContractError("JSON artifact must be an object: %s" % path)
    return value


def load_strict_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Phase5ContractError("blank JSONL line at %s:%d" % (path, line_number))
                value = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(value, dict):
                    raise Phase5ContractError(
                        "JSONL row is not an object at %s:%d" % (path, line_number)
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, Phase5ContractError):
            raise
        raise Phase5ContractError("cannot load strict JSONL: %s" % path) from exc
    if not rows:
        raise Phase5ContractError("trajectory JSONL is empty")
    return rows


def validate_extension_card(card: Mapping[str, Any]) -> None:
    if card.get("schema_version") != "loopscope.phase5.multiwidth-mid40-card.v1":
        raise Phase5ContractError("unsupported Gate E card")
    if card.get("status") != "PLANNING_FROZEN_POST_TERMINAL_EXPLORATORY":
        raise Phase5ContractError("Gate E card is not planning-frozen")
    universe = card.get("candidate_universe", {})
    if universe.get("widths") != list(WIDTHS):
        raise Phase5ContractError("Gate E widths differ")
    if universe.get("candidate_count") != EXPECTED_CANDIDATE_COUNT:
        raise Phase5ContractError("Gate E candidate count differs")
    observed = universe.get("starts_by_width", {})
    for width, starts in CANDIDATE_STARTS.items():
        if observed.get(str(width)) != [starts[0], starts[-1]]:
            raise Phase5ContractError("Gate E start range differs for width %d" % width)
    bootstrap = card.get("bootstrap", {})
    if (
        bootstrap.get("replicates") != EXPECTED_REPLICATES
        or bootstrap.get("seed") != EXPECTED_SEED
        or bootstrap.get("standard_error_ddof") != 1
        or bootstrap.get("shared_draw_stream_across_all_widths_starts_metrics") is not True
    ):
        raise Phase5ContractError("Gate E bootstrap contract differs")
    boundary = card.get("information_boundary", {})
    required_false = (
        "outcome_values_consumed",
        "test_split_consumed",
        "geometry_fields_consumed",
        "model_or_data_loaded",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise Phase5ContractError("Gate E information boundary differs")
    if (
        boundary.get("post_terminal_exploratory_no_selection") is not True
        or boundary.get("selected_window_field_forbidden") is not True
    ):
        raise Phase5ContractError("Gate E no-selection boundary differs")


def extract_boundary_samples(
    records: Sequence[Mapping[str, Any]], *, enforce_population: bool = True
) -> List[Dict[str, Any]]:
    """Validate Phase 5 trajectory records and retain only H/D scalars."""

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        validate_trajectory_record(record)
        identity = canonical_json_bytes(record["identity"]).decode("utf-8")
        if identity in seen:
            raise Phase5ContractError("trajectory identities must be unique")
        seen.add(identity)
        boundaries = record["boundaries"]
        normalized.append(
            {
                "identity": identity,
                "subject": str(record["subject"]),
                "H": [float(row["choice_entropy"]) for row in boundaries],
                "D": [float(row["kl_to_final"]) for row in boundaries],
            }
        )
    normalized.sort(key=lambda row: (row["subject"], row["identity"]))
    return normalize_boundary_samples(normalized, enforce_population=enforce_population)


def normalize_boundary_samples(
    samples: Sequence[Mapping[str, Any]], *, enforce_population: bool = True
) -> List[Dict[str, Any]]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence) or not samples:
        raise Phase5ContractError("Gate E requires boundary samples")
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for index, sample in enumerate(samples):
        if set(sample) != {"identity", "subject", "H", "D"}:
            raise Phase5ContractError("boundary sample keys differ")
        identity = str(sample["identity"] or "sample-%d" % index)
        subject = str(sample["subject"] or "").strip()
        if not subject or identity in seen:
            raise Phase5ContractError("boundary sample identity/subject closure failed")
        seen.add(identity)
        h_values = [_finite(value, "H") for value in sample["H"]]
        d_values = [_finite(value, "D") for value in sample["D"]]
        if len(h_values) != EXPECTED_BOUNDARY_COUNT or len(d_values) != EXPECTED_BOUNDARY_COUNT:
            raise Phase5ContractError("boundary samples must contain B0...B36")
        normalized.append(
            {"identity": identity, "subject": subject, "H": h_values, "D": d_values}
        )
    normalized.sort(key=lambda row: (row["subject"], row["identity"]))
    if enforce_population:
        if len(normalized) != EXPECTED_RECORD_COUNT:
            raise Phase5ContractError("formal Gate E requires 1,531 records")
        if len({row["subject"] for row in normalized}) != EXPECTED_SUBJECT_COUNT:
            raise Phase5ContractError("formal Gate E requires 57 subjects")
    return normalized


def required_reference_starts(width: int) -> Tuple[int, ...]:
    values = set()
    last_legal_start = 36 - width
    for start in CANDIDATE_STARTS[width]:
        references = (start - 1, start + 1, start - width, start + width)
        if any(value < 0 or value > last_legal_start for value in references):
            raise Phase5ContractError("Gate E reference start is not legal")
        values.add(start)
        values.update(references)
    return tuple(sorted(values))


def analyze_boundary_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    replicates: int = EXPECTED_REPLICATES,
    seed: int = EXPECTED_SEED,
    enforce_population: bool = True,
) -> Dict[str, Any]:
    """Compute the frozen 42-row score-only ranking from boundary scalars."""

    normalized = normalize_boundary_samples(samples, enforce_population=enforce_population)
    if enforce_population and (replicates != EXPECTED_REPLICATES or seed != EXPECTED_SEED):
        raise Phase5ContractError("formal Gate E requires R=2000 seed=20260722")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise Phase5ContractError("bootstrap replicates must be an integer >=2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Phase5ContractError("bootstrap seed must be an integer")

    required_keys = [
        (width, start)
        for width in WIDTHS
        for start in required_reference_starts(width)
    ]
    per_sample_e: Dict[Tuple[int, int], List[float]] = {}
    per_sample_k: Dict[Tuple[int, int], List[float]] = {}
    for width, start in required_keys:
        per_sample_e[(width, start)] = [
            (row["H"][start] - row["H"][start + width]) / width for row in normalized
        ]
        per_sample_k[(width, start)] = [
            (row["D"][start + width] - row["D"][start]) / width for row in normalized
        ]
    sample_count = len(normalized)
    point_e = {
        key: math.fsum(values) / sample_count for key, values in per_sample_e.items()
    }
    point_k = {
        key: math.fsum(values) / sample_count for key, values in per_sample_k.items()
    }

    groups: Dict[str, List[int]] = {}
    for index, row in enumerate(normalized):
        groups.setdefault(row["subject"], []).append(index)
    rng = random.Random(seed)
    bootstrap_e = {key: [] for key in required_keys}
    bootstrap_k = {key: [] for key in required_keys}
    draw_digest = hashlib.sha256()
    draw_digest.update(b"[")
    for replicate_index in range(replicates):
        drawn: List[int] = []
        for subject in sorted(groups):
            indices = groups[subject]
            drawn.extend(indices[rng.randrange(len(indices))] for _ in indices)
        if replicate_index:
            draw_digest.update(b",")
        draw_digest.update(canonical_json_bytes(drawn))
        for key in required_keys:
            bootstrap_e[key].append(
                math.fsum(per_sample_e[key][index] for index in drawn) / sample_count
            )
            bootstrap_k[key].append(
                math.fsum(per_sample_k[key][index] for index in drawn) / sample_count
            )
    draw_digest.update(b"]")

    rows: List[Dict[str, Any]] = []
    for width in WIDTHS:
        for start in CANDIDATE_STARTS[width]:
            key = (width, start)
            comparisons = (start - 1, start + 1, start - width, start + width)
            points = {
                "OH": point_e[key]
                - 0.5 * (point_e[(width, start - 1)] + point_e[(width, start + 1)]),
                "OK": point_k[key]
                - 0.5 * (point_k[(width, start - 1)] + point_k[(width, start + 1)]),
                "FH": point_e[key]
                - 0.5 * (point_e[(width, start - width)] + point_e[(width, start + width)]),
                "FK": point_k[key]
                - 0.5 * (point_k[(width, start - width)] + point_k[(width, start + width)]),
            }
            replicate_values = {
                "OH": [
                    bootstrap_e[key][i]
                    - 0.5
                    * (
                        bootstrap_e[(width, start - 1)][i]
                        + bootstrap_e[(width, start + 1)][i]
                    )
                    for i in range(replicates)
                ],
                "OK": [
                    bootstrap_k[key][i]
                    - 0.5
                    * (
                        bootstrap_k[(width, start - 1)][i]
                        + bootstrap_k[(width, start + 1)][i]
                    )
                    for i in range(replicates)
                ],
                "FH": [
                    bootstrap_e[key][i]
                    - 0.5
                    * (
                        bootstrap_e[(width, start - width)][i]
                        + bootstrap_e[(width, start + width)][i]
                    )
                    for i in range(replicates)
                ],
                "FK": [
                    bootstrap_k[key][i]
                    - 0.5
                    * (
                        bootstrap_k[(width, start - width)][i]
                        + bootstrap_k[(width, start + width)][i]
                    )
                    for i in range(replicates)
                ],
            }
            summaries = {
                name: _bootstrap_summary(points[name], replicate_values[name])
                for name in CONTRASTS
            }
            failures = _eligibility_failures(
                width, start, comparisons, point_e, point_k, summaries
            )
            e_rate = point_e[key]
            k_rate = point_k[key]
            row: Dict[str, Any] = {
                "global_rank": None,
                "width": width,
                "width_rank": None,
                "start": start,
                "window": "%d:%d" % (start, start + width - 1),
                "boundary_transition": "B_%d->B_%d" % (start, start + width),
                "E_raw": e_rate * width,
                "K_raw": k_rate * width,
                "CEdrop_raw": (e_rate - k_rate) * width,
                "E_rate": e_rate,
                "K_rate": k_rate,
                "CEdrop_rate": e_rate - k_rate,
                "O_H": points["OH"],
                "O_K": points["OK"],
                "F_H": points["FH"],
                "F_K": points["FK"],
                "z_OH": summaries["OH"]["z"],
                "z_OK": summaries["OK"]["z"],
                "z_FH": summaries["FH"]["z"],
                "z_FK": summaries["FK"]["z"],
                "score": min(summaries[name]["z"] for name in CONTRASTS),
                "point_eligible": not failures,
                "eligibility_failures": failures,
                "comparison_starts": {
                    "s_minus_1": comparisons[0],
                    "s_plus_1": comparisons[1],
                    "s_minus_width": comparisons[2],
                    "s_plus_width": comparisons[3],
                },
                "contrast_standard_errors": {
                    "O_H": summaries["OH"]["standard_error"],
                    "O_K": summaries["OK"]["standard_error"],
                    "F_H": summaries["FH"]["standard_error"],
                    "F_K": summaries["FK"]["standard_error"],
                },
                "contrast_ci95": {
                    "O_H": summaries["OH"]["ci95"],
                    "O_K": summaries["OK"]["ci95"],
                    "F_H": summaries["FH"]["ci95"],
                    "F_K": summaries["FK"]["ci95"],
                },
            }
            rows.append(row)

    ranked = sorted(
        rows, key=lambda row: (-float(row["score"]), int(row["width"]), int(row["start"]))
    )
    for rank, row in enumerate(ranked, start=1):
        row["global_rank"] = rank
    for width in WIDTHS:
        width_rows = sorted(
            (row for row in ranked if int(row["width"]) == width),
            key=lambda row: (-float(row["score"]), int(row["start"])),
        )
        for rank, row in enumerate(width_rows, start=1):
            row["width_rank"] = rank
    if len(ranked) != EXPECTED_CANDIDATE_COUNT:
        raise Phase5ContractError("published Gate E ranking must contain exactly 42 rows")
    return {
        "rows": ranked,
        "record_count": len(normalized),
        "subject_count": len(groups),
        "candidate_count": len(ranked),
        "candidate_counts_by_width": {
            str(width): len(CANDIDATE_STARTS[width]) for width in WIDTHS
        },
        "bootstrap": {
            "method": "joint_subject_stratified_sample_bootstrap",
            "replicates": replicates,
            "seed": seed,
            "ddof": 1,
            "draw_index_sha256": draw_digest.hexdigest(),
            "shared_across_all_widths_starts_metrics": True,
        },
        "outcome_values_consumed": False,
        "test_split_consumed": False,
        "geometry_fields_consumed": False,
        "post_terminal_exploratory_no_selection": True,
    }


def verify_width4_backward_compatibility(
    rows: Sequence[Mapping[str, Any]],
    selector_report: Mapping[str, Any],
    *,
    tolerance: float = 1e-10,
) -> Dict[str, Any]:
    current = {
        int(row["start"]): row for row in rows if int(row["width"]) == 4
    }
    previous_rows = selector_report.get("published_ranking")
    if not isinstance(previous_rows, list):
        raise Phase5ContractError("width4 selector report lacks published_ranking")
    previous = {int(row["start"]): row for row in previous_rows}
    required = set(CANDIDATE_STARTS[4])
    if set(current) != required or not required.issubset(previous):
        raise Phase5ContractError("width4 central start closure differs")
    max_difference = 0.0
    for start in CANDIDATE_STARTS[4]:
        left = current[start]
        right = previous[start]
        comparisons = [(float(left["score"]), float(right["score"]))]
        for new_name, old_name in OLD_CONTRAST_NAMES.items():
            old_payload = right["contrasts"][old_name]
            old_z = float(old_payload["point"]) / max(
                float(old_payload["standard_error"]), 1e-12
            )
            comparisons.append((float(left["z_" + new_name]), old_z))
        for observed, expected in comparisons:
            difference = abs(observed - expected)
            max_difference = max(max_difference, difference)
            if difference > tolerance:
                raise Phase5ContractError(
                    "width4 numeric mismatch at start %d: %.17g" % (start, difference)
                )
        if bool(left["point_eligible"]) is not bool(right["point_eligible"]):
            raise Phase5ContractError("width4 eligibility mismatch at start %d" % start)
        if list(left["eligibility_failures"]) != list(right["failure_reasons"]):
            raise Phase5ContractError("width4 failure reasons mismatch at start %d" % start)
    return {
        "status": "WIDTH4_CENTRAL11_EXACT_REPRODUCTION_PASS",
        "row_count": len(required),
        "starts": [min(required), max(required)],
        "absolute_tolerance": tolerance,
        "max_absolute_difference": max_difference,
        "score_z_eligibility_failure_reasons_exact": True,
    }


def build_output_payloads(
    analysis: Mapping[str, Any],
    *,
    card_file_sha256: str,
    input_provenance: Mapping[str, Any],
    implementation_provenance: Mapping[str, Any],
    width4_compatibility: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str, Dict[str, Any], str]:
    rows = [dict(row) for row in analysis["rows"]]
    common = {
        "card_file_sha256": card_file_sha256,
        "input_provenance": dict(input_provenance),
        "implementation_provenance": dict(implementation_provenance),
        "record_count": int(analysis["record_count"]),
        "subject_count": int(analysis["subject_count"]),
        "bootstrap": dict(analysis["bootstrap"]),
        "outcome_values_consumed": False,
        "test_split_consumed": False,
        "geometry_fields_consumed": False,
        "post_terminal_exploratory_no_selection": True,
    }
    scores = {
        "schema_version": "loopscope.phase5.multiwidth-mid40-scores.v1",
        **common,
        "candidate_count": len(rows),
        "rows": rows,
    }
    _attach_manifest(scores)
    eligible = [row for row in rows if bool(row["point_eligible"])]
    per_width = {}
    for width in WIDTHS:
        width_rows = [row for row in rows if int(row["width"]) == width]
        per_width[str(width)] = {
            "candidate_count": len(width_rows),
            "start_range": [CANDIDATE_STARTS[width][0], CANDIDATE_STARTS[width][-1]],
            "top5": [_rank_entry(row) for row in width_rows[:5]],
        }
    summary = {
        "schema_version": "loopscope.phase5.multiwidth-mid40-summary.v1",
        **common,
        "candidate_count": len(rows),
        "candidate_counts_by_width": dict(analysis["candidate_counts_by_width"]),
        "ranking_top1": _rank_entry(rows[0]),
        "global_top10": [_rank_entry(row) for row in rows[:10]],
        "per_width": per_width,
        "eligible_count": len(eligible),
        "eligible_rows": [_rank_entry(row) for row in eligible],
        "width4_backward_compatibility": dict(width4_compatibility),
    }
    _attach_manifest(summary)
    return scores, render_scores_csv(rows), summary, render_summary_zh(summary)


def render_scores_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(CSV_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        projected = {}
        for field in CSV_FIELDS:
            value = row[field]
            if field in ("eligibility_failures", "comparison_starts"):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            projected[field] = value
        writer.writerow(projected)
    return output.getvalue()


def render_summary_zh(summary: Mapping[str, Any]) -> str:
    lines = [
        "# LoopScope 第五阶段扩展 Gate E：多宽度中央 40% score-only 排名",
        "",
        "本结果仅复用 validation-1531 的 native no-loop trajectory；未读取 test、accuracy、gain、flip 或任何 outcome。",
        "这是 Phase 5 终结后的探索性排名，不产生 prospective selected window。",
        "",
        "## 闭合摘要",
        "",
        "- 候选总数：`%d`（width3/4/5/6 = `12/11/10/9`）。"
        % int(summary["candidate_count"]),
        "- bootstrap：R=`%d`，seed=`%d`，draw digest=`%s`。"
        % (
            int(summary["bootstrap"]["replicates"]),
            int(summary["bootstrap"]["seed"]),
            summary["bootstrap"]["draw_index_sha256"],
        ),
        "- eligible 数量：`%d`。" % int(summary["eligible_count"]),
        "- width4 向后兼容：`%s`，max abs diff=`%.17g`。"
        % (
            summary["width4_backward_compatibility"]["status"],
            float(summary["width4_backward_compatibility"]["max_absolute_difference"]),
        ),
        "",
        "## 全局 Top 10",
        "",
        "| rank | width | window | score | eligible |",
        "|---:|---:|:---:|---:|:---:|",
    ]
    for row in summary["global_top10"]:
        lines.append(
            "| %d | %d | `%s` | %.12g | %s |"
            % (
                int(row["global_rank"]),
                int(row["width"]),
                row["window"],
                float(row["score"]),
                str(bool(row["point_eligible"])).lower(),
            )
        )
    for width in WIDTHS:
        lines.extend(
            [
                "",
                "## Width %d Top 5" % width,
                "",
                "| width rank | global rank | window | score | eligible |",
                "|---:|---:|:---:|---:|:---:|",
            ]
        )
        for row in summary["per_width"][str(width)]["top5"]:
            lines.append(
                "| %d | %d | `%s` | %.12g | %s |"
                % (
                    int(row["width_rank"]),
                    int(row["global_rank"]),
                    row["window"],
                    float(row["score"]),
                    str(bool(row["point_eligible"])).lower(),
                )
            )
    if summary["eligible_rows"]:
        lines.extend(["", "## Eligible rows", ""])
        for row in summary["eligible_rows"]:
            lines.append(
                "- width `%d` window `%s`, score=`%.12g`"
                % (int(row["width"]), row["window"], float(row["score"]))
            )
    lines.extend(
        [
            "",
            "`outcome_values_consumed=false`",
            "",
            "`post_terminal_exploratory_no_selection=true`",
            "",
        ]
    )
    return "\n".join(lines)


def write_primary_outputs(
    output_root: Path,
    payloads: Tuple[Mapping[str, Any], str, Mapping[str, Any], str],
) -> Dict[str, str]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    scores, csv_text, summary, zh_text = payloads
    return {
        SCORES_JSON: write_new_json(output_root / SCORES_JSON, scores),
        SCORES_CSV: write_new_text(output_root / SCORES_CSV, csv_text),
        SUMMARY_JSON: write_new_json(output_root / SUMMARY_JSON, summary),
        SUMMARY_ZH: write_new_text(output_root / SUMMARY_ZH, zh_text),
    }


def verify_and_close_outputs(
    output_root: Path,
    expected_payloads: Tuple[Mapping[str, Any], str, Mapping[str, Any], str],
    *,
    input_provenance: Mapping[str, Any],
    implementation_provenance: Mapping[str, Any],
    bootstrap_draw_digest_matches_width4_selector: bool,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    scores, csv_text, summary, zh_text = expected_payloads
    observed_scores = load_strict_json(output_root / SCORES_JSON)
    observed_summary = load_strict_json(output_root / SUMMARY_JSON)
    observed_csv = (output_root / SCORES_CSV).read_text(encoding="utf-8")
    observed_zh = (output_root / SUMMARY_ZH).read_text(encoding="utf-8")
    checks = {
        "scores_exact_raw_recomputation": canonical_json_bytes(observed_scores)
        == canonical_json_bytes(scores),
        "csv_exact_raw_recomputation": observed_csv == csv_text,
        "summary_exact_raw_recomputation": canonical_json_bytes(observed_summary)
        == canonical_json_bytes(summary),
        "summary_zh_exact_recomputation": observed_zh == zh_text,
        "candidate_count_42": len(observed_scores.get("rows", [])) == 42,
        "candidate_counts_12_11_10_9": observed_summary.get("candidate_counts_by_width")
        == {"3": 12, "4": 11, "5": 10, "6": 9},
        "width4_backward_compatibility_pass": observed_summary.get(
            "width4_backward_compatibility", {}
        ).get("status")
        == "WIDTH4_CENTRAL11_EXACT_REPRODUCTION_PASS",
        "bootstrap_draw_digest_matches_width4_selector": bool(
            bootstrap_draw_digest_matches_width4_selector
        ),
        "outcome_values_consumed_false": observed_summary.get("outcome_values_consumed")
        is False,
        "post_terminal_no_selection_true": observed_summary.get(
            "post_terminal_exploratory_no_selection"
        )
        is True,
        "selected_window_absent": "selected_window" not in canonical_json_bytes(
            observed_summary
        ).decode("utf-8")
        and "selected_window" not in canonical_json_bytes(observed_scores).decode("utf-8"),
    }
    if not all(checks.values()):
        raise Phase5ContractError(
            "Gate E verifier failed: %s"
            % sorted(name for name, value in checks.items() if not value)
        )
    primary_hashes = {
        name: file_sha256(output_root / name)
        for name in (SCORES_JSON, SCORES_CSV, SUMMARY_JSON, SUMMARY_ZH)
    }
    verifier = {
        "schema_version": "loopscope.phase5.multiwidth-mid40-verifier-receipt.v1",
        "result": "PASS",
        "checks": checks,
        "recomputed_row_count": 42,
        "input_provenance": dict(input_provenance),
        "implementation_provenance": dict(implementation_provenance),
        "verified_primary_output_sha256": primary_hashes,
        "outcome_values_consumed": False,
        "test_split_consumed": False,
    }
    _attach_manifest(verifier)
    verifier_sha = write_new_json(output_root / VERIFIER_RECEIPT, verifier)
    all_output_hashes = dict(primary_hashes, **{VERIFIER_RECEIPT: verifier_sha})
    manifest = {
        "schema_version": "loopscope.phase5.multiwidth-mid40-manifest-receipt.v1",
        "result": "READY_FOR_PLANNING_AUDIT",
        "input_provenance": dict(input_provenance),
        "implementation_provenance": dict(implementation_provenance),
        "output_file_sha256": all_output_hashes,
        "verifier_manifest_sha256": verifier["manifest_sha256"],
        "candidate_count": 42,
        "outcome_values_consumed": False,
        "test_split_consumed": False,
        "model_or_data_loaded": False,
        "gpu_or_slurm_used": False,
        "post_terminal_exploratory_no_selection": True,
    }
    _attach_manifest(manifest)
    write_new_json(output_root / MANIFEST_RECEIPT, manifest)
    return manifest


def write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return write_new_text(path, text + "\n")


def write_new_text(path: Path, text: str) -> str:
    if Path(path).exists():
        raise FileExistsError("refusing to overwrite Gate E artifact: %s" % path)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def _eligibility_failures(
    width: int,
    start: int,
    comparisons: Iterable[int],
    point_e: Mapping[Tuple[int, int], float],
    point_k: Mapping[Tuple[int, int], float],
    summaries: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    key = (width, start)
    failures: List[str] = []
    if not point_e[key] > 0.0:
        failures.append("CENTER_E_NOT_STRICTLY_POSITIVE")
    if not point_k[key] > 0.0:
        failures.append("CENTER_K_NOT_STRICTLY_POSITIVE")
    for reference in comparisons:
        if not point_e[(width, reference)] < 0.0:
            failures.append("COMPARISON_S%d_E_NOT_STRICTLY_NEGATIVE" % reference)
        if not point_k[(width, reference)] < 0.0:
            failures.append("COMPARISON_S%d_K_NOT_STRICTLY_NEGATIVE" % reference)
    for name in CONTRASTS:
        if not float(summaries[name]["ci95"][0]) > 0.0:
            failures.append("%s_LOWER95_NOT_STRICTLY_POSITIVE" % OLD_CONTRAST_NAMES[name])
    return failures


def _bootstrap_summary(point: float, values: Sequence[float]) -> Dict[str, Any]:
    mean = math.fsum(values) / len(values)
    standard_error = math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )
    return {
        "point": point,
        "standard_error": standard_error,
        "ci95": [_linear_percentile(values, 0.025), _linear_percentile(values, 0.975)],
        "z": point / max(standard_error, 1e-12),
    }


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _rank_entry(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "global_rank": int(row["global_rank"]),
        "width": int(row["width"]),
        "width_rank": int(row["width_rank"]),
        "start": int(row["start"]),
        "window": str(row["window"]),
        "score": float(row["score"]),
        "point_eligible": bool(row["point_eligible"]),
    }


def _attach_manifest(payload: Dict[str, Any]) -> None:
    payload["manifest_sha256"] = semantic_sha256(payload)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase5ContractError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise Phase5ContractError("%s must be finite" % label)
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


__all__ = [
    "CANDIDATE_STARTS",
    "MANIFEST_RECEIPT",
    "SCORES_CSV",
    "SCORES_JSON",
    "SUMMARY_JSON",
    "SUMMARY_ZH",
    "VERIFIER_RECEIPT",
    "WIDTHS",
    "analyze_boundary_samples",
    "build_output_payloads",
    "extract_boundary_samples",
    "file_sha256",
    "load_strict_json",
    "load_strict_jsonl",
    "render_scores_csv",
    "render_summary_zh",
    "required_reference_starts",
    "validate_extension_card",
    "verify_and_close_outputs",
    "verify_width4_backward_compatibility",
    "write_primary_outputs",
]
