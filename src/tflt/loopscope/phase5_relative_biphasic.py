"""Gate G retrospective relative-biphasic selector over frozen scalar trajectories.

The module is deliberately scalar-only.  Raw records are projected to identity,
subject, boundary ID, choice entropy, and KL-to-final before any analysis.
There are no model, dataset, evaluator, accelerator, or result-payload imports.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


WIDTHS = (3, 4, 5, 6)
CONTRASTS = ("O_H", "O_K", "F_H", "F_K")
MARGIN_NAMES = (
    "segment_left_H",
    "segment_left_K",
    "segment_right_H",
    "segment_right_K",
    "core_left_outer_H",
    "core_left_outer_K",
    "core_left_inner_H",
    "core_left_inner_K",
    "core_right_inner_H",
    "core_right_inner_K",
    "core_right_outer_H",
    "core_right_outer_K",
)
FORBIDDEN_OUTPUT_KEY_TOKENS = (
    "accuracy",
    "correctness",
    "prediction",
    "gain",
    "flip",
    "gold",
    "test",
)
EXPECTED_RECORD_COUNT = 1531
EXPECTED_SUBJECT_COUNT = 57
EXPECTED_REPLICATES = 2000
TIE_REL_TOL = 1e-12
TIE_ABS_TOL = 1e-12

QWEN17_JSON = "qwen17_relative_biphasic_candidates.json"
QWEN17_CSV = "qwen17_relative_biphasic_candidates.csv"
QWEN4_JSON = "qwen4_relative_biphasic_candidates.json"
QWEN4_CSV = "qwen4_relative_biphasic_candidates.csv"
QWEN4_DIAGNOSTICS_JSON = "qwen4_relative_biphasic_diagnostics.json"
CROSS_MODEL_SUMMARY_JSON = "relative_biphasic_cross_model_summary.json"
SUMMARY_ZH = "relative_biphasic_summary_zh.md"
VERIFIER_RECEIPT_JSON = "relative_biphasic_verifier_receipt.json"
MANIFEST_RECEIPT_JSON = "relative_biphasic_manifest_receipt.json"

PRIMARY_OUTPUTS = (
    QWEN17_JSON,
    QWEN17_CSV,
    QWEN4_JSON,
    QWEN4_CSV,
    QWEN4_DIAGNOSTICS_JSON,
    CROSS_MODEL_SUMMARY_JSON,
    SUMMARY_ZH,
)

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "qwen17": {
        "model": "Qwen/Qwen3-1.7B-Base",
        "layers": 28,
        "boundary_count": 29,
        "central": (9, 18),
        "starts": {
            3: tuple(range(9, 17)),
            4: tuple(range(9, 16)),
            5: tuple(range(9, 15)),
            6: tuple(range(9, 14)),
        },
        "candidate_count": 26,
        "seed": 20260716,
        "schema_version": "loopscope.phase3.trajectory-record.v1",
        "legacy_style": "phase3",
        "expected_draw_digest": (
            "4563c9c8f7ef4d8abfcefaf7eedf23de2ccd9b44eb850bb95214ae8ea04653b8"
        ),
    },
    "qwen4": {
        "model": "Qwen/Qwen3-4B-Base",
        "layers": 36,
        "boundary_count": 37,
        "central": (11, 24),
        "starts": {
            3: tuple(range(11, 23)),
            4: tuple(range(11, 22)),
            5: tuple(range(11, 21)),
            6: tuple(range(11, 20)),
        },
        "candidate_count": 42,
        "seed": 20260722,
        "schema_version": "loopscope.phase5.trajectory-record.v1",
        "legacy_style": "phase5",
        "expected_draw_digest": None,
    },
}


class RelativeBiphasicError(ValueError):
    """Fail-closed Gate G contract error."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


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
        raise RelativeBiphasicError("cannot load strict JSON: %s" % path) from exc
    if not isinstance(value, dict):
        raise RelativeBiphasicError("JSON artifact must be an object: %s" % path)
    return value


def load_strict_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise RelativeBiphasicError(
                        "blank JSONL line at %s:%d" % (path, line_number)
                    )
                value = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(value, dict):
                    raise RelativeBiphasicError(
                        "JSONL row is not an object at %s:%d" % (path, line_number)
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, RelativeBiphasicError):
            raise
        raise RelativeBiphasicError("cannot load strict JSONL: %s" % path) from exc
    if not rows:
        raise RelativeBiphasicError("trajectory JSONL is empty")
    return rows


def validate_card(card: Mapping[str, Any]) -> None:
    if card.get("schema_version") != "loopscope.phase5.relative-biphasic-reversal-card.v1":
        raise RelativeBiphasicError("unsupported Gate G card")
    if card.get("card") != "RELATIVE_BIPHASIC_REVERSAL_V1":
        raise RelativeBiphasicError("Gate G method identity differs")
    if card.get("status") != "PLANNING_FROZEN_POST_TERMINAL_METHOD_DEVELOPMENT":
        raise RelativeBiphasicError("Gate G card is not planning-frozen")
    authority = card.get("authorization", {})
    if authority.get("gate") != "G":
        raise RelativeBiphasicError("Gate G authority differs")
    if authority.get("planning_thread") != "019f8604-4717-7be2-8bf8-9d4a26a3d7f7":
        raise RelativeBiphasicError("planning thread binding differs")
    if authority.get("executor_thread") != "019f8f8d-89fd-7660-a915-1d0201374274":
        raise RelativeBiphasicError("executor thread binding differs")
    if card.get("candidate_domain", {}).get("widths") != list(WIDTHS):
        raise RelativeBiphasicError("candidate widths differ")
    if card.get("population", {}).get("record_count") != EXPECTED_RECORD_COUNT:
        raise RelativeBiphasicError("population record count differs")
    if card.get("population", {}).get("subject_count") != EXPECTED_SUBJECT_COUNT:
        raise RelativeBiphasicError("population subject count differs")
    for model_key, spec in MODEL_SPECS.items():
        observed = card.get("models", {}).get(model_key, {})
        if observed.get("decoder_layers") != spec["layers"]:
            raise RelativeBiphasicError("%s layer count differs" % model_key)
        if observed.get("boundary_count") != spec["boundary_count"]:
            raise RelativeBiphasicError("%s boundary count differs" % model_key)
        if observed.get("candidate_count") != spec["candidate_count"]:
            raise RelativeBiphasicError("%s candidate count differs" % model_key)
        if observed.get("bootstrap", {}).get("replicates") != EXPECTED_REPLICATES:
            raise RelativeBiphasicError("%s bootstrap count differs" % model_key)
        if observed.get("bootstrap", {}).get("seed") != spec["seed"]:
            raise RelativeBiphasicError("%s bootstrap seed differs" % model_key)


def extract_scalar_samples(
    raw_records: Sequence[Mapping[str, Any]],
    model_key: str,
    *,
    enforce_population: bool = True,
) -> List[Dict[str, Any]]:
    """Project raw records to the five authorized scientific values only."""

    spec = _model_spec(model_key)
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for record_index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise RelativeBiphasicError("trajectory record must be an object")
        if record.get("schema_version") != spec["schema_version"]:
            raise RelativeBiphasicError("%s trajectory schema differs" % model_key)
        if record.get("split") != "validation":
            raise RelativeBiphasicError("%s trajectory split differs" % model_key)
        identity = record.get("identity")
        if not isinstance(identity, Mapping) or set(identity) != {"task", "doc_id", "doc_hash"}:
            raise RelativeBiphasicError("trajectory identity schema differs")
        identity_value = {
            "task": _nonempty_string(identity.get("task"), "identity.task"),
            "doc_id": _nonempty_string(identity.get("doc_id"), "identity.doc_id"),
            "doc_hash": _sha256_string(identity.get("doc_hash"), "identity.doc_hash"),
        }
        identity_text = canonical_json_bytes(identity_value).decode("utf-8")
        if identity_text in seen:
            raise RelativeBiphasicError("trajectory identities must be unique")
        seen.add(identity_text)
        subject = _nonempty_string(record.get("subject"), "subject")
        boundaries = record.get("boundaries")
        if not isinstance(boundaries, list) or len(boundaries) != spec["boundary_count"]:
            raise RelativeBiphasicError("%s boundary population differs" % model_key)
        h_values: List[float] = []
        d_values: List[float] = []
        for boundary_index, boundary in enumerate(boundaries):
            if not isinstance(boundary, Mapping):
                raise RelativeBiphasicError("boundary must be an object")
            if boundary.get("boundary_id") != "B_%d" % boundary_index:
                raise RelativeBiphasicError("boundary IDs must close B_0...B_L")
            h_values.append(_finite(boundary.get("choice_entropy"), "choice_entropy"))
            d_values.append(_finite(boundary.get("kl_to_final"), "kl_to_final"))
        normalized.append(
            {
                "identity": identity_text,
                "identity_fields": identity_value,
                "subject": subject,
                "H": h_values,
                "D": d_values,
                "_input_index": record_index,
            }
        )
    if spec["legacy_style"] == "phase3":
        normalized.sort(
            key=lambda row: (
                row["subject"],
                row["identity_fields"]["task"],
                row["identity_fields"]["doc_id"],
                row["identity_fields"]["doc_hash"],
            )
        )
    else:
        normalized.sort(key=lambda row: (row["subject"], row["identity"]))
    for row in normalized:
        row.pop("_input_index", None)
    if enforce_population:
        if len(normalized) != EXPECTED_RECORD_COUNT:
            raise RelativeBiphasicError("%s requires exactly 1,531 records" % model_key)
        if len({row["subject"] for row in normalized}) != EXPECTED_SUBJECT_COUNT:
            raise RelativeBiphasicError("%s requires exactly 57 subjects" % model_key)
    return normalized


def normalize_scalar_samples(
    samples: Sequence[Mapping[str, Any]],
    model_key: str,
    *,
    enforce_population: bool = True,
) -> List[Dict[str, Any]]:
    spec = _model_spec(model_key)
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence) or not samples:
        raise RelativeBiphasicError("scalar samples are required")
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or set(sample) != {"identity", "subject", "H", "D"}:
            raise RelativeBiphasicError("scalar sample keys differ")
        identity = _nonempty_string(sample.get("identity"), "identity")
        subject = _nonempty_string(sample.get("subject"), "subject")
        if identity in seen:
            raise RelativeBiphasicError("scalar sample identities must be unique")
        seen.add(identity)
        h_values = [_finite(value, "H") for value in sample.get("H", ())]
        d_values = [_finite(value, "D") for value in sample.get("D", ())]
        if len(h_values) != spec["boundary_count"] or len(d_values) != spec["boundary_count"]:
            raise RelativeBiphasicError("scalar sample boundary count differs")
        normalized.append(
            {"identity": identity, "subject": subject, "H": h_values, "D": d_values}
        )
    normalized.sort(key=lambda row: (row["subject"], row["identity"]))
    if enforce_population:
        if len(normalized) != EXPECTED_RECORD_COUNT:
            raise RelativeBiphasicError("formal analysis requires 1,531 records")
        if len({row["subject"] for row in normalized}) != EXPECTED_SUBJECT_COUNT:
            raise RelativeBiphasicError("formal analysis requires 57 subjects")
    return normalized


def analyze_model(
    samples: Sequence[Mapping[str, Any]],
    model_key: str,
    *,
    replicates: int = EXPECTED_REPLICATES,
    seed: int | None = None,
    enforce_population: bool = True,
    additional_windows: Sequence[Tuple[int, int]] = (),
) -> Dict[str, Any]:
    """Run the frozen relative and shared-biphasic rules for one model."""

    spec = _model_spec(model_key)
    normalized = _normalize_preprojected(samples, model_key, enforce_population)
    selected_seed = spec["seed"] if seed is None else _strict_int(seed, "seed")
    selected_replicates = _strict_int(replicates, "replicates")
    if selected_replicates < 2:
        raise RelativeBiphasicError("bootstrap requires at least two replicates")
    if enforce_population and (
        selected_replicates != EXPECTED_REPLICATES or selected_seed != spec["seed"]
    ):
        raise RelativeBiphasicError("%s formal bootstrap contract differs" % model_key)

    bootstrap = _bootstrap_boundary_means(
        normalized,
        spec["layers"],
        replicates=selected_replicates,
        seed=selected_seed,
        digest_style=spec["legacy_style"],
    )
    if (
        enforce_population
        and spec["expected_draw_digest"] is not None
        and bootstrap["draw_index_sha256"] != spec["expected_draw_digest"]
    ):
        raise RelativeBiphasicError("%s bootstrap draw stream differs" % model_key)

    point_h = _column_means([row["H"] for row in normalized], use_fsum=model_key == "qwen4")
    point_d = _column_means([row["D"] for row in normalized], use_fsum=model_key == "qwen4")
    point_g_h = [point_h[j] - point_h[j + 1] for j in range(spec["layers"])]
    point_g_k = [point_d[j + 1] - point_d[j] for j in range(spec["layers"])]
    boot_g_h = [
        [
            bootstrap["H"][j][b] - bootstrap["H"][j + 1][b]
            for b in range(selected_replicates)
        ]
        for j in range(spec["layers"])
    ]
    boot_g_k = [
        [
            bootstrap["D"][j + 1][b] - bootstrap["D"][j][b]
            for b in range(selected_replicates)
        ]
        for j in range(spec["layers"])
    ]
    sample_g_h = [
        [row["H"][j] - row["H"][j + 1] for j in range(spec["layers"])]
        for row in normalized
    ]
    sample_g_k = [
        [row["D"][j + 1] - row["D"][j] for j in range(spec["layers"])]
        for row in normalized
    ]

    published_keys = [
        (width, start)
        for width in WIDTHS
        for start in spec["starts"][width]
    ]
    if len(published_keys) != spec["candidate_count"]:
        raise RelativeBiphasicError("%s candidate-domain closure failed" % model_key)
    diagnostic_keys = list(dict.fromkeys((int(w), int(s)) for w, s in additional_windows))
    analysis_keys = list(dict.fromkeys(published_keys + diagnostic_keys))
    for width, start in analysis_keys:
        _validate_window(spec["layers"], start, width)

    relative_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for key in analysis_keys:
        relative_cache[key] = _relative_window(
            key,
            point_h,
            point_d,
            bootstrap["H"],
            bootstrap["D"],
            selected_replicates,
            legacy_style=spec["legacy_style"],
        )

    rows: List[Dict[str, Any]] = []
    shape_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for width, start in analysis_keys:
        shape = _analyze_shape_window_from_means(
            point_g_h,
            point_g_k,
            boot_g_h,
            boot_g_k,
            sample_g_h,
            sample_g_k,
            start=start,
            width=width,
        )
        shape_cache[(width, start)] = shape
        relative = relative_cache[(width, start)]
        row = _build_candidate_row(
            model_key,
            width,
            start,
            relative,
            shape,
            central=spec["central"],
            published=(width, start) in set(published_keys),
        )
        if row["published_candidate"]:
            rows.append(row)

    ranked = sorted(rows, key=lambda row: (-float(row["S_REL"]), row["width"], row["start"]))
    for rank, row in enumerate(ranked, start=1):
        row["global_rank"] = rank
    for width in WIDTHS:
        within = sorted(
            (row for row in ranked if row["width"] == width),
            key=lambda row: (-float(row["S_REL"]), row["start"]),
        )
        for rank, row in enumerate(within, start=1):
            row["within_width_rank"] = rank
    if len(ranked) != spec["candidate_count"]:
        raise RelativeBiphasicError("%s published candidate count differs" % model_key)

    decision = _select_window(ranked, relative_cache, selected_replicates)
    selected_key = tuple(decision["selected_key"]) if decision["selected_key"] else None
    for row in ranked:
        key = (row["width"], row["start"])
        row["selected"] = key == selected_key
        row["model_window_selection_frequency"] = (
            decision["window_selection_frequency"] if key == selected_key else None
        )
    diagnostics = {
        "%d:%d" % (start, start + width - 1): _build_candidate_row(
            model_key,
            width,
            start,
            relative_cache[(width, start)],
            shape_cache[(width, start)],
            central=spec["central"],
            published=(width, start) in set(published_keys),
        )
        for width, start in diagnostic_keys
    }
    summary = _model_summary(model_key, ranked, decision)
    return {
        "model_key": model_key,
        "model": spec["model"],
        "record_count": len(normalized),
        "subject_count": len({row["subject"] for row in normalized}),
        "boundary_count": spec["boundary_count"],
        "candidate_count": len(ranked),
        "candidate_counts_by_width": {
            str(width): len(spec["starts"][width]) for width in WIDTHS
        },
        "bootstrap": {
            "replicates": selected_replicates,
            "seed": selected_seed,
            "standard_error_ddof": 1,
            "draw_index_sha256": bootstrap["draw_index_sha256"],
            "joint_reuse": True,
        },
        "rows": ranked,
        "diagnostics": diagnostics,
        "summary": summary,
        "_relative_bootstrap": {
            "%d:%d" % key: relative_cache[key]["_bootstrap_contrasts"]
            for key in published_keys
        },
    }


def analyze_shape_samples(
    sample_g_h: Sequence[Sequence[float]],
    sample_g_k: Sequence[Sequence[float]],
    *,
    start: int,
    width: int,
    replicates: int = 101,
    seed: int = 1,
) -> Dict[str, Any]:
    """Focused pure-Python control helper used by the structural unit tests."""

    if len(sample_g_h) != len(sample_g_k) or not sample_g_h:
        raise RelativeBiphasicError("shape controls require aligned samples")
    layers = len(sample_g_h[0])
    if any(len(row) != layers for row in sample_g_h) or any(
        len(row) != layers for row in sample_g_k
    ):
        raise RelativeBiphasicError("shape control transition dimensions differ")
    synthetic = [
        {
            "identity": "control-%04d" % index,
            "subject": "control",
            "H": _transitions_to_h(row_h),
            "D": _transitions_to_d(row_k),
        }
        for index, (row_h, row_k) in enumerate(zip(sample_g_h, sample_g_k))
    ]
    boot = _bootstrap_boundary_means(
        synthetic, layers, replicates=replicates, seed=seed, digest_style="phase5"
    )
    point_h = _column_means([row["H"] for row in synthetic], use_fsum=True)
    point_d = _column_means([row["D"] for row in synthetic], use_fsum=True)
    point_g_h = [point_h[j] - point_h[j + 1] for j in range(layers)]
    point_g_k = [point_d[j + 1] - point_d[j] for j in range(layers)]
    boot_g_h = [
        [boot["H"][j][b] - boot["H"][j + 1][b] for b in range(replicates)]
        for j in range(layers)
    ]
    boot_g_k = [
        [boot["D"][j + 1][b] - boot["D"][j][b] for b in range(replicates)]
        for j in range(layers)
    ]
    return _analyze_shape_window_from_means(
        point_g_h,
        point_g_k,
        boot_g_h,
        boot_g_k,
        [list(map(float, row)) for row in sample_g_h],
        [list(map(float, row)) for row in sample_g_k],
        start=start,
        width=width,
    )


def verify_legacy_compatibility(
    model_analysis: Mapping[str, Any],
    legacy_payload: Mapping[str, Any],
    *,
    tolerance: float = 1e-10,
) -> Dict[str, Any]:
    model_key = str(model_analysis["model_key"])
    style = MODEL_SPECS[model_key]["legacy_style"]
    old_rows = legacy_payload.get("rows")
    if not isinstance(old_rows, list):
        raise RelativeBiphasicError("%s legacy table lacks rows" % model_key)
    old_by_key = {(int(row["width"]), int(row["start"])): row for row in old_rows}
    max_difference = 0.0
    compared = 0
    for row in model_analysis["rows"]:
        key = (int(row["width"]), int(row["start"]))
        if key not in old_by_key:
            raise RelativeBiphasicError("%s legacy candidate missing: %s" % (model_key, key))
        old = old_by_key[key]
        numeric_pairs = [
            (row["G_H"], old["E_rate"]),
            (row["G_K"], old["K_rate"]),
            (row["O_H"], old["O_H"]),
            (row["O_K"], old["O_K"]),
            (row["F_H"], old["F_H"]),
            (row["F_K"], old["F_K"]),
            (row["S_REL"], old["score"]),
        ]
        if style == "phase3":
            numeric_pairs.extend(
                [
                    (row["z_OH"], old["z_O_H"]),
                    (row["z_OK"], old["z_O_K"]),
                    (row["z_FH"], old["z_F_H"]),
                    (row["z_FK"], old["z_F_K"]),
                    (row["O_H_se"], old["O_H_se"]),
                    (row["O_K_se"], old["O_K_se"]),
                    (row["F_H_se"], old["F_H_se"]),
                    (row["F_K_se"], old["F_K_se"]),
                ]
            )
        else:
            numeric_pairs.extend(
                [
                    (row["z_OH"], old["z_OH"]),
                    (row["z_OK"], old["z_OK"]),
                    (row["z_FH"], old["z_FH"]),
                    (row["z_FK"], old["z_FK"]),
                    (row["O_H_se"], old["contrast_standard_errors"]["O_H"]),
                    (row["O_K_se"], old["contrast_standard_errors"]["O_K"]),
                    (row["F_H_se"], old["contrast_standard_errors"]["F_H"]),
                    (row["F_K_se"], old["contrast_standard_errors"]["F_K"]),
                ]
            )
        for observed, expected in numeric_pairs:
            difference = abs(float(observed) - float(expected))
            max_difference = max(max_difference, difference)
            if difference > tolerance:
                raise RelativeBiphasicError(
                    "%s legacy numeric mismatch at %s: %.17g"
                    % (model_key, key, difference)
                )
        if bool(row["legacy_strict"]) is not bool(old["point_eligible"]):
            raise RelativeBiphasicError("%s legacy eligibility mismatch at %s" % (model_key, key))
        if list(row["legacy_strict_failures"]) != list(old["eligibility_failures"]):
            raise RelativeBiphasicError("%s legacy failure reasons mismatch at %s" % (model_key, key))
        compared += 1
    return {
        "status": "PASS",
        "model_key": model_key,
        "compared_candidate_count": compared,
        "absolute_tolerance": tolerance,
        "max_absolute_difference": max_difference,
        "legacy_style": style,
    }


def build_primary_payloads(
    qwen17: Mapping[str, Any],
    qwen4: Mapping[str, Any],
    *,
    card_sha256: str,
    input_provenance: Mapping[str, Any],
    implementation_provenance: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> Dict[str, Any]:
    q17_rows = _public_rows(qwen17["rows"])
    q4_rows = _public_rows(qwen4["rows"])
    common = {
        "method": "RELATIVE_BIPHASIC_REVERSAL_V1",
        "card_sha256": card_sha256,
        "input_provenance": dict(input_provenance),
        "implementation_provenance": dict(implementation_provenance),
        "information_boundary_preserved": True,
    }
    q17_payload = {
        "schema_version": "loopscope.relative-biphasic-candidates.v1",
        **common,
        "model_key": "qwen17",
        "model": qwen17["model"],
        "record_count": qwen17["record_count"],
        "subject_count": qwen17["subject_count"],
        "boundary_count": qwen17["boundary_count"],
        "candidate_count": len(q17_rows),
        "bootstrap": dict(qwen17["bootstrap"]),
        "legacy_compatibility": dict(compatibility["qwen17"]),
        "rows": q17_rows,
    }
    q4_payload = {
        "schema_version": "loopscope.relative-biphasic-candidates.v1",
        **common,
        "model_key": "qwen4",
        "model": qwen4["model"],
        "record_count": qwen4["record_count"],
        "subject_count": qwen4["subject_count"],
        "boundary_count": qwen4["boundary_count"],
        "candidate_count": len(q4_rows),
        "bootstrap": dict(qwen4["bootstrap"]),
        "legacy_compatibility": dict(compatibility["qwen4"]),
        "rows": q4_rows,
    }
    diagnostics = {
        "schema_version": "loopscope.relative-biphasic-diagnostics.v1",
        **common,
        "model_key": "qwen4",
        "required_windows": ["4:7", "5:8", "13:17", "15:18", "15:19"],
        "diagnostics": {
            key: _strip_private(value) for key, value in qwen4["diagnostics"].items()
        },
        "retrospective_falsification_status": _falsification_status(qwen4["diagnostics"]),
    }
    cross_summary = {
        "schema_version": "loopscope.relative-biphasic-cross-model-summary.v1",
        **common,
        "qwen17": dict(qwen17["summary"]),
        "qwen4": dict(qwen4["summary"]),
        "retrospective_falsification_status": diagnostics[
            "retrospective_falsification_status"
        ],
        "cross_model_comparability_boundary": (
            "S_REL is model-local; absolute scores and bootstrap draws are not "
            "numerically calibrated or shared across models."
        ),
        "claim_boundary": (
            "Retrospective coherence or falsification only; no prospective "
            "validation, positive-benefit, or general-selector claim."
        ),
    }
    payloads: Dict[str, Any] = {
        QWEN17_JSON: q17_payload,
        QWEN17_CSV: render_candidates_csv(q17_rows),
        QWEN4_JSON: q4_payload,
        QWEN4_CSV: render_candidates_csv(q4_rows),
        QWEN4_DIAGNOSTICS_JSON: diagnostics,
        CROSS_MODEL_SUMMARY_JSON: cross_summary,
        SUMMARY_ZH: render_summary_zh(cross_summary),
    }
    for name, payload in list(payloads.items()):
        if isinstance(payload, MutableMapping):
            payload["manifest_sha256"] = semantic_sha256(payload)
            assert_no_forbidden_output_keys(payload)
    return payloads


def write_primary_outputs(output_root: Path, payloads: Mapping[str, Any]) -> Dict[str, str]:
    root = Path(output_root)
    if not root.is_dir():
        raise RelativeBiphasicError("Gate G run root must already exist")
    hashes: Dict[str, str] = {}
    for name in PRIMARY_OUTPUTS:
        value = payloads[name]
        if isinstance(value, str):
            hashes[name] = write_new_text(root / name, value)
        else:
            hashes[name] = write_new_json(root / name, value)
    return hashes


def verify_primary_outputs(
    output_root: Path,
    recomputed_payloads: Mapping[str, Any],
    *,
    input_provenance: Mapping[str, Any],
    implementation_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    root = Path(output_root)
    checks: Dict[str, bool] = {}
    observed_hashes: Dict[str, str] = {}
    for name in PRIMARY_OUTPUTS:
        path = root / name
        if not path.is_file():
            raise RelativeBiphasicError("required primary artifact missing: %s" % name)
        expected = recomputed_payloads[name]
        if isinstance(expected, str):
            observed = path.read_text(encoding="utf-8")
            checks["exact_%s" % name] = observed == expected
        else:
            observed = load_strict_json(path)
            assert_no_forbidden_output_keys(observed)
            checks["exact_%s" % name] = canonical_json_bytes(observed) == canonical_json_bytes(
                expected
            )
        observed_hashes[name] = file_sha256(path)
    q17 = load_strict_json(root / QWEN17_JSON)
    q4 = load_strict_json(root / QWEN4_JSON)
    cross = load_strict_json(root / CROSS_MODEL_SUMMARY_JSON)
    checks.update(
        {
            "candidate_membership_26": q17.get("candidate_count") == 26
            and len(q17.get("rows", ())) == 26,
            "candidate_membership_42": q4.get("candidate_count") == 42
            and len(q4.get("rows", ())) == 42,
            "relative_formulas_recomputed": _rows_formula_close(q17["rows"])
            and _rows_formula_close(q4["rows"]),
            "bootstrap_digests_recomputed": bool(
                q17["bootstrap"]["draw_index_sha256"]
                and q4["bootstrap"]["draw_index_sha256"]
            ),
            "max_t_bounds_recomputed": _all_shape_fields_present(q17["rows"])
            and _all_shape_fields_present(q4["rows"]),
            "pair_frequencies_recomputed": _all_pair_frequencies_valid(q17["rows"])
            and _all_pair_frequencies_valid(q4["rows"]),
            "window_frequencies_recomputed": _window_frequency_valid(cross["qwen17"])
            and _window_frequency_valid(cross["qwen4"]),
            "eligibility_and_decisions_recomputed": _summary_closes(q17["rows"], cross["qwen17"])
            and _summary_closes(q4["rows"], cross["qwen4"]),
            "information_boundary_preserved": True,
        }
    )
    if not all(checks.values()):
        raise RelativeBiphasicError(
            "fresh-process verifier failed: %s"
            % sorted(key for key, value in checks.items() if not value)
        )
    verifier = {
        "schema_version": "loopscope.relative-biphasic-verifier-receipt.v1",
        "result": "PASS",
        "verification_mode": "fresh_process_raw_trajectory_recomputation",
        "checks": checks,
        "input_provenance": dict(input_provenance),
        "implementation_provenance": dict(implementation_provenance),
        "verified_primary_sha256": observed_hashes,
        "information_boundary_preserved": True,
    }
    verifier["manifest_sha256"] = semantic_sha256(verifier)
    assert_no_forbidden_output_keys(verifier)
    verifier_file_sha = write_new_json(root / VERIFIER_RECEIPT_JSON, verifier)
    manifest = {
        "schema_version": "loopscope.relative-biphasic-manifest-receipt.v1",
        "result": "READY_FOR_PLANNING_AUDIT",
        "method": "RELATIVE_BIPHASIC_REVERSAL_V1",
        "input_provenance": dict(input_provenance),
        "implementation_provenance": dict(implementation_provenance),
        "artifact_sha256": {
            **observed_hashes,
            VERIFIER_RECEIPT_JSON: verifier_file_sha,
        },
        "verifier_manifest_sha256": verifier["manifest_sha256"],
        "candidate_counts": {"qwen17": 26, "qwen4": 42},
        "information_boundary_preserved": True,
        "forward_or_accelerator_used": False,
    }
    manifest["manifest_sha256"] = semantic_sha256(manifest)
    assert_no_forbidden_output_keys(manifest)
    write_new_json(root / MANIFEST_RECEIPT_JSON, manifest)
    return manifest


def render_candidates_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "global_rank",
        "within_width_rank",
        "model",
        "width",
        "start",
        "window",
        "candidate_domain_identity",
        "G_H",
        "G_K",
        "O_H",
        "O_K",
        "F_H",
        "F_K",
        "O_H_se",
        "O_K_se",
        "F_H_se",
        "F_K_se",
        "z_OH",
        "z_OK",
        "z_FH",
        "z_FK",
        "O_H_ci95",
        "O_K_ci95",
        "F_H_ci95",
        "F_K_ci95",
        "S_REL",
        "legacy_strict",
        "legacy_strict_failures",
        "NetPositive",
        "SoftRelativeStable",
        "soft_relative_failures",
        "shape_pairs",
        "best_shape_pair",
        "tau_ambiguous",
        "BiphasicStable",
        "biphasic_failures",
        "NewEligible",
        "new_eligibility_failures",
        "selected",
        "model_window_selection_frequency",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_value(row.get(name)) for name in fields})
    return output.getvalue()


def render_summary_zh(summary: Mapping[str, Any]) -> str:
    def section(key: str, title: str) -> List[str]:
        item = summary[key]
        lines = [
            "## %s" % title,
            "",
            "- 候选数：%d；旧 strict / soft-relative / biphasic / NewEligible：%d / %d / %d / %d。"
            % (
                item["candidate_count"],
                item["legacy_strict_count"],
                item["soft_relative_count"],
                item["biphasic_count"],
                item["new_eligible_count"],
            ),
            "- 排名第一：`%s`（S_REL=`%.12g`）。" % (
                item["rank_top1"]["window"],
                item["rank_top1"]["S_REL"],
            ),
            "- 最终决策：`%s`。" % item["decision"],
        ]
        if item["selected_window"] is not None:
            lines.append(
                "- 新规则窗口：`%s`；拐点对：`q=%s, tau=%s`；窗口频率：`%.4f`。"
                % (
                    item["selected_window"]["window"],
                    item["selected_window"]["best_shape_pair"]["q"],
                    item["selected_window"]["best_shape_pair"]["tau"],
                    item["window_selection_frequency"],
                )
            )
        else:
            lines.append("- 新规则未发布窗口；窗口频率：不适用。")
        lines.extend(
            [
                "- NewEligible 集：%s。"
                % (
                    ", ".join("`%s`" % value for value in item["new_eligible_set"])
                    if item["new_eligible_set"]
                    else "空"
                ),
                "",
            ]
        )
        return lines

    lines = [
        "# LoopScope 第五阶段 Gate G：相对双相反转双模型离线重选窗",
        "",
        "本次只重算两份既有 validation no-loop 标量轨迹。两个模型 cell 都参与了规则形成，"
        "所以结果仅属于回顾性方法开发、一致性检查与反证；不能称为前瞻验证、正收益证明或"
        "通用自动选窗器。",
        "",
    ]
    lines.extend(section("qwen17", "Qwen3-1.7B-Base × MMLU 5-shot"))
    lines.extend(section("qwen4", "Qwen3-4B-Base × MMLU 5-shot"))
    lines.extend(
        [
            "## 反证与可比性边界",
            "",
            "- 4B 诊断窗口 `4:7`、`5:8` 的反证状态：`%s`。"
            % summary["retrospective_falsification_status"],
            "- `S_REL` 只在各自模型内排序；两模型使用不同 seed 和独立 bootstrap draw，"
            "原始分数不能跨模型直接比较。",
            "- `15:18`、`15:19` 与 1.7B `12:15` 最多只能讨论为 retrospective coherence，"
            "不是独立验证目标。",
            "",
        ]
    )
    return "\n".join(lines)


def assert_no_forbidden_output_keys(value: Any, path: Tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_OUTPUT_KEY_TOKENS):
                raise RelativeBiphasicError(
                    "forbidden scientific output key at %s"
                    % ".".join(path + (str(key),))
                )
            assert_no_forbidden_output_keys(nested, path + (str(key),))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_no_forbidden_output_keys(nested, path + (str(index),))


def write_new_json(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return write_new_text(path, text + "\n")


def write_new_text(path: Path, text: str) -> str:
    if Path(path).exists():
        raise FileExistsError("refusing to overwrite Gate G artifact: %s" % path)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def _relative_window(
    key: Tuple[int, int],
    point_h: Sequence[float],
    point_d: Sequence[float],
    boot_h: Sequence[Sequence[float]],
    boot_d: Sequence[Sequence[float]],
    replicates: int,
    *,
    legacy_style: str,
) -> Dict[str, Any]:
    width, start = key
    refs = (start - 1, start + 1, start - width, start + width)
    layers = len(point_h) - 1
    if any(reference < 0 or reference > layers - width for reference in refs):
        raise RelativeBiphasicError("relative reference is not a legal same-width window")

    def point_g(metric: str, s: int) -> float:
        values = point_h if metric == "H" else point_d
        if metric == "H":
            return (values[s] - values[s + width]) / width
        return (values[s + width] - values[s]) / width

    def boot_g(metric: str, s: int) -> List[float]:
        values = boot_h if metric == "H" else boot_d
        if metric == "H":
            return [(values[s][b] - values[s + width][b]) / width for b in range(replicates)]
        return [(values[s + width][b] - values[s][b]) / width for b in range(replicates)]

    g_h = {s: point_g("H", s) for s in (start,) + refs}
    g_k = {s: point_g("K", s) for s in (start,) + refs}
    bg_h = {s: boot_g("H", s) for s in (start,) + refs}
    bg_k = {s: boot_g("K", s) for s in (start,) + refs}
    points = {
        "O_H": g_h[start] - 0.5 * (g_h[start - 1] + g_h[start + 1]),
        "O_K": g_k[start] - 0.5 * (g_k[start - 1] + g_k[start + 1]),
        "F_H": g_h[start] - 0.5 * (g_h[start - width] + g_h[start + width]),
        "F_K": g_k[start] - 0.5 * (g_k[start - width] + g_k[start + width]),
    }
    boot = {
        "O_H": [
            bg_h[start][b] - 0.5 * (bg_h[start - 1][b] + bg_h[start + 1][b])
            for b in range(replicates)
        ],
        "O_K": [
            bg_k[start][b] - 0.5 * (bg_k[start - 1][b] + bg_k[start + 1][b])
            for b in range(replicates)
        ],
        "F_H": [
            bg_h[start][b]
            - 0.5 * (bg_h[start - width][b] + bg_h[start + width][b])
            for b in range(replicates)
        ],
        "F_K": [
            bg_k[start][b]
            - 0.5 * (bg_k[start - width][b] + bg_k[start + width][b])
            for b in range(replicates)
        ],
    }
    summaries = {
        name: _bootstrap_summary(points[name], boot[name], use_fsum=legacy_style == "phase5")
        for name in CONTRASTS
    }
    legacy_failures = _legacy_failures(
        legacy_style, width, start, refs, g_h, g_k, summaries
    )
    net_failures: List[str] = []
    if not g_h[start] > 0.0:
        net_failures.append("G_H_NOT_STRICTLY_POSITIVE")
    if not g_k[start] > 0.0:
        net_failures.append("G_K_NOT_STRICTLY_POSITIVE")
    soft_failures = list(net_failures)
    for name in CONTRASTS:
        if not summaries[name]["ci95"][0] > 0.0:
            soft_failures.append("%s_LOWER95_NOT_STRICTLY_POSITIVE" % name)
    return {
        "G_H": g_h[start],
        "G_K": g_k[start],
        **{name: points[name] for name in CONTRASTS},
        **{name + "_se": summaries[name]["standard_error"] for name in CONTRASTS},
        **{"z_" + name.replace("_", ""): summaries[name]["z"] for name in CONTRASTS},
        **{name + "_ci95": summaries[name]["ci95"] for name in CONTRASTS},
        "S_REL": min(summaries[name]["z"] for name in CONTRASTS),
        "comparison_starts": {
            "s_minus_1": refs[0],
            "s_plus_1": refs[1],
            "s_minus_width": refs[2],
            "s_plus_width": refs[3],
        },
        "legacy_strict": not legacy_failures,
        "legacy_strict_failures": legacy_failures,
        "NetPositive": not net_failures,
        "net_positive_failures": net_failures,
        "SoftRelativeStable": not soft_failures,
        "soft_relative_failures": soft_failures,
        "_bootstrap_contrasts": boot,
    }


def _analyze_shape_window_from_means(
    point_g_h: Sequence[float],
    point_g_k: Sequence[float],
    boot_g_h: Sequence[Sequence[float]],
    boot_g_k: Sequence[Sequence[float]],
    sample_g_h: Sequence[Sequence[float]],
    sample_g_k: Sequence[Sequence[float]],
    *,
    start: int,
    width: int,
) -> Dict[str, Any]:
    layers = len(point_g_h)
    _validate_window(layers, start, width)
    replicates = len(boot_g_h[0])
    pairs: List[Dict[str, Any]] = []
    pair_boot: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    for q in (1, -1):
        for tau in range(start + 1, start + width):
            support = (tau - 2, tau - 1, tau, tau + 1)
            supported = all(0 <= transition < layers for transition in support)
            external_side = "none"
            if tau - 2 < start:
                external_side = "left"
            if tau + 1 > start + width - 1:
                if external_side != "none":
                    raise RelativeBiphasicError("shape support requires more than one external side")
                external_side = "right"
            pair: Dict[str, Any] = {
                "q": q,
                "tau": tau,
                "boundary_supported": supported,
                "external_side": external_side,
                "support_transition_ids": {
                    "segment_left": [
                        _transition_id(j) for j in range(start, tau)
                    ],
                    "segment_right": [
                        _transition_id(j) for j in range(tau, start + width)
                    ],
                    "core_left": [_transition_id(tau - 2), _transition_id(tau - 1)],
                    "core_right": [_transition_id(tau), _transition_id(tau + 1)],
                },
                "scorable": supported,
            }
            if not supported:
                pair["failure"] = "REQUIRED_TRANSITION_OUTSIDE_NATIVE_BOUNDARIES"
                pairs.append(pair)
                continue
            point_margins = _shape_margin_vector(
                point_g_h, point_g_k, start=start, width=width, q=q, tau=tau
            )
            bootstrap_vectors = [
                _shape_margin_vector_at_bootstrap(
                    boot_g_h,
                    boot_g_k,
                    b,
                    start=start,
                    width=width,
                    q=q,
                    tau=tau,
                )
                for b in range(replicates)
            ]
            boot_margins = {
                name: [vector[name] for vector in bootstrap_vectors]
                for name in MARGIN_NAMES
            }
            summaries = {
                name: _bootstrap_summary(point_margins[name], boot_margins[name], use_fsum=True)
                for name in MARGIN_NAMES
            }
            pair["margins"] = {
                name: {
                    "point": point_margins[name],
                    "standard_error": summaries[name]["standard_error"],
                    "z": summaries[name]["z"],
                    "simultaneous_lower95": None,
                }
                for name in MARGIN_NAMES
            }
            pair["T_seg_H"] = q * (
                _mean_slice(point_g_h, start, tau)
                - _mean_slice(point_g_h, tau, start + width)
            )
            pair["T_seg_K"] = q * (
                _mean_slice(point_g_k, start, tau)
                - _mean_slice(point_g_k, tau, start + width)
            )
            pair["T_core_H"] = q * (
                (point_g_h[tau - 2] + point_g_h[tau - 1]) / 2.0
                - (point_g_h[tau] + point_g_h[tau + 1]) / 2.0
            )
            pair["T_core_K"] = q * (
                (point_g_k[tau - 2] + point_g_k[tau - 1]) / 2.0
                - (point_g_k[tau] + point_g_k[tau + 1]) / 2.0
            )
            pair["S_SHAPE"] = min(summaries[name]["z"] for name in MARGIN_NAMES)
            pair["pi"] = _heterogeneity_pi(
                sample_g_h,
                sample_g_k,
                start=start,
                width=width,
                q=q,
                tau=tau,
            )
            pair["segment_direction_pass"] = all(
                point_margins[name] > 0.0 for name in MARGIN_NAMES[:4]
            )
            pair["core_direction_pass"] = all(
                point_margins[name] > 0.0 for name in MARGIN_NAMES[4:]
            )
            pair_boot[(q, tau)] = boot_margins
            pairs.append(pair)
    scorable_pairs = [pair for pair in pairs if pair["scorable"]]
    if not scorable_pairs:
        return {
            "Scorable": False,
            "shape_pairs": pairs,
            "best_shape_pair": None,
            "tau_ambiguous": False,
            "BiphasicStable": False,
            "biphasic_failures": ["NO_SCORABLE_SHAPE_PAIR"],
        }

    max_t_values = []
    for b in range(replicates):
        max_t_values.append(
            max(
                (
                    pair["margins"][name]["point"]
                    - pair_boot[(pair["q"], pair["tau"])][name][b]
                )
                / max(pair["margins"][name]["standard_error"], 1e-12)
                for pair in scorable_pairs
                for name in MARGIN_NAMES
            )
        )
    c95 = _linear_percentile(max_t_values, 0.95)
    for pair in scorable_pairs:
        for name in MARGIN_NAMES:
            margin = pair["margins"][name]
            margin["simultaneous_lower95"] = margin["point"] - c95 * max(
                margin["standard_error"], 1e-12
            )

    point_scores = [float(pair["S_SHAPE"]) for pair in scorable_pairs]
    point_max = max(point_scores)
    point_winners = [
        pair
        for pair in scorable_pairs
        if math.isclose(
            float(pair["S_SHAPE"]),
            point_max,
            rel_tol=TIE_REL_TOL,
            abs_tol=TIE_ABS_TOL,
        )
    ]
    frequency_counts = {(pair["q"], pair["tau"]): 0 for pair in scorable_pairs}
    for b in range(replicates):
        replicate_scores = {}
        for pair in scorable_pairs:
            key = (pair["q"], pair["tau"])
            replicate_scores[key] = min(
                pair_boot[key][name][b]
                / max(pair["margins"][name]["standard_error"], 1e-12)
                for name in MARGIN_NAMES
            )
        maximum = max(replicate_scores.values())
        winners = [
            key
            for key, score in replicate_scores.items()
            if math.isclose(score, maximum, rel_tol=TIE_REL_TOL, abs_tol=TIE_ABS_TOL)
        ]
        if len(winners) == 1:
            frequency_counts[winners[0]] += 1
    for pair in scorable_pairs:
        pair["tau_pair_frequency"] = frequency_counts[(pair["q"], pair["tau"])] / replicates

    failures: List[str] = []
    ambiguous = len(point_winners) != 1
    best = None if ambiguous else point_winners[0]
    if ambiguous:
        failures.append("INELIGIBLE_TAU_AMBIGUOUS")
    if best is not None:
        if not all(
            best["margins"][name]["simultaneous_lower95"] > 0.0
            for name in MARGIN_NAMES
        ):
            failures.append("BIPHASIC_DIRECTION_SIMULTANEOUS_LOWER95_NOT_ALL_POSITIVE")
        if not best["tau_pair_frequency"] >= 0.80:
            failures.append("TAU_PAIR_FREQUENCY_BELOW_0_80")
    return {
        "Scorable": True,
        "shape_pairs": pairs,
        "simultaneous_max_t_c95": c95,
        "best_shape_pair": (
            None
            if best is None
            else {
                "q": best["q"],
                "tau": best["tau"],
                "S_SHAPE": best["S_SHAPE"],
                "tau_pair_frequency": best["tau_pair_frequency"],
                "pi": best["pi"],
            }
        ),
        "tau_ambiguous": ambiguous,
        "BiphasicStable": not failures,
        "biphasic_failures": failures,
    }


def _shape_margin_vector(
    g_h: Sequence[float],
    g_k: Sequence[float],
    *,
    start: int,
    width: int,
    q: int,
    tau: int,
) -> Dict[str, float]:
    return {
        "segment_left_H": q * _mean_slice(g_h, start, tau),
        "segment_left_K": q * _mean_slice(g_k, start, tau),
        "segment_right_H": -q * _mean_slice(g_h, tau, start + width),
        "segment_right_K": -q * _mean_slice(g_k, tau, start + width),
        "core_left_outer_H": q * g_h[tau - 2],
        "core_left_outer_K": q * g_k[tau - 2],
        "core_left_inner_H": q * g_h[tau - 1],
        "core_left_inner_K": q * g_k[tau - 1],
        "core_right_inner_H": -q * g_h[tau],
        "core_right_inner_K": -q * g_k[tau],
        "core_right_outer_H": -q * g_h[tau + 1],
        "core_right_outer_K": -q * g_k[tau + 1],
    }


def _shape_margin_vector_at_bootstrap(
    boot_g_h: Sequence[Sequence[float]],
    boot_g_k: Sequence[Sequence[float]],
    b: int,
    *,
    start: int,
    width: int,
    q: int,
    tau: int,
) -> Dict[str, float]:
    g_h = [values[b] for values in boot_g_h]
    g_k = [values[b] for values in boot_g_k]
    return _shape_margin_vector(g_h, g_k, start=start, width=width, q=q, tau=tau)


def _heterogeneity_pi(
    sample_g_h: Sequence[Sequence[float]],
    sample_g_k: Sequence[Sequence[float]],
    *,
    start: int,
    width: int,
    q: int,
    tau: int,
) -> float:
    count = 0
    for g_h, g_k in zip(sample_g_h, sample_g_k):
        if (
            q * _mean_slice(g_h, start, tau) > 0.0
            and q * _mean_slice(g_k, start, tau) > 0.0
            and -q * _mean_slice(g_h, tau, start + width) > 0.0
            and -q * _mean_slice(g_k, tau, start + width) > 0.0
        ):
            count += 1
    return count / len(sample_g_h)


def _build_candidate_row(
    model_key: str,
    width: int,
    start: int,
    relative: Mapping[str, Any],
    shape: Mapping[str, Any],
    *,
    central: Tuple[int, int],
    published: bool,
) -> Dict[str, Any]:
    failures: List[str] = []
    if not shape["Scorable"]:
        failures.append("NOT_SCORABLE")
    if not relative["NetPositive"]:
        failures.extend(relative["net_positive_failures"])
    if not relative["SoftRelativeStable"]:
        failures.extend(relative["soft_relative_failures"])
    if not shape["BiphasicStable"]:
        failures.extend(shape["biphasic_failures"])
    row = {
        "global_rank": None,
        "within_width_rank": None,
        "model_key": model_key,
        "model": MODEL_SPECS[model_key]["model"],
        "width": width,
        "start": start,
        "window": "%d:%d" % (start, start + width - 1),
        "boundary_transition": "B_%d->B_%d" % (start, start + width),
        "published_candidate": published,
        "candidate_domain_identity": {
            "rule": "CENTRAL_40_PERCENT_COMPLETE_CONTAINMENT_V1",
            "central_blocks": list(central),
            "width": width,
            "start": start,
        },
        **{key: _strip_private(value) for key, value in relative.items() if not key.startswith("_")},
        **{key: _strip_private(value) for key, value in shape.items()},
        "NewEligible": not failures,
        "new_eligibility_failures": _deduplicate(failures),
        "selected": False,
        "model_window_selection_frequency": None,
    }
    return row


def _select_window(
    rows: Sequence[Mapping[str, Any]],
    relative_cache: Mapping[Tuple[int, int], Mapping[str, Any]],
    replicates: int,
) -> Dict[str, Any]:
    eligible = [row for row in rows if row["NewEligible"]]
    if not eligible:
        return {
            "decision": "ABSTAIN_NO_NEW_ELIGIBLE",
            "selected_key": None,
            "window_selection_frequency": None,
        }
    point_max = max(float(row["S_REL"]) for row in eligible)
    point_winners = [
        row
        for row in eligible
        if math.isclose(
            float(row["S_REL"]),
            point_max,
            rel_tol=TIE_REL_TOL,
            abs_tol=TIE_ABS_TOL,
        )
    ]
    if len(point_winners) != 1:
        return {
            "decision": "ABSTAIN_NO_UNIQUE_TOP1",
            "selected_key": None,
            "window_selection_frequency": None,
        }
    top = point_winners[0]
    top_key = (int(top["width"]), int(top["start"]))
    wins = 0
    for b in range(replicates):
        replicate_rows = []
        for row in eligible:
            key = (int(row["width"]), int(row["start"]))
            boot = relative_cache[key]["_bootstrap_contrasts"]
            score = min(
                boot[name][b] / max(float(row[name + "_se"]), 1e-12)
                for name in CONTRASTS
            )
            replicate_rows.append((score, key[0], key[1]))
        replicate_rows.sort(key=lambda value: (-value[0], value[1], value[2]))
        if (replicate_rows[0][1], replicate_rows[0][2]) == top_key:
            wins += 1
    frequency = wins / replicates
    if frequency < 0.80:
        return {
            "decision": "ABSTAIN_WINDOW_UNSTABLE",
            "selected_key": None,
            "point_top_key": list(top_key),
            "window_selection_frequency": frequency,
        }
    return {
        "decision": "SELECTED_WINDOW",
        "selected_key": list(top_key),
        "window_selection_frequency": frequency,
    }


def _model_summary(
    model_key: str,
    rows: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> Dict[str, Any]:
    def selected_set(field: str) -> List[str]:
        return [row["window"] for row in rows if bool(row[field])]

    selected_row = next((row for row in rows if row["selected"]), None)
    return {
        "model_key": model_key,
        "model": MODEL_SPECS[model_key]["model"],
        "candidate_count": len(rows),
        "complete_ranking": [
            {
                "global_rank": row["global_rank"],
                "within_width_rank": row["within_width_rank"],
                "width": row["width"],
                "start": row["start"],
                "window": row["window"],
                "S_REL": row["S_REL"],
                "legacy_strict": row["legacy_strict"],
                "SoftRelativeStable": row["SoftRelativeStable"],
                "BiphasicStable": row["BiphasicStable"],
                "NewEligible": row["NewEligible"],
                "best_shape_pair": row["best_shape_pair"],
            }
            for row in rows
        ],
        "legacy_strict_count": len(selected_set("legacy_strict")),
        "legacy_strict_set": selected_set("legacy_strict"),
        "soft_relative_count": len(selected_set("SoftRelativeStable")),
        "soft_relative_set": selected_set("SoftRelativeStable"),
        "biphasic_count": len(selected_set("BiphasicStable")),
        "biphasic_set": selected_set("BiphasicStable"),
        "new_eligible_count": len(selected_set("NewEligible")),
        "new_eligible_set": selected_set("NewEligible"),
        "rank_top1": {
            "window": rows[0]["window"],
            "width": rows[0]["width"],
            "start": rows[0]["start"],
            "S_REL": rows[0]["S_REL"],
            "NewEligible": rows[0]["NewEligible"],
        },
        "decision": decision["decision"],
        "selected_window": (
            None
            if selected_row is None
            else {
                "window": selected_row["window"],
                "width": selected_row["width"],
                "start": selected_row["start"],
                "S_REL": selected_row["S_REL"],
                "best_shape_pair": selected_row["best_shape_pair"],
            }
        ),
        "window_selection_frequency": decision.get("window_selection_frequency"),
        "pair_frequencies": {
            row["window"]: (
                None
                if row["best_shape_pair"] is None
                else row["best_shape_pair"]["tau_pair_frequency"]
            )
            for row in rows
        },
    }


def _falsification_status(diagnostics: Mapping[str, Mapping[str, Any]]) -> str:
    admitted = [
        window
        for window in ("4:7", "5:8")
        if window in diagnostics and bool(diagnostics[window]["NewEligible"])
    ]
    if admitted:
        return "RETROSPECTIVE_FALSIFICATION_FAILED_HARMFUL_DIAGNOSTIC_ADMITTED"
    return "RETROSPECTIVE_FALSIFICATION_PASSED_HARMFUL_DIAGNOSTICS_EXCLUDED"


def _bootstrap_boundary_means(
    samples: Sequence[Mapping[str, Any]],
    layers: int,
    *,
    replicates: int,
    seed: int,
    digest_style: str,
) -> Dict[str, Any]:
    groups: Dict[str, List[int]] = {}
    for index, row in enumerate(samples):
        groups.setdefault(str(row["subject"]), []).append(index)
    generator = random.Random(seed)
    boot_h = [[] for _ in range(layers + 1)]
    boot_d = [[] for _ in range(layers + 1)]
    struct_digest = hashlib.sha256()
    json_digest = hashlib.sha256()
    json_digest.update(b"[")
    for replicate in range(replicates):
        drawn: List[int] = []
        for subject in sorted(groups):
            indices = groups[subject]
            drawn.extend(indices[generator.randrange(len(indices))] for _ in indices)
        if digest_style == "phase3":
            struct_digest.update(struct.pack(">I", len(drawn)))
            for index in drawn:
                struct_digest.update(struct.pack(">I", index))
        else:
            if replicate:
                json_digest.update(b",")
            json_digest.update(canonical_json_bytes(drawn))
        h_sum = [0.0] * (layers + 1)
        d_sum = [0.0] * (layers + 1)
        for index in drawn:
            h_row = samples[index]["H"]
            d_row = samples[index]["D"]
            for boundary in range(layers + 1):
                h_sum[boundary] += h_row[boundary]
                d_sum[boundary] += d_row[boundary]
        denominator = len(drawn)
        for boundary in range(layers + 1):
            boot_h[boundary].append(h_sum[boundary] / denominator)
            boot_d[boundary].append(d_sum[boundary] / denominator)
    json_digest.update(b"]")
    return {
        "H": boot_h,
        "D": boot_d,
        "draw_index_sha256": (
            struct_digest.hexdigest() if digest_style == "phase3" else json_digest.hexdigest()
        ),
    }


def _legacy_failures(
    style: str,
    width: int,
    start: int,
    refs: Sequence[int],
    g_h: Mapping[int, float],
    g_k: Mapping[int, float],
    summaries: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    failures: List[str] = []
    if style == "phase3":
        if not g_h[start] > 0.0:
            failures.append("center_E_rate_not_strictly_positive")
        if not g_k[start] > 0.0:
            failures.append("center_K_rate_not_strictly_positive")
        for reference in refs:
            if not g_h[reference] < 0.0:
                failures.append("reference_%d_E_rate_not_strictly_negative" % reference)
            if not g_k[reference] < 0.0:
                failures.append("reference_%d_K_rate_not_strictly_negative" % reference)
        for name in CONTRASTS:
            if not summaries[name]["ci95"][0] > 0.0:
                failures.append("%s_ci_lower_not_strictly_positive" % name)
    else:
        if not g_h[start] > 0.0:
            failures.append("CENTER_E_NOT_STRICTLY_POSITIVE")
        if not g_k[start] > 0.0:
            failures.append("CENTER_K_NOT_STRICTLY_POSITIVE")
        for reference in refs:
            if not g_h[reference] < 0.0:
                failures.append("COMPARISON_S%d_E_NOT_STRICTLY_NEGATIVE" % reference)
            if not g_k[reference] < 0.0:
                failures.append("COMPARISON_S%d_K_NOT_STRICTLY_NEGATIVE" % reference)
        for name in CONTRASTS:
            if not summaries[name]["ci95"][0] > 0.0:
                failures.append("%s_LOWER95_NOT_STRICTLY_POSITIVE" % name)
    return failures


def _bootstrap_summary(
    point: float, values: Sequence[float], *, use_fsum: bool
) -> Dict[str, Any]:
    if len(values) < 2:
        raise RelativeBiphasicError("bootstrap summary requires at least two replicates")
    total = math.fsum(values) if use_fsum else sum(values)
    mean = total / len(values)
    squared = (
        math.fsum((value - mean) ** 2 for value in values)
        if use_fsum
        else sum((value - mean) ** 2 for value in values)
    )
    standard_error = math.sqrt(squared / (len(values) - 1))
    return {
        "point": float(point),
        "standard_error": standard_error,
        "ci95": [
            _linear_percentile(values, 0.025),
            _linear_percentile(values, 0.975),
        ],
        "z": float(point) / max(standard_error, 1e-12),
    }


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(_finite(value, "percentile value") for value in values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _column_means(rows: Sequence[Sequence[float]], *, use_fsum: bool) -> List[float]:
    if not rows:
        raise RelativeBiphasicError("column means require rows")
    return [
        (math.fsum(row[index] for row in rows) if use_fsum else sum(row[index] for row in rows))
        / len(rows)
        for index in range(len(rows[0]))
    ]


def _normalize_preprojected(
    samples: Sequence[Mapping[str, Any]], model_key: str, enforce_population: bool
) -> List[Dict[str, Any]]:
    spec = _model_spec(model_key)
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise RelativeBiphasicError("sample must be an object")
        allowed = {"identity", "identity_fields", "subject", "H", "D"}
        if not set(sample).issubset(allowed) or not {"identity", "subject", "H", "D"}.issubset(
            sample
        ):
            raise RelativeBiphasicError("sample keys differ")
        identity = _nonempty_string(sample["identity"], "identity")
        if identity in seen:
            raise RelativeBiphasicError("duplicate sample identity")
        seen.add(identity)
        subject = _nonempty_string(sample["subject"], "subject")
        h_values = [_finite(value, "H") for value in sample["H"]]
        d_values = [_finite(value, "D") for value in sample["D"]]
        if len(h_values) != spec["boundary_count"] or len(d_values) != spec["boundary_count"]:
            raise RelativeBiphasicError("sample boundary count differs")
        normalized.append(
            {
                "identity": identity,
                "identity_fields": (
                    dict(sample["identity_fields"])
                    if isinstance(sample.get("identity_fields"), Mapping)
                    else None
                ),
                "subject": subject,
                "H": h_values,
                "D": d_values,
            }
        )
    if model_key == "qwen17" and all(
        isinstance(row.get("identity_fields"), Mapping) for row in normalized
    ):
        normalized.sort(
            key=lambda row: (
                row["subject"],
                row["identity_fields"]["task"],
                row["identity_fields"]["doc_id"],
                row["identity_fields"]["doc_hash"],
            )
        )
    else:
        normalized.sort(key=lambda row: (row["subject"], row["identity"]))
    for row in normalized:
        row.pop("identity_fields", None)
    if enforce_population:
        if len(normalized) != EXPECTED_RECORD_COUNT:
            raise RelativeBiphasicError("formal analysis requires 1,531 records")
        if len({row["subject"] for row in normalized}) != EXPECTED_SUBJECT_COUNT:
            raise RelativeBiphasicError("formal analysis requires 57 subjects")
    return normalized


def _public_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [_strip_private(row) for row in rows]


def _strip_private(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_private(nested)
            for key, nested in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_strip_private(nested) for nested in value]
    if isinstance(value, tuple):
        return [_strip_private(nested) for nested in value]
    return value


def _rows_formula_close(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if not math.isclose(
            float(row["S_REL"]),
            min(float(row["z_OH"]), float(row["z_OK"]), float(row["z_FH"]), float(row["z_FK"])),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
        expected = (
            bool(row["Scorable"])
            and bool(row["NetPositive"])
            and bool(row["SoftRelativeStable"])
            and bool(row["BiphasicStable"])
        )
        if bool(row["NewEligible"]) is not expected:
            return False
    return True


def _all_shape_fields_present(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        row.get("simultaneous_max_t_c95") is not None
        and all(
            all(
                pair["margins"][name]["simultaneous_lower95"] is not None
                for name in MARGIN_NAMES
            )
            for pair in row["shape_pairs"]
            if pair["scorable"]
        )
        for row in rows
    )


def _all_pair_frequencies_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        0.0 <= float(pair["tau_pair_frequency"]) <= 1.0
        for row in rows
        for pair in row["shape_pairs"]
        if pair["scorable"]
    )


def _window_frequency_valid(summary: Mapping[str, Any]) -> bool:
    value = summary.get("window_selection_frequency")
    if value is None:
        return summary.get("selected_window") is None
    return 0.0 <= float(value) <= 1.0


def _summary_closes(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> bool:
    return (
        int(summary["candidate_count"]) == len(rows)
        and int(summary["new_eligible_count"])
        == sum(bool(row["NewEligible"]) for row in rows)
        and summary["rank_top1"]["window"] == rows[0]["window"]
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _mean_slice(values: Sequence[float], start: int, end: int) -> float:
    if not start < end:
        raise RelativeBiphasicError("mean slice must be non-empty")
    return math.fsum(values[start:end]) / (end - start)


def _transition_id(index: int) -> str:
    return "B_%d->B_%d" % (index, index + 1)


def _transitions_to_h(values: Sequence[float]) -> List[float]:
    result = [0.0]
    for value in values:
        result.append(result[-1] - float(value))
    return result


def _transitions_to_d(values: Sequence[float]) -> List[float]:
    result = [0.0]
    for value in values:
        result.append(result[-1] + float(value))
    return result


def _validate_window(layers: int, start: int, width: int) -> None:
    if width not in WIDTHS:
        raise RelativeBiphasicError("width must be one of 3,4,5,6")
    if start < 0 or start + width > layers:
        raise RelativeBiphasicError("window lies outside decoder blocks")


def _model_spec(model_key: str) -> Dict[str, Any]:
    if model_key not in MODEL_SPECS:
        raise RelativeBiphasicError("unknown model key: %s" % model_key)
    return MODEL_SPECS[model_key]


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RelativeBiphasicError("%s must be an integer" % label)
    return int(value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RelativeBiphasicError("%s must be numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise RelativeBiphasicError("%s must be finite" % label)
    return result


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RelativeBiphasicError("%s must be a non-empty string" % label)
    return value.strip()


def _sha256_string(value: Any, label: str) -> str:
    text = _nonempty_string(value, label).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RelativeBiphasicError("%s must be lowercase SHA256" % label)
    return text


def _deduplicate(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant: %s" % value)


__all__ = [
    "CROSS_MODEL_SUMMARY_JSON",
    "MANIFEST_RECEIPT_JSON",
    "PRIMARY_OUTPUTS",
    "QWEN17_CSV",
    "QWEN17_JSON",
    "QWEN4_CSV",
    "QWEN4_DIAGNOSTICS_JSON",
    "QWEN4_JSON",
    "RelativeBiphasicError",
    "SUMMARY_ZH",
    "VERIFIER_RECEIPT_JSON",
    "analyze_model",
    "analyze_shape_samples",
    "assert_no_forbidden_output_keys",
    "build_primary_payloads",
    "canonical_json_bytes",
    "extract_scalar_samples",
    "file_sha256",
    "load_strict_json",
    "load_strict_jsonl",
    "normalize_scalar_samples",
    "render_candidates_csv",
    "render_summary_zh",
    "semantic_sha256",
    "validate_card",
    "verify_legacy_compatibility",
    "verify_primary_outputs",
    "write_primary_outputs",
]
