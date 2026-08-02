"""Pure CPU implementation of the frozen Phase 6 Gate N V3 rescore.

The module deliberately exposes only the selector projection ``identity``,
``category``, ``H`` and ``D`` to the V3 math.  The producer reads the already
verified Gate M JSONL trajectory, never runs a model, and never reads an
outcome artifact.  The independent verifier in
``phase6_mmlu5_v3_verifier.py`` repeats the math in a fresh process.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


LAYERS = 36
BOUNDARY_COUNT = 37
WIDTHS = (3, 4, 5, 6)
STARTS = {
    3: tuple(range(11, 23)),
    4: tuple(range(11, 22)),
    5: tuple(range(11, 21)),
    6: tuple(range(11, 20)),
}
FORMAL_REPLICATES = 2000
FORMAL_SEED = 20260801
Q_THRESHOLD = 0.5
FREQUENCY_THRESHOLD = 0.80
REL_TOL = 1e-12
ABS_TOL = 1e-12
EPSILON = 1e-12

METHOD_ID = "AGGREGATE_COMMON_TURN_V3_ABSOLUTE_RATE"
METHOD_VERSION = "3.1.0"
METHOD_FILE_SHA256 = "2a7aaac6bcd4e0d4995758c8e25a720469c1d6e2afd65e7a1143103f31556644"
SOURCE_GATE_M_COMMIT = "9446f0e4940a7ae762c95625f8bd891ac4090f6c"
# The current Gate N implementation commit is supplied by the launcher.  It
# is distinct from the frozen Gate M source commit above.
EXPECTED_COMMIT = SOURCE_GATE_M_COMMIT
EXPECTED_TRAJECTORY_SHA256 = "0c6989abac8fab56d1661ffdfad9e291e2591b5eb6c87738cfd92282631a1d73"
EXPECTED_SOURCE_ROOT_NAME = "phase6-gate-m-mmlu5-formal-20260802T085444Z"
EXPECTED_RECORD_COUNT = 1531
EXPECTED_CATEGORY_COUNT = 57

INPUT_CLOSURE_SHA256 = {
    "manifest/formal_manifest.json": "5fec780360b38676bda4cbb9cd7b8ac43ffa9480bce3283acb4f267cb288c6be",
    "manifest/freeze_receipt.json": "d57de8faed5e66a0f6c36f1af43602bc008349bd8d68426cd85dbd3a695b1468",
    "merge/merge_receipt.json": "c1b3b5f2dc3330caaf22bc4f81ec5a6fa5b367495094f9a8d451bf6ee6fde6df",
    "merge/merged_trajectory_records.jsonl": EXPECTED_TRAJECTORY_SHA256,
    "merge/verifier_receipt.json": "13b81088f15428161afdfcd97ec7583cce42cb9f043199ffb76d6c4ed87634ec",
    "selector/selector_freeze.json": "a05ab56e9d9c38c2d98034bda829a6930660184759f27f6c9af38c3fad8ba95f",
}

LEGAL_TERMINAL_STATES = (
    "SELECTED_WINDOW",
    "ABSTAIN_NO_V3_ELIGIBLE",
    "ABSTAIN_COMBINED_RANK_UNSTABLE",
)
PROJECTION_FIELDS = ("identity", "category", "H", "D")
FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "gold",
        "label",
        "correctness",
        "accuracy",
        "gain",
        "flip",
        "outcome",
        "prediction",
        "answer",
        "target",
    }
)


class GateNV3Error(ValueError):
    """Fail-closed Gate N error with a planning-readable BLOCK reason."""


def _validate_commit(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise GateNV3Error("BLOCK_REPOSITORY_ADMISSION_MISMATCH: commit is not a SHA-1")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GateNV3Error("BLOCK_REPOSITORY_ADMISSION_MISMATCH: commit is not hexadecimal") from exc
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def write_new_bytes(path: Path, data: bytes) -> str:
    path = Path(path)
    if path.exists():
        raise GateNV3Error("BLOCK_WRITE_ONCE_VIOLATION: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return file_sha256(path)


def write_new_json(path: Path, value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    return write_new_bytes(path, payload)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateNV3Error("BLOCK_INPUT_CLOSURE_READ: %s" % path) from exc
    if not isinstance(value, dict):
        raise GateNV3Error("BLOCK_INPUT_CLOSURE_SCHEMA: %s" % path)
    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GateNV3Error("BLOCK_INPUT_CLOSURE_READ: %s" % path) from exc
    rows: List[Dict[str, Any]] = []
    try:
        for line in lines:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row is not an object")
                rows.append(value)
    except (json.JSONDecodeError, ValueError) as exc:
        raise GateNV3Error("BLOCK_INVALID_V3_INPUT: malformed trajectory JSONL") from exc
    return rows


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateNV3Error("BLOCK_INVALID_V3_INPUT: %s is not numeric" % label)
    result = float(value)
    if not math.isfinite(result):
        raise GateNV3Error("BLOCK_INVALID_V3_INPUT: %s is non-finite" % label)
    return result


def _canonical_identity(identity: Any) -> str:
    if not isinstance(identity, Mapping):
        raise GateNV3Error("BLOCK_INVALID_V3_INPUT: identity is not an object")
    try:
        return json.dumps(
            dict(identity),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise GateNV3Error("BLOCK_INVALID_V3_INPUT: identity is not canonical") from exc


def enumerate_candidates() -> List[Dict[str, Any]]:
    candidates = [
        {
            "width": width,
            "start": start,
            "stop_exclusive": start + width,
            "end_inclusive": start + width - 1,
            "window_half_open": "%d:%d" % (start, start + width),
            "window_layers_inclusive": "%d:%d" % (start, start + width - 1),
            "boundary_entry": "B_%d" % start,
            "boundary_exit": "B_%d" % (start + width),
        }
        for width in WIDTHS
        for start in STARTS[width]
    ]
    pairs = [(row["width"], row["start"]) for row in candidates]
    expected = [(width, start) for width in WIDTHS for start in STARTS[width]]
    if len(candidates) != 42 or pairs != expected or len(set(pairs)) != 42:
        raise GateNV3Error("BLOCK_CANDIDATE_DOMAIN_MISMATCH")
    return candidates


def project_records(
    records: Sequence[Mapping[str, Any]], *, formal: bool = False
) -> List[Dict[str, Any]]:
    """Project safe trajectory records to exactly identity/category/H/D."""

    projected: List[Dict[str, Any]] = []
    seen = set()
    for ordinal, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise GateNV3Error("BLOCK_INVALID_V3_INPUT: record is not an object")
        if FORBIDDEN_OUTCOME_KEYS.intersection(record):
            raise GateNV3Error("BLOCK_INFORMATION_BARRIER_VIOLATION: outcome key in record")
        required = {"identity", "subject", "split", "boundaries"}
        if not required.issubset(record):
            raise GateNV3Error("BLOCK_INVALID_V3_INPUT: trajectory record fields are incomplete")
        if record.get("split") != "validation":
            raise GateNV3Error("BLOCK_INVALID_V3_INPUT: trajectory split differs")
        identity = _canonical_identity(record["identity"])
        if not isinstance(record["subject"], str) or not record["subject"]:
            raise GateNV3Error("BLOCK_INVALID_V3_INPUT: category is empty")
        if identity in seen:
            raise GateNV3Error("BLOCK_INVALID_V3_INPUT: duplicate identity")
        seen.add(identity)
        boundaries = record["boundaries"]
        if isinstance(boundaries, (str, bytes)) or not isinstance(boundaries, Sequence):
            raise GateNV3Error("BLOCK_INVALID_V3_INPUT: boundaries are not a sequence")
        if len(boundaries) != BOUNDARY_COUNT:
            raise GateNV3Error("BLOCK_INVALID_V3_INPUT: boundary count differs")
        values_h: List[float] = []
        values_d: List[float] = []
        for index, boundary in enumerate(boundaries):
            if not isinstance(boundary, Mapping):
                raise GateNV3Error("BLOCK_INVALID_V3_INPUT: boundary is not an object")
            if boundary.get("boundary_id") != "B_%d" % index:
                raise GateNV3Error("BLOCK_INVALID_V3_INPUT: boundary IDs are not canonical")
            h_value = _finite(boundary.get("choice_entropy"), "H[%d]" % index)
            d_value = _finite(boundary.get("kl_to_final"), "D[%d]" % index)
            if h_value < 0.0 or d_value < 0.0:
                raise GateNV3Error("BLOCK_INVALID_V3_INPUT: H/D is negative")
            values_h.append(h_value)
            values_d.append(d_value)
        projected.append(
            {
                "identity": identity,
                "category": record["subject"],
                "H": values_h,
                "D": values_d,
            }
        )
    if formal:
        if len(projected) != EXPECTED_RECORD_COUNT:
            raise GateNV3Error("BLOCK_POPULATION_OR_HASH_MISMATCH: record count differs")
        if len({row["category"] for row in projected}) != EXPECTED_CATEGORY_COUNT:
            raise GateNV3Error("BLOCK_POPULATION_OR_HASH_MISMATCH: category count differs")
    return projected


def _category_layout(projected: Sequence[Mapping[str, Any]]) -> Tuple[List[str], Dict[str, List[Tuple[int, Mapping[str, Any]]]]]:
    groups: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = {}
    for row in projected:
        groups.setdefault(str(row["category"]), []).append((0, row))
    categories = sorted(groups)
    indexed: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = {}
    global_index = 0
    for category in categories:
        ordered = sorted(groups[category], key=lambda item: str(item[1]["identity"]))
        indexed[category] = []
        for _old, row in ordered:
            indexed[category].append((global_index, row))
            global_index += 1
    return categories, indexed


def _macro_curves(
    projected: Sequence[Mapping[str, Any]],
) -> Tuple[List[str], List[float], List[float]]:
    categories, groups = _category_layout(projected)
    if not categories or any(not groups[category] for category in categories):
        raise GateNV3Error("BLOCK_INVALID_V3_INPUT: empty category")
    h_bar: List[float] = []
    d_bar: List[float] = []
    for layer in range(BOUNDARY_COUNT):
        category_h = [
            math.fsum(float(row["H"][layer]) for _idx, row in groups[category])
            / len(groups[category])
            for category in categories
        ]
        category_d = [
            math.fsum(float(row["D"][layer]) for _idx, row in groups[category])
            / len(groups[category])
            for category in categories
        ]
        h_bar.append(math.fsum(category_h) / len(categories))
        d_bar.append(math.fsum(category_d) / len(categories))
    return categories, h_bar, d_bar


def _bootstrap_macro_curves(
    projected: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> Dict[str, Any]:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise GateNV3Error("BLOCK_INVALID_BOOTSTRAP: at least two replicates required")
    categories, groups = _category_layout(projected)
    rng = random.Random(seed)
    draw_index: List[List[int]] = []
    h_curves: List[List[float]] = []
    d_curves: List[List[float]] = []
    for _replicate in range(replicates):
        drawn_global: List[int] = []
        category_h: List[List[float]] = []
        category_d: List[List[float]] = []
        for category in categories:
            group = groups[category]
            choices = [rng.randrange(len(group)) for _ in group]
            drawn_global.extend(group[index][0] for index in choices)
            category_h.append(
                [math.fsum(group[index][1]["H"][layer] for index in choices) / len(group) for layer in range(BOUNDARY_COUNT)]
            )
            category_d.append(
                [math.fsum(group[index][1]["D"][layer] for index in choices) / len(group) for layer in range(BOUNDARY_COUNT)]
            )
        draw_index.append(drawn_global)
        h_curves.append([math.fsum(row[layer] for row in category_h) / len(categories) for layer in range(BOUNDARY_COUNT)])
        d_curves.append([math.fsum(row[layer] for row in category_d) / len(categories) for layer in range(BOUNDARY_COUNT)])
    return {
        "draw_index": draw_index,
        "draw_index_sha256": hashlib.sha256(canonical_json_bytes(draw_index)).hexdigest(),
        "H": h_curves,
        "D": d_curves,
    }


def _rates(h_bar: Sequence[float], d_bar: Sequence[float]) -> Tuple[List[float], List[float]]:
    if len(h_bar) != BOUNDARY_COUNT or len(d_bar) != BOUNDARY_COUNT:
        raise GateNV3Error("BLOCK_INVALID_V3_INPUT: macro boundary count differs")
    r_h = [float(h_bar[index]) - float(h_bar[index + 1]) for index in range(LAYERS)]
    r_k = [float(d_bar[index + 1]) - float(d_bar[index]) for index in range(LAYERS)]
    if not all(math.isfinite(value) for value in r_h + r_k):
        raise GateNV3Error("BLOCK_INVALID_V3_TURN_ANALYSIS: non-finite rate")
    return r_h, r_k


def _fit_at_tau(series: Sequence[float], start: int, width: int, tau: int) -> Dict[str, Any]:
    values = [float(value) for value in series[start : start + width]]
    split = tau - start
    if split <= 0 or split >= width:
        raise GateNV3Error("BLOCK_INVALID_V3_TURN_ANALYSIS: tau is not internal")
    left = values[:split]
    right = values[split:]
    mu_left = math.fsum(left) / len(left)
    mu_right = math.fsum(right) / len(right)
    mu_all = math.fsum(values) / len(values)
    delta = mu_right - mu_left
    turn = abs(delta)
    sst = math.fsum((value - mu_all) ** 2 for value in values)
    bss = (len(left) * len(right) / width) * (mu_left - mu_right) ** 2
    wss = math.fsum((value - mu_left) ** 2 for value in left) + math.fsum(
        (value - mu_right) ** 2 for value in right
    )
    decomposition_tolerance = 1e-10 * max(sst, bss, wss, 1e-24)
    if bss < -decomposition_tolerance or wss < -decomposition_tolerance:
        raise GateNV3Error("BLOCK_INVALID_V3_TURN_ANALYSIS: negative BSS/WSS")
    if abs(sst - bss - wss) > decomposition_tolerance:
        raise GateNV3Error("BLOCK_INVALID_V3_TURN_ANALYSIS: SST decomposition mismatch")
    q: float | None
    if sst == 0.0:
        q = None
    else:
        q = bss / sst
        if q < 0.0 and abs(q) <= 1e-12:
            q = 0.0
        if q > 1.0 and abs(q - 1.0) <= 1e-12:
            q = 1.0
        if q < -1e-12 or q > 1.0 + 1e-12:
            raise GateNV3Error("BLOCK_INVALID_V3_TURN_ANALYSIS: Q is outside [0,1]")
    return {
        "tau": tau,
        "n_L": len(left),
        "n_R": len(right),
        "mu_L": mu_left,
        "mu_R": mu_right,
        "delta": delta,
        "T": turn,
        "BSS": bss,
        "WSS": wss,
        "Q": q,
        "SST": sst,
    }


def _fit_turn(series: Sequence[float], start: int, width: int) -> Dict[str, Any]:
    values = [float(value) for value in series[start : start + width]]
    if len(values) != width or not all(math.isfinite(value) for value in values):
        raise GateNV3Error("BLOCK_INVALID_V3_TURN_ANALYSIS: incomplete rate vector")
    scale = max(1.0, max(abs(value) for value in values))
    rate_tol = 1e-12 * scale
    sst_tol = width * rate_tol**2
    entries = [_fit_at_tau(series, start, width, tau) for tau in range(start + 1, start + width)]
    measurable = sum((float(value) - (math.fsum(values) / width)) ** 2 for value in values) > sst_tol and any(
        float(entry["T"]) > rate_tol for entry in entries
    )
    if not measurable:
        return {
            "entries": entries,
            "tau_candidates": [entry["tau"] for entry in entries],
            "tau": None,
            "ties": [],
            "turn_strength_pass": False,
            "failure": "NO_MEASURABLE_TURN",
            "rate_tol": rate_tol,
            "sst_tol": sst_tol,
        }
    q_entries = [entry for entry in entries if entry["Q"] is not None]
    q_max = max(float(entry["Q"]) for entry in q_entries)
    ties = [entry["tau"] for entry in q_entries if math.isclose(float(entry["Q"]), q_max, rel_tol=REL_TOL, abs_tol=ABS_TOL)]
    if len(ties) != 1:
        return {
            "entries": entries,
            "tau_candidates": [entry["tau"] for entry in entries],
            "tau": None,
            "ties": ties,
            "turn_strength_pass": False,
            "failure": "TURN_TAU_TIE",
            "rate_tol": rate_tol,
            "sst_tol": sst_tol,
        }
    tau = ties[0]
    winner = next(entry for entry in entries if entry["tau"] == tau)
    dominant = float(winner["Q"]) > Q_THRESHOLD and not math.isclose(
        float(winner["Q"]), Q_THRESHOLD, rel_tol=REL_TOL, abs_tol=ABS_TOL
    )
    return {
        "entries": entries,
        "tau_candidates": [entry["tau"] for entry in entries],
        "tau": tau,
        "ties": ties,
        "turn_strength_pass": dominant,
        "failure": None if dominant else "TURN_NOT_DOMINANT",
        "rate_tol": rate_tol,
        "sst_tol": sst_tol,
    }


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise GateNV3Error("BLOCK_INVALID_BOOTSTRAP: ddof=1 needs two values")
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _linear_quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise GateNV3Error("BLOCK_INVALID_BOOTSTRAP: empty quantile input")
    ordered = sorted(float(value) for value in values)
    position = q * (len(ordered) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    fraction = position - left
    return ordered[left] + fraction * (ordered[right] - ordered[left])


def compute_rate_statistics(
    point_h: float, point_k: float, bootstrap_h: Sequence[float], bootstrap_k: Sequence[float]
) -> Dict[str, Any]:
    point_h = _finite(point_h, "G_H")
    point_k = _finite(point_k, "G_K")
    if len(bootstrap_h) != len(bootstrap_k) or len(bootstrap_h) < 2:
        raise GateNV3Error("BLOCK_INVALID_BOOTSTRAP: H/K draw count differs")
    h_values = [_finite(value, "bootstrap G_H") for value in bootstrap_h]
    k_values = [_finite(value, "bootstrap G_K") for value in bootstrap_k]
    se_h = _sample_sd(h_values)
    se_k = _sample_sd(k_values)
    max_t = [
        max(
            (point_h - h_value) / max(se_h, EPSILON),
            (point_k - k_value) / max(se_k, EPSILON),
        )
        for h_value, k_value in zip(h_values, k_values)
    ]
    c95 = max(0.0, _linear_quantile(max_t, 0.95))
    lcb_h = point_h - c95 * max(se_h, EPSILON)
    lcb_k = point_k - c95 * max(se_k, EPSILON)
    values = (se_h, se_k, c95, lcb_h, lcb_k)
    if not all(math.isfinite(value) for value in values):
        raise GateNV3Error("BLOCK_NONFINITE_RATE_ANALYSIS")
    return {
        "rate_SE_H": se_h,
        "rate_SE_K": se_k,
        "rate_c95": c95,
        "rate_LCB_H": lcb_h,
        "rate_LCB_K": lcb_k,
        "RateStable": lcb_h > 0.0 and lcb_k > 0.0,
    }


def decide_terminal(
    eligible_count: int, point_frequency: float | None
) -> Tuple[str, List[int] | None]:
    """Apply the exact three-state V3 terminal rule."""

    if eligible_count == 0:
        return "ABSTAIN_NO_V3_ELIGIBLE", None
    if point_frequency is None or point_frequency < FREQUENCY_THRESHOLD:
        return "ABSTAIN_COMBINED_RANK_UNSTABLE", None
    return "SELECTED_WINDOW", []


def _candidate_key(row: Mapping[str, Any]) -> Tuple[int, int]:
    return int(row["width"]), int(row["start"])


def _rank_anchored(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[List[Tuple[int, int]]], Dict[Tuple[int, int], int]]:
    remaining = [row for row in rows if row.get("EligibleV3")]
    groups: List[List[Tuple[int, int]]] = []
    ranks: Dict[Tuple[int, int], int] = {}
    rank = 1
    while remaining:
        anchor = max(float(row["S_RATE_TURN"]) for row in remaining)
        tied = [
            row
            for row in remaining
            if math.isclose(float(row["S_RATE_TURN"]), anchor, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        ]
        tied.sort(key=_candidate_key)
        keys = [_candidate_key(row) for row in tied]
        groups.append(keys)
        for row in tied:
            ranks[_candidate_key(row)] = rank
        remaining = [row for row in remaining if _candidate_key(row) not in set(keys)]
        rank += 1
    return groups, ranks


def _fixed_tau_q(series: Sequence[float], start: int, width: int, tau: int) -> float | None:
    fit = _fit_at_tau(series, start, width, tau)
    scale = max(1.0, max(abs(float(value)) for value in series[start : start + width]))
    rate_tol = 1e-12 * scale
    sst_tol = width * rate_tol**2
    if fit["SST"] <= sst_tol or fit["T"] <= rate_tol or fit["Q"] is None:
        return None
    q = float(fit["Q"])
    if q <= 0.0:
        return None
    return q


def _row_failure_reasons(row: Mapping[str, Any], turn_h: Mapping[str, Any], turn_k: Mapping[str, Any]) -> List[str]:
    reasons: List[str] = []
    if not row["Scorable"]:
        reasons.append("NOT_SCORABLE")
    if not row["NetPositive"]:
        reasons.append("NET_NOT_POSITIVE")
    if not row["RateStable"]:
        reasons.append("RATE_NOT_STABLE")
    if turn_h.get("failure"):
        reasons.append("%s_H" % turn_h["failure"])
    if turn_k.get("failure"):
        reasons.append("%s_K" % turn_k["failure"])
    if turn_h.get("tau") is not None and turn_k.get("tau") is not None and turn_h["tau"] != turn_k["tau"]:
        reasons.append("TURN_LOCATION_MISMATCH")
    return reasons


def analyze_projected(
    projected: Sequence[Mapping[str, Any]],
    *,
    source_root: str = "",
    source_hashes: Mapping[str, str] | None = None,
    expected_commit: str = EXPECTED_COMMIT,
    selector_projection_file_sha256: str = "",
    replicates: int = FORMAL_REPLICATES,
    seed: int = FORMAL_SEED,
    formal: bool = False,
) -> Dict[str, Any]:
    """Compute one V3 analysis from a projected trajectory."""

    if formal and (replicates != FORMAL_REPLICATES or seed != FORMAL_SEED):
        raise GateNV3Error("BLOCK_METHOD_CONTRACT_MISMATCH: formal bootstrap differs")
    _validate_commit(expected_commit)
    projected = [dict(row) for row in projected]
    projected = project_records(
        [
            {
                "identity": json.loads(row["identity"]),
                "subject": row["category"],
                "split": "validation",
                "boundaries": [
                    {"boundary_id": "B_%d" % index, "choice_entropy": row["H"][index], "kl_to_final": row["D"][index]}
                    for index in range(BOUNDARY_COUNT)
                ],
            }
            for row in projected
        ],
        formal=formal,
    )
    categories, h_bar, d_bar = _macro_curves(projected)
    r_h, r_k = _rates(h_bar, d_bar)
    bootstrap = _bootstrap_macro_curves(projected, replicates, seed)
    rows: List[Dict[str, Any]] = []
    internal_turns: Dict[Tuple[int, int], Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for candidate in enumerate_candidates():
        width = int(candidate["width"])
        start = int(candidate["start"])
        g_h = (h_bar[start] - h_bar[start + width]) / width
        g_k = (d_bar[start + width] - d_bar[start]) / width
        turn_h = _fit_turn(r_h, start, width)
        turn_k = _fit_turn(r_k, start, width)
        internal_turns[(width, start)] = (turn_h, turn_k)
        net_positive = math.isfinite(g_h) and math.isfinite(g_k) and g_h > 0.0 and g_k > 0.0
        common_turn = (
            turn_h["tau"] is not None
            and turn_k["tau"] is not None
            and turn_h["turn_strength_pass"]
            and turn_k["turn_strength_pass"]
            and turn_h["tau"] == turn_k["tau"]
        )
        s_rate = math.sqrt(g_h * g_k) if net_positive else None
        s_turn = (
            math.sqrt(float(turn_h["entries"][turn_h["tau"] - start]["Q"]) * float(turn_k["entries"][turn_k["tau"] - start]["Q"]))
            if common_turn
            else None
        )
        s_rate_turn = s_rate * s_turn if s_rate is not None and s_turn is not None else None
        row = {
            **candidate,
            "Scorable": True,
            "NetPositive": net_positive,
            "RateStable": False,
            "AggregateCommonTurn": common_turn,
            "EligibleV3": False,
            "G_H": g_h,
            "G_K": g_k,
            "S_RATE": s_rate,
            "S_TURN": s_turn,
            "S_RATE_TURN": s_rate_turn,
            "rate_SE_H": None,
            "rate_SE_K": None,
            "rate_c95": None,
            "rate_LCB_H": None,
            "rate_LCB_K": None,
            "rH_window": r_h[start : start + width],
            "rK_window": r_k[start : start + width],
            "tau_candidates": list(turn_h["tau_candidates"]),
            "turn_H_by_tau": turn_h["entries"],
            "turn_K_by_tau": turn_k["entries"],
            "tau_H": turn_h["tau"],
            "tau_K": turn_k["tau"],
            "common_tau": turn_h["tau"] if common_turn else None,
            "turn_T_H": None,
            "turn_T_K": None,
            "turn_delta_H": None,
            "turn_delta_K": None,
            "turn_BSS_H": None,
            "turn_BSS_K": None,
            "turn_WSS_H": None,
            "turn_WSS_K": None,
            "turn_Q_H": None,
            "turn_Q_K": None,
            "turn_tau_tie_H": list(turn_h["ties"]),
            "turn_tau_tie_K": list(turn_k["ties"]),
            "turn_strength_pass_H": bool(turn_h["turn_strength_pass"]),
            "turn_strength_pass_K": bool(turn_k["turn_strength_pass"]),
            "point_rank": None,
            "point_top_tie": False,
            "tie_break_applied": False,
            "combined_rank_selection_frequency": None,
            "selected": False,
            "v3_failure_reasons": [],
            "legacy_diagnostics": {},
            "hidden_diagnostics": {},
        }
        if turn_h["tau"] is not None:
            fit = turn_h["entries"][turn_h["tau"] - start]
            row.update({"turn_T_H": fit["T"], "turn_delta_H": fit["delta"], "turn_BSS_H": fit["BSS"], "turn_WSS_H": fit["WSS"], "turn_Q_H": fit["Q"]})
        if turn_k["tau"] is not None:
            fit = turn_k["entries"][turn_k["tau"] - start]
            row.update({"turn_T_K": fit["T"], "turn_delta_K": fit["delta"], "turn_BSS_K": fit["BSS"], "turn_WSS_K": fit["WSS"], "turn_Q_K": fit["Q"]})
        rows.append(row)

    for row in rows:
        width, start = _candidate_key(row)
        boot_h = [(curve[start] - curve[start + width]) / width for curve in bootstrap["H"]]
        boot_k = [(curve[start + width] - curve[start]) / width for curve in bootstrap["D"]]
        statistics = compute_rate_statistics(row["G_H"], row["G_K"], boot_h, boot_k)
        row.update(statistics)
        row["EligibleV3"] = bool(row["Scorable"] and row["NetPositive"] and row["RateStable"] and row["AggregateCommonTurn"])

    ranking_groups, ranks = _rank_anchored(rows)
    point_top_tie_set = ranking_groups[0] if ranking_groups else []
    point_operational_winner = point_top_tie_set[0] if point_top_tie_set else None
    for row in rows:
        key = _candidate_key(row)
        row["point_rank"] = ranks.get(key)
        row["point_top_tie"] = key in set(point_top_tie_set)
        row["tie_break_applied"] = bool(row["EligibleV3"] and len(next((group for group in ranking_groups if key in group), [])) > 1)

    winner_keys: List[List[int] | None] = []
    winner_counts: Dict[Tuple[int, int], int] = {_candidate_key(row): 0 for row in rows if row["EligibleV3"]}
    eligible_rows = [row for row in rows if row["EligibleV3"]]
    for replicate_index in range(replicates):
        available: List[Tuple[float, Tuple[int, int]]] = []
        r_h_boot, r_k_boot = _rates(bootstrap["H"][replicate_index], bootstrap["D"][replicate_index])
        for row in eligible_rows:
            width, start = _candidate_key(row)
            g_h = (bootstrap["H"][replicate_index][start] - bootstrap["H"][replicate_index][start + width]) / width
            g_k = (bootstrap["D"][replicate_index][start + width] - bootstrap["D"][replicate_index][start]) / width
            if not math.isfinite(g_h) or not math.isfinite(g_k):
                raise GateNV3Error("BLOCK_NONFINITE_RATE_ANALYSIS")
            if g_h <= 0.0 or g_k <= 0.0:
                continue
            tau_h = int(row["tau_H"])
            tau_k = int(row["tau_K"])
            q_h = _fixed_tau_q(r_h_boot, start, width, tau_h)
            q_k = _fixed_tau_q(r_k_boot, start, width, tau_k)
            if q_h is None or q_k is None:
                continue
            score = math.sqrt(g_h * g_k * q_h * q_k)
            if not math.isfinite(score):
                raise GateNV3Error("BLOCK_INVALID_V3_BOOTSTRAP_TURN_ANALYSIS")
            available.append((score, (width, start)))
        if not available:
            winner_keys.append(None)
            continue
        anchor = max(score for score, _key in available)
        ties = sorted(
            key for score, key in available if math.isclose(score, anchor, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        )
        winner = ties[0]
        winner_counts[winner] += 1
        winner_keys.append([winner[0], winner[1]])

    for row in rows:
        key = _candidate_key(row)
        if row["EligibleV3"]:
            row["combined_rank_selection_frequency"] = winner_counts[key] / replicates

    top_frequency = (winner_counts[point_operational_winner] / replicates) if point_operational_winner else None
    decision, selected_key = decide_terminal(len(eligible_rows), top_frequency)
    if decision == "SELECTED_WINDOW":
        selected_key = list(point_operational_winner) if point_operational_winner else None
    for row in rows:
        row["selected"] = bool(decision == "SELECTED_WINDOW" and _candidate_key(row) == point_operational_winner)
        turn_h, turn_k = internal_turns[_candidate_key(row)]
        reasons = _row_failure_reasons(row, turn_h, turn_k)
        if row["EligibleV3"] and decision == "ABSTAIN_COMBINED_RANK_UNSTABLE":
            reasons.append("COMBINED_RANK_UNSTABLE")
        row["v3_failure_reasons"] = reasons

    rows.sort(key=_candidate_key)
    ranking_digest = semantic_sha256(rows)
    payload: Dict[str, Any] = {
        "schema_version": "loopscope.phase6.mmlu5-gate-n-v3-selector-freeze.v1",
        "artifact_role": "gate_n_post_hoc_v3_selector_freeze",
        "gate": "N",
        "planning_thread_id": "019fb3de-2298-75f2-a083-0dca453ea79c",
        "executor_thread_id": "019fc2f1-812c-73a3-b5e3-011222a2f539",
        "expected_commit": expected_commit,
        "source_gate_m_root": str(source_root),
        "source_input_closure": dict(source_hashes or INPUT_CLOSURE_SHA256),
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
        "method_file_sha256": METHOD_FILE_SHA256,
        "input_distribution_contract": "full-vocabulary entropy and KL-to-final from sanitized Gate M MMLU 5-shot trajectory",
        "probe_contract": "Gate M validation-1531 native no-loop scalar trajectory; post-hoc rescore",
        "trajectory_file_sha256": EXPECTED_TRAJECTORY_SHA256,
        "identity_manifest_sha256": semantic_sha256([json.loads(row["identity"]) for row in projected]),
        "record_count": len(projected),
        "category_count": len(categories),
        "ordered_category_sha256": semantic_sha256(categories),
        "layer_count": LAYERS,
        "boundary_count": BOUNDARY_COUNT,
        "central_blocks": [11, 24],
        "widths": list(WIDTHS),
        "candidate_count": len(rows),
        "aggregation": "equal_category_macro",
        "selector_projection_fields": list(PROJECTION_FIELDS),
        "hidden_diagnostics_in_selector": False,
        "outcome_fields_consumed": False,
        "bootstrap_method": "within-category resample, equal-category-macro joint bootstrap",
        "replicates": replicates,
        "seed": seed,
        "prng": "random.Random",
        "draw_api": "randrange",
        "standard_deviation_ddof": 1,
        "epsilon": EPSILON,
        "quantile_method": "linear_q_times_R_minus_1",
        "bootstrap_draw_index_sha256": bootstrap["draw_index_sha256"],
        "bootstrap_combined_rank_winner_sha256": semantic_sha256(winner_keys),
        "turn_strength_Q_threshold": "0.5_strict",
        "numeric_tolerances": {"rel_tol": REL_TOL, "abs_tol": ABS_TOL, "decomposition_rel": 1e-10},
        "point_rank_score": "S_RATE_TURN",
        "combined_rank_frequency_threshold": FREQUENCY_THRESHOLD,
        "selector_decision": decision,
        "selected_key": selected_key,
        "point_operational_winner": list(point_operational_winner) if point_operational_winner else None,
        "point_top_tie_set": [list(key) for key in point_top_tie_set],
        "point_tie_break_applied": len(point_top_tie_set) > 1,
        "combined_rank_selection_frequency": top_frequency,
        "combined_rank_winner_count": sum(winner is not None for winner in winner_keys),
        "combined_rank_usable_replicates": sum(winner is not None for winner in winner_keys),
        "selected_window": None,
        "ranking_digest": ranking_digest,
        "selector_projection_file_sha256": selector_projection_file_sha256,
        "information_barrier": {
            "validation_target_gold_read": False,
            "test_split_read": False,
            "outcome_read": False,
            "model_weights_loaded": False,
            "model_forward_executed": False,
            "cuda_gpu_slurm": False,
            "loop_executed": False,
            "hidden_diagnostics_consumed": False,
        },
        "rows": rows,
    }
    if selected_key is not None:
        selected_row = next(row for row in rows if _candidate_key(row) == tuple(selected_key))
        payload["selected_window"] = {
            "window_half_open": selected_row["window_half_open"],
            "window_layers_inclusive": selected_row["window_layers_inclusive"],
            "boundary_entry": selected_row["boundary_entry"],
            "boundary_exit": selected_row["boundary_exit"],
        }
    payload["analysis_semantic_digest"] = semantic_sha256(payload)
    validate_freeze_payload(payload, enforce_population=formal)
    return payload


def validate_freeze_payload(payload: Mapping[str, Any], *, enforce_population: bool = True) -> None:
    if not isinstance(payload, Mapping):
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: freeze is not an object")
    required = {
        "schema_version", "artifact_role", "gate", "planning_thread_id", "executor_thread_id", "expected_commit",
        "source_gate_m_root", "source_input_closure", "method_id", "method_version", "method_file_sha256",
        "input_distribution_contract", "probe_contract", "trajectory_file_sha256", "identity_manifest_sha256",
        "record_count", "category_count", "ordered_category_sha256", "layer_count", "boundary_count", "central_blocks",
        "widths", "candidate_count", "aggregation", "selector_projection_fields", "hidden_diagnostics_in_selector",
        "outcome_fields_consumed", "bootstrap_method", "replicates", "seed", "prng", "draw_api", "standard_deviation_ddof",
        "epsilon", "quantile_method", "bootstrap_draw_index_sha256", "bootstrap_combined_rank_winner_sha256",
        "turn_strength_Q_threshold", "numeric_tolerances", "point_rank_score", "combined_rank_frequency_threshold",
        "selector_decision", "selected_key", "point_operational_winner", "point_top_tie_set", "point_tie_break_applied",
        "combined_rank_selection_frequency", "combined_rank_winner_count", "combined_rank_usable_replicates", "selected_window",
        "ranking_digest", "selector_projection_file_sha256", "information_barrier", "rows", "analysis_semantic_digest",
    }
    if set(payload) != required:
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: freeze root fields differ")
    if payload["schema_version"] != "loopscope.phase6.mmlu5-gate-n-v3-selector-freeze.v1" or payload["artifact_role"] != "gate_n_post_hoc_v3_selector_freeze" or payload["gate"] != "N":
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: freeze identity differs")
    if payload["method_id"] != METHOD_ID or payload["method_version"] != METHOD_VERSION or payload["method_file_sha256"] != METHOD_FILE_SHA256:
        raise GateNV3Error("BLOCK_METHOD_CONTRACT_MISMATCH")
    if payload["selector_projection_fields"] != list(PROJECTION_FIELDS) or payload["hidden_diagnostics_in_selector"] is not False or payload["outcome_fields_consumed"] is not False:
        raise GateNV3Error("BLOCK_INFORMATION_BARRIER_VIOLATION")
    if payload["central_blocks"] != [11, 24] or payload["widths"] != list(WIDTHS) or payload["candidate_count"] != 42:
        raise GateNV3Error("BLOCK_CANDIDATE_DOMAIN_MISMATCH")
    if payload["selector_decision"] not in LEGAL_TERMINAL_STATES:
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: illegal V3 terminal")
    if enforce_population and (payload["record_count"] != EXPECTED_RECORD_COUNT or payload["category_count"] != EXPECTED_CATEGORY_COUNT):
        raise GateNV3Error("BLOCK_POPULATION_OR_HASH_MISMATCH")
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != 42:
        raise GateNV3Error("BLOCK_CANDIDATE_DOMAIN_MISMATCH")
    expected_pairs = [(width, start) for width in WIDTHS for start in STARTS[width]]
    observed_pairs = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: row is not an object")
        observed_pairs.append((row.get("width"), row.get("start")))
        if row.get("legacy_diagnostics") != {} or row.get("hidden_diagnostics") != {}:
            raise GateNV3Error("BLOCK_INFORMATION_BARRIER_VIOLATION: diagnostic namespace consumed")
    if observed_pairs != expected_pairs:
        raise GateNV3Error("BLOCK_CANDIDATE_DOMAIN_MISMATCH")
    barrier = payload["information_barrier"]
    if not isinstance(barrier, Mapping) or any(value is not False for value in barrier.values()):
        raise GateNV3Error("BLOCK_INFORMATION_BARRIER_VIOLATION")
    if payload["selector_decision"] == "SELECTED_WINDOW" and payload["selected_key"] is None:
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: selected key missing")
    if payload["selector_decision"] != "SELECTED_WINDOW" and payload["selected_key"] is not None:
        raise GateNV3Error("BLOCK_VERIFIER_MISMATCH: ABSTAIN carries selected key")


def validate_input_closure(source_root: Path) -> Dict[str, Any]:
    root = Path(source_root).resolve()
    if root.name != EXPECTED_SOURCE_ROOT_NAME:
        raise GateNV3Error("BLOCK_INPUT_CLOSURE_HASH_MISMATCH: Gate M root name differs")
    observed: Dict[str, str] = {}
    for relative, expected in INPUT_CLOSURE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise GateNV3Error("BLOCK_INPUT_CLOSURE_READ: missing %s" % relative)
        observed[relative] = file_sha256(path)
        if observed[relative] != expected:
            raise GateNV3Error("BLOCK_INPUT_CLOSURE_HASH_MISMATCH: %s" % relative)
    safe_payloads = [
        load_json(root / "manifest/formal_manifest.json"),
        load_json(root / "manifest/freeze_receipt.json"),
        load_json(root / "merge/merge_receipt.json"),
        load_json(root / "merge/verifier_receipt.json"),
    ]
    for payload in safe_payloads:
        if payload.get("status") not in (None, "PASS"):
            raise GateNV3Error("BLOCK_INPUT_CLOSURE_STATUS: Gate M receipt is not PASS")
        for key in ("selector_executed", "loop_executed", "outcome_read", "validation_target_gold_read", "test_split_read", "model_weights_loaded", "model_forward_executed", "cuda_gpu_slurm"):
            if payload.get(key) is True:
                raise GateNV3Error("BLOCK_INFORMATION_BARRIER_VIOLATION: source receipt %s" % key)
    manifest = safe_payloads[0]
    if manifest.get("expected_commit") not in (None, SOURCE_GATE_M_COMMIT):
        raise GateNV3Error("BLOCK_INPUT_CLOSURE_HASH_MISMATCH: Gate M source commit differs")
    merge = safe_payloads[2]
    verifier = safe_payloads[3]
    if merge.get("record_file_sha256") != EXPECTED_TRAJECTORY_SHA256 or verifier.get("record_file_sha256") != EXPECTED_TRAJECTORY_SHA256:
        raise GateNV3Error("BLOCK_INPUT_CLOSURE_HASH_MISMATCH: receipt trajectory hash differs")
    if merge.get("record_count") != EXPECTED_RECORD_COUNT or merge.get("subject_count") != EXPECTED_CATEGORY_COUNT or verifier.get("record_count") != EXPECTED_RECORD_COUNT or verifier.get("subject_count") != EXPECTED_CATEGORY_COUNT:
        raise GateNV3Error("BLOCK_POPULATION_OR_HASH_MISMATCH: source receipt population differs")
    return observed


def analyze_source(
    source_root: Path,
    *,
    expected_commit: str = EXPECTED_COMMIT,
    replicates: int = FORMAL_REPLICATES,
    seed: int = FORMAL_SEED,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, str]]:
    _validate_commit(expected_commit)
    closure = validate_input_closure(source_root)
    trajectory_path = Path(source_root).resolve() / "merge/merged_trajectory_records.jsonl"
    records = load_jsonl(trajectory_path)
    projected = project_records(records, formal=True)
    projection_bytes = _jsonl_bytes(projected)
    projection_sha = hashlib.sha256(projection_bytes).hexdigest()
    payload = analyze_projected(
        projected,
        source_root=str(Path(source_root).resolve()),
        source_hashes=closure,
        expected_commit=expected_commit,
        selector_projection_file_sha256=projection_sha,
        replicates=replicates,
        seed=seed,
        formal=True,
    )
    return payload, projected, closure


__all__ = [
    "ABS_TOL",
    "BOUNDARY_COUNT",
    "EXPECTED_COMMIT",
    "EXPECTED_RECORD_COUNT",
    "EXPECTED_CATEGORY_COUNT",
    "EXPECTED_SOURCE_ROOT_NAME",
    "EXPECTED_TRAJECTORY_SHA256",
    "FORMAL_REPLICATES",
    "FORMAL_SEED",
    "FREQUENCY_THRESHOLD",
    "GateNV3Error",
    "INPUT_CLOSURE_SHA256",
    "LAYERS",
    "METHOD_FILE_SHA256",
    "METHOD_ID",
    "METHOD_VERSION",
    "SOURCE_GATE_M_COMMIT",
    "PROJECTION_FIELDS",
    "Q_THRESHOLD",
    "REL_TOL",
    "STARTS",
    "WIDTHS",
    "analyze_projected",
    "analyze_source",
    "canonical_json_bytes",
    "compute_rate_statistics",
    "decide_terminal",
    "enumerate_candidates",
    "file_sha256",
    "load_json",
    "load_jsonl",
    "project_records",
    "semantic_sha256",
    "validate_freeze_payload",
    "validate_input_closure",
    "write_new_bytes",
    "write_new_json",
]
