"""Pure Phase 6 selector and diagnostic-panel functions.

The public adapter deliberately destroys access to producer diagnostics: after
``project_selector_records`` returns, every downstream computation receives only
identity, category, H[37], and D[37].
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from tflt.loopscope import phase5_relative_biphasic as v1
from tflt.loopscope.phase5_relative_biphasic_v2_absolute_rate import (
    AbsoluteRateError,
    compute_rate_statistics,
    rate_score,
    select_rate_decision,
)
from tflt.loopscope.phase6_schema import (
    METHOD,
    SELECTOR_SCHEMA_VERSION,
    Phase6ContractError,
    selector_sample,
    validate_selector_freeze,
)


LAYERS = 36
BOUNDARY_COUNT = LAYERS + 1
WIDTHS = (3, 4, 5, 6)
STARTS = {
    3: tuple(range(11, 23)),
    4: tuple(range(11, 22)),
    5: tuple(range(11, 21)),
    6: tuple(range(11, 20)),
}
FORMAL_REPLICATES = 2000
FORMAL_SEED = 20260801
SELECTION_FREQUENCY_THRESHOLD = 0.80

SCIENTIFIC_STATES = (
    "SELECTED_WINDOW",
    "ABSTAIN_NO_RATE_STABLE_ELIGIBLE",
    "ABSTAIN_NO_UNIQUE_TOP1",
    "ABSTAIN_RATE_RANK_UNSTABLE",
)

KNOWN_OUTCOME_REGISTRY_ORDER = (
    "no-loop",
    "15:18",
    "6:9",
    "10:13",
    "25:28",
    "4:7",
    "5:8",
    "22:25",
)
KNOWN_OUTCOME_REGISTRY = frozenset(KNOWN_OUTCOME_REGISTRY_ORDER)

PUBLIC_ROW_FIELDS = (
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
)

V1_COMPATIBILITY_FIELDS = (
    "width",
    "start",
    "Scorable",
    "NetPositive",
    "SoftRelativeStable",
    "BiphasicStable",
    "NewEligible",
    "G_H",
    "G_K",
    "best_q",
    "best_tau",
    "tau_pair_frequency",
    "v1_failure_reasons",
)


def enumerate_candidates() -> List[Dict[str, Any]]:
    """Return the exact 42-candidate central-40% domain in canonical order."""

    candidates = [
        {
            "width": width,
            "start": start,
            "end": start + width - 1,
            "window": "%d:%d" % (start, start + width),
            "boundary_entry": "B_%d" % start,
            "boundary_exit": "B_%d" % (start + width),
        }
        for width in WIDTHS
        for start in STARTS[width]
    ]
    if len(candidates) != 42 or len({row["window"] for row in candidates}) != 42:
        raise Phase6ContractError("Phase 6 candidate-domain closure failed")
    return candidates


def project_selector_records(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate and irreversibly project producer records to selector inputs."""

    projected: List[Dict[str, Any]] = []
    for record in records:
        sample = selector_sample(record)
        if set(sample) != {"identity", "category", "H", "D"}:
            raise Phase6ContractError("selector_sample projection keys differ")
        projected.append(
            {
                "identity": sample["identity"],
                "category": sample["category"],
                "H": list(sample["H"]),
                "D": list(sample["D"]),
            }
        )
    return projected


def v1_compatibility_view(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return the compact Phase 5 V1 semantic fixture retained by Phase 6."""

    view: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (int(item["width"]), int(item["start"]))):
        best = row.get("best_shape_pair")
        view.append(
            {
                "width": int(row["width"]),
                "start": int(row["start"]),
                "Scorable": bool(row["Scorable"]),
                "NetPositive": bool(row["NetPositive"]),
                "SoftRelativeStable": bool(row["SoftRelativeStable"]),
                "BiphasicStable": bool(row["BiphasicStable"]),
                "NewEligible": bool(row["NewEligible"]),
                "G_H": float(row["G_H"]),
                "G_K": float(row["G_K"]),
                "best_q": None if best is None else best["q"],
                "best_tau": None if best is None else best["tau"],
                "tau_pair_frequency": (
                    None if best is None else best.get("tau_pair_frequency")
                ),
                "v1_failure_reasons": list(row["new_eligibility_failures"]),
            }
        )
        if tuple(view[-1]) != V1_COMPATIBILITY_FIELDS:
            raise Phase6ContractError("V1 compatibility view fields differ")
    if len(view) != 42:
        raise Phase6ContractError("V1 compatibility view candidate count differs")
    return view


def select_scientific_state(
    rows: Sequence[MutableMapping[str, Any]],
    bootstrap_rates: Mapping[Tuple[int, int], Mapping[str, Sequence[Any]]],
    replicates: int,
) -> Dict[str, Any]:
    """Apply the frozen Gate H point/tie/frequency decision semantics."""

    try:
        decision = select_rate_decision(rows, bootstrap_rates, replicates)
    except (AbsoluteRateError, KeyError, TypeError, ValueError) as exc:
        raise Phase6ContractError(str(exc)) from exc
    if decision["decision"] not in SCIENTIFIC_STATES:
        raise Phase6ContractError("unexpected Phase 6 scientific state")
    return decision


def analyze_selector(
    records: Sequence[Mapping[str, Any]],
    *,
    replicates: int = FORMAL_REPLICATES,
    seed: int = FORMAL_SEED,
    formal: bool = True,
) -> Dict[str, Any]:
    """Analyze Phase 6 records with V1 admission and Gate H V2 rate ranking.

    ``formal=False`` permits small deterministic bootstrap counts for unit tests.
    V1 population enforcement remains disabled in both modes because Phase 6 has
    its own population/freeze validator.
    """

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise Phase6ContractError("bootstrap requires at least two replicates")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Phase6ContractError("bootstrap seed must be an integer")
    if formal and (replicates != FORMAL_REPLICATES or seed != FORMAL_SEED):
        raise Phase6ContractError("formal Phase 6 bootstrap contract differs")

    projected = project_selector_records(records)
    legacy_samples = [
        {
            "identity": row["identity"],
            "subject": row["category"],
            "H": row["H"],
            "D": row["D"],
        }
        for row in projected
    ]
    try:
        v1_analysis = v1.analyze_model(
            legacy_samples,
            "qwen4",
            replicates=replicates,
            seed=seed,
            enforce_population=False,
        )
        normalized = v1._normalize_preprojected(legacy_samples, "qwen4", False)
        bootstrap = v1._bootstrap_boundary_means(
            normalized,
            LAYERS,
            replicates=replicates,
            seed=seed,
            digest_style=v1.MODEL_SPECS["qwen4"]["legacy_style"],
        )
    except (v1.RelativeBiphasicError, KeyError, TypeError, ValueError) as exc:
        raise Phase6ContractError(str(exc)) from exc
    if bootstrap["draw_index_sha256"] != v1_analysis["bootstrap"]["draw_index_sha256"]:
        raise Phase6ContractError("Phase 6 bootstrap draw stream differs from V1")

    compatibility = v1_compatibility_view(v1_analysis["rows"])
    working_rows: List[Dict[str, Any]] = []
    bootstrap_rates: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    for source in v1_analysis["rows"]:
        width = int(source["width"])
        start = int(source["start"])
        boot_h = [
            (bootstrap["H"][start][index] - bootstrap["H"][start + width][index])
            / width
            for index in range(replicates)
        ]
        boot_k = [
            (bootstrap["D"][start + width][index] - bootstrap["D"][start][index])
            / width
            for index in range(replicates)
        ]
        try:
            statistics = compute_rate_statistics(
                source["G_H"], source["G_K"], boot_h, boot_k
            )
        except (AbsoluteRateError, KeyError, TypeError, ValueError) as exc:
            raise Phase6ContractError(str(exc)) from exc
        best = source.get("best_shape_pair")
        row = {
            "width": width,
            "start": start,
            "end": start + width - 1,
            "window": "%d:%d" % (start, start + width),
            "boundary_entry": "B_%d" % start,
            "boundary_exit": "B_%d" % (start + width),
            "Scorable": bool(source["Scorable"]),
            "NetPositive": bool(source["NetPositive"]),
            "SoftRelativeStable": bool(source["SoftRelativeStable"]),
            "BiphasicStable": bool(source["BiphasicStable"]),
            "NewEligible": bool(source["NewEligible"]),
            "G_H": float(source["G_H"]),
            "G_K": float(source["G_K"]),
            "S_RATE": statistics["S_RATE"],
            "RateStable": bool(statistics["RateStable"]),
            "rate_SE_H": statistics["rate_SE_H"],
            "rate_SE_K": statistics["rate_SE_K"],
            "rate_c95": statistics["rate_c95"],
            "rate_LCB_H": statistics["rate_LCB_H"],
            "rate_LCB_K": statistics["rate_LCB_K"],
            "ranking_candidate": False,
            "point_top_tie": False,
            "selected": False,
            "display_rank": None,
            "selection_frequency": None,
            "best_q": None if best is None else best["q"],
            "best_tau": None if best is None else best["tau"],
            "tau_pair_frequency": (
                None if best is None else best.get("tau_pair_frequency")
            ),
            "v1_failure_reasons": list(source["new_eligibility_failures"]),
            "v2_failure_reasons": [],
        }
        working_rows.append(row)
        bootstrap_rates[(width, start)] = {
            "H": statistics["_bootstrap_rate_H"],
            "K": statistics["_bootstrap_rate_K"],
        }

    decision = select_scientific_state(working_rows, bootstrap_rates, replicates)
    for row in working_rows:
        row["selection_frequency"] = row.pop("model_rate_selection_frequency")
        failures: List[str] = []
        if not row["NewEligible"]:
            failures.append("NOT_NEW_ELIGIBLE")
        if not row["RateStable"]:
            failures.append("RATE_NOT_STABLE")
        if row["ranking_candidate"] and not row["selected"]:
            if row["point_top_tie"]:
                failures.append("POINT_TOP_TIE")
            elif row["display_rank"] != 1:
                failures.append("NOT_POINT_TOP")
            elif decision["decision"] == "ABSTAIN_RATE_RANK_UNSTABLE":
                failures.append("RATE_RANK_UNSTABLE")
        row["v2_failure_reasons"] = failures
        if tuple(row) != PUBLIC_ROW_FIELDS:
            raise Phase6ContractError("public selector row fields differ")

    canonical_rows = sorted(
        working_rows, key=lambda row: (int(row["width"]), int(row["start"]))
    )
    payload = {
        "selector_decision": decision["decision"],
        "selected_key": decision["selected_key"],
        "selection_frequency": decision["selection_frequency"],
        "point_top_key": decision["point_top_key"],
        "point_top_tied_keys": decision["point_top_tied_keys"],
        "record_count": len(projected),
        "category_count": len({row["category"] for row in projected}),
        "candidate_count": len(canonical_rows),
        "candidate_counts_by_width": {
            str(width): len(STARTS[width]) for width in WIDTHS
        },
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "standard_deviation_ddof": 1,
            "draw_index_sha256": bootstrap["draw_index_sha256"],
            "category_stratified": True,
            "joint_reuse": True,
        },
        "v1_compatibility": {
            "status": "PASS",
            "view": compatibility,
        },
        "rows": canonical_rows,
    }
    return payload


def freeze_panel(selector_freeze: Mapping[str, Any]) -> Dict[str, Any]:
    """Freeze the registry-excluded High3/Low3 and unique outcome panel."""

    rows = selector_freeze.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise Phase6ContractError("selector freeze rows are missing")
    by_window: Dict[str, Mapping[str, Any]] = {}
    diagnostic: List[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise Phase6ContractError("selector row must be an object")
        width = _strict_int(row.get("width"), "width")
        start = _strict_int(row.get("start"), "start")
        window = "%d:%d" % (start, start + width)
        if row.get("window") != window or window in by_window:
            raise Phase6ContractError("selector candidate identity differs")
        by_window[window] = row
        try:
            score = rate_score(row.get("G_H"), row.get("G_K"))
        except AbsoluteRateError as exc:
            raise Phase6ContractError(str(exc)) from exc
        observed_score = row.get("S_RATE")
        if score is None:
            if observed_score is not None:
                raise Phase6ContractError("nonpositive candidate S_RATE differs")
        else:
            if (
                isinstance(observed_score, bool)
                or not isinstance(observed_score, (int, float))
                or not math.isfinite(float(observed_score))
                or not math.isclose(
                    float(observed_score), score, rel_tol=1e-12, abs_tol=1e-12
                )
            ):
                raise Phase6ContractError("positive candidate S_RATE differs")
            if window not in KNOWN_OUTCOME_REGISTRY:
                diagnostic.append(row)
    if len(diagnostic) < 6:
        raise Phase6ContractError("BLOCK_DIAGNOSTIC_PANEL_UNDERPOPULATED")

    descending = sorted(
        diagnostic,
        key=lambda row: (
            -float(row["S_RATE"]),
            int(row["width"]),
            int(row["start"]),
        ),
    )
    high3 = [str(row["window"]) for row in descending[:3]]
    low3 = [
        str(row["window"])
        for row in sorted(
            diagnostic,
            key=lambda row: (
                float(row["S_RATE"]),
                int(row["width"]),
                int(row["start"]),
            ),
        )[:3]
    ]
    if set(high3) & set(low3):
        raise Phase6ContractError("diagnostic High3 and Low3 overlap")

    decision = selector_freeze.get("selector_decision")
    if decision not in SCIENTIFIC_STATES:
        raise Phase6ContractError("selector scientific state differs")
    selected_key = selector_freeze.get("selected_key")
    selected_window = None
    if decision == "SELECTED_WINDOW":
        if (
            not isinstance(selected_key, Sequence)
            or isinstance(selected_key, (str, bytes))
            or len(selected_key) != 2
        ):
            raise Phase6ContractError("selected key is missing")
        width = _strict_int(selected_key[0], "selected width")
        start = _strict_int(selected_key[1], "selected start")
        selected_window = "%d:%d" % (start, start + width)
        selected_row = by_window.get(selected_window)
        if selected_row is None or not bool(selected_row.get("selected")):
            raise Phase6ContractError("selected row closure differs")
    elif selected_key is not None:
        raise Phase6ContractError("ABSTAIN cannot carry a selected key")

    if selected_window is None:
        selected_known_status = "NOT_APPLICABLE"
    elif selected_window in KNOWN_OUTCOME_REGISTRY:
        selected_known_status = "SELECTED_KNOWN_OUTCOME_WINDOW"
    else:
        selected_known_status = "SELECTED_NEW_OUTCOME_WINDOW"

    roles: Dict[str, List[str]] = {}

    def add(cell: str, role: str) -> None:
        roles.setdefault(cell, []).append(role)

    add("no-loop", "baseline")
    add("15:18", "fixed_comparator")
    for window in high3:
        add(window, "High3")
    for window in low3:
        add(window, "Low3")
    if selected_window is not None:
        add(selected_window, "selected")

    payload = {
        "selector_decision": decision,
        "selected_window": selected_window,
        "selected_known_outcome_status": selected_known_status,
        "diagnostic_universe": [
            str(row["window"])
            for row in sorted(
                diagnostic, key=lambda row: (int(row["width"]), int(row["start"]))
            )
        ],
        "high3": high3,
        "low3": low3,
        "panel_cells": list(roles),
        "cell_roles": [
            {"cell": cell, "roles": cell_roles} for cell, cell_roles in roles.items()
        ],
        "loop_cell_count": len(roles) - 1,
    }
    return payload


def build_selector_freeze(
    analysis: Mapping[str, Any],
    *,
    card_sha256: str,
    input_manifest_sha256: str,
) -> Dict[str, Any]:
    """Assemble the closed Phase 6 selector-freeze payload."""

    panel = freeze_panel(analysis)
    point_top_key = analysis.get("point_top_key")
    point_top_window = None
    if point_top_key is not None:
        if (
            not isinstance(point_top_key, Sequence)
            or isinstance(point_top_key, (str, bytes))
            or len(point_top_key) != 2
        ):
            raise Phase6ContractError("point top key differs")
        width = _strict_int(point_top_key[0], "point top width")
        start = _strict_int(point_top_key[1], "point top start")
        point_top_window = "%d:%d" % (start, start + width)
    bootstrap = analysis.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise Phase6ContractError("selector bootstrap metadata are missing")
    rows = analysis.get("rows")
    if not isinstance(rows, list):
        raise Phase6ContractError("selector candidate rows are missing")
    payload = {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "card_sha256": card_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "method": METHOD,
        "population": analysis.get("record_count"),
        "category_count": analysis.get("category_count"),
        "candidate_count": analysis.get("candidate_count"),
        "candidate_counts_by_width": analysis.get("candidate_counts_by_width"),
        "replicates": bootstrap.get("replicates"),
        "selector_seed": bootstrap.get("seed"),
        "decision": analysis.get("selector_decision"),
        "selected_window": panel["selected_window"],
        "point_top_window": point_top_window,
        "selection_frequency": analysis.get("selection_frequency"),
        "selected_known_outcome_status": panel[
            "selected_known_outcome_status"
        ],
        "hidden_fields_consumed": False,
        "outcome_fields_consumed": False,
        "known_outcome_registry": list(KNOWN_OUTCOME_REGISTRY_ORDER),
        "high3": panel["high3"],
        "low3": panel["low3"],
        "panel_cells": panel["panel_cells"],
        "candidates": rows,
    }
    validate_selector_freeze(payload)
    return payload


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase6ContractError("%s must be an integer" % label)
    return value
