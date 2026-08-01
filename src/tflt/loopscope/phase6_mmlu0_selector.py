"""Outcome-blind MMLU 0-shot adapter for the frozen Phase 5/V2 selector math.

Only the closed projection ``identity, category, H[37], D[37]`` reaches the
selector.  The adapter deliberately does not import the historical Phase 6
MMLU-Pro schema or any outcome/panel machinery.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from tflt.loopscope import phase5_relative_biphasic as v1
from tflt.loopscope.phase5_relative_biphasic_v2_absolute_rate import (
    AbsoluteRateError,
    compute_rate_statistics,
    select_rate_decision,
)
from tflt.loopscope.phase6_mmlu0_schema import (
    BOUNDARY_COUNT,
    FORMAL_REPLICATES,
    FORMAL_SEED,
    LEGAL_SELECTOR_STATES,
    LAYERS,
    STARTS,
    VALIDATION_RECORD_COUNT,
    VALIDATION_SUBJECT_COUNT,
    WIDTHS,
    MMLU0ContractError,
    selector_sample,
    validate_selector_sample,
)


class MMLU0SelectorError(MMLU0ContractError):
    """Selector adapter contract error."""


def enumerate_candidates() -> List[Dict[str, Any]]:
    """Return the exact 42 central-block candidates in canonical order."""

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
    if len(candidates) != 42 or len({(row["width"], row["start"]) for row in candidates}) != 42:
        raise MMLU0SelectorError("MMLU 0-shot candidate domain is not exactly 42 cells")
    return candidates


def project_selector_records(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Irreversibly project producer records to the V2 selector surface."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise MMLU0SelectorError("trajectory records must be a sequence")
    projected: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        sample = selector_sample(record)
        validate_selector_sample(sample)
        if sample["identity"] in seen:
            raise MMLU0SelectorError("selector identities must be unique")
        seen.add(sample["identity"])
        projected.append(
            {
                "identity": sample["identity"],
                "category": sample["category"],
                "H": list(sample["H"]),
                "D": list(sample["D"]),
            }
        )
    return projected


def _compatibility_view(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: (int(value["width"]), int(value["start"]))):
        best = row.get("best_shape_pair")
        result.append(
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
                "tau_pair_frequency": None if best is None else best.get("tau_pair_frequency"),
                "v1_failure_reasons": list(row["new_eligibility_failures"]),
            }
        )
    if len(result) != 42:
        raise MMLU0SelectorError("V1 compatibility candidate count differs")
    return result


def _decision(
    rows: Sequence[MutableMapping[str, Any]],
    bootstrap_rates: Mapping[Tuple[int, int], Mapping[str, Sequence[Any]]],
    replicates: int,
) -> Dict[str, Any]:
    try:
        result = select_rate_decision(rows, bootstrap_rates, replicates)
    except (AbsoluteRateError, KeyError, TypeError, ValueError) as exc:
        raise MMLU0SelectorError(str(exc)) from exc
    if result["decision"] not in LEGAL_SELECTOR_STATES:
        raise MMLU0SelectorError("unknown MMLU 0-shot selector terminal state")
    return result


def analyze_selector(
    records: Sequence[Mapping[str, Any]],
    *,
    replicates: int = FORMAL_REPLICATES,
    seed: int = FORMAL_SEED,
    formal: bool = True,
) -> Dict[str, Any]:
    """Run pure V1/V2 selection over already sanitized trajectory records.

    Gate H only validates this path and runs synthetic smoke checks.  A formal
    invocation is reserved for Gate J by the card and CLI boundaries.
    """

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise MMLU0SelectorError("bootstrap requires at least two replicates")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise MMLU0SelectorError("bootstrap seed must be an integer")
    if formal and (replicates != FORMAL_REPLICATES or seed != FORMAL_SEED):
        raise MMLU0SelectorError("formal MMLU 0-shot bootstrap contract differs")

    projected = project_selector_records(records)
    if formal:
        if len(projected) != VALIDATION_RECORD_COUNT:
            raise MMLU0SelectorError("formal selector requires exactly 1,531 records")
        if len({row["category"] for row in projected}) != VALIDATION_SUBJECT_COUNT:
            raise MMLU0SelectorError("formal selector requires exactly 57 subjects")
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
        raise MMLU0SelectorError(str(exc)) from exc
    if bootstrap["draw_index_sha256"] != v1_analysis["bootstrap"]["draw_index_sha256"]:
        raise MMLU0SelectorError("joint bootstrap draw stream differs from V1")

    working_rows: List[Dict[str, Any]] = []
    bootstrap_rates: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
    for source in v1_analysis["rows"]:
        width = int(source["width"])
        start = int(source["start"])
        boot_h = [
            (bootstrap["H"][start][index] - bootstrap["H"][start + width][index]) / width
            for index in range(replicates)
        ]
        boot_k = [
            (bootstrap["D"][start + width][index] - bootstrap["D"][start][index]) / width
            for index in range(replicates)
        ]
        try:
            statistics = compute_rate_statistics(source["G_H"], source["G_K"], boot_h, boot_k)
        except (AbsoluteRateError, KeyError, TypeError, ValueError) as exc:
            raise MMLU0SelectorError(str(exc)) from exc
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
            "tau_pair_frequency": None if best is None else best.get("tau_pair_frequency"),
            "v1_failure_reasons": list(source["new_eligibility_failures"]),
            "v2_failure_reasons": [],
        }
        working_rows.append(row)
        bootstrap_rates[(width, start)] = {
            "H": statistics["_bootstrap_rate_H"],
            "K": statistics["_bootstrap_rate_K"],
        }

    decision = _decision(working_rows, bootstrap_rates, replicates)
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

    rows = sorted(working_rows, key=lambda row: (int(row["width"]), int(row["start"])))
    if len(rows) != 42 or {(row["width"], row["start"]) for row in rows} != set(expected for expected in ((width, start) for width in WIDTHS for start in STARTS[width])):
        raise MMLU0SelectorError("MMLU 0-shot V2 candidate closure failed")
    return {
        "selector_decision": decision["decision"],
        "selected_key": decision["selected_key"],
        "selection_frequency": decision["selection_frequency"],
        "point_top_key": decision["point_top_key"],
        "point_top_tied_keys": decision["point_top_tied_keys"],
        "record_count": len(projected),
        "category_count": len({row["category"] for row in projected}),
        "candidate_count": len(rows),
        "candidate_counts_by_width": {str(width): len(STARTS[width]) for width in WIDTHS},
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "standard_deviation_ddof": 1,
            "draw_index_sha256": bootstrap["draw_index_sha256"],
            "category_stratified": True,
            "joint_reuse": True,
        },
        "v1_compatibility": {"status": "PASS", "view": _compatibility_view(v1_analysis["rows"])},
        "rows": rows,
    }


__all__ = ["MMLU0SelectorError", "analyze_selector", "enumerate_candidates", "project_selector_records"]
