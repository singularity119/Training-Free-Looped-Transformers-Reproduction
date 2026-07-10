"""Offline evaluation analysis for a frozen LoopScope window grid."""

from __future__ import annotations

import json
import math
import random
import shlex
from hashlib import sha256
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.loopscope.grid import (
    candidate_windows,
    format_window,
    parse_window,
    validate_window_grid,
)
from tflt.loopscope.schema import (
    ANALYSIS_SCHEMA_VERSION,
    SELECTION_SCHEMA_VERSION,
    canonical_json_bytes,
    ensure_new_directory,
    manifest_sha256,
    write_new_json,
)
from tflt.loopscope.selection import PRIMARY_SIGNALS


HARMFUL_THRESHOLD_PP = -0.3
PHASE1_RUN_SCHEMA_VERSION = "loopscope.phase1-run.v1"
PHASE1_FULL_STAGE = "gate-e-full"
PHASE1_RECIPE = {
    "model": "qwen3-1.7b-base",
    "repo_id": "Qwen/Qwen3-1.7B-Base",
    "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    "task": "mmlu",
    "num_fewshot": 5,
    "dtype": "float16",
    "k": 2,
    "iteration_mode": "block",
    "strategy": "damped_euler",
    "alpha": 1.0,
    "beta": 0.0,
    "cache_strategy": "last",
    "decode_mode": "bypass",
    "window_width": 4,
}
TIE_POLICY = {
    "ranking_ties": "retain_all",
    "display_tiebreak": "frozen_candidate_order",
    "phase_gate_regret": "worst_case",
}


class AnalysisError(ValueError):
    """Raised when evaluation artifacts are ambiguous or internally inconsistent."""


def add_analyze_window_grid_args(parser: Any) -> None:
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument(
        "--window-result",
        action="append",
        required=True,
        metavar="WINDOW=PATH",
    )
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--window-grid", required=True)
    parser.add_argument("--run-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="mmlu")
    parser.add_argument("--metric", default="acc,none")
    parser.add_argument("--paired-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--paired-bootstrap-seed", type=int, default=20260710)
    parser.add_argument(
        "--analysis-scope",
        choices=("full", "limit-smoke"),
        required=True,
    )


def cmd_analyze_window_grid(args: Any) -> int:
    """CLI adapter that writes the final, write-once selection audit."""

    if args.analysis_scope == "full" and not args.run_manifest:
        raise AnalysisError("full analysis requires --run-manifest")
    require_provenance = args.analysis_scope == "full"
    baseline_bundle = _load_result_bundle(
        Path(args.baseline_results), require_provenance=require_provenance
    )
    baseline_path = baseline_bundle["results_path"]
    baseline = baseline_bundle["results"]
    window_results: Dict[str, Mapping[str, Any]] = {}
    window_paths: Dict[str, str] = {}
    window_bundles: Dict[str, Mapping[str, Any]] = {}
    for specification in args.window_result:
        window, path = _parse_window_result_spec(specification)
        if window in window_results:
            raise AnalysisError("duplicate --window-result for %s" % window)
        bundle = _load_result_bundle(path, require_provenance=require_provenance)
        window_results[window] = bundle["results"]
        window_paths[window] = str(bundle["results_path"])
        window_bundles[window] = bundle
    score_report = _load_json_object(Path(args.selection_report), "selection report")
    window_grid = _load_json_object(Path(args.window_grid), "window grid")
    execution_provenance = None
    if args.analysis_scope == "full":
        run_manifest_path = Path(args.run_manifest)
        run_manifest = _load_json_object(run_manifest_path, "phase-one run manifest")
        execution_provenance = _validate_full_cli_provenance(
            run_manifest_path=run_manifest_path,
            run_manifest=run_manifest,
            window_grid_path=Path(args.window_grid),
            window_grid=window_grid,
            score_report_path=Path(args.selection_report),
            score_report=score_report,
            baseline_bundle=baseline_bundle,
            window_bundles=window_bundles,
            task=args.task,
            metric=args.metric,
        )
    report = analyze_window_grid(
        baseline_result=baseline,
        window_results=window_results,
        score_report=score_report,
        window_grid=window_grid,
        task=args.task,
        metric=args.metric,
        paired_bootstrap_replicates=args.paired_bootstrap_replicates,
        paired_bootstrap_seed=args.paired_bootstrap_seed,
        analysis_scope=args.analysis_scope,
        input_paths={
            "baseline_results": str(baseline_path),
            "window_results": window_paths,
            "window_scores": str(Path(args.selection_report)),
            "window_grid": str(Path(args.window_grid)),
            "run_manifest": str(Path(args.run_manifest)) if args.run_manifest else None,
        },
    )
    report["execution_provenance"] = execution_provenance
    output_dir = ensure_new_directory(Path(args.output_dir))
    report_path = output_dir / "selection_report.json"
    summary_path = output_dir / "selection_summary.md"
    write_new_json(report_path, report)
    _write_new_text(summary_path, render_analysis_summary(report))
    print(str(report_path))
    print(str(summary_path))
    return 0


def analyze_window_grid(
    baseline_result: Mapping[str, Any],
    window_results: Mapping[str, Mapping[str, Any]],
    score_report: Mapping[str, Any],
    window_grid: Optional[Mapping[str, Any]] = None,
    task: str = "mmlu",
    metric: str = "acc,none",
    paired_bootstrap_replicates: int = 2000,
    paired_bootstrap_seed: int = 20260710,
    analysis_scope: str = "full",
    input_paths: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Combine frozen signal scores with per-window labelled evaluations.

    Accuracy differences are expressed in percentage points.  The function
    reports missing/misaligned samples instead of inventing paired results.
    """

    if score_report.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise AnalysisError("unsupported offline selection report schema")
    _validate_score_report_integrity(score_report)
    _validate_tie_policy(score_report)
    if analysis_scope not in ("full", "limit-smoke"):
        raise AnalysisError("analysis_scope must be full or limit-smoke")
    if paired_bootstrap_replicates < 1:
        raise AnalysisError("paired bootstrap replicate count must be positive")
    if window_grid is not None:
        validate_window_grid(window_grid)
        score_grid = score_report.get("window_grid")
        if not isinstance(score_grid, Mapping):
            raise AnalysisError("selection report is missing frozen window-grid provenance")
        if score_grid.get("manifest_sha256") != window_grid.get("manifest_sha256"):
            raise AnalysisError("selection report and window grid hashes disagree")
        score_order = [
            format_window(parse_window(str(window)))
            for window in score_grid.get("candidate_windows", [])
        ]
        frozen_order = [format_window(window) for window in candidate_windows(window_grid)]
        if score_order != frozen_order:
            raise AnalysisError(
                "selection report candidate order disagrees with the frozen window grid"
            )
        scored_windows = {
            parse_window(str(candidate.get("window")))
            for candidate in score_report.get("candidates", [])
            if isinstance(candidate, Mapping) and candidate.get("window")
        }
        frozen_windows = set(candidate_windows(window_grid))
        if scored_windows != frozen_windows:
            raise AnalysisError(
                "selection candidates do not exactly match the frozen window grid"
            )
    normalized_results = _normalize_window_results(window_results)
    if not normalized_results:
        raise AnalysisError("at least one window result is required")
    if window_grid is not None:
        required = set(format_window(window) for window in candidate_windows(window_grid))
        if analysis_scope == "full":
            comparison = window_grid.get("comparison_windows", {})
            if isinstance(comparison, Mapping):
                fixed = comparison.get("fixed_depth")
                if fixed:
                    required.add(format_window(parse_window(str(fixed))))
                required.update(
                    format_window(parse_window(str(window)))
                    for window in comparison.get("random_in_band", [])
                )
        missing_results = sorted(required.difference(normalized_results), key=_window_sort_key)
        if missing_results:
            raise AnalysisError(
                "%s analysis is missing required window results: %s"
                % (analysis_scope, ", ".join(missing_results))
            )

    warnings: List[str] = []
    candidate_order = _frozen_candidate_order(score_report, window_grid)
    baseline_accuracy = extract_accuracy(baseline_result, task=task, metric=metric)
    roles = _window_roles(window_grid)
    result_rows: List[Dict[str, Any]] = []
    delta_by_window: Dict[str, float] = {}
    for window in sorted(normalized_results, key=_window_sort_key):
        accuracy = extract_accuracy(normalized_results[window], task=task, metric=metric)
        delta = (accuracy - baseline_accuracy) * 100.0
        harmful = None
        if analysis_scope == "full":
            harmful = delta < HARMFUL_THRESHOLD_PP and not math.isclose(
                delta, HARMFUL_THRESHOLD_PP, rel_tol=0.0, abs_tol=1e-12
            )
        delta_by_window[window] = delta
        result_rows.append(
            {
                "window": window,
                "accuracy": accuracy,
                "delta_acc_pp": delta,
                "harmful": harmful,
                "roles": roles.get(window, []),
            }
        )

    score_maps = _signal_score_maps(score_report)
    correlations: Dict[str, Dict[str, Any]] = {}
    regrets: Dict[str, Dict[str, Any]] = {}
    for signal in PRIMARY_SIGNALS:
        score_map = score_maps[signal]
        shared = sorted(set(score_map).intersection(delta_by_window), key=_window_sort_key)
        missing = sorted(set(score_map).difference(delta_by_window), key=_window_sort_key)
        if missing:
            warnings.append(
                "Signal %s is missing labelled results for: %s"
                % (signal, ", ".join(missing))
            )
        if analysis_scope == "limit-smoke":
            correlations[signal] = {
                "available": False,
                "n": len(shared),
                "rho": None,
                "reason": "limit-smoke accuracy must not rank signals",
            }
            regrets[signal] = {
                "available": False,
                "absolute_regret_pp": None,
                "reason": "limit-smoke accuracy must not select a window",
            }
        else:
            correlations[signal] = spearman_report(
                [score_map[window] for window in shared],
                [delta_by_window[window] for window in shared],
                windows=shared,
            )
            regrets[signal] = _signal_regret(
                score_map, delta_by_window, candidate_order=candidate_order
            )

    paired: Dict[str, Dict[str, Any]] = {}
    try:
        baseline_samples = extract_correctness_samples(
            baseline_result, task=task, metric=metric
        )
    except AnalysisError as exc:
        baseline_samples = {}
        warnings.append("Baseline paired samples unavailable: %s" % exc)
    for window in sorted(normalized_results, key=_window_sort_key):
        if not baseline_samples:
            paired[window] = {
                "available": False,
                "reason": "baseline correctness samples unavailable",
            }
            continue
        try:
            candidate_samples = extract_correctness_samples(
                normalized_results[window], task=task, metric=metric
            )
            paired[window] = paired_comparison(
                baseline_samples,
                candidate_samples,
                bootstrap_replicates=paired_bootstrap_replicates,
                bootstrap_seed=_window_seed(paired_bootstrap_seed, window),
            )
            if not paired[window]["sample_alignment"]["exact_match"]:
                warnings.append("Paired sample IDs do not exactly align for window %s" % window)
        except AnalysisError as exc:
            paired[window] = {"available": False, "reason": str(exc)}
            warnings.append("Paired analysis unavailable for window %s: %s" % (window, exc))

    calibration_bootstrap = calibration_bootstrap_stability(
        score_report, candidate_order=candidate_order
    )
    if not calibration_bootstrap.get("available"):
        warnings.append(
            "Calibration bootstrap unavailable: %s"
            % calibration_bootstrap.get("reason", "unknown reason")
        )

    primary_signal = str(score_report.get("primary_signal"))
    paired_alignment_ready = bool(paired) and all(
        item.get("available")
        and item.get("sample_alignment", {}).get("exact_match")
        for item in paired.values()
    )
    comparisons = _comparison_report(
        window_grid=window_grid,
        result_rows=result_rows,
        primary_regret=regrets[primary_signal],
    )
    if comparisons.get("random_in_band", {}).get("complete") is False:
        warnings.append("Random-in-band comparison is incomplete")

    if analysis_scope == "limit-smoke":
        warnings.append(
            "This is a --limit engineering smoke; its accuracy, ranking, and regret are not "
            "scientific evidence and must not select a window."
        )

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "report_kind": "window_grid_selection_analysis",
        "analysis_scope": analysis_scope,
        "inputs": dict(input_paths or {}),
        "accuracy": {
            "task": task,
            "metric": metric,
            "unit": "fraction",
            "delta_unit": "percentage_points",
            "baseline": baseline_accuracy,
        },
        "harmful_definition": {
            "operator": "strictly_less_than",
            "threshold_pp": HARMFUL_THRESHOLD_PP,
        },
        "windows": result_rows,
        "signal_correlations": correlations,
        "regret_by_signal": regrets,
        "primary_signal": primary_signal,
        "criterion": score_report.get("criterion"),
        "criterion_sha256": score_report.get("criterion_sha256"),
        "offline_rankings": score_report.get("rankings"),
        "offline_selections_by_signal": score_report.get("selections_by_signal"),
        "offline_candidates": score_report.get("candidates"),
        "comparisons": comparisons,
        "paired_analysis": paired,
        "calibration_bootstrap": calibration_bootstrap,
        "phase_assessment": _phase_assessment(
            correlations.get(primary_signal, {}),
            regrets.get(primary_signal, {}),
            comparisons,
            calibration_bootstrap,
            paired_alignment_ready,
            analysis_scope,
        ),
        "warnings": _deduplicate(warnings),
    }


def extract_accuracy(
    payload: Mapping[str, Any], task: str = "mmlu", metric: str = "acc,none"
) -> float:
    """Extract a fraction-valued lm-eval metric without using stderr fields."""

    for container_name in ("results", "groups"):
        container = payload.get(container_name)
        if isinstance(container, Mapping):
            entry = container.get(task)
            if isinstance(entry, Mapping) and metric in entry:
                context = "%s.%s.%s" % (container_name, task, metric)
                return _accuracy_fraction(entry[metric], context)

    direct = payload.get(metric)
    if direct is not None:
        return _accuracy_fraction(direct, metric)
    if "accuracy" in payload:
        return _accuracy_fraction(payload["accuracy"], "accuracy")

    results = payload.get("results")
    if isinstance(results, Mapping):
        matching = [
            (name, entry)
            for name, entry in results.items()
            if str(name).startswith(task + "_") and isinstance(entry, Mapping) and metric in entry
        ]
        if matching:
            weighted = _weighted_group_accuracy(payload, matching, metric)
            if weighted is not None:
                return weighted
            raise AnalysisError(
                "task %s has only subgroup metrics and no sample counts/groups aggregate" % task
            )
    raise AnalysisError("could not find metric %s for task %s" % (metric, task))


def extract_correctness_samples(
    payload: Mapping[str, Any], task: str = "mmlu", metric: str = "acc,none"
) -> Dict[str, bool]:
    """Extract explicit per-sample correctness keyed by stable task-qualified IDs."""

    raw_samples = payload.get("samples")
    groups: List[Tuple[str, Sequence[Any]]] = []
    if isinstance(raw_samples, Mapping):
        for name, items in raw_samples.items():
            if str(name) == task or str(name).startswith(task + "_"):
                if not isinstance(items, list):
                    raise AnalysisError("samples.%s must be a list" % name)
                groups.append((str(name), items))
    elif isinstance(raw_samples, list):
        groups.append((task, raw_samples))
    if not groups:
        raise AnalysisError("results artifact contains no samples for task %s" % task)

    result: Dict[str, bool] = {}
    for namespace, items in groups:
        for item in items:
            if not isinstance(item, Mapping):
                raise AnalysisError("sample entries must be objects")
            raw_id = _sample_id(item)
            sample_id = "%s:%s" % (namespace, raw_id)
            if sample_id in result:
                raise AnalysisError("duplicate paired sample id: %s" % sample_id)
            result[sample_id] = _sample_correctness(item, metric)
    if not result:
        raise AnalysisError("no correctness samples were extracted")
    return result


def paired_comparison(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260710,
) -> Dict[str, Any]:
    """Compute paired counts, exact McNemar, and paired bootstrap delta."""

    if bootstrap_replicates < 1:
        raise AnalysisError("paired bootstrap replicate count must be positive")
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    common = sorted(baseline_ids.intersection(candidate_ids))
    if not common:
        raise AnalysisError("baseline and window have no shared sample IDs")
    counts = {"both_correct": 0, "baseline_only": 0, "window_only": 0, "both_wrong": 0}
    differences = []
    for sample_id in common:
        base = bool(baseline[sample_id])
        window = bool(candidate[sample_id])
        if base and window:
            counts["both_correct"] += 1
        elif base:
            counts["baseline_only"] += 1
        elif window:
            counts["window_only"] += 1
        else:
            counts["both_wrong"] += 1
        differences.append((int(window) - int(base)) * 100.0)

    generator = random.Random(bootstrap_seed)
    bootstrap = []
    for _ in range(bootstrap_replicates):
        bootstrap.append(mean(generator.choice(differences) for _ in differences))
    bootstrap.sort()
    discordant = counts["baseline_only"] + counts["window_only"]
    return {
        "available": True,
        "sample_alignment": {
            "exact_match": baseline_ids == candidate_ids,
            "matched_count": len(common),
            "baseline_count": len(baseline_ids),
            "window_count": len(candidate_ids),
            "missing_from_window": sorted(baseline_ids.difference(candidate_ids)),
            "missing_from_baseline": sorted(candidate_ids.difference(baseline_ids)),
        },
        "counts": counts,
        "contingency_table": {
            "n11_both_correct": counts["both_correct"],
            "n10_baseline_only": counts["baseline_only"],
            "n01_window_only": counts["window_only"],
            "n00_both_wrong": counts["both_wrong"],
        },
        "discordant_count": discordant,
        "discordance_rate": discordant / float(len(common)),
        "mcnemar": {
            "method": "exact_two_sided_binomial",
            "p_value": exact_mcnemar_p(
                counts["baseline_only"], counts["window_only"]
            ),
        },
        "paired_bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "delta_acc_pp_mean": mean(bootstrap),
            "delta_acc_pp_median": median(bootstrap),
            "ci95_pp": [_quantile(bootstrap, 0.025), _quantile(bootstrap, 0.975)],
            "probability_delta_gt_zero": sum(value > 0.0 for value in bootstrap)
            / float(bootstrap_replicates),
        },
    }


def exact_mcnemar_p(baseline_only: int, window_only: int) -> float:
    """Two-sided exact McNemar p-value, stable for large discordant counts."""

    if baseline_only < 0 or window_only < 0:
        raise AnalysisError("McNemar counts must be non-negative")
    total = baseline_only + window_only
    if total == 0:
        return 1.0
    tail = min(baseline_only, window_only)
    logs = [
        math.lgamma(total + 1)
        - math.lgamma(index + 1)
        - math.lgamma(total - index + 1)
        - total * math.log(2.0)
        for index in range(tail + 1)
    ]
    peak = max(logs)
    one_sided = math.exp(peak) * sum(math.exp(value - peak) for value in logs)
    return min(1.0, 2.0 * one_sided)


def spearman_report(
    left: Sequence[float], right: Sequence[float], windows: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """Spearman rho using average ranks for ties."""

    if len(left) != len(right):
        raise AnalysisError("Spearman vectors must have equal length")
    if len(left) < 2:
        return {"available": False, "n": len(left), "rho": None, "reason": "need >=2 windows"}
    x = [_finite_number(value, "Spearman score") for value in left]
    y = [_finite_number(value, "Spearman delta") for value in right]
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    rho = _pearson(x_rank, y_rank)
    if rho is None:
        return {
            "available": False,
            "n": len(x),
            "rho": None,
            "reason": "constant rank vector",
            "windows": list(windows or []),
        }
    return {
        "available": True,
        "n": len(x),
        "rho": rho,
        "windows": list(windows or []),
    }


def calibration_bootstrap_stability(
    score_report: Mapping[str, Any],
    candidate_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run the criterion's five calibration resamples with no gold labels."""

    primary = str(score_report.get("primary_signal", ""))
    if primary not in PRIMARY_SIGNALS:
        return {"available": False, "reason": "unsupported primary signal"}
    candidates = score_report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return {"available": False, "reason": "selection report has no candidates"}
    criterion = score_report.get("criterion", {})
    bootstrap_config = criterion.get("bootstrap", {}) if isinstance(criterion, Mapping) else {}
    replicates = int(bootstrap_config.get("replicates", 5))
    seed = int(bootstrap_config.get("seed", 20260710))
    stable_at_least = int(
        bootstrap_config.get("stable_if_same_or_adjacent_at_least", 4)
    )
    if replicates != 5:
        return {
            "available": False,
            "reason": "phase-one calibration bootstrap must use exactly 5 replicates",
        }
    if stable_at_least != 4:
        return {
            "available": False,
            "reason": "phase-one stability threshold must be exactly 4 of 5",
        }

    by_window: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or "window" not in candidate:
            return {"available": False, "reason": "malformed candidate"}
        window = format_window(parse_window(str(candidate["window"])))
        sample_map: Dict[str, Mapping[str, Any]] = {}
        sample_rows = candidate.get("sample_scores")
        if not isinstance(sample_rows, list):
            return {"available": False, "reason": "candidate sample_scores are unavailable"}
        for row in sample_rows:
            if not isinstance(row, Mapping) or "id" not in row or not isinstance(
                row.get("scores"), Mapping
            ):
                continue
            if _sample_supports_signal(row["scores"], primary):
                sample_map[str(row["id"])] = row["scores"]
        if not sample_map:
            return {
                "available": False,
                "reason": "candidate %s has no samples for %s" % (window, primary),
            }
        by_window[window] = sample_map

    common_ids = set.intersection(*(set(values) for values in by_window.values()))
    if not common_ids:
        return {"available": False, "reason": "candidates share no calibration sample IDs"}
    ordered_windows = [
        format_window(parse_window(str(window)))
        for window in (candidate_order or [])
    ]
    if not ordered_windows:
        return {
            "available": False,
            "reason": "frozen candidate-window order is unavailable",
        }
    if len(ordered_windows) != len(set(ordered_windows)) or set(ordered_windows) != set(
        by_window
    ):
        return {
            "available": False,
            "reason": "frozen candidate-window order disagrees with scored candidates",
        }
    reference_windows = list(score_report.get("primary_top_windows", []))
    reference_windows = [
        format_window(parse_window(str(window)))
        for window in reference_windows
        if format_window(parse_window(str(window))) in by_window
    ]
    if not reference_windows:
        full_scores = {
            window: _aggregate_bootstrap_signal(
                list(sample_map.values()), primary
            )
            for window, sample_map in by_window.items()
        }
        reference_windows = _top_windows(full_scores)

    ids = sorted(common_ids)
    generator = random.Random(seed)
    outcomes = []
    stable_count = 0
    for replicate in range(replicates):
        resampled_ids = [generator.choice(ids) for _ in ids]
        scores = {
            window: _aggregate_bootstrap_signal(
                [sample_map[sample_id] for sample_id in resampled_ids], primary
            )
            for window, sample_map in by_window.items()
        }
        selected = _top_windows(scores)
        same_or_adjacent = _same_or_adjacent(
            selected, reference_windows, ordered_windows
        )
        stable_count += int(same_or_adjacent)
        outcomes.append(
            {
                "replicate": replicate + 1,
                "top_windows": selected,
                "scores": scores,
                "same_or_adjacent": same_or_adjacent,
            }
        )
    return {
        "available": True,
        "signal": primary,
        "replicates": replicates,
        "seed": seed,
        "sample_count": len(ids),
        "common_sample_ids": ids,
        "reference_top_windows": reference_windows,
        "candidate_order": ordered_windows,
        "adjacency_definition": "index distance <= 1 in the frozen candidate-window order",
        "same_or_adjacent_count": stable_count,
        "stable_if_at_least": stable_at_least,
        "stable": stable_count >= stable_at_least,
        "outcomes": outcomes,
    }


def render_analysis_summary(report: Mapping[str, Any]) -> str:
    """Render the final report in concise Chinese Markdown."""

    lines = [
        "# LoopScope 第一阶段窗口分析摘要",
        "",
        "- 分析范围：`%s`" % report["analysis_scope"],
        "- baseline accuracy：`%.6f`" % report["accuracy"]["baseline"],
        "- harmful 定义：`DeltaAcc < %.1f pp`"
        % report["harmful_definition"]["threshold_pp"],
        "",
        "## 窗口结果",
        "",
        "| window | accuracy | DeltaAcc (pp) | harmful |",
        "|---|---:|---:|:---:|",
    ]
    for item in report["windows"]:
        lines.append(
            "| %s | %.6f | %+.4f | %s |"
            % (item["window"], item["accuracy"], item["delta_acc_pp"], item["harmful"])
        )
    lines.extend(["", "## 信号相关性与 regret", ""])
    for signal in PRIMARY_SIGNALS:
        correlation = report["signal_correlations"][signal]
        regret = report["regret_by_signal"][signal]
        rho = "unavailable" if correlation.get("rho") is None else "%.4f" % correlation["rho"]
        regret_value = regret.get("worst_case_regret_pp")
        regret_text = "unavailable" if regret_value is None else "%.4f pp" % regret_value
        lines.append(
            "- `%s`: Spearman=%s, worst-case regret=%s"
            % (signal, rho, regret_text)
        )
    stability = report["calibration_bootstrap"]
    lines.extend(["", "## 校准稳定性", ""])
    if stability.get("available"):
        lines.append(
            "- 5 次 bootstrap 中同窗或相邻窗 %d 次；稳定=%s。"
            % (stability["same_or_adjacent_count"], stability["stable"])
        )
    else:
        lines.append("- unavailable：%s" % stability.get("reason", "unknown"))
    if report.get("warnings"):
        lines.extend(["", "## 警告", ""])
        lines.extend("- %s" % warning for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _signal_score_maps(score_report: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    candidates = score_report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise AnalysisError("selection report contains no candidates")
    result = {signal: {} for signal in PRIMARY_SIGNALS}
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or "window" not in candidate:
            raise AnalysisError("malformed candidate in selection report")
        window = format_window(parse_window(str(candidate["window"])))
        scores = candidate.get("scores")
        if not isinstance(scores, Mapping):
            raise AnalysisError("candidate %s has no scores" % window)
        for signal in PRIMARY_SIGNALS:
            result[signal][window] = _finite_number(scores.get(signal), signal)
    return result


def _signal_regret(
    scores: Mapping[str, float],
    delta_by_window: Mapping[str, float],
    candidate_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    shared = sorted(set(scores).intersection(delta_by_window), key=_window_sort_key)
    if not shared:
        return {"available": False, "absolute_regret_pp": None, "reason": "no evaluated candidates"}
    top_score = max(scores[window] for window in scores)
    tied = {window for window, value in scores.items() if value == top_score}
    frozen_order = [
        format_window(parse_window(str(window))) for window in (candidate_order or [])
    ]
    selected_all = [window for window in frozen_order if window in tied]
    selected_all.extend(sorted(tied.difference(selected_all), key=_window_sort_key))
    selected_evaluated = [window for window in selected_all if window in delta_by_window]
    if not selected_evaluated:
        return {
            "available": False,
            "absolute_regret_pp": None,
            "selected_windows": selected_all,
            "reason": "top-scored windows were not evaluated",
        }
    best_delta = max(delta_by_window[window] for window in shared)
    best_windows = [
        window for window in shared if math.isclose(delta_by_window[window], best_delta)
    ]
    deterministic = selected_evaluated[0]
    per_selected = {
        window: best_delta - delta_by_window[window] for window in selected_evaluated
    }
    return {
        "available": True,
        "selected_window": deterministic,
        "selected_windows": selected_all,
        "selected_tie_count": len(selected_all),
        "tie_policy": dict(TIE_POLICY),
        "display_tiebreak": {
            "selected_window": deterministic,
            "policy": "first tied window in frozen candidate order",
            "phase_gate_uses_this_window": False,
        },
        "best_window": best_windows[0],
        "best_windows": best_windows,
        "best_delta_acc_pp": best_delta,
        "selected_delta_acc_pp": delta_by_window[deterministic],
        "absolute_regret_pp": per_selected[deterministic],
        "best_case_regret_pp": min(per_selected.values()),
        "worst_case_regret_pp": max(per_selected.values()),
        "phase_gate_regret_pp": max(per_selected.values()),
        "per_selected_window_regret_pp": per_selected,
    }


def _comparison_report(
    window_grid: Optional[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    primary_regret: Mapping[str, Any],
) -> Dict[str, Any]:
    by_window = {str(row["window"]): row for row in result_rows}
    comparison = (
        window_grid.get("comparison_windows", {})
        if isinstance(window_grid, Mapping)
        else {}
    )
    anchors = window_grid.get("anchors", {}) if isinstance(window_grid, Mapping) else {}
    fixed = comparison.get("fixed_depth") if isinstance(comparison, Mapping) else None
    random_windows = comparison.get("random_in_band", []) if isinstance(comparison, Mapping) else []
    fixed = format_window(parse_window(str(fixed))) if fixed else None
    random_windows = [format_window(parse_window(str(window))) for window in random_windows]
    anchor = anchors.get("required") if isinstance(anchors, Mapping) else None
    anchor = format_window(parse_window(str(anchor))) if anchor else None
    selected = primary_regret.get("selected_window") if primary_regret.get("available") else None
    selected_windows = (
        list(primary_regret.get("selected_windows", []))
        if primary_regret.get("available")
        else []
    )
    selected_delta = by_window[selected]["delta_acc_pp"] if selected in by_window else None
    selected_deltas = {
        window: float(by_window[window]["delta_acc_pp"])
        for window in selected_windows
        if window in by_window
    }

    fixed_row = by_window.get(fixed) if fixed else None
    random_rows = [by_window[window] for window in random_windows if window in by_window]
    random_deltas = [float(row["delta_acc_pp"]) for row in random_rows]
    random_complete = bool(random_windows) and len(random_rows) == len(random_windows)
    random_median = median(random_deltas) if random_deltas else None
    return {
        "selected_primary_window": selected,
        "selected_primary_windows": selected_windows,
        "selected_window_tie_policy": dict(TIE_POLICY),
        "fixed_depth": {
            "window": fixed,
            "available": fixed_row is not None,
            "delta_acc_pp": fixed_row.get("delta_acc_pp") if fixed_row else None,
            "selected_minus_fixed_pp": (
                selected_delta - float(fixed_row["delta_acc_pp"])
                if selected_delta is not None and fixed_row is not None
                else None
            ),
            "worst_case_selected_minus_fixed_pp": (
                min(
                    delta - float(fixed_row["delta_acc_pp"])
                    for delta in selected_deltas.values()
                )
                if selected_deltas and fixed_row is not None
                else None
            ),
        },
        "anchor": {
            "window": anchor,
            "available": anchor in by_window if anchor else False,
            "delta_acc_pp": by_window[anchor]["delta_acc_pp"] if anchor in by_window else None,
        },
        "random_in_band": {
            "expected_windows": random_windows,
            "available_windows": [row["window"] for row in random_rows],
            "complete": random_complete,
            "delta_acc_pp": {row["window"]: row["delta_acc_pp"] for row in random_rows},
            "median_delta_acc_pp": random_median,
            "selected_minus_median_pp": (
                selected_delta - random_median
                if selected_delta is not None and random_complete and random_median is not None
                else None
            ),
            "selected_minus_median_pp_by_window": (
                {
                    window: delta - random_median
                    for window, delta in selected_deltas.items()
                }
                if random_complete and random_median is not None
                else {}
            ),
            "selector_outperforms_median": (
                all(delta > random_median for delta in selected_deltas.values())
                if selected_deltas and random_complete and random_median is not None
                else None
            ),
            "outperformance_policy": "all tied top windows must beat the random median",
        },
    }


def _phase_assessment(
    correlation: Mapping[str, Any],
    regret: Mapping[str, Any],
    comparisons: Mapping[str, Any],
    calibration_bootstrap: Mapping[str, Any],
    paired_alignment_ready: bool,
    analysis_scope: str,
) -> Dict[str, Any]:
    if analysis_scope == "limit-smoke":
        return {
            "eligible": False,
            "decision": "engineering_smoke_only",
            "reason": "limit accuracy must not choose or validate a window",
        }
    rho = correlation.get("rho") if correlation.get("available") else None
    regret_pp = regret.get("worst_case_regret_pp") if regret.get("available") else None
    random_result = comparisons.get("random_in_band", {})
    if rho is None or regret_pp is None:
        return {"eligible": False, "decision": "insufficient_evidence"}
    if rho < 0.3:
        signal_decision = "proxy_failure_or_wide_basin_analysis"
    elif rho < 0.5:
        signal_decision = "uncertain_audit_stability"
    elif regret_pp <= 0.3:
        signal_decision = "eligible_to_request_multimodel_validation"
    else:
        signal_decision = "correlated_but_regret_too_large"
    random_outperformance = random_result.get("selector_outperforms_median")
    claim_allowed = signal_decision == "eligible_to_request_multimodel_validation"
    if random_outperformance is not True:
        claim_allowed = False
    calibration_stable = (
        calibration_bootstrap.get("stable")
        if calibration_bootstrap.get("available")
        else None
    )
    if calibration_stable is not True:
        claim_allowed = False
    if not paired_alignment_ready:
        claim_allowed = False
    return {
        "eligible": claim_allowed,
        "decision": signal_decision,
        "rho": rho,
        "phase_gate_regret_pp": regret_pp,
        "phase_gate_regret_policy": "worst_case_across_all_tied_top_windows",
        "display_absolute_regret_pp": regret.get("absolute_regret_pp"),
        "selector_outperforms_random_median": random_outperformance,
        "calibration_bootstrap_stable": calibration_stable,
        "paired_sample_alignment_ready": paired_alignment_ready,
    }


def _window_roles(window_grid: Optional[Mapping[str, Any]]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    if not isinstance(window_grid, Mapping):
        return result
    for item in window_grid.get("windows", []):
        if isinstance(item, Mapping) and item.get("window"):
            window = format_window(parse_window(str(item["window"])))
            result[window] = [str(role) for role in item.get("roles", [])]
    comparison = window_grid.get("comparison_windows", {})
    if isinstance(comparison, Mapping):
        fixed = comparison.get("fixed_depth")
        if fixed:
            result.setdefault(format_window(parse_window(str(fixed))), []).append("fixed_depth")
        for window in comparison.get("random_in_band", []):
            result.setdefault(format_window(parse_window(str(window))), []).append(
                "random_in_band"
            )
    return {window: sorted(set(values)) for window, values in result.items()}


def _normalize_window_results(
    values: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Mapping[str, Any]]:
    result = {}
    for raw_window, payload in values.items():
        window = format_window(parse_window(str(raw_window)))
        if window in result:
            raise AnalysisError("duplicate normalized window result: %s" % window)
        if not isinstance(payload, Mapping):
            raise AnalysisError("window result %s must be an object" % window)
        result[window] = payload
    return result


def _weighted_group_accuracy(
    payload: Mapping[str, Any],
    matching: Sequence[Tuple[Any, Mapping[str, Any]]],
    metric: str,
) -> Optional[float]:
    sample_counts = payload.get("n-samples")
    if not isinstance(sample_counts, Mapping):
        return None
    numerator = 0.0
    denominator = 0.0
    for name, entry in matching:
        count_entry = sample_counts.get(name)
        if isinstance(count_entry, Mapping):
            raw_count = count_entry.get("effective", count_entry.get("original"))
        else:
            raw_count = count_entry
        if raw_count is None:
            return None
        count = _finite_number(raw_count, "n-samples.%s" % name)
        if count <= 0.0:
            return None
        numerator += _accuracy_fraction(entry[metric], "%s.%s" % (name, metric)) * count
        denominator += count
    return numerator / denominator if denominator else None


def _sample_id(item: Mapping[str, Any]) -> str:
    for key in ("sample_id", "id", "doc_id"):
        if item.get(key) is not None:
            return str(item[key])
    doc = item.get("doc")
    if isinstance(doc, Mapping):
        for key in ("id", "doc_id"):
            if doc.get(key) is not None:
                return str(doc[key])
    raise AnalysisError("sample has no stable id/sample_id/doc_id")


def _sample_correctness(item: Mapping[str, Any], metric: str) -> bool:
    base_metric = metric.split(",", 1)[0]
    keys = (metric, base_metric, "correct", "exact_match", "acc", "acc_norm")
    for container in (item, item.get("metrics")):
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            if key in container:
                value = container[key]
                if isinstance(value, bool):
                    return value
                number = _finite_number(value, "sample correctness")
                if number in (0.0, 1.0):
                    return bool(number)
                raise AnalysisError("sample correctness must be exactly 0 or 1")
    raise AnalysisError("sample has no explicit correctness metric %s" % metric)


def _sample_supports_signal(scores: Mapping[str, Any], signal: str) -> bool:
    if signal == "r_times_one_minus_q":
        required = ("activity_r", "negative_contraction_q")
    else:
        required = (signal,)
    try:
        for key in required:
            _finite_number(scores.get(key), key)
    except AnalysisError:
        return False
    return True


def _aggregate_bootstrap_signal(
    score_rows: Sequence[Mapping[str, Any]], signal: str
) -> float:
    if not score_rows:
        raise AnalysisError("cannot aggregate an empty calibration resample")
    if signal == "r_times_one_minus_q":
        activity = median(
            _finite_number(row.get("activity_r"), "activity_r") for row in score_rows
        )
        contraction = median(
            -_finite_number(row.get("negative_contraction_q"), "negative_contraction_q")
            for row in score_rows
        )
        return activity * max(0.0, 1.0 - contraction)
    values = [_finite_number(row.get(signal), signal) for row in score_rows]
    if signal in ("activity_r", "negative_contraction_q"):
        return median(values)
    return mean(values)


def _top_windows(scores: Mapping[str, float]) -> List[str]:
    top = max(scores.values())
    return sorted(
        [window for window, value in scores.items() if value == top],
        key=_window_sort_key,
    )


def _same_or_adjacent(
    selected: Sequence[str], reference: Sequence[str], ordered_windows: Sequence[str]
) -> bool:
    normalized_order = [
        format_window(parse_window(str(window))) for window in ordered_windows
    ]
    if not normalized_order or len(normalized_order) != len(set(normalized_order)):
        return False
    positions = {window: index for index, window in enumerate(normalized_order)}
    return any(
        format_window(parse_window(str(selected_window))) in positions
        and format_window(parse_window(str(reference_window))) in positions
        and abs(
            positions[format_window(parse_window(str(selected_window)))]
            - positions[format_window(parse_window(str(reference_window)))]
        )
        <= 1
        for selected_window in selected
        for reference_window in reference
    )


def _frozen_candidate_order(
    score_report: Mapping[str, Any], window_grid: Optional[Mapping[str, Any]]
) -> List[str]:
    if isinstance(window_grid, Mapping):
        return [format_window(window) for window in candidate_windows(window_grid)]
    score_grid = score_report.get("window_grid")
    if not isinstance(score_grid, Mapping):
        return []
    return [
        format_window(parse_window(str(window)))
        for window in score_grid.get("candidate_windows", [])
    ]


def _validate_tie_policy(score_report: Mapping[str, Any]) -> None:
    criterion = score_report.get("criterion")
    if not isinstance(criterion, Mapping) or criterion.get("tie_policy") != TIE_POLICY:
        raise AnalysisError(
            "selection criterion must freeze retain-all ties, frozen-order display, "
            "and worst-case phase-gate regret"
        )


def _validate_score_report_integrity(score_report: Mapping[str, Any]) -> None:
    expected = score_report.get("manifest_sha256")
    actual = manifest_sha256(score_report)
    if not isinstance(expected, str) or expected != actual:
        raise AnalysisError("selection report manifest_sha256 mismatch")
    criterion = score_report.get("criterion")
    if not isinstance(criterion, Mapping):
        raise AnalysisError("selection report criterion must be an object")
    if score_report.get("criterion_sha256") != manifest_sha256(criterion):
        raise AnalysisError("selection report criterion_sha256 mismatch")


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        rank = (cursor + 1 + stop) / 2.0
        for offset in range(cursor, stop):
            ranks[order[offset]] = rank
        cursor = stop
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0.0 or right_ss == 0.0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise AnalysisError("cannot take a quantile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def _accuracy_fraction(value: Any, context: str) -> float:
    number = _finite_number(value, context)
    if not 0.0 <= number <= 1.0:
        raise AnalysisError("%s must be a fraction in [0, 1]" % context)
    return number


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise AnalysisError("%s must be numeric, not boolean" % context)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("%s must be numeric" % context) from exc
    if not math.isfinite(number):
        raise AnalysisError("%s must be finite" % context)
    return number


def _window_seed(seed: int, window: str) -> int:
    digest = sha256(window.encode("utf-8")).hexdigest()
    return int(seed) + int(digest[:8], 16)


def _window_sort_key(window: str) -> Tuple[int, int]:
    return parse_window(window)


def _parse_window_result_spec(specification: str) -> Tuple[str, Path]:
    if "=" not in specification:
        raise AnalysisError("--window-result must use WINDOW=PATH")
    raw_window, raw_path = specification.split("=", 1)
    if not raw_path:
        raise AnalysisError("--window-result path must not be empty")
    return format_window(parse_window(raw_window)), Path(raw_path)


def _load_result_bundle(path: Path, require_provenance: bool) -> Dict[str, Any]:
    resolved, payload = _load_result_artifact(path)
    bundle: Dict[str, Any] = {
        "results_path": resolved.resolve(),
        "results": payload,
        "artifact_sha256": {"results.json": _file_sha256(resolved)},
    }
    if not require_provenance:
        return bundle
    if resolved.name != "results.json":
        raise AnalysisError("full analysis requires an artifact named results.json")
    command_path = resolved.parent / "command_args.json"
    revision_path = resolved.parent / "model_revision.json"
    if not command_path.is_file() or not revision_path.is_file():
        raise AnalysisError(
            "full result %s requires adjacent command_args.json and model_revision.json"
            % resolved
        )
    bundle.update(
        {
            "command_args_path": command_path.resolve(),
            "command_args": _load_json_object(command_path, "command args"),
            "model_revision_path": revision_path.resolve(),
            "model_revision": _load_json_object(revision_path, "model revision"),
        }
    )
    bundle["artifact_sha256"].update(
        {
            "command_args.json": _file_sha256(command_path),
            "model_revision.json": _file_sha256(revision_path),
        }
    )
    sample_hashes = {
        path.name: _file_sha256(path)
        for path in sorted(resolved.parent.glob("samples_*.jsonl"))
    }
    if sample_hashes:
        bundle["artifact_sha256"]["sample_sidecars"] = sample_hashes
    return bundle


def _validate_full_cli_provenance(
    run_manifest_path: Path,
    run_manifest: Mapping[str, Any],
    window_grid_path: Path,
    window_grid: Mapping[str, Any],
    score_report_path: Path,
    score_report: Mapping[str, Any],
    baseline_bundle: Mapping[str, Any],
    window_bundles: Mapping[str, Mapping[str, Any]],
    task: str,
    metric: str,
) -> Dict[str, Any]:
    """Fail closed unless every full result is one frozen Gate E job artifact."""

    if task != "mmlu" or metric != "acc,none":
        raise AnalysisError("phase-one full analysis requires task=mmlu and metric=acc,none")
    _validate_score_report_integrity(score_report)
    _validate_tie_policy(score_report)
    if run_manifest.get("schema_version") != PHASE1_RUN_SCHEMA_VERSION:
        raise AnalysisError("unsupported phase-one run manifest schema")
    if run_manifest.get("manifest_sha256") != manifest_sha256(run_manifest):
        raise AnalysisError("phase-one run manifest SHA256 mismatch")
    run_root = Path(str(run_manifest.get("run_root", ""))).resolve()
    expected_manifest_path = run_root / "control" / "phase1_run_manifest.json"
    if run_manifest_path.resolve() != expected_manifest_path:
        raise AnalysisError("--run-manifest path does not match its frozen run_root")

    recipe = run_manifest.get("frozen_recipe")
    if not isinstance(recipe, Mapping):
        raise AnalysisError("phase-one run manifest is missing frozen_recipe")
    for key, expected in PHASE1_RECIPE.items():
        if recipe.get(key) != expected:
            raise AnalysisError(
                "phase-one frozen recipe mismatch at %s: expected %r, got %r"
                % (key, expected, recipe.get(key))
            )

    validate_window_grid(window_grid)
    grid_input = run_manifest.get("inputs", {}).get("window_grid")
    if not isinstance(grid_input, Mapping):
        raise AnalysisError("run manifest is missing inputs.window_grid")
    if grid_input.get("manifest_sha256") != window_grid.get("manifest_sha256"):
        raise AnalysisError("run manifest and supplied window-grid hashes disagree")
    if Path(str(grid_input.get("snapshot_path", ""))).resolve() != window_grid_path.resolve():
        raise AnalysisError("--window-grid must be the run manifest's frozen snapshot")
    if score_report.get("window_grid", {}).get("manifest_sha256") != window_grid.get(
        "manifest_sha256"
    ):
        raise AnalysisError("window scores are bound to a different window-grid manifest")
    score_freeze = _validate_gate_e_score_freeze(
        run_root=run_root,
        run_manifest=run_manifest,
        score_report_path=score_report_path,
        score_report=score_report,
        window_grid=window_grid,
    )
    full_probe_pool = _validate_full_probe_pool_provenance(
        run_root=run_root,
        run_manifest=run_manifest,
        score_report=score_report,
    )
    _validate_frozen_full_probe_evidence(score_freeze, full_probe_pool)

    stages = run_manifest.get("stages")
    stage = stages.get(PHASE1_FULL_STAGE) if isinstance(stages, Mapping) else None
    if not isinstance(stage, Mapping) or stage.get("scientific_selection_allowed") is not True:
        raise AnalysisError("run manifest does not authorize scientific Gate E analysis")
    if stage.get("limit") is not None:
        raise AnalysisError("Gate E full stage must not declare a limit")
    jobs = stage.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise AnalysisError("Gate E full stage has no frozen jobs")

    expected_windows = _expected_full_windows(window_grid)
    baseline_job: Optional[Dict[str, Any]] = None
    jobs_by_window: Dict[str, Dict[str, Any]] = {}
    job_ids = set()
    for raw_job in jobs:
        job = _validate_manifest_eval_job(raw_job, run_root, recipe)
        if job["job_id"] in job_ids:
            raise AnalysisError("duplicate Gate E job_id: %s" % job["job_id"])
        job_ids.add(job["job_id"])
        if job["window"] is None:
            if baseline_job is not None:
                raise AnalysisError("Gate E manifest contains multiple baseline jobs")
            baseline_job = job
        else:
            if job["window"] in jobs_by_window:
                raise AnalysisError("Gate E manifest contains duplicate window jobs")
            jobs_by_window[job["window"]] = job
    if baseline_job is None:
        raise AnalysisError("Gate E manifest is missing the baseline job")
    if set(jobs_by_window) != set(expected_windows):
        raise AnalysisError("Gate E job/window mapping disagrees with the frozen grid")
    if set(window_bundles) != set(expected_windows):
        raise AnalysisError("provided full window artifacts do not match Gate E jobs")

    artifact_records = {
        "baseline": _validate_eval_bundle(baseline_bundle, baseline_job, recipe),
        "windows": {},
    }
    revisions = {tuple(artifact_records["baseline"]["revision_closure"].values())}
    for window in expected_windows:
        record = _validate_eval_bundle(
            window_bundles[window], jobs_by_window[window], recipe
        )
        artifact_records["windows"][window] = record
        revisions.add(tuple(record["revision_closure"].values()))
    if len(revisions) != 1:
        raise AnalysisError(
            "mixed model/tokenizer/manifest revisions across full artifacts: %s"
            % sorted(revisions)
        )
    revision_closure = artifact_records["baseline"]["revision_closure"]
    model_revision = revision_closure["model_commit"]
    _validate_score_probe_provenance(
        score_report, run_manifest, recipe, model_revision
    )

    return {
        "validated": True,
        "policy": "phase1_gate_e_fail_closed",
        "run_manifest": {
            "path": str(run_manifest_path.resolve()),
            "manifest_sha256": run_manifest["manifest_sha256"],
            "file_sha256": _file_sha256(run_manifest_path),
        },
        "window_grid": {
            "path": str(window_grid_path.resolve()),
            "manifest_sha256": window_grid["manifest_sha256"],
            "file_sha256": _file_sha256(window_grid_path),
        },
        "window_scores": {
            "path": str(score_report_path.resolve()),
            "manifest_sha256": score_report["manifest_sha256"],
            "criterion_sha256": score_report["criterion_sha256"],
            "file_sha256": _file_sha256(score_report_path),
        },
        "score_freeze": score_freeze,
        "full_probe_pool": full_probe_pool,
        "stage": PHASE1_FULL_STAGE,
        "frozen_recipe": dict(recipe),
        "model_revision": model_revision,
        "revision_closure": revision_closure,
        "artifacts": artifact_records,
    }


def _validate_full_probe_pool_provenance(
    run_root: Path,
    run_manifest: Mapping[str, Any],
    score_report: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest_commit = run_manifest.get("frozen_recipe", {}).get("revision")
    if manifest_commit != PHASE1_RECIPE["revision"]:
        raise AnalysisError("run manifest does not freeze the phase-one model revision")
    inputs = run_manifest.get("inputs")
    pool_input = inputs.get("probe_pool") if isinstance(inputs, Mapping) else None
    if not isinstance(pool_input, Mapping):
        raise AnalysisError("run manifest is missing frozen probe-pool provenance")
    manifest_path = Path(str(pool_input.get("manifest_snapshot_path", "")))
    pool_path = Path(str(pool_input.get("snapshot_path", "")))
    expected_manifest_path = run_root / "manifests" / "probe_pool_manifest.json"
    expected_pool_path = run_root / "manifests" / "probe_pool.jsonl"
    if manifest_path.resolve() != expected_manifest_path.resolve():
        raise AnalysisError("probe-pool manifest is not the frozen run snapshot")
    if pool_path.resolve() != expected_pool_path.resolve():
        raise AnalysisError("probe-pool JSONL is not the frozen run snapshot")
    if not manifest_path.is_file() or not pool_path.is_file():
        raise AnalysisError("frozen probe-pool snapshot artifacts are missing")
    pool_manifest = _load_json_object(manifest_path, "frozen probe-pool manifest")
    if pool_manifest.get("schema_version") != "loopscope.probe-pool-manifest.v1":
        raise AnalysisError("unsupported frozen probe-pool manifest schema")
    if pool_manifest.get("manifest_sha256") != manifest_sha256(pool_manifest):
        raise AnalysisError("frozen probe-pool manifest canonical hash mismatch")
    if pool_input.get("manifest_sha256") != pool_manifest.get("manifest_sha256"):
        raise AnalysisError("run manifest probe-pool SHA256 mismatch")
    pool_file_hash = _file_sha256(pool_path)
    if pool_input.get("pool_sha256") != pool_file_hash:
        raise AnalysisError("frozen probe-pool JSONL SHA256 mismatch")

    provenance = score_report.get("probe_provenance")
    score_pool = provenance.get("probe_pool") if isinstance(provenance, Mapping) else None
    if not isinstance(score_pool, Mapping):
        raise AnalysisError("window scores are missing full probe-pool provenance")
    full_count = int(pool_manifest.get("count", -1))
    if (
        int(score_pool.get("count", -1)) != full_count
        or int(score_pool.get("source_manifest_count", -1)) != full_count
        or int(pool_input.get("count", -1)) != full_count
    ):
        raise AnalysisError("Gate E score does not use the complete frozen probe pool")
    sample_ids = [str(value) for value in pool_manifest.get("sample_ids", [])]
    if score_pool.get("sample_ids") != sample_ids:
        raise AnalysisError("Gate E score sample IDs/order differ from the full pool")
    records = pool_manifest.get("records")
    rendering_records = pool_manifest.get("rendering_records")
    if score_pool.get("records") != records or score_pool.get(
        "rendering_records"
    ) != rendering_records:
        raise AnalysisError("Gate E score record/rendering provenance differs from full pool")
    source_hash = pool_manifest["manifest_sha256"]
    if score_pool.get("source_manifest_sha256") != source_hash:
        raise AnalysisError("Gate E score source manifest SHA256 differs from full pool")
    full_render_hash = pool_manifest.get("render_contract_subset_sha256")
    if (
        score_pool.get("source_render_contract_subset_sha256") != full_render_hash
        or score_pool.get("selected_render_contract_subset_sha256") != full_render_hash
    ):
        raise AnalysisError("Gate E score uses a rendering-contract subset, not the full pool")
    actual_render_hash = sha256(canonical_json_bytes(rendering_records)).hexdigest()
    if actual_render_hash != full_render_hash:
        raise AnalysisError("frozen probe-pool rendering subset hash mismatch")
    renderer = score_pool.get("renderer")
    if not isinstance(renderer, Mapping) or renderer.get(
        "render_contract_sha256"
    ) != pool_manifest.get("renderer", {}).get("render_contract_sha256"):
        raise AnalysisError("Gate E score renderer contract differs from frozen pool")

    selected_manifest = {
        "schema_version": "loopscope.probe-pool-selection.v1",
        "source": score_pool.get("source"),
        "split": score_pool.get("split"),
        "count": full_count,
        "seed": score_pool.get("seed"),
        "sample_ids": sample_ids,
        "records": records,
        "task_group": score_pool.get("task_group"),
        "num_fewshot": score_pool.get("num_fewshot"),
        "uses_target_gold_labels": score_pool.get("uses_target_gold_labels"),
        "fewshot_answers_present": score_pool.get("fewshot_answers_present"),
        "renderer": renderer,
        "render_contract_sha256": score_pool.get("render_contract_sha256"),
        "rendering_records": rendering_records,
        "render_contract_subset_sha256": full_render_hash,
        "source_manifest_sha256": source_hash,
    }
    selected_hash = manifest_sha256(selected_manifest)
    if score_pool.get("selected_subset_sha256") != selected_hash:
        raise AnalysisError("Gate E score selected-subset canonical SHA256 mismatch")

    probe_jobs = run_manifest.get("stages", {}).get("gate-e-probe", {}).get("jobs")
    if not isinstance(probe_jobs, list) or len(probe_jobs) != 2:
        raise AnalysisError("run manifest must freeze exactly two Gate E probe jobs")
    jobs_by_id = {
        str(job.get("job_id")): job for job in probe_jobs if isinstance(job, Mapping)
    }
    expected_ids = {"probe-layers-full", "probe-windows-full"}
    if set(jobs_by_id) != expected_ids:
        raise AnalysisError("Gate E probe job identities differ from the frozen contract")
    layer_path = Path(str(jobs_by_id["probe-layers-full"].get("revision_artifact", "")))
    window_path = Path(str(jobs_by_id["probe-windows-full"].get("revision_artifact", "")))
    score_paths = score_report.get("inputs", {}).get("paths", {})
    if Path(str(score_paths.get("layer_probe", ""))).resolve() != layer_path.resolve():
        raise AnalysisError("score layer-probe path differs from Gate E job output")
    window_paths = score_paths.get("window_probes")
    if not isinstance(window_paths, list) or len(window_paths) != 1:
        raise AnalysisError("score must use exactly one frozen full-window probe report")
    if Path(str(window_paths[0])).resolve() != window_path.resolve():
        raise AnalysisError("score window-probe path differs from Gate E job output")
    report_provenance = {}
    for label, path, expected_hash in (
        (
            "layer_probe",
            layer_path,
            score_report.get("inputs", {}).get("layer_probe_manifest_sha256"),
        ),
        (
            "window_probe",
            window_path,
            (score_report.get("inputs", {}).get("window_probe_manifest_sha256") or [None])[0],
        ),
    ):
        if not path.is_file():
            raise AnalysisError("frozen Gate E %s report is missing" % label)
        report = _load_json_object(path, "Gate E %s report" % label)
        report_pool = report.get("probe_pool")
        if not isinstance(report_pool, Mapping):
            raise AnalysisError("Gate E %s report has no probe_pool" % label)
        if report_pool.get("manifest_sha256") != expected_hash:
            raise AnalysisError("score input manifest hash differs from %s report" % label)
        for key in (
            "count",
            "source_manifest_count",
            "sample_ids",
            "source_manifest_sha256",
            "selected_subset_sha256",
            "source_render_contract_subset_sha256",
            "selected_render_contract_subset_sha256",
        ):
            if report_pool.get(key) != score_pool.get(key):
                raise AnalysisError("Gate E %s probe-pool mismatch at %s" % (label, key))
        revision_closure = report.get("revision_closure")
        if not isinstance(revision_closure, Mapping):
            raise AnalysisError("Gate E %s report lacks revision_closure" % label)
        for key, expected in {
            "manifest_commit": manifest_commit,
            "model_commit": manifest_commit,
            "tokenizer_commit": manifest_commit,
            "match": True,
        }.items():
            if revision_closure.get(key) != expected:
                raise AnalysisError("Gate E %s revision mismatch at %s" % (label, key))
        if report.get("model", {}).get("revision") != manifest_commit or report.get(
            "tokenizer", {}
        ).get("revision") != manifest_commit:
            raise AnalysisError("Gate E %s model/tokenizer revision mismatch" % label)
        report_provenance[label] = {
            "path": str(path.resolve()),
            "file_sha256": _file_sha256(path),
            "canonical_sha256": manifest_sha256(report),
            "probe_pool_manifest_sha256": report_pool["manifest_sha256"],
            "revision_closure": dict(revision_closure),
        }
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": source_hash,
        "manifest_file_sha256": _file_sha256(manifest_path),
        "pool_path": str(pool_path.resolve()),
        "pool_file_sha256": pool_file_hash,
        "count": full_count,
        "source_manifest_sha256": source_hash,
        "selected_subset_sha256": selected_hash,
        "render_contract_subset_sha256": full_render_hash,
        "sample_ids_sha256": sha256(canonical_json_bytes(sample_ids)).hexdigest(),
        "revision_binding": {
            "manifest_commit": manifest_commit,
            "model_commit": manifest_commit,
            "tokenizer_commit": manifest_commit,
        },
        "probe_reports": report_provenance,
    }


def _validate_frozen_full_probe_evidence(
    score_freeze: Mapping[str, Any], full_probe_pool: Mapping[str, Any]
) -> None:
    """Require the current full probes to be byte-identical to the frozen score inputs."""

    frozen = score_freeze.get("full_probe_evidence")
    if not isinstance(frozen, Mapping):
        raise AnalysisError("Gate E score freeze is missing full_probe_evidence")

    for key in (
        "count",
        "source_manifest_sha256",
        "selected_subset_sha256",
        "render_contract_subset_sha256",
        "sample_ids_sha256",
        "revision_binding",
    ):
        if key not in frozen:
            raise AnalysisError("Gate E score freeze full-probe evidence is missing %s" % key)
        if frozen.get(key) != full_probe_pool.get(key):
            raise AnalysisError("Gate E frozen full-probe evidence mismatch at %s" % key)

    frozen_reports = frozen.get("probe_reports")
    current_reports = full_probe_pool.get("probe_reports")
    if not isinstance(frozen_reports, Mapping) or not isinstance(current_reports, Mapping):
        raise AnalysisError("Gate E score freeze has malformed full-probe report evidence")
    expected_reports = {
        "layer": current_reports.get("layer_probe"),
        "window": current_reports.get("window_probe"),
    }
    if set(frozen_reports) != set(expected_reports):
        raise AnalysisError("Gate E score freeze must bind exactly two full probe reports")
    for label, current in expected_reports.items():
        frozen_report = frozen_reports.get(label)
        if not isinstance(frozen_report, Mapping) or not isinstance(current, Mapping):
            raise AnalysisError("Gate E %s full-probe evidence is malformed" % label)
        for key in (
            "path",
            "file_sha256",
            "canonical_sha256",
            "probe_pool_manifest_sha256",
            "revision_closure",
        ):
            if key not in frozen_report:
                raise AnalysisError(
                    "Gate E %s full-probe evidence is missing %s" % (label, key)
                )
            if frozen_report.get(key) != current.get(key):
                raise AnalysisError(
                    "Gate E frozen %s full-probe evidence mismatch at %s" % (label, key)
                )


def _validate_gate_e_score_freeze(
    run_root: Path,
    run_manifest: Mapping[str, Any],
    score_report_path: Path,
    score_report: Mapping[str, Any],
    window_grid: Mapping[str, Any],
) -> Dict[str, Any]:
    freeze_path = run_root / "control" / "provenance" / "gate-e-score-freeze.json"
    if not freeze_path.is_file():
        raise AnalysisError("immutable gate-e score freeze is missing")
    freeze = _load_json_object(freeze_path, "Gate E score freeze")
    if freeze.get("schema_version") != "loopscope.score-freeze.v1":
        raise AnalysisError("unsupported Gate E score-freeze schema")
    if freeze.get("manifest_sha256") != manifest_sha256(freeze):
        raise AnalysisError("Gate E score-freeze canonical hash mismatch")
    if freeze.get("run_manifest_sha256") != run_manifest.get("manifest_sha256"):
        raise AnalysisError("Gate E score freeze is bound to another run manifest")

    stages = run_manifest.get("stages")
    score_stage = stages.get("gate-e-score") if isinstance(stages, Mapping) else None
    score_jobs = score_stage.get("jobs") if isinstance(score_stage, Mapping) else None
    if not isinstance(score_jobs, list) or len(score_jobs) != 1:
        raise AnalysisError("run manifest must contain exactly one gate-e-score job")
    score_job = score_jobs[0]
    if not isinstance(score_job, Mapping):
        raise AnalysisError("gate-e-score job must be an object")
    expected_score_path = Path(str(score_job.get("output_dir", ""))) / "window_scores.json"
    if freeze.get("score_job_id") != score_job.get("job_id"):
        raise AnalysisError("Gate E score freeze job id mismatch")
    if freeze.get("score_report_path") != str(expected_score_path):
        raise AnalysisError("Gate E score freeze job/path mismatch")
    if score_report_path.resolve() != expected_score_path.resolve():
        raise AnalysisError("--selection-report must be the exact frozen score path")
    score_file_hash = _file_sha256(score_report_path)
    if freeze.get("score_report_sha256") != score_file_hash:
        raise AnalysisError("window score file SHA256 changed after freeze")
    if freeze.get("score_report_manifest_sha256") != score_report.get(
        "manifest_sha256"
    ):
        raise AnalysisError("window score manifest SHA256 differs from freeze")

    inputs = run_manifest.get("inputs")
    criterion_input = inputs.get("criterion") if isinstance(inputs, Mapping) else None
    if not isinstance(criterion_input, Mapping):
        raise AnalysisError("run manifest is missing frozen criterion provenance")
    criterion_path = Path(str(criterion_input.get("snapshot_path", "")))
    if freeze.get("criterion_path") != str(criterion_path):
        raise AnalysisError("Gate E score freeze criterion path mismatch")
    if not criterion_path.is_file():
        raise AnalysisError("frozen criterion snapshot is missing")
    criterion = _load_json_object(criterion_path, "frozen criterion")
    criterion_file_hash = _file_sha256(criterion_path)
    criterion_hash = manifest_sha256(criterion)
    if freeze.get("criterion_file_sha256") != criterion_file_hash or criterion_input.get(
        "file_sha256"
    ) != criterion_file_hash:
        raise AnalysisError("frozen criterion file SHA256 mismatch")
    if (
        freeze.get("criterion_sha256") != criterion_hash
        or criterion_input.get("criterion_sha256") != criterion_hash
        or score_report.get("criterion_sha256") != criterion_hash
    ):
        raise AnalysisError("frozen criterion canonical SHA256 mismatch")

    expected_candidates = [str(item["window"]) for item in window_grid["windows"]]
    if freeze.get("window_grid_manifest_sha256") != window_grid.get("manifest_sha256"):
        raise AnalysisError("score freeze window-grid SHA256 mismatch")
    if freeze.get("candidate_windows") != expected_candidates:
        raise AnalysisError("score freeze candidate list differs from frozen grid")
    if score_report.get("window_grid", {}).get("candidate_windows") != expected_candidates:
        raise AnalysisError("window score candidate list differs from score freeze")

    revision_path = (
        run_root
        / "control"
        / "provenance"
        / "gate-e-probe-model-revisions.json"
    )
    if not revision_path.is_file():
        raise AnalysisError("gate-e-probe revision report is missing")
    revision_report = _load_json_object(revision_path, "gate-e-probe revision report")
    expected_revision = PHASE1_RECIPE["revision"]
    if revision_report.get("schema_version") != "loopscope.model-tokenizer-revision-check.v2":
        raise AnalysisError("unsupported gate-e-probe revision report schema")
    if revision_report.get("manifest_sha256") != manifest_sha256(revision_report):
        raise AnalysisError("gate-e-probe revision report canonical hash mismatch")
    if revision_report.get("run_manifest_sha256") != run_manifest.get("manifest_sha256"):
        raise AnalysisError("gate-e-probe revision report belongs to another run")
    if revision_report.get("stage") != "gate-e-probe" or revision_report.get("match") is not True:
        raise AnalysisError("gate-e-probe revision report is not a matching stage proof")
    if revision_report.get("manifest_commit") != expected_revision or revision_report.get(
        "unique_revisions"
    ) != [expected_revision]:
        raise AnalysisError("gate-e-probe revision report differs from frozen manifest commit")
    observations = revision_report.get("observations")
    if not isinstance(observations, list) or not observations:
        raise AnalysisError("gate-e-probe revision report has no observations")
    for observation in observations:
        commits = {
            observation.get("model_commit"),
            observation.get("tokenizer_commit"),
            observation.get("manifest_commit"),
        }
        if commits != {expected_revision} or observation.get("match") is not True:
            raise AnalysisError("gate-e-probe observation fails three-party revision closure")
    if freeze.get("gate_e_probe_revision_report_sha256") != revision_report.get(
        "manifest_sha256"
    ):
        raise AnalysisError("score freeze gate-e-probe revision hash mismatch")
    full_probe_evidence = freeze.get("full_probe_evidence")
    if not isinstance(full_probe_evidence, Mapping):
        raise AnalysisError("Gate E score freeze is missing full_probe_evidence")
    return {
        "path": str(freeze_path),
        "manifest_sha256": freeze["manifest_sha256"],
        "file_sha256": _file_sha256(freeze_path),
        "score_job_id": freeze["score_job_id"],
        "full_probe_evidence": dict(full_probe_evidence),
        "gate_e_probe_revision_report": {
            "path": str(revision_path),
            "manifest_sha256": revision_report["manifest_sha256"],
            "file_sha256": _file_sha256(revision_path),
        },
    }


def _expected_full_windows(window_grid: Mapping[str, Any]) -> List[str]:
    values = [format_window(window) for window in candidate_windows(window_grid)]
    comparison = window_grid.get("comparison_windows", {})
    if isinstance(comparison, Mapping):
        values.extend(
            format_window(parse_window(str(window)))
            for window in comparison.get("random_in_band", [])
        )
    result = []
    seen = set()
    for window in values:
        if window not in seen:
            result.append(window)
            seen.add(window)
    return result


def _validate_manifest_eval_job(
    raw_job: Any, run_root: Path, recipe: Mapping[str, Any]
) -> Dict[str, Any]:
    if not isinstance(raw_job, Mapping):
        raise AnalysisError("Gate E jobs must be objects")
    job_id = str(raw_job.get("job_id", ""))
    if not job_id or raw_job.get("stage") != PHASE1_FULL_STAGE:
        raise AnalysisError("Gate E job has invalid id/stage")
    if raw_job.get("automatic_retry") is not False:
        raise AnalysisError("Gate E job must freeze automatic_retry=false")
    expected_revision_keys = {
        "model_commit": ["model_commit"],
        "tokenizer_commit": ["tokenizer_commit"],
        "manifest_commit": ["manifest_commit"],
    }
    if raw_job.get("revision_keys") != expected_revision_keys:
        raise AnalysisError("Gate E job must freeze model/tokenizer/manifest revision keys")
    output_dir = Path(str(raw_job.get("output_dir", ""))).resolve()
    expected_stage_root = (run_root / PHASE1_FULL_STAGE).resolve()
    if not _path_is_within(output_dir, expected_stage_root):
        raise AnalysisError("Gate E job output is outside the frozen stage root")
    revision_artifact = Path(str(raw_job.get("revision_artifact", ""))).resolve()
    if revision_artifact != output_dir / "model_revision.json":
        raise AnalysisError("Gate E job revision_artifact does not match output_dir")
    argv = raw_job.get("argv")
    options = _parse_eval_job_argv(argv)
    if raw_job.get("command") != shlex.join([str(value) for value in argv]):
        raise AnalysisError("Gate E job command disagrees with argv")
    _validate_eval_options(options, output_dir, recipe, context="run manifest job")
    window = options.get("window") if options.get("loop") else None
    return {
        "job_id": job_id,
        "output_dir": output_dir,
        "revision_artifact": revision_artifact,
        "window": window,
        "options": options,
    }


def _parse_eval_job_argv(argv: Any) -> Dict[str, Any]:
    if not isinstance(argv, list) or [str(value) for value in argv[:3]] != [
        "python",
        "-m",
        "tflt.eval_runner",
    ]:
        raise AnalysisError("Gate E argv must invoke python -m tflt.eval_runner")
    boolean_flags = {"--loop": "loop"}
    value_flags = {
        "--model": "model",
        "--revision": "revision",
        "--tasks": "tasks",
        "--output-dir": "output_dir",
        "--limit": "limit",
        "--num-fewshot": "num_fewshot",
        "--batch-size": "batch_size",
        "--dtype": "dtype",
        "--window": "window",
        "--k": "k",
        "--iteration-mode": "iteration_mode",
        "--strategy": "strategy",
        "--alpha": "alpha",
        "--beta": "beta",
        "--cache-strategy": "cache_strategy",
        "--decode-mode": "decode_mode",
        "--first-n": "first_n",
    }
    result: Dict[str, Any] = {"loop": False}
    index = 3
    while index < len(argv):
        flag = str(argv[index])
        if flag in boolean_flags:
            key = boolean_flags[flag]
            if result.get(key) is True:
                raise AnalysisError("duplicate Gate E argv flag: %s" % flag)
            result[key] = True
            index += 1
            continue
        if flag not in value_flags or index + 1 >= len(argv):
            raise AnalysisError("unknown or valueless Gate E argv flag: %s" % flag)
        key = value_flags[flag]
        if key in result:
            raise AnalysisError("duplicate Gate E argv flag: %s" % flag)
        result[key] = str(argv[index + 1])
        index += 2
    return result


def _validate_eval_options(
    options: Mapping[str, Any],
    output_dir: Path,
    recipe: Mapping[str, Any],
    context: str,
    allow_inactive_window_default: bool = False,
) -> None:
    expected_common = {
        "model": recipe["model"],
        "revision": recipe["revision"],
        "tasks": recipe["task"],
        "batch_size": "auto",
        "dtype": recipe["dtype"],
    }
    for key, expected in expected_common.items():
        if options.get(key) != expected:
            raise AnalysisError("%s recipe mismatch at %s" % (context, key))
    if Path(str(options.get("output_dir", ""))).resolve() != output_dir:
        raise AnalysisError("%s output_dir mismatch" % context)
    if _strict_int(options.get("num_fewshot"), "%s num_fewshot" % context) != 5:
        raise AnalysisError("%s requires MMLU 5-shot" % context)
    if options.get("limit") is not None:
        raise AnalysisError("%s is a limit artifact; full analysis rejects it" % context)
    if bool(options.get("loop")):
        window = format_window(parse_window(str(options.get("window", ""))))
        if window != options.get("window"):
            raise AnalysisError("%s window must use canonical start:end form" % context)
        if parse_window(window)[1] - parse_window(window)[0] + 1 != recipe["window_width"]:
            raise AnalysisError("%s window width differs from frozen recipe" % context)
        loop_expected = {
            "k": recipe["k"],
            "iteration_mode": recipe["iteration_mode"],
            "strategy": recipe["strategy"],
            "alpha": recipe["alpha"],
            "beta": recipe["beta"],
            "cache_strategy": recipe["cache_strategy"],
            "decode_mode": recipe["decode_mode"],
        }
        for key, expected in loop_expected.items():
            actual = options.get(key)
            if isinstance(expected, int):
                actual = _strict_int(actual, "%s %s" % (context, key))
            elif isinstance(expected, float):
                actual = _strict_float(actual, "%s %s" % (context, key))
            if actual != expected:
                raise AnalysisError("%s loop recipe mismatch at %s" % (context, key))
    elif options.get("window") is not None and not allow_inactive_window_default:
        raise AnalysisError("baseline Gate E argv must not freeze a window")


def _validate_eval_bundle(
    bundle: Mapping[str, Any], job: Mapping[str, Any], recipe: Mapping[str, Any]
) -> Dict[str, Any]:
    results_path = Path(str(bundle.get("results_path", ""))).resolve()
    output_dir = Path(str(job["output_dir"])).resolve()
    if results_path != output_dir / "results.json":
        raise AnalysisError("result artifact path does not match its Gate E job")
    if Path(str(bundle.get("command_args_path", ""))).resolve() != output_dir / "command_args.json":
        raise AnalysisError("command_args.json is not adjacent to its Gate E result")
    if Path(str(bundle.get("model_revision_path", ""))).resolve() != job[
        "revision_artifact"
    ]:
        raise AnalysisError("model_revision.json is not the frozen revision artifact")

    command_args = bundle.get("command_args")
    if not isinstance(command_args, Mapping):
        raise AnalysisError("command_args.json must contain an object")
    command_options = dict(command_args)
    command_options["output_dir"] = command_args.get("output_dir")
    _validate_eval_options(
        command_options,
        output_dir,
        recipe,
        context="command_args.json",
        allow_inactive_window_default=True,
    )
    expected_loop = job["window"] is not None
    if command_args.get("loop") is not expected_loop:
        raise AnalysisError("command_args.json loop flag disagrees with Gate E job")
    if expected_loop:
        command_window = format_window(parse_window(str(command_args.get("window", ""))))
        if command_window != job["window"]:
            raise AnalysisError("command_args.json window disagrees with Gate E job")

    revision = bundle.get("model_revision")
    if not isinstance(revision, Mapping):
        raise AnalysisError("model_revision.json must contain an object")
    if revision.get("repo_id") != recipe["repo_id"]:
        raise AnalysisError("model_revision.json repo_id differs from frozen recipe")
    closure = {
        key: revision.get(key)
        for key in ("model_commit", "tokenizer_commit", "manifest_commit")
    }
    if revision.get("match") is not True or len(set(closure.values())) != 1:
        raise AnalysisError("model_revision.json does not close model/tokenizer/manifest revisions")
    if next(iter(closure.values())) != recipe["revision"]:
        raise AnalysisError("model_revision.json differs from frozen manifest revision")
    if revision.get("commit_hash") != closure["model_commit"]:
        raise AnalysisError("legacy commit_hash disagrees with model_commit")

    results = bundle.get("results")
    result_config = results.get("config") if isinstance(results, Mapping) else None
    if isinstance(result_config, Mapping):
        if result_config.get("limit") is not None:
            raise AnalysisError("results.json config identifies a limit artifact")
        if result_config.get("num_fewshot") not in (None, 5):
            raise AnalysisError("results.json config is not MMLU 5-shot")

    return {
        "job_id": job["job_id"],
        "window": job["window"],
        "output_dir": str(output_dir),
        "model_revision": closure["model_commit"],
        "revision_closure": closure,
        "artifact_sha256": dict(bundle.get("artifact_sha256", {})),
    }


def _validate_score_probe_provenance(
    score_report: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    recipe: Mapping[str, Any],
    model_revision: str,
) -> None:
    provenance = score_report.get("probe_provenance")
    if not isinstance(provenance, Mapping):
        raise AnalysisError("window scores are missing probe provenance")
    model = provenance.get("model")
    git = provenance.get("git")
    probe_pool = provenance.get("probe_pool")
    if not isinstance(model, Mapping) or not isinstance(git, Mapping):
        raise AnalysisError("window-score model/git provenance is malformed")
    if not isinstance(probe_pool, Mapping):
        raise AnalysisError("window-score probe-pool provenance is malformed")
    expected_model = {
        "alias": recipe["model"],
        "repo_id": recipe["repo_id"],
        "revision": model_revision,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise AnalysisError("window-score probe model mismatch at %s" % key)
    if provenance.get("tokenizer_revision") != recipe["revision"]:
        raise AnalysisError("window-score probe tokenizer revision mismatch")
    closure = provenance.get("revision_closure")
    if not isinstance(closure, Mapping):
        raise AnalysisError("window-score probe revision closure is missing")
    expected_closure = {
        "manifest_commit": recipe["revision"],
        "model_commit": recipe["revision"],
        "tokenizer_commit": recipe["revision"],
        "match": True,
    }
    for key, expected in expected_closure.items():
        if closure.get(key) != expected:
            raise AnalysisError("window-score probe revision closure mismatch at %s" % key)
    if provenance.get("runtime_dtype") != recipe["dtype"]:
        raise AnalysisError("window-score probe dtype differs from frozen recipe")
    manifest_git = run_manifest.get("git")
    if isinstance(manifest_git, Mapping) and manifest_git.get("commit") != git.get("commit"):
        raise AnalysisError("window scores and Gate E use different code revisions")
    manifest_pool = run_manifest.get("inputs", {}).get("probe_pool")
    if isinstance(manifest_pool, Mapping):
        if manifest_pool.get("manifest_sha256") != probe_pool.get(
            "source_manifest_sha256"
        ):
            raise AnalysisError("window scores use a different probe-pool manifest")
        renderer = probe_pool.get("renderer")
        if not isinstance(renderer, Mapping):
            raise AnalysisError("window-score renderer provenance is malformed")
        expected_render = manifest_pool.get("render_contract_sha256")
        if expected_render is not None and expected_render != renderer.get(
            "render_contract_sha256"
        ):
            raise AnalysisError("window scores use a different render contract")
        expected_subset = manifest_pool.get("render_contract_subset_sha256")
        if expected_subset is not None and expected_subset != probe_pool.get(
            "source_render_contract_subset_sha256"
        ):
            raise AnalysisError("window scores use a different render-contract subset")


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise AnalysisError("%s must be an integer" % context)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("%s must be an integer" % context) from exc
    if str(result) != str(value):
        raise AnalysisError("%s must use an exact integer value" % context)
    return result


def _strict_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise AnalysisError("%s must be numeric" % context)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("%s must be numeric" % context) from exc
    if not math.isfinite(result):
        raise AnalysisError("%s must be finite" % context)
    return result


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_result_artifact(path: Path) -> Tuple[Path, Dict[str, Any]]:
    resolved = path / "results.json" if path.is_dir() else path
    payload = _load_json_object(resolved, "results artifact")
    if "samples" not in payload:
        task_names = []
        raw_results = payload.get("results")
        if isinstance(raw_results, Mapping):
            task_names = [str(name) for name in raw_results]
        sidecars = _load_sample_sidecars(resolved.parent, task_names)
        if sidecars:
            payload = dict(payload)
            payload["samples"] = sidecars
    return resolved, payload


def _load_sample_sidecars(
    directory: Path, task_names: Sequence[str]
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(directory.glob("samples_*.jsonl")):
        raw_namespace = path.stem[len("samples_") :]
        matching_names = [
            name
            for name in task_names
            if raw_namespace == name or raw_namespace.startswith(name + "_")
        ]
        namespace = max(matching_names, key=len) if matching_names else raw_namespace
        rows = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise AnalysisError(
                            "%s line %d must be a JSON object" % (path, line_number)
                        )
                    rows.append(item)
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisError("could not read sample sidecar %s" % path) from exc
        result[namespace] = rows
    return result


def _load_json_object(path: Path, context: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError("could not read %s from %s" % (context, path)) from exc
    if not isinstance(payload, dict):
        raise AnalysisError("%s must be a JSON object: %s" % (context, path))
    return payload


def _write_new_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError("refusing to overwrite existing artifact: %s" % path)
    path.write_text(text, encoding="utf-8")


def _deduplicate(values: Iterable[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
