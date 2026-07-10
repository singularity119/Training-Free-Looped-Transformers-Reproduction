"""Offline, label-free scoring for LoopScope candidate windows.

This module intentionally depends only on the Python standard library and the
pure LoopScope schema/grid helpers.  It never loads a model and never consumes
evaluation labels.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from tflt.loopscope.grid import format_window, parse_window
from tflt.loopscope.metrics import activity_contraction_score
from tflt.loopscope.schema import (
    SELECTION_SCHEMA_VERSION,
    ensure_new_directory,
    manifest_sha256,
    validate_probe_report,
    write_new_json,
)


PRIMARY_SIGNALS = (
    "entropy_drop",
    "entropy_flatness",
    "kl_to_final_drop",
    "activity_r",
    "negative_contraction_q",
    "r_times_one_minus_q",
)

DEFAULT_CRITERION: Dict[str, Any] = {
    "schema_version": "loopscope.criterion.v0",
    "primary_signal": "r_times_one_minus_q",
    "tie_policy": {
        "ranking_ties": "retain_all",
        "display_tiebreak": "frozen_candidate_order",
        "phase_gate_regret": "worst_case",
    },
    "signals": {
        "entropy_drop": {"higher_is_better": True},
        "entropy_flatness": {
            "higher_is_better": True,
            "transform": "negative_population_std",
        },
        "kl_to_final_drop": {"higher_is_better": True},
        "activity_r": {"higher_is_better": True},
        "negative_contraction_q": {
            "higher_is_better": True,
            "transform": "negative_q",
        },
        "r_times_one_minus_q": {
            "higher_is_better": True,
            "transform": "r * max(0, 1-q)",
        },
        "effective_rank_delta": {
            "higher_is_better": None,
            "auxiliary_only": True,
        },
    },
    "effective_rank_gate": {
        "enabled": False,
        "reason": (
            "Auxiliary diagnostic only in phase one; no threshold is tuned on MMLU test."
        ),
    },
    "bootstrap": {
        "replicates": 5,
        "seed": 20260710,
        "stable_if_same_or_adjacent_at_least": 4,
    },
}


class SelectionError(ValueError):
    """Raised when probe artifacts cannot support an honest window score."""


def add_score_windows_args(parser: Any) -> None:
    parser.add_argument("--layer-probe", required=True)
    parser.add_argument("--window-probe", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--criterion")
    parser.add_argument("--boundary-stat", choices=("mean", "median"), default="mean")
    parser.add_argument("--window-stat", choices=("mean", "median"), default="median")


def cmd_score_windows(args: Any) -> int:
    """CLI adapter that writes a new, immutable offline score artifact."""

    layer_path, layer_report = _load_probe_path(Path(args.layer_probe))
    window_paths_and_reports = [
        _load_probe_path(Path(path)) for path in list(args.window_probe)
    ]
    criterion = (
        _load_json_object(Path(args.criterion), "criterion")
        if args.criterion
        else copy.deepcopy(DEFAULT_CRITERION)
    )
    report = score_windows(
        layer_report=layer_report,
        window_reports=[item[1] for item in window_paths_and_reports],
        criterion=criterion,
        boundary_stat=args.boundary_stat,
        window_stat=args.window_stat,
        input_paths={
            "layer_probe": str(layer_path),
            "window_probes": [str(item[0]) for item in window_paths_and_reports],
            "criterion": str(Path(args.criterion)) if args.criterion else "built-in-v0",
        },
    )
    output_dir = ensure_new_directory(Path(args.output_dir))
    report_path = output_dir / "window_scores.json"
    summary_path = output_dir / "window_scores_summary.md"
    write_new_json(report_path, report)
    _write_new_text(summary_path, render_score_summary(report))
    print(str(report_path))
    print(str(summary_path))
    return 0


def score_windows(
    layer_report: Mapping[str, Any],
    window_reports: Sequence[Mapping[str, Any]],
    criterion: Optional[Mapping[str, Any]] = None,
    boundary_stat: str = "mean",
    window_stat: str = "median",
    input_paths: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Score every valid candidate without collapsing the signals into one rank.

    Boundary-curve scores use the requested aggregate from ``boundary_metrics``.
    Activity/contraction use the requested aggregate at the answer position.
    Per-example score components are retained so the later analysis command can
    perform a calibration-pool bootstrap without touching gold labels.
    """

    validate_probe_report(layer_report)
    if not window_reports:
        raise SelectionError("at least one window probe report is required")
    for report in window_reports:
        validate_probe_report(report)
    _validate_probe_alignment(layer_report, window_reports)
    if boundary_stat not in ("mean", "median"):
        raise SelectionError("boundary_stat must be mean or median")
    if window_stat not in ("mean", "median"):
        raise SelectionError("window_stat must be mean or median")

    frozen_criterion = copy.deepcopy(dict(criterion or DEFAULT_CRITERION))
    _validate_criterion(frozen_criterion)
    boundaries = _index_boundaries(layer_report.get("boundary_metrics", []))
    boundary_examples = _index_boundary_examples(layer_report.get("examples", []))
    windows = _index_window_metrics(window_reports)
    expected_windows = {
        format_window(parse_window(str(window)))
        for window in window_reports[0]["window_grid"]["candidate_windows"]
    }
    if set(windows) != expected_windows:
        missing = sorted(expected_windows.difference(windows), key=_window_sort_key)
        extra = sorted(set(windows).difference(expected_windows), key=_window_sort_key)
        raise SelectionError(
            "window probes must cover the frozen candidate grid exactly; missing=%s extra=%s"
            % (missing, extra)
        )
    warnings: List[str] = []
    candidates: List[Dict[str, Any]] = []

    for window_key in sorted(windows, key=_window_sort_key):
        metric = windows[window_key]
        if not metric.get("valid"):
            raise SelectionError(
                "window %s is invalid and cannot be silently scored: %r"
                % (window_key, metric.get("errors", []))
            )
        start, end = parse_window(window_key)
        entry_boundary = start
        exit_boundary = end + 1
        missing_boundaries = [
            index
            for index in range(entry_boundary, exit_boundary + 1)
            if index not in boundaries
        ]
        if missing_boundaries:
            raise SelectionError(
                "window %s is missing boundary metrics for %s"
                % (window_key, missing_boundaries)
            )

        entropy_curve = [
            _summary_number(
                boundaries[index].get("choice_entropy"),
                boundary_stat,
                "choice_entropy",
            )
            for index in range(entry_boundary, exit_boundary + 1)
        ]
        kl_curve = [
            _summary_number(
                boundaries[index].get("kl_to_final"), boundary_stat, "kl_to_final"
            )
            for index in range(entry_boundary, exit_boundary + 1)
        ]
        effective_rank_curve = [
            _finite_number(boundaries[index].get("effective_rank"), "effective_rank")
            for index in range(entry_boundary, exit_boundary + 1)
        ]
        activity = _window_summary_number(metric, "r", window_stat)
        contraction = _window_summary_number(metric, "q", window_stat)
        if activity < 0.0 or contraction < 0.0:
            raise SelectionError("window activity and contraction must be non-negative")

        effective_rank_delta = effective_rank_curve[-1] - effective_rank_curve[0]
        scores = {
            "entropy_drop": entropy_curve[0] - entropy_curve[-1],
            "entropy_flatness": -pstdev(entropy_curve),
            "kl_to_final_drop": kl_curve[0] - kl_curve[-1],
            "activity_r": activity,
            "negative_contraction_q": -contraction,
            "r_times_one_minus_q": activity_contraction_score(activity, contraction),
        }
        sample_scores = _candidate_sample_scores(
            start=start,
            end=end,
            boundary_examples=boundary_examples,
            window_metric=metric,
        )
        if not sample_scores:
            warnings.append(
                "Window %s has no joined per-example scores; calibration bootstrap is unavailable."
                % window_key
            )
        candidates.append(
            {
                "window": window_key,
                "start": start,
                "end": end,
                "scores": scores,
                "ranks": {},
                "auxiliary_effective_rank": {
                    "delta": effective_rank_delta,
                    "phase": _effective_rank_phase(effective_rank_delta),
                    "gate": _effective_rank_gate(frozen_criterion, effective_rank_delta),
                    "curve": effective_rank_curve,
                    "entry_boundary": entry_boundary,
                    "exit_boundary": exit_boundary,
                },
                "evidence": {
                    "boundary_stat": boundary_stat,
                    "window_stat": window_stat,
                    "entry_boundary": entry_boundary,
                    "exit_boundary": exit_boundary,
                    "entropy_boundary_curve": entropy_curve,
                    "kl_to_final_boundary_curve": kl_curve,
                    "activity_r": activity,
                    "contraction_q": contraction,
                },
                "sample_scores": sample_scores,
            }
        )

    rankings: Dict[str, List[Dict[str, Any]]] = {}
    selections: Dict[str, Dict[str, Any]] = {}
    for signal in PRIMARY_SIGNALS:
        ranking = rank_candidates(candidates, signal)
        rankings[signal] = ranking
        rank_by_window = {item["window"]: item["rank"] for item in ranking}
        for candidate in candidates:
            candidate["ranks"][signal] = rank_by_window[candidate["window"]]
        top_rank = min(item["rank"] for item in ranking)
        top = [item for item in ranking if item["rank"] == top_rank]
        selections[signal] = {
            "top_windows": [item["window"] for item in top],
            "top_score": top[0]["score"],
            "tie_count": len(top),
        }

    primary_signal = str(frozen_criterion["primary_signal"])
    grid = window_reports[0]["window_grid"]
    report = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "report_kind": "offline_window_scores",
        "label_free": True,
        "criterion": frozen_criterion,
        "criterion_sha256": manifest_sha256(frozen_criterion),
        "window_grid": {
            "manifest_sha256": grid["manifest_sha256"],
            "layer_count": grid["layer_count"],
            "candidate_windows": list(grid["candidate_windows"]),
        },
        "probe_provenance": {
            "boundary_contract": dict(layer_report["boundary_contract"]),
            "git": {
                key: layer_report["git"].get(key)
                for key in ("branch", "commit", "dirty")
            },
            "model": {
                key: layer_report["model"].get(key)
                for key in ("alias", "repo_id", "revision", "layer_count")
            },
            "tokenizer_revision": layer_report["tokenizer"].get("revision"),
            "runtime_dtype": layer_report["runtime"].get("dtype"),
            "probe_pool": {
                key: layer_report["probe_pool"].get(key)
                for key in (
                    "source",
                    "split",
                    "count",
                    "seed",
                    "source_manifest_count",
                    "sample_ids",
                    "records",
                    "rendering_records",
                    "task_group",
                    "num_fewshot",
                    "uses_target_gold_labels",
                    "fewshot_answers_present",
                    "source_manifest_sha256",
                    "selected_subset_sha256",
                    "source_render_contract_subset_sha256",
                    "selected_render_contract_subset_sha256",
                    "render_contract_sha256",
                    "renderer",
                )
            },
        },
        "aggregation": {"boundary_stat": boundary_stat, "window_stat": window_stat},
        "inputs": {
            "paths": dict(input_paths or {}),
            "layer_probe_manifest_sha256": _manifest_hash(layer_report),
            "window_probe_manifest_sha256": [
                _manifest_hash(report) for report in window_reports
            ],
        },
        "signal_definitions": {
            "entropy_drop": "entropy(B_a) - entropy(B_(b+1))",
            "entropy_flatness": (
                "negative population std of entropy on boundaries B_a..B_(b+1)"
            ),
            "kl_to_final_drop": "KL-to-final(B_a) - KL-to-final(B_(b+1))",
            "activity_r": "answer-position first-step relative activity r",
            "negative_contraction_q": "negative answer-position contraction q",
            "r_times_one_minus_q": "r * max(0, 1-q)",
            "effective_rank_delta": (
                "auxiliary effective-rank(B_(b+1)) - effective-rank(B_a)"
            ),
        },
        "primary_signal": primary_signal,
        "primary_top_windows": selections[primary_signal]["top_windows"],
        "candidates": candidates,
        "rankings": rankings,
        "selections_by_signal": selections,
        "warnings": _deduplicate(warnings),
    }
    report["manifest_sha256"] = manifest_sha256(report)
    return report


def rank_candidates(
    candidates: Sequence[Mapping[str, Any]], signal: str
) -> List[Dict[str, Any]]:
    """Return descending competition rankings with average ranks for ties."""

    if signal not in PRIMARY_SIGNALS:
        raise SelectionError("unsupported primary signal: %s" % signal)
    rows = []
    for candidate in candidates:
        window = str(candidate.get("window", ""))
        score = _finite_number(candidate.get("scores", {}).get(signal), signal)
        start, end = parse_window(window)
        rows.append({"window": window, "start": start, "end": end, "score": score})
    if not rows:
        raise SelectionError("cannot rank an empty candidate set")
    rows.sort(key=lambda item: (-item["score"], item["start"], item["end"]))
    index = 0
    while index < len(rows):
        stop = index + 1
        while stop < len(rows) and rows[stop]["score"] == rows[index]["score"]:
            stop += 1
        average_rank = (index + 1 + stop) / 2.0
        for offset in range(index, stop):
            rows[offset]["rank"] = average_rank
        index = stop
    return rows


def render_score_summary(report: Mapping[str, Any]) -> str:
    """Render a compact Chinese summary without making a test-set claim."""

    lines = [
        "# LoopScope 离线窗口打分摘要",
        "",
        "本报告只使用校准 probe 信号，不含正式 MMLU 测试标签或准确率。",
        "所有候选与所有并列排名均保留；小样本结果不得用于科学结论。",
        "",
        "## 各信号排名",
        "",
    ]
    for signal in PRIMARY_SIGNALS:
        rendered = ", ".join(
            "%s (rank %.1f, score %.6g)"
            % (item["window"], item["rank"], item["score"])
            for item in report["rankings"][signal]
        )
        lines.append("- `%s`: %s" % (signal, rendered))
    lines.extend(["", "## 有效秩辅助诊断", ""])
    for candidate in report["candidates"]:
        auxiliary = candidate["auxiliary_effective_rank"]
        lines.append(
            "- `%s`: delta=%.6g, phase=%s, gate=%s"
            % (
                candidate["window"],
                auxiliary["delta"],
                auxiliary["phase"],
                auxiliary["gate"]["passed"],
            )
        )
    if report.get("warnings"):
        lines.extend(["", "## 警告", ""])
        lines.extend("- %s" % warning for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _candidate_sample_scores(
    start: int,
    end: int,
    boundary_examples: Mapping[str, Mapping[int, Mapping[str, Any]]],
    window_metric: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    window_examples = _index_window_examples(window_metric.get("examples", []))
    sample_ids = sorted(set(boundary_examples).union(window_examples))
    rows = []
    for sample_id in sample_ids:
        scores: Dict[str, float] = {}
        boundary_by_index = boundary_examples.get(sample_id)
        boundary_range = range(start, end + 2)
        if boundary_by_index and all(index in boundary_by_index for index in boundary_range):
            entropy_curve = [
                _finite_number(
                    boundary_by_index[index].get("choice_entropy"), "choice_entropy"
                )
                for index in range(start, end + 2)
            ]
            kl_curve = [
                _finite_number(
                    boundary_by_index[index].get("kl_to_final"), "kl_to_final"
                )
                for index in range(start, end + 2)
            ]
            scores.update(
                {
                    "entropy_drop": entropy_curve[0] - entropy_curve[-1],
                    "entropy_flatness": -pstdev(entropy_curve),
                    "kl_to_final_drop": kl_curve[0] - kl_curve[-1],
                }
            )
        window_example = window_examples.get(sample_id)
        if window_example is not None:
            answer_metrics = window_example.get("answer_position_metrics")
            if isinstance(answer_metrics, Mapping):
                activity = _finite_number(answer_metrics.get("r"), "sample activity r")
                contraction = _finite_number(answer_metrics.get("q"), "sample contraction q")
                if activity < 0.0 or contraction < 0.0:
                    raise SelectionError("sample activity and contraction must be non-negative")
                scores.update(
                    {
                        "activity_r": activity,
                        "negative_contraction_q": -contraction,
                        "r_times_one_minus_q": activity_contraction_score(
                            activity, contraction
                        ),
                    }
                )
        if scores:
            rows.append({"id": sample_id, "scores": scores})
    return rows


def _index_boundaries(items: Any) -> Dict[int, Mapping[str, Any]]:
    if not isinstance(items, list) or not items:
        raise SelectionError("boundary probe contains no boundary_metrics")
    result: Dict[int, Mapping[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping) or "boundary_index" not in item:
            raise SelectionError("each boundary metric must contain boundary_index")
        index = int(item["boundary_index"])
        if index in result:
            raise SelectionError("duplicate boundary_index: %d" % index)
        result[index] = item
    return result


def _validate_probe_alignment(
    layer_report: Mapping[str, Any],
    window_reports: Sequence[Mapping[str, Any]],
) -> None:
    """Require every probe to describe the same code/model/pool/grid state."""

    reference_paths = (
        ("git", "branch"),
        ("git", "commit"),
        ("model", "alias"),
        ("model", "repo_id"),
        ("model", "revision"),
        ("model", "layer_count"),
        ("tokenizer", "revision"),
        ("runtime", "dtype"),
        ("probe_pool", "manifest_sha256"),
        ("probe_pool", "source_manifest_sha256"),
        ("probe_pool", "selected_subset_sha256"),
        ("probe_pool", "source_render_contract_subset_sha256"),
        ("probe_pool", "selected_render_contract_subset_sha256"),
        ("probe_pool", "render_contract_sha256"),
        ("probe_pool", "renderer"),
        ("probe_pool", "sample_ids"),
        (None, "position_rule"),
    )
    first_grid = window_reports[0].get("window_grid")
    if not isinstance(first_grid, Mapping):
        raise SelectionError("window probe is missing frozen grid provenance")
    renderer = layer_report["probe_pool"].get("renderer")
    if not isinstance(renderer, Mapping):
        raise SelectionError("probe_pool.renderer must be an object")
    render_contract_hash = str(renderer.get("render_contract_sha256", ""))
    if len(render_contract_hash) != 64 or any(
        char not in "0123456789abcdef" for char in render_contract_hash
    ):
        raise SelectionError(
            "probe_pool.renderer.render_contract_sha256 must be a lowercase SHA-256"
        )
    if layer_report["probe_pool"].get("render_contract_sha256") != render_contract_hash:
        raise SelectionError(
            "probe_pool.render_contract_sha256 disagrees with renderer provenance"
        )
    grid_identity = {
        "manifest_sha256": first_grid.get("manifest_sha256"),
        "layer_count": first_grid.get("layer_count"),
        "candidate_windows": list(first_grid.get("candidate_windows", [])),
    }
    for report_index, report in enumerate(window_reports):
        for section, key in reference_paths:
            expected = layer_report.get(key) if section is None else layer_report[section].get(key)
            actual = report.get(key) if section is None else report[section].get(key)
            if actual != expected:
                name = key if section is None else "%s.%s" % (section, key)
                raise SelectionError(
                    "window probe %d disagrees with layer probe at %s"
                    % (report_index, name)
                )
        grid = report.get("window_grid")
        if not isinstance(grid, Mapping):
            raise SelectionError("window probe %d is missing window_grid" % report_index)
        actual_grid = {
            "manifest_sha256": grid.get("manifest_sha256"),
            "layer_count": grid.get("layer_count"),
            "candidate_windows": list(grid.get("candidate_windows", [])),
        }
        if actual_grid != grid_identity:
            raise SelectionError("window probe reports disagree on frozen window grid")


def _index_boundary_examples(items: Any) -> Dict[str, Dict[int, Mapping[str, Any]]]:
    result: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    if items is None:
        return result
    if not isinstance(items, list):
        raise SelectionError("layer probe examples must be a list")
    for item in items:
        if not isinstance(item, Mapping) or "id" not in item:
            raise SelectionError("layer probe example must contain id")
        sample_id = str(item["id"])
        if sample_id in result:
            raise SelectionError("duplicate layer probe example id: %s" % sample_id)
        by_index: Dict[int, Mapping[str, Any]] = {}
        for metric in item.get("boundary_metrics", []):
            if not isinstance(metric, Mapping) or "boundary_index" not in metric:
                raise SelectionError(
                    "example boundary metric must contain boundary_index"
                )
            index = int(metric["boundary_index"])
            if index in by_index:
                raise SelectionError(
                    "duplicate boundary %d in example %s" % (index, sample_id)
                )
            by_index[index] = metric
        result[sample_id] = by_index
    return result


def _index_window_metrics(
    reports: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for report in reports:
        metrics = report.get("window_metrics")
        if not isinstance(metrics, list):
            raise SelectionError("window probe window_metrics must be a list")
        for metric in metrics:
            if not isinstance(metric, Mapping) or "window" not in metric:
                raise SelectionError("window metric must contain window")
            window = format_window(parse_window(str(metric["window"])))
            if window in result:
                raise SelectionError("duplicate window metric: %s" % window)
            result[window] = metric
    if not result:
        raise SelectionError("window probes contain no candidate metrics")
    return result


def _index_window_examples(items: Any) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    if items is None:
        return result
    if not isinstance(items, list):
        raise SelectionError("window examples must be a list")
    for item in items:
        if not isinstance(item, Mapping):
            raise SelectionError("window example must be an object")
        if not item.get("valid", False):
            continue
        raw_id = item.get("sample_id", item.get("id"))
        if raw_id is None:
            raise SelectionError("valid window example must contain sample_id")
        sample_id = str(raw_id)
        if sample_id in result:
            raise SelectionError("duplicate window example id: %s" % sample_id)
        result[sample_id] = item
    return result


def _window_summary_number(metric: Mapping[str, Any], name: str, stat: str) -> float:
    answer_position = metric.get("answer_position")
    if not isinstance(answer_position, Mapping):
        raise SelectionError("window metric is missing answer_position")
    return _summary_number(answer_position.get(name), stat, "answer_position.%s" % name)


def _summary_number(value: Any, stat: str, context: str) -> float:
    if not isinstance(value, Mapping):
        raise SelectionError("%s must be a summary object" % context)
    if stat not in value:
        raise SelectionError("%s summary is missing %s" % (context, stat))
    return _finite_number(value[stat], "%s.%s" % (context, stat))


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise SelectionError("%s must be numeric, not boolean" % context)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectionError("%s must be numeric" % context) from exc
    if not math.isfinite(number):
        raise SelectionError("%s must be finite" % context)
    return number


def _effective_rank_phase(delta: float, tolerance: float = 1e-9) -> str:
    if delta > tolerance:
        return "expanding"
    if delta < -tolerance:
        return "contracting"
    return "flat"


def _effective_rank_gate(
    criterion: Mapping[str, Any], delta: float
) -> Dict[str, Any]:
    config = criterion.get("effective_rank_gate", {})
    if not isinstance(config, Mapping):
        raise SelectionError("effective_rank_gate must be an object")
    enabled = bool(config.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "passed": None,
            "reason": str(config.get("reason", "auxiliary only")),
        }
    minimum = config.get("minimum_delta")
    maximum = config.get("maximum_delta")
    if minimum is None and maximum is None:
        raise SelectionError("enabled effective-rank gate needs minimum_delta or maximum_delta")
    passed = True
    if minimum is not None:
        passed = passed and delta >= _finite_number(minimum, "minimum_delta")
    if maximum is not None:
        passed = passed and delta <= _finite_number(maximum, "maximum_delta")
    return {"enabled": True, "passed": passed, "reason": str(config.get("reason", ""))}


def _validate_criterion(criterion: MutableMapping[str, Any]) -> None:
    signals = criterion.get("signals")
    if not isinstance(signals, Mapping):
        raise SelectionError("criterion signals must be an object")
    for signal in PRIMARY_SIGNALS:
        config = signals.get(signal)
        if not isinstance(config, Mapping):
            raise SelectionError("criterion is missing signal %s" % signal)
        if config.get("higher_is_better") is not True:
            raise SelectionError("phase-one score %s must use higher-is-better direction" % signal)
    primary = str(criterion.get("primary_signal", ""))
    if primary not in PRIMARY_SIGNALS:
        raise SelectionError("criterion primary_signal is unsupported: %s" % primary)
    expected_tie_policy = {
        "ranking_ties": "retain_all",
        "display_tiebreak": "frozen_candidate_order",
        "phase_gate_regret": "worst_case",
    }
    if criterion.get("tie_policy") != expected_tie_policy:
        raise SelectionError(
            "criterion tie_policy must retain all ties, use frozen candidate order only "
            "for display, and gate on worst-case regret"
        )


def _manifest_hash(report: Mapping[str, Any]) -> Optional[str]:
    probe_pool = report.get("probe_pool")
    if isinstance(probe_pool, Mapping):
        value = probe_pool.get("manifest_sha256")
        return str(value) if value is not None else None
    return None


def _window_sort_key(window: str) -> Tuple[int, int]:
    return parse_window(window)


def _load_probe_path(path: Path) -> Tuple[Path, Dict[str, Any]]:
    resolved = path / "probe_report.json" if path.is_dir() else path
    return resolved, _load_json_object(resolved, "probe report")


def _load_json_object(path: Path, context: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError("could not read %s from %s" % (context, path)) from exc
    if not isinstance(payload, dict):
        raise SelectionError("%s must be a JSON object: %s" % (context, path))
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
