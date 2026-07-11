#!/usr/bin/env python3
"""Run frozen Gate E development selection and one-shot holdout analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tflt.looppilot.analysis import mcnemar_exact_p
from tflt.looppilot.full import GATE_E_CANDIDATE_FIELDS, GATE_E_COVERAGES, GATE_E_DIRECTIONS, select_candidate, terminal_recommendation, validate_gate_e_config
from tflt.looppilot.schema import write_json_once, write_jsonl_once


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/looppilot/gate_e_full.json")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = _json(Path(args.config))
    validate_gate_e_config(config)
    root = Path(args.run_root).resolve()
    if not (root / "receipts/shards_verified.json").is_file():
        raise RuntimeError("Gate E shards must be verified before analysis")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    splits = {row["doc_key"]: row["split"] for row in _jsonl(root / "preflight/split_manifest.jsonl")}
    arms = {name: {} for name in ("baseline_a", "baseline_b", "always_loop")}
    decisions: Dict[str, Mapping[str, Any]] = {}
    diagnostics: Dict[str, Mapping[str, Any]] = {}
    for shard_id in range(8):
        shard = root / ("shards/shard-%d" % shard_id)
        for name in arms:
            for row in _jsonl(shard / (name + "_samples.jsonl")):
                if row["doc_key"] in arms[name]:
                    raise RuntimeError("duplicate aggregate arm key")
                arms[name][row["doc_key"]] = row
        for row in _jsonl(shard / "decisions.jsonl"):
            if row["doc_key"] in decisions:
                raise RuntimeError("duplicate aggregate decision key")
            decisions[row["doc_key"]] = row
        signal_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in _jsonl(shard / "signal_records.jsonl"):
            signal_groups[row["doc_key"]].append(row)
        for key, group in signal_groups.items():
            values = sorted(group, key=lambda row: row["choice_index"])
            if [row["choice_index"] for row in values] != [0, 1, 2, 3] or any(values[0][field] != row[field] for row in values[1:] for field in ("q1", "residual_cosine")):
                raise RuntimeError("Gate E paid diagnostics do not agree across choices")
            if key in diagnostics:
                raise RuntimeError("duplicate aggregate diagnostic key")
            diagnostics[key] = {"q1": float(values[0]["q1"]), "residual_cosine": float(values[0]["residual_cosine"])}
    keys = set(arms["baseline_a"])
    if not keys or any(set(rows) != keys for rows in arms.values()) or set(decisions) != keys or set(diagnostics) != keys or set(splits) != keys:
        raise RuntimeError("Gate E aggregate join is not closed")

    # Development selection accesses only development correctness. Holdout correctness is
    # deliberately not dereferenced until selected_policy.json exists exclusively.
    development = [_analysis_row(key, arms, decisions, diagnostics, splits, include_labels=True) for key in sorted(keys) if splits[key] == "development"]
    selected, candidate_grid = select_candidate(development)
    dev_best_fixed = max(_accuracy(development, "baseline_correct"), _accuracy(development, "loop_correct"))
    selected["development_gain_over_best_fixed"] = selected["development_accuracy"] - dev_best_fixed
    selected["holdout_labels_read_before_freeze"] = False
    selected["selection_contract_sha256"] = _sha(Path(args.config))
    write_json_once(output / "selected_policy.json", selected)
    selected_hash = _sha(output / "selected_policy.json")
    write_json_once(output / "pre_holdout_freeze_proof.json", {"selected_policy_sha256": selected_hash, "exclusive_create": True, "holdout_labels_read": False})
    write_json_once(output / "candidate_grid.json", {"schema_version": 1, "development": candidate_grid})

    holdout = [_analysis_row(key, arms, decisions, diagnostics, splits, include_labels=True) for key in sorted(keys) if splits[key] == "holdout"]
    paired = development + holdout
    write_jsonl_once(output / "paired_samples.jsonl", paired)
    policy = [_policy_baseline(row, selected) for row in holdout]
    metrics = _metrics(paired, holdout, policy, config, selected)
    comparators = _comparators(development, holdout, config)
    random_null = _random_null(holdout, policy, int(config["random_null_replicates"]), int(config["random_null_seed"]))
    curves = _coverage_curves(candidate_grid, holdout)
    strata = _strata(holdout, policy)
    oracle = _oracle(holdout, policy, comparators["subject_only"]["actions"])

    assessable = metrics["changed_docs"] >= 200 and metrics["loop_flip_rate"] >= max(0.005, 3 * metrics["self_flip_rate"]) and oracle["oracle_headroom_over_best_fixed"] >= 0.002
    candidate_loop_ci = metrics["candidate_vs_loop_bootstrap_ci"]
    candidate_base_ci = metrics["candidate_vs_baseline_bootstrap_ci"]
    max_share = strata["max_absolute_subject_gain_share"]
    nonnegative_fraction = strata["changed_subject_nonnegative_gain_fraction"]
    pass_flags = {
        "development_gain_positive": selected["development_gain_over_best_fixed"] > 0,
        "holdout_above_both_fixed": metrics["candidate_accuracy"] > metrics["holdout_baseline_accuracy"] and metrics["candidate_accuracy"] > metrics["holdout_loop_accuracy"],
        "candidate_vs_loop_ci_lower_positive": candidate_loop_ci[0] > 0,
        "candidate_vs_baseline_point_positive_ci_tolerated": metrics["candidate_vs_baseline_delta"] > 0 and candidate_base_ci[0] >= -0.0005,
        "harm_block_above_null_97_5": metrics["harm_block_rate"] > random_null["harm_block_rate_p97_5"],
        "help_retention_ge_80pct": metrics["help_retention"] >= 0.80,
        "oracle_capture_ge_10pct": oracle["candidate_capture"] >= 0.10,
        "not_single_subject_dominated": max_share <= 0.25 and nonnegative_fraction >= 0.60,
    }
    pass_to_online = assessable and all(pass_flags.values())
    subject_capture = oracle["subject_capture"]
    decision_flags = {
        "technical_closed": True, "assessable": assessable, "pass_to_online": pass_to_online,
        "subject_capture_ge_80pct": subject_capture >= 0.80,
        "hidden_capture_le_subject": oracle["candidate_capture"] <= subject_capture,
    }
    recommendation = terminal_recommendation(decision_flags)
    decision_trace = {"schema_version": 1, **decision_flags, "pass_conditions": pass_flags, "terminal_recommendation": recommendation}
    summary = {
        "schema_version": 1, "gate": "E", "document_count": len(paired), "development_count": len(development), "holdout_count": len(holdout),
        "selected_policy_sha256": selected_hash, "selected_policy": selected, "metrics": metrics,
        "random_null": random_null, "comparators": {name: {k: v for k, v in value.items() if k != "actions"} for name, value in comparators.items()},
        "oracle": oracle, "strata": strata, "decision_trace": decision_trace,
    }
    write_json_once(output / "coverage_curves.json", curves)
    write_json_once(output / "bootstrap.json", {"candidate_vs_loop_ci": candidate_loop_ci, "candidate_vs_baseline_ci": candidate_base_ci, "replicates": config["bootstrap_replicates"], "seed": config["bootstrap_seed"]})
    write_json_once(output / "random_null.json", random_null)
    write_json_once(output / "comparators.json", summary["comparators"])
    write_json_once(output / "oracle.json", oracle)
    write_json_once(output / "strata.json", strata)
    write_json_once(output / "decision_trace.json", decision_trace)
    write_json_once(output / "gate_e_summary.json", summary)
    _write_text_once(output / "gate_e_report.md", _report(summary))
    print(json.dumps({"recommendation": recommendation, "documents": len(paired)}, sort_keys=True))
    return 0


def _analysis_row(key: str, arms: Mapping[str, Mapping[str, Mapping[str, Any]]], decisions: Mapping[str, Mapping[str, Any]], diagnostics: Mapping[str, Mapping[str, Any]], splits: Mapping[str, str], include_labels: bool) -> Dict[str, Any]:
    a, b, loop = arms["baseline_a"][key], arms["baseline_b"][key], arms["always_loop"][key]
    scores = sorted((float(value) for value in a["choice_loglikelihoods"]), reverse=True)
    row = {
        "doc_key": key, "task": a["task"], "subject": a["subject"], "split": splits[key],
        "prompt_length": int(a["prompt_length"]), "baseline_margin": scores[0] - scores[1],
        **{field: float(decisions[key][field]) for field in GATE_E_CANDIDATE_FIELDS},
        "q1": float(diagnostics[key]["q1"]), "residual_cosine": float(diagnostics[key]["residual_cosine"]),
    }
    if include_labels:
        gold = int(a["gold"])
        row.update(
            baseline_prediction=int(a["prediction"]), baseline_b_prediction=int(b["prediction"]), loop_prediction=int(loop["prediction"]), gold=gold,
            baseline_correct=int(a["prediction"]) == gold, baseline_b_correct=int(b["prediction"]) == gold, loop_correct=int(loop["prediction"]) == gold,
        )
        row["outcome"] = "both_correct" if row["baseline_correct"] and row["loop_correct"] else "both_wrong" if not row["baseline_correct"] and not row["loop_correct"] else "loop_helps" if row["loop_correct"] else "loop_hurts"
    return row


def _policy_baseline(row: Mapping[str, Any], selected: Mapping[str, Any]) -> bool:
    value, threshold = float(row[selected["signal"]]), float(selected["threshold"])
    return value >= threshold if selected["direction"] == "BASELINE_IF_HIGH" else value <= threshold


def _metrics(all_rows: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], policy: Sequence[bool], config: Mapping[str, Any], selected: Mapping[str, Any]) -> Dict[str, Any]:
    outcomes = Counter(row["outcome"] for row in all_rows)
    holdout_outcomes = Counter(row["outcome"] for row in holdout)
    candidate_correct = [bool(row["baseline_correct"] if baseline else row["loop_correct"]) for row, baseline in zip(holdout, policy)]
    base = [bool(row["baseline_correct"]) for row in holdout]
    loop = [bool(row["loop_correct"]) for row in holdout]
    harm = [i for i, row in enumerate(holdout) if row["outcome"] == "loop_hurts"]
    helps = [i for i, row in enumerate(holdout) if row["outcome"] == "loop_helps"]
    multiplier = 1.0 if selected["direction"] == "BASELINE_IF_HIGH" else -1.0
    score = [multiplier * float(row[selected["signal"]]) for row in holdout]
    signal_diagnostics = {}
    for field in GATE_E_CANDIDATE_FIELDS:
        values = [float(row[field]) for row in holdout]
        signal_diagnostics[field] = {
            "loop_hurts_pr_auc_high": _average_precision([row["outcome"] == "loop_hurts" for row in holdout], values),
            "loop_hurts_pr_auc_low": _average_precision([row["outcome"] == "loop_hurts" for row in holdout], [-value for value in values]),
        }
    paid_diagnostics = {
        field: {outcome: _mean([float(row[field]) for row in holdout if row["outcome"] == outcome]) for outcome in ("both_correct", "both_wrong", "loop_helps", "loop_hurts")}
        for field in ("q1", "residual_cosine")
    }
    return {
        "baseline_a_accuracy": _accuracy(all_rows, "baseline_correct"), "baseline_b_accuracy": _accuracy(all_rows, "baseline_b_correct"), "always_loop_accuracy": _accuracy(all_rows, "loop_correct"),
        "self_flip_rate": sum(row["baseline_prediction"] != row["baseline_b_prediction"] for row in all_rows) / len(all_rows),
        "loop_flip_rate": sum(row["baseline_prediction"] != row["loop_prediction"] for row in all_rows) / len(all_rows),
        "changed_docs": outcomes["loop_helps"] + outcomes["loop_hurts"], "n01": outcomes["loop_helps"], "n10": outcomes["loop_hurts"],
        "outcomes": dict(outcomes), "paired_delta": (_accuracy(all_rows, "loop_correct") - _accuracy(all_rows, "baseline_correct")), "mcnemar_exact_p": mcnemar_exact_p(outcomes["loop_helps"], outcomes["loop_hurts"]),
        "holdout_outcomes": dict(holdout_outcomes), "holdout_baseline_accuracy": sum(base) / len(base), "holdout_loop_accuracy": sum(loop) / len(loop), "candidate_accuracy": sum(candidate_correct) / len(candidate_correct),
        "candidate_vs_baseline_delta": (sum(candidate_correct) - sum(base)) / len(base), "candidate_vs_loop_delta": (sum(candidate_correct) - sum(loop)) / len(loop),
        "candidate_vs_baseline_bootstrap_ci": _cluster_ci(holdout, candidate_correct, base, int(config["bootstrap_replicates"]), int(config["bootstrap_seed"])),
        "candidate_vs_loop_bootstrap_ci": _cluster_ci(holdout, candidate_correct, loop, int(config["bootstrap_replicates"]), int(config["bootstrap_seed"])),
        "candidate_vs_baseline_mcnemar_p": _paired_mcnemar(candidate_correct, base), "candidate_vs_loop_mcnemar_p": _paired_mcnemar(candidate_correct, loop),
        "harm_block_rate": sum(policy[i] for i in harm) / len(harm) if harm else 0.0,
        "help_retention": sum(not policy[i] for i in helps) / len(helps) if helps else 0.0,
        "loop_hurts_pr_auc": _average_precision([row["outcome"] == "loop_hurts" for row in holdout], score),
        "helps_vs_hurts_auroc": _auroc([row["outcome"] == "loop_hurts" for row in holdout if row["outcome"] in ("loop_helps", "loop_hurts")], [score[i] for i, row in enumerate(holdout) if row["outcome"] in ("loop_helps", "loop_hurts")]),
        "candidate_signal_diagnostics": signal_diagnostics, "paid_q1_cos_diagnostics": paid_diagnostics,
    }


def _comparators(dev: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, Any]:
    subject_choice = {}
    for subject in sorted({row["subject"] for row in dev}):
        rows = [row for row in dev if row["subject"] == subject]
        subject_choice[subject] = sum(row["baseline_correct"] for row in rows) >= sum(row["loop_correct"] for row in rows)
    subject_actions = [subject_choice[row["subject"]] for row in holdout]
    result = {"subject_only": {"development_rule": subject_choice, "actions": subject_actions, "holdout_accuracy": _policy_accuracy(holdout, subject_actions)}}
    for name, field in (("prompt_length_only", "prompt_length"), ("baseline_margin_only", "baseline_margin")):
        transformed = [dict(row, **{"r0_median": float(row[field]), "c0": float(row[field]), "n0_median": float(row[field])}) for row in dev]
        selected, _ = select_candidate(transformed)
        selected["signal"] = field
        actions = [_policy_baseline(row, selected) for row in holdout]
        result[name] = {"selected": selected, "actions": actions, "holdout_accuracy": _policy_accuracy(holdout, actions)}
    return result


def _random_null(rows: Sequence[Mapping[str, Any]], policy: Sequence[bool], replicates: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["subject"]].append(index)
    baseline_counts = {subject: sum(policy[i] for i in indices) for subject, indices in groups.items()}
    accuracies, harm_blocks = [], []
    harm = [i for i, row in enumerate(rows) if row["outcome"] == "loop_hurts"]
    for _ in range(replicates):
        action = [False] * len(rows)
        for subject, indices in groups.items():
            for index in rng.sample(indices, baseline_counts[subject]):
                action[index] = True
        accuracies.append(_policy_accuracy(rows, action))
        harm_blocks.append(sum(action[i] for i in harm) / len(harm) if harm else 0.0)
    return {"replicates": replicates, "seed": seed, "match": "per-subject-baseline-count", "accuracy_mean": sum(accuracies) / len(accuracies), "accuracy_p97_5": _quantile(sorted(accuracies), 0.975), "harm_block_rate_mean": sum(harm_blocks) / len(harm_blocks), "harm_block_rate_p97_5": _quantile(sorted(harm_blocks), 0.975)}


def _coverage_curves(grid: Sequence[Mapping[str, Any]], holdout: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for candidate in grid:
        actions = [_policy_baseline(row, candidate) for row in holdout]
        rows.append({**candidate, "holdout_accuracy": _policy_accuracy(holdout, actions), "holdout_baseline_coverage": sum(actions) / len(actions)})
    return {"schema_version": 1, "rows": rows}


def _oracle(rows: Sequence[Mapping[str, Any]], candidate: Sequence[bool], subject: Sequence[bool]) -> Dict[str, Any]:
    baseline = _accuracy(rows, "baseline_correct")
    loop = _accuracy(rows, "loop_correct")
    best = max(baseline, loop)
    oracle = sum(row["baseline_correct"] or row["loop_correct"] for row in rows) / len(rows)
    headroom = oracle - best
    candidate_acc, subject_acc = _policy_accuracy(rows, candidate), _policy_accuracy(rows, subject)
    capture = lambda value: (value - best) / headroom if headroom > 0 else 0.0
    return {"baseline_accuracy": baseline, "loop_accuracy": loop, "best_fixed_accuracy": best, "oracle_accuracy": oracle, "oracle_headroom_over_best_fixed": headroom, "candidate_accuracy": candidate_acc, "candidate_capture": capture(candidate_acc), "subject_accuracy": subject_acc, "subject_capture": capture(subject_acc)}


def _strata(rows: Sequence[Mapping[str, Any]], policy: Sequence[bool]) -> Dict[str, Any]:
    by_subject = []
    contributions = []
    changed_nonnegative = []
    for subject in sorted({row["subject"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["subject"] == subject]
        gain = sum((row["baseline_correct"] if policy[i] else row["loop_correct"]) - row["loop_correct"] for i, row in enumerate(rows) if i in indices)
        changed = sum(rows[i]["outcome"] in ("loop_helps", "loop_hurts") for i in indices)
        by_subject.append({"subject": subject, "count": len(indices), "changed_docs": changed, "gain_count_vs_loop": gain})
        contributions.append(abs(gain))
        if changed:
            changed_nonnegative.append(gain >= 0)
    total_abs = sum(contributions)
    return {"by_subject": by_subject, "max_absolute_subject_gain_share": max(contributions, default=0) / total_abs if total_abs else 0.0, "changed_subject_nonnegative_gain_fraction": sum(changed_nonnegative) / len(changed_nonnegative) if changed_nonnegative else 0.0, "prompt_length_quartiles": _binned(rows, policy, "prompt_length"), "baseline_margin_quartiles": _binned(rows, policy, "baseline_margin")}


def _binned(rows: Sequence[Mapping[str, Any]], policy: Sequence[bool], field: str) -> List[Dict[str, Any]]:
    values = sorted(float(row[field]) for row in rows)
    cuts = [_quantile(values, q) for q in (0.25, 0.5, 0.75)]
    groups = [[] for _ in range(4)]
    for i, row in enumerate(rows):
        value = float(row[field])
        bucket = sum(value > cut for cut in cuts)
        groups[bucket].append(i)
    return [{"bin": i, "count": len(indices), "candidate_accuracy": _policy_accuracy([rows[j] for j in indices], [policy[j] for j in indices]) if indices else None} for i, indices in enumerate(groups)]


def _cluster_ci(rows: Sequence[Mapping[str, Any]], left: Sequence[bool], right: Sequence[bool], replicates: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    subjects = sorted({row["subject"] for row in rows})
    indices = {subject: [i for i, row in enumerate(rows) if row["subject"] == subject] for subject in subjects}
    values = []
    for _ in range(replicates):
        sampled = [rng.choice(subjects) for _ in subjects]
        chosen = [i for subject in sampled for i in indices[subject]]
        values.append(sum(int(left[i]) - int(right[i]) for i in chosen) / len(chosen))
    values.sort()
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def _paired_mcnemar(left: Sequence[bool], right: Sequence[bool]) -> float:
    n01 = sum((not r) and l for l, r in zip(left, right))
    n10 = sum(r and (not l) for l, r in zip(left, right))
    return mcnemar_exact_p(n01, n10)


def _average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if not positives:
        return 0.0
    ranked = sorted(zip(scores, labels), reverse=True)
    found = 0
    total = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            found += 1
            total += found / rank
    return total / positives


def _auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    pos = [score for score, label in zip(scores, labels) if label]
    neg = [score for score, label in zip(scores, labels) if not label]
    if not pos or not neg:
        return 0.5
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg) / (len(pos) * len(neg))


def _policy_accuracy(rows: Sequence[Mapping[str, Any]], actions: Sequence[bool]) -> float:
    return sum(bool(row["baseline_correct"] if baseline else row["loop_correct"]) for row, baseline in zip(rows, actions)) / len(rows)


def _accuracy(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _quantile(values: Sequence[float], q: float) -> float:
    position = (len(values) - 1) * q
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (position - lo)


def _report(summary: Mapping[str, Any]) -> str:
    metrics, decision = summary["metrics"], summary["decision_trace"]
    return "\n".join([
        "# LoopPilot 第一阶段 Gate E 报告", "", "本报告是有标签开发、冻结后一次性 holdout 验证的信号可行性分析，不是最终无标签或 training-free controller。", "",
        "- 总样本：%d；development：%d；holdout：%d" % (summary["document_count"], summary["development_count"], summary["holdout_count"]),
        "- baseline A / baseline B / AlwaysLoop 准确率：%.6f / %.6f / %.6f" % (metrics["baseline_a_accuracy"], metrics["baseline_b_accuracy"], metrics["always_loop_accuracy"]),
        "- self-flip / loop-flip：%.6f / %.6f" % (metrics["self_flip_rate"], metrics["loop_flip_rate"]),
        "- loop helps / hurts：%d / %d" % (metrics["n01"], metrics["n10"]),
        "- holdout candidate 准确率：%.6f" % metrics["candidate_accuracy"],
        "- harm-block / help-retention：%.6f / %.6f" % (metrics["harm_block_rate"], metrics["help_retention"]), "",
        "## 冻结终态建议", "", "`%s`" % decision["terminal_recommendation"], "",
        "该建议严格按 BLOCK → assessable → PASS_TO_ONLINE → PIVOT_TO_COARSE → PIVOT_TO_PHASE3 的预冻结顺序产生。", "",
    ])


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> List[Any]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_text_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
