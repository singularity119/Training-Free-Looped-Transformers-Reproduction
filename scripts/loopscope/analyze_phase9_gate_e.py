#!/usr/bin/env python3
"""One-shot, post-outcome exploratory analysis for the Phase 9 Gate E panel."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from tflt.loopscope.phase8_accuracy_stats import (
    BOOTSTRAP_REPLICATES,
    cell_accuracy,
    exact_mcnemar_p,
    holm_adjust,
    paired_contrast,
    score_predictions,
)
from tflt.loopscope.phase9_accuracy import EXPECTED_TEST_COUNT, load_config as load_phase9_config, logical_panel
from tflt.loopscope.phase9_gate_e_accuracy import load_config as load_gate_e_config, new_cells

from analyze_phase9 import load_frozen_gold, load_raw_cells
from verify_phase9_history import verify as verify_history
from verify_phase9_scores import verify as verify_phase9_new
from verify_phase9_gate_e_scores import verify as verify_gate_e


BOOTSTRAP_SEED = 20260923
PRIMARY_FAMILY = "GATE_E_LAG1_PRIMARY_12"


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_gate_e_cells(closure: Mapping[str, Any], pool_rows: Sequence[Mapping[str, Any]], cells: Sequence[Mapping[str, Any]]):
    canonical_ids = [row["identity"] for row in pool_rows]
    aligned = []
    for cell in cells:
        scores_by_id: Dict[str, list[float]] = {}
        roots = [Path(root) for root in closure["cell_roots"]
                 if read_json(Path(root) / "command_args.json")["cell"]["cell_id"] == cell["cell_id"]]
        require(bool(roots), "Gate E closure has no roots for %s" % cell["cell_id"])
        for root in roots:
            with (root / "scores.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    identity = row["identity"]
                    require(identity not in scores_by_id, "duplicate Gate E raw score identity")
                    scores = row["scores"]
                    require(isinstance(scores, list) and len(scores) == 4 and
                            all(type(value) in (int, float) and math.isfinite(value) for value in scores),
                            "Gate E raw scores must be four finite values")
                    scores_by_id[identity] = [float(value) for value in scores]
        require(set(scores_by_id) == set(canonical_ids), "Gate E raw scores do not cover canonical identities")
        aligned.append(dict(cell, scores=[scores_by_id[identity] for identity in canonical_ids]))
    return aligned


def build_analysis(cells: Sequence[Mapping[str, Any]], gold: Sequence[int], subjects: Sequence[str]) -> Dict[str, Any]:
    accuracy = []
    correctness: Dict[str, list[int]] = {}
    by_key: Dict[tuple[Any, Any, Any, Any], str] = {}
    by_id: Dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        key = (cell["model"], tuple(cell["window"]) if cell["window"] else None,
               cell["k"], cell["arm"])
        require(key not in by_key, "duplicate original-plus-Gate-E configuration")
        by_key[key] = cell["cell_id"]
        by_id[cell["cell_id"]] = cell
        stats = cell_accuracy(cell["scores"], gold, subjects)
        accuracy.append({
            "cell_id": cell["cell_id"], "model": cell["model"], "window": cell["window"],
            "k": cell["k"], "arm": cell["arm"],
            "direction_policy": cell.get("direction_policy"),
            "intervention_mode": cell.get("intervention_mode"),
            "source": cell.get("source"), **stats,
        })
        correctness[cell["cell_id"]] = [
            int(prediction == answer) for prediction, answer in zip(score_predictions(cell["scores"]), gold)
        ]

    def id_for(model: str, window: Sequence[int] | None, k: int | None, arm: str) -> str:
        key = (model, tuple(window) if window is not None else None, k, arm)
        require(key in by_key, "missing frozen Gate E analysis arm: %r" % (key,))
        return by_key[key]

    contrasts = []

    def add(reference_id: str, treatment_id: str, family: str, comparison: str) -> None:
        reference = correctness[reference_id]
        treatment = correctness[treatment_id]
        cell = by_id[treatment_id]
        value = paired_contrast(
            reference, treatment, subjects,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED,
        )
        contrasts.append({
            "reference": reference_id,
            "treatment": treatment_id,
            "family": family,
            "comparison": comparison,
            "model": cell["model"],
            "window": cell["window"],
            "k": cell["k"],
            **value,
            "holm_adjusted_p": None,
            "holm_reject_alpha_0_05": None,
        })

    e_cells = [cell for cell in cells if cell.get("source") == "phase9_gate_e"]
    require(len(e_cells) == 12, "Gate E analysis requires twelve newly scored cells")
    require(all(cell.get("direction_policy") == "lag1" for cell in e_cells),
            "Gate E formal analysis may include only direction_policy=lag1 cells")
    require(all(cell.get("intervention_mode") ==
                ("spectral" if cell["arm"] == "Lag1-Online" else "matched_norm")
                for cell in e_cells),
            "Gate E formal analysis mode labels differ from the frozen arm names")
    for cell in e_cells:
        if cell["arm"] != "Lag1-Online":
            continue
        model, window, k = cell["model"], cell["window"], cell["k"]
        lag1_online = cell["cell_id"]
        old_online = id_for(model, window, k, "Online-t0")
        lag1_matched = id_for(model, window, k, "Lag1-Matched-norm")
        add(old_online, lag1_online, PRIMARY_FAMILY, "Lag1-Online minus Online-t0")
        add(lag1_matched, lag1_online, PRIMARY_FAMILY, "Lag1-Online minus Lag1-Matched-norm")
        add(id_for(model, window, k, "Loop"), lag1_online,
            "GATE_E_DESCRIPTIVE", "Lag1-Online minus Loop")
        add(id_for(model, window, k, "Matched-norm"), lag1_online,
            "GATE_E_DESCRIPTIVE", "Lag1-Online minus original Matched-norm")
        add(id_for(model, None, None, "Native"), lag1_online,
            "GATE_E_DESCRIPTIVE", "Lag1-Online minus Native")

    primary = [row for row in contrasts if row["family"] == PRIMARY_FAMILY]
    require(len(primary) == 12, "Gate E primary family must contain exactly twelve comparisons")
    adjusted = holm_adjust([row["mcnemar_exact_p"] for row in primary])
    for row, p_value in zip(primary, adjusted):
        row["holm_adjusted_p"] = p_value
        row["holm_reject_alpha_0_05"] = p_value <= 0.05
        row["holm_family_size"] = 12
    support = {}
    for cell in e_cells:
        if cell["arm"] != "Lag1-Online":
            continue
        selected = [row for row in primary if row["treatment"] == cell["cell_id"]]
        require(len(selected) == 2, "each Lag1-Online cell requires its two primary comparisons")
        support[cell["cell_id"]] = {
            "both_primary_holm_reject": all(row["holm_reject_alpha_0_05"] for row in selected),
            "comparisons": [row["comparison"] for row in selected],
        }
    return {
        "schema": "loopscope.phase9.gate_e.analysis.v2",
        "status": "ANALYZED_POST_OUTCOME_EXPLORATORY",
        "target_gold_loaded": True,
        "panel_cell_count": len(accuracy),
        "new_gate_e_cell_count": len(e_cells),
        "gate_e_configuration": {
            "formal_direction_policy": "lag1",
            "supported_direction_policies": ["fixed_t0", "lag1"],
            "supported_intervention_modes": ["spectral", "matched_norm"],
            "fixed_t0_full_scores": "reused from verified original Phase 9 cells; not rerun",
        },
        "cells": accuracy,
        "contrasts": contrasts,
        "primary_family": {
            "name": PRIMARY_FAMILY,
            "size": len(primary),
            "alpha": 0.05,
            "method": "exact two-sided McNemar with one Holm adjustment over twelve predeclared contrasts",
            "support_by_lag1_online_cell": support,
        },
        "bootstrap": {
            "method": "fixed_subject_stratified_paired_bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ci_level": 0.95,
            "ci_kind": "nominal_percentile_linear",
        },
        "interpretation": {
            "domain": "post-outcome exploratory comparison on previously evaluated MMLU test identities",
            "positive_result": "not new held-out confirmation",
            "cell_support": "both Lag1-Online contrasts must pass the same twelve-item Holm family",
            "trajectory_scope": "K>2 arms use their own diverging residual trajectories",
            "descriptive_only": "Lag1-Online versus Loop, original Matched-norm, and Native",
        },
    }


def independent_check(result: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], gold: Sequence[int], subjects: Sequence[str]) -> Dict[str, Any]:
    by_cell = {cell["cell_id"]: cell for cell in cells}
    reported = {cell["cell_id"]: cell for cell in result["cells"]}
    correctness = {}
    for cell in cells:
        stats = cell_accuracy(cell["scores"], gold, subjects)
        saved = reported[cell["cell_id"]]
        require(stats["correct"] == saved["correct"] and stats["n"] == saved["n"],
                "independent Gate E cell accuracy check failed")
        require(stats["micro_accuracy"] == saved["micro_accuracy"],
                "independent Gate E micro accuracy check failed")
        correctness[cell["cell_id"]] = [
            int(prediction == label) for prediction, label in zip(score_predictions(cell["scores"]), gold)
        ]
    primary = [row for row in result["contrasts"] if row["family"] == PRIMARY_FAMILY]
    fresh_p = []
    for row in primary:
        reference = correctness[row["reference"]]
        treatment = correctness[row["treatment"]]
        gains = sum(int(t and not r) for r, t in zip(reference, treatment))
        losses = sum(int(r and not t) for r, t in zip(reference, treatment))
        require((gains, losses) == (row["wrong_to_right"], row["right_to_wrong"]),
                "independent Gate E discordant-pair check failed")
        p_value = exact_mcnemar_p(gains, losses)
        require(math.isclose(p_value, row["mcnemar_exact_p"], rel_tol=1e-12, abs_tol=1e-14),
                "independent Gate E McNemar check failed")
        fresh_p.append(p_value)
    fresh_holm = holm_adjust(fresh_p)
    for row, p_value in zip(primary, fresh_holm):
        require(math.isclose(p_value, row["holm_adjusted_p"], rel_tol=1e-12, abs_tol=1e-14),
                "independent Gate E Holm check failed")
    return {
        "status": "VERIFIED",
        "cell_count": len(cells),
        "primary_contrast_count": len(primary),
        "n_per_cell": len(gold),
        "checks": ["raw-score accuracy", "paired discordant counts", "exact McNemar", "Holm adjustment"],
    }


def write_tables(output_dir: Path, result: Mapping[str, Any]) -> None:
    with (output_dir / "cells.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = ["cell_id", "model", "window", "k", "arm", "direction_policy",
                  "intervention_mode", "source", "correct", "n",
                  "micro_accuracy", "subject_macro_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["cells"])
    for file_name, family in (("primary_contrasts.csv", PRIMARY_FAMILY),
                              ("descriptive_contrasts.csv", "GATE_E_DESCRIPTIVE")):
        with (output_dir / file_name).open("x", newline="", encoding="utf-8") as handle:
            fields = ["reference", "treatment", "comparison", "model", "window", "k", "n",
                      "reference_correct", "treatment_correct", "wrong_to_right", "right_to_wrong",
                      "delta_pp", "subject_macro_delta_pp", "ci_low_pp", "ci_high_pp",
                      "mcnemar_exact_p", "holm_adjusted_p", "holm_reject_alpha_0_05"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({**row, **row["bootstrap"]}
                             for row in result["contrasts"] if row["family"] == family)


def write_report(path: Path, result: Mapping[str, Any], resource_summary: Mapping[str, Any] | None) -> None:
    primary = [row for row in result["contrasts"] if row["family"] == PRIMARY_FAMILY]
    lines = [
        "# LoopScope 第九阶段 Gate E：逐轮方向更新探索性比较",
        "",
        "本报告记录 Phase 9 主线 A–D 结项后追加的单独 Gate E。新增比较使用原 MMLU test identities；该测试集已有历史结果，因此所有准确率和检验均为事后探索性结果，不构成新的 held-out 确认。原 47-cell 主面板与阶段报告保持原结论。",
        "",
        "## 冻结设计",
        "",
        "- 新增 12 cells：4B `13:16`/`15:18` 与 1.7B `12:15`，各自 K=3/4 × Lag1-Online/Lag1-Matched-norm。每 cell 14,042 题、57 subjects；新增 168,504 份题目分数。结合已核验原 47 cells，展示面板为 59 个逻辑 cells、828,478 条 cell-sample 记录，其中只有 12 cells 是本 Gate 新采集。",
        "- 实现使用同一 Gate E runtime/scoring adapter，显式组合 `direction_policy=fixed_t0|lag1` 与 `intervention_mode=spectral|matched_norm`；K2 debug 对应原 Phase 9 两臂作分数与答案残差等价检查。正式只新采集 `lag1` 的 12 cells；fixed_t0 的全量正式原始分数复用原 47-cell 面板，不重跑。",
        "- 每题在 `t=0` 不干预并从全体有效 prompt-token 原生残差拟合 D0；`t=1` 使用 D0，当前原生残差再拟合 D1；`t=2` 使用 D1 并为 K4 拟合 D2；`t=3`（K4）使用 D2。最终轮不拟合不会再使用的方向。方向为 FP32、不中心化、不逐行归一化的精确 reduced SVD；只处理答案前位置，`lambda=0.5`、`alpha=1`、步长 `1/K`。",
        "- Lag1-Matched-norm 按各自轨迹中的上一轮方向计算定向阻尼范数，再均匀缩放当前答案位置残差到该范数。K>2 时两新臂的轨迹会分化，Matched-norm 是自身规则的逐状态等范数对照。",
        "",
        "## 十二项预设比较",
        "",
        "主 family 对每个 Lag1-Online cell 同时比较原 Online-t0 与 Lag1-Matched-norm，共 12 项；报告 exact 双侧 McNemar 和一个 Holm family（α=0.05）。只有同 cell 的两项都通过该 family，才称该 cell 有支持逐轮方向更新额外价值的探索性证据。区间为按 subject 分层 paired bootstrap 10,000 次、seed `20260923` 的 nominal 95% CI；CI 不作多重校正。Lag1-Online 对 Loop、原 Matched-norm 和 Native 仅描述。",
        "",
        "| 模型 | 窗口 | K | 比较 | Δ 准确率 (pp) | 95% CI (pp) | exact p | Holm p |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| {model} | {window} | {k} | {comparison} | {delta:+.4f} | [{low:+.4f}, {high:+.4f}] | {p:.6g} | {holm:.6g} |".format(
                model=row["model"].split("/")[-1], window="–".join(map(str, row["window"])),
                k=row["k"], comparison=row["comparison"], delta=row["delta_pp"],
                low=row["bootstrap"]["ci_low_pp"], high=row["bootstrap"]["ci_high_pp"],
                p=row["mcnemar_exact_p"], holm=row["holm_adjusted_p"],
            )
        )
    lines.extend(["", "## 逐 cell 判定", "", "| 模型 | 窗口 | K | 两项 Holm 均通过 |", "|---|---:|---:|---|"])
    support = result["primary_family"]["support_by_lag1_online_cell"]
    for cell_id, supported in support.items():
        cell = next(row for row in result["cells"] if row["cell_id"] == cell_id)
        lines.append("| {} | {} | {} | {} |".format(
            cell["model"].split("/")[-1], "–".join(map(str, cell["window"])), cell["k"],
            "是" if supported["both_primary_holm_reject"] else "否",
        ))
    lines.extend(["", "## 资源和工程结果", ""])
    if resource_summary:
        lines.extend([
            "- 累计 GPU 时间：{:.3f} 小时（包含 Gate E debug、canary、正式运行和记录在账的失败尝试）。".format(float(resource_summary.get("cumulative_gpu_hours", 0.0))),
            "- 峰值并发 GPU：{}；OOM：{}。".format(resource_summary.get("max_concurrent_gpus", "未提供"), resource_summary.get("oom_count", "未提供")),
            "- 实际 A800 packing/canary 说明：{}".format(resource_summary.get("packing_decision", "未提供")),
            "- 新评分 SVD 累计 {:.3f} 小时；最大 GPU reserved 显存 {:.3f} GiB。".format(
                float(resource_summary.get("svd_hours", 0.0)),
                float(resource_summary.get("max_peak_reserved_gib", 0.0))),
            "- 各 score shard 的峰值显存与吞吐见 `resource_summary.json` 及原始 summary。",
        ])
    else:
        lines.append("资源账本未提供；提交最终报告前须补入已核实的 GPU 小时、并发、OOM 与 packing 结果。")
    lines.extend([
        "",
        "## 限制与结果文件",
        "",
        "结果是已看过 test 后的追加探索。点估计、区间和显著性不可称为新样本泛化或确认性证据；不显著不代表等价。完整 59-cell 准确率表见 `cells.csv`，十二项预设比较见 `primary_contrasts.csv`，描述性对照见 `descriptive_contrasts.csv`。",
        "",
        "分析完成时间：{}。".format(result.get("completed_at", "")),
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    old_closure = read_json(args.phase9_new_closure)
    old_history = read_json(args.phase9_history_verification)
    gate_e_closure = read_json(args.gate_e_closure)
    require(old_closure.get("status") == "FULL_NEW_SCORE_PANEL_CLOSED" and old_closure.get("target_gold_loaded") is False,
            "original Phase 9 new score closure is not closed and gold-free")
    require(old_history.get("status") == "HISTORICAL_29_CELLS_VERIFIED" and old_history.get("target_gold_loaded") is False,
            "original Phase 9 historical score verification is not closed and gold-free")
    require(gate_e_closure.get("status") == "FULL_GATE_E_SCORE_PANEL_CLOSED" and gate_e_closure.get("target_gold_loaded") is False,
            "Gate E score closure is not closed and gold-free")

    config = load_phase9_config()
    gate_config = load_gate_e_config(args.gate_e_config)
    original_panel = logical_panel(config)
    gate_cells = new_cells(gate_config)
    pool = read_json(args.pool)
    from tflt.loopscope.phase8_test_pool import validate_test_pool
    validate_test_pool(pool)
    pool_rows = pool["rows"]
    require(len(pool_rows) == EXPECTED_TEST_COUNT and len({row["subject"] for row in pool_rows}) == 57,
            "Gate E requires the canonical 14042/57 gold-free pool")

    fresh_old_new = verify_phase9_new(
        [Path(root) for root in old_closure["cell_roots"]],
        Path(old_closure["manifest"]), args.pool, old_closure["scope"],
        "combined", args.original_canary_indices, full_panel=True,
    )
    fresh_history = verify_history(args.original_history_map, args.pool)
    fresh_gate_e = verify_gate_e(
        [Path(root) for root in gate_e_closure["cell_roots"]],
        Path(gate_e_closure["manifest"]), args.pool, gate_e_closure["scope"],
        "combined", args.gate_e_canary_indices, full_panel=True,
    )
    require(fresh_old_new["status"] == "FULL_NEW_SCORE_PANEL_CLOSED" and
            fresh_history["status"] == "HISTORICAL_29_CELLS_VERIFIED" and
            fresh_gate_e["status"] == "FULL_GATE_E_SCORE_PANEL_CLOSED",
            "fresh pre-outcome raw-score verification did not close")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_once(args.output_dir / "pre_outcome_verification.json", {
        "schema": "loopscope.phase9.gate_e.pre_outcome_verification.v2",
        "target_gold_loaded": False,
        "verified_at": now(),
        "original_new_scores": fresh_old_new,
        "original_historical_scores": fresh_history,
        "gate_e_scores": fresh_gate_e,
    })

    original_cells = load_raw_cells(old_closure, old_history, pool_rows, original_panel)
    gate_e_cells = load_gate_e_cells(gate_e_closure, pool_rows, gate_cells)
    cells = original_cells + gate_e_cells
    subjects = [row["subject"] for row in pool_rows]
    gold_started = now()
    gold, gold_sources = load_frozen_gold(pool_rows, args.cache_dir)
    result = build_analysis(cells, gold, subjects)
    result.update({
        "completed_at": now(),
        "gold_load_started": gold_started,
        "dataset": {"repo": config["dataset_repo"], "revision": config["dataset_revision"], "split": "test"},
        "original_phase9_new_closure": str(args.phase9_new_closure),
        "original_phase9_history_verification": str(args.phase9_history_verification),
        "gate_e_closure": str(args.gate_e_closure),
        "pool": str(args.pool),
        "gold_source_evidence": gold_sources,
    })
    result["fresh_count_verification"] = independent_check(result, cells, gold, subjects)
    write_json_once(args.output_dir / "analysis.json", result)
    write_tables(args.output_dir, result)
    if args.report_path:
        resource_summary = read_json(args.resource_summary) if args.resource_summary else None
        write_report(args.report_path, result, resource_summary)
    print(json.dumps({
        "status": result["status"], "panel_cells": result["panel_cell_count"],
        "new_gate_e_cells": result["new_gate_e_cell_count"],
        "primary_contrasts": result["primary_family"]["size"],
        "n_per_cell": EXPECTED_TEST_COUNT,
        "target_gold_loaded": result["target_gold_loaded"],
        "output_dir": str(args.output_dir),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase9-new-closure", type=Path, required=True)
    parser.add_argument("--phase9-history-verification", type=Path, required=True)
    parser.add_argument("--original-history-map", type=Path, required=True)
    parser.add_argument("--original-canary-indices", type=Path, required=True)
    parser.add_argument("--gate-e-closure", type=Path, required=True)
    parser.add_argument("--gate-e-canary-indices", type=Path, required=True)
    parser.add_argument("--gate-e-config", type=Path)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--resource-summary", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
