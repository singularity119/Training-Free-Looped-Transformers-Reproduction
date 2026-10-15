#!/usr/bin/env python3
"""Freeze all 5376 D6 S2 predictions without gold, then perform one full analysis."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path

from tflt.loopscope.phase11_analysis import gold_indices
from tflt.loopscope.phase11_d6_analysis import CELLS, analyze
from tflt.loopscope.phase11_s2 import extract_s2

COUNT = 448
BINDING = ("cell_key", "cell_id", "seed", "source_commit", "prompt_mode", "damping_scope")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def validate_cells(value, *, closure=False):
    require(value.get("schema") == ("loopscope.phase11.d6.raw_prediction_closure.v1" if closure else
                                   "loopscope.phase11.d6.plan.v1"), "D6 schema differs")
    require(value.get("record_count") == 5376 and value.get("target_gold_loaded") is False,
            "all 5376 raw records/predictions must close before gold")
    if closure:
        require(value.get("status") == "GOLD_UNREAD" and value.get("prediction_closed") is True and
                value.get("cell_count") == 12, "D6 prediction closure is incomplete")
    cells = value["cells"]
    require(len(cells) == 12 and {c["cell_key"] for c in cells} == set(CELLS), "D6 membership differs from 12 cells")
    for cell in cells:
        cid = cell["cell_key"]
        require(cell.get("population_count") == COUNT and cell.get("seed") == 20261015 and
                cell.get("synthetic") is False and cell.get("engineering") is False,
                "D6 formal cell count/seed/scope differs")
        expected_scope = "prefill_only" if cid.endswith("prefill") else "full" if cid.endswith("full") else "none"
        require(cell.get("prompt_mode") == cid.split("_")[0] and cell.get("damping_scope") == expected_scope and
                isinstance(cell.get("source_commit"), str) and bool(cell["source_commit"]),
                "D6 prompt/scope/source binding differs")
        require(not closure or cell.get("record_count") == COUNT, "D6 closed cell is incomplete")


def canonical_pool(path):
    pool = read(path)
    rows = pool["rows"]
    identities = [row["identity"] for row in rows]
    require(len(rows) == COUNT and len(set(identities)) == COUNT, "D6 canonical identity population differs")
    require(all(row["choice_labels"] == list("ABCD") and len(row["choices"]) == 4 for row in rows),
            "D6 canonical choice labels differ")
    return rows, identities


def align(rows, identities):
    indexed = {}
    indices = {identity: index for index, identity in enumerate(identities)}
    for row in rows:
        identity = row["identity"]
        require(identity in indices and identity not in indexed and
                type(row.get("canonical_index")) is int and row["canonical_index"] == indices[identity],
                "D6 identity/canonical index missing, unknown or duplicated")
        indexed[identity] = row
    require(set(indexed) == set(identities), "D6 cell is incomplete")
    return [indexed[identity] for identity in identities]


def load_raw_cells(cells):
    """Read inputs/raw only. No gold path is consulted here."""
    data, canonical, checked_pools, common_rows = {}, None, {}, None
    for cell in cells:
        path = cell["pool"]
        if path not in checked_pools:
            checked_pools[path] = canonical_pool(path)
        pool_rows, identities = checked_pools[path]
        if canonical is None:
            canonical, common_rows = identities, pool_rows
        else:
            require(identities == canonical, "D6 prompt pools differ in canonical identities/order")
            for a, b in zip(common_rows, pool_rows):
                require(all(a[field] == b[field] for field in ("choices", "choice_labels", "subject")),
                        "D6 prompt pools differ in original choices/subject")
        raw = align([row for root in cell["raw_roots"] for row in lines(root["records_path"])], identities)
        parsed, predictions = [], []
        for row in raw:
            require(all(row.get("d6_binding", {}).get(field) == cell[field] for field in BINDING),
                    "D6 raw runtime/config binding differs")
            require(row.get("candidate_labels") == list("ABCD") and
                    isinstance(row.get("generated_token_ids"), list) and
                    type(row.get("generated_token_count")) is int and
                    0 < row["generated_token_count"] <= 8192 and
                    len(row["generated_token_ids"]) == row["generated_token_count"] and
                    row.get("end_reason") in ("eos", "length_limit"), "D6 raw token/termination fields differ")
            result = extract_s2(row["generated_text"])
            parsed.append(result)
            predictions.append(result["prediction_index"])
        data[cell["cell_key"]] = {"raw": raw, "predictions": predictions, "auxiliary": parsed}
    require(len(data) == 12 and sum(len(c["raw"]) for c in data.values()) == 5376,
            "all 5376 raw records must close before gold")
    return data, common_rows, canonical


def close(plan):
    validate_cells(plan)
    data, _, identities = load_raw_cells(plan["cells"])
    predictions = []
    for cell in plan["cells"]:
        content = data[cell["cell_key"]]
        for index, (identity, row, parsed) in enumerate(zip(identities, content["raw"], content["auxiliary"])):
            predictions.append({"cell_key": cell["cell_key"], "identity": identity, "canonical_index": index,
                "seed": cell["seed"], **parsed, "generated_token_count": row["generated_token_count"],
                "end_reason": row["end_reason"]})
    closure = {"schema": "loopscope.phase11.d6.raw_prediction_closure.v1", "status": "GOLD_UNREAD",
        "target_gold_loaded": False, "prediction_closed": True, "record_count": 5376, "cell_count": 12,
        "cells": [{**cell, "record_count": COUNT} for cell in plan["cells"]],
        "prediction_rules": "unchanged phase11_s2.extract_s2; consistent repeat merge, no semantic rescue"}
    return closure, predictions


def load_before_gold(closure):
    validate_cells(closure, closure=True)
    data, pool_rows, identities = load_raw_cells(closure["cells"])
    saved = lines(closure["predictions_path"])
    require(len(saved) == 5376, "D6 closed prediction population differs")
    by_cell = {}
    for row in saved:
        by_cell.setdefault(row["cell_key"], []).append(row)
    require(set(by_cell) == set(CELLS), "D6 closed prediction cell membership differs")
    for cid, value in data.items():
        frozen = align(by_cell[cid], identities)
        for p, raw, parsed in zip(frozen, value["raw"], value["auxiliary"]):
            require(p["seed"] == 20261015 and all(p.get(k) == v for k, v in parsed.items()) and
                    p["generated_token_count"] == raw["generated_token_count"] and p["end_reason"] == raw["end_reason"],
                    "D6 frozen S2 prediction/length/termination differs from raw")
    return data, pool_rows, identities


def load_gold(path, format, pool_rows, identities):
    require(format in ("p11_jsonl", "p11_json"), "unknown D6 gold format")
    rows = lines(path) if format == "p11_jsonl" else read(path)
    gold = gold_indices(rows, identities,
        {identity: row["choice_labels"] for identity, row in zip(identities, pool_rows)},
        {identity: row["subject"] for identity, row in zip(identities, pool_rows)})
    require(len(gold) == COUNT and all(type(g) is int and 0 <= g < 4 for g in gold), "D6 gold population differs")
    return gold


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def write_report(path, result):
    text = ["# Gate D-6：GPQA 提示与衰减阶段对照", "",
        "同一448题、12个fresh cells，实际运行seed20261015，输入排列seed20261008。配置来自同题集事后选择，本次为探索性机制检验，不能视为新题集确认。S2沿用既定整行提取；冲突与无匹配均计错。", "",
        "| 配置 | 正确数 | Accuracy | ΔNative pp | Δ同KLoop pp | token均值/中位/p90 | EOS/上限 | S2成功/无匹配/冲突 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["cells"]:
        loop = "—" if row["delta_loop_pp"] is None else f"{row['delta_loop_pp']:+.4f}"
        text.append(f"| {row['cell']} | {row['correct']}/{row['n']} | {row['accuracy_percent']:.4f}% | {row['delta_native_pp']:+.4f} | {loop} | {row['generated_tokens_mean']:.2f}/{row['generated_tokens_median']:.2f}/{row['generated_tokens_p90']:.2f} | {row['eos_count']}/{row['length_limit_count']} | {row['S2_success_count']}/{row['S2_no_match_count']}/{row['S2_conflict_count']} |")
    text += ["", "主family共4项：两个提示交互与两个CoT full−prefill-only，共同Holm校正；交互为100000次双侧paired sign-flip，seed20261008。其余主比较为exact双侧McNemar。所有CI为10000次同题配对bootstrap名义95%CI，seed20261008。", "",
        "| 主比较 | Δpp [名义95%CI] | 原始p | Holm p（4项） |", "|---|---:|---:|---:|"]
    for row in result["primary_contrasts"]:
        ci = row["bootstrap"]
        text.append(f"| {row['contrast']} | {row['delta_pp']:+.4f} [{ci['ci_low_pp']:+.4f},{ci['ci_high_pp']:+.4f}] | {row['p_value']:.6g} | {row['holm_adjusted_p']:.6g} |")
    text += ["", "交互定义为(CoT方法full−CoT同KLoop)−(direct方法full−direct同KLoop)。Sign-flip依赖题级符号可交换性，属于探索性假设；不能以一侧显著、另一侧不显著代替交互检验。", "",
        "| 辅助比较 | Δpp [名义95%CI] | Holm p（10项） |", "|---|---:|---:|"]
    for row in result["auxiliary_contrasts"]:
        ci = row["bootstrap"]
        text.append(f"| {row['treatment']} − {row['reference']} | {row['delta_pp']:+.4f} [{ci['ci_low_pp']:+.4f},{ci['ci_high_pp']:+.4f}] | {row['holm_adjusted_p']:.6g} |")
    text += ["", "| 方法相对同KLoop | 全448题平均token差 [名义95%CI] |", "|---|---:|"]
    for row in result["length_contrasts"]:
        ci = row["bootstrap"]
        text.append(f"| {row['treatment']} − {row['reference']} | {row['mean_delta_tokens']:+.2f} [{ci['ci_low_tokens']:+.2f},{ci['ci_high_tokens']:+.2f}] |")
    text += ["", "Prefill-only仍在decode执行同K普通Loop与cachefirst，只关闭decode residual damping；prefill可影响首个输出token logits。直接作答只改变指令，不能证明关闭内部推理；未遵循指令的输出照计。EOS-only长度见cells.csv，仅为条件性描述；不按格式成功子集或EOS子集替换448题主分母。普通Loop−Native和其他跨prompt变化仅描述，不另挑显著检验。", "",
        "原题、逐题输出和gold保留HPC，仅交付聚合统计。负结果亦完整保留，不据收益正负裁定工程Gate。"]
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write("\n".join(text) + "\n")


def run(args):
    if args.close:
        require(args.plan is not None and args.closure is None and args.gold is None,
                "gold-free closure requires --plan only; no --gold")
        closure, predictions = close(read(args.plan))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        path = args.output_dir / "predictions.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        closure.update(source_plan=str(args.plan), predictions_path=str(path))
        write(args.output_dir / "closure.json", closure)
        print(json.dumps({"status": "GOLD_UNREAD", "record_count": 5376, "cell_count": 12,
                          "closure": str(args.output_dir / "closure.json")}))
        return 0
    require(args.closure is not None and args.gold is not None and args.plan is None,
            "analysis requires --closure and --gold")
    closure = read(args.closure)
    # Every raw record and saved prediction is verified before any target gold read.
    data, pool_rows, identities = load_before_gold(closure)
    require(Path(args.gold).is_file(), "D6 gold is absent; no gold has been read")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir / "analysis_spec.json", {"schema": "loopscope.phase11.d6.analysis_spec.v1",
        "source_closure": str(args.closure), "gold_read": False, "record_count": 5376, "cell_count": 12,
        "primary_family_size": 4, "auxiliary_family_size": 10, "bootstrap_replicates": 10000,
        "sign_flip_draws": 100000, "statistical_seed": 20261008,
        "primary": ["prompt_interaction_k2", "prompt_interaction_k3", "full_minus_prefill_k2", "full_minus_prefill_k3"]})
    gold = load_gold(args.gold, args.gold_format, pool_rows, identities)
    write(args.output_dir / "gold_read_receipt.json", {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_closure": str(args.closure), "gold_path": str(args.gold), "n": len(gold),
        "prior_record_count": 5376, "prior_prediction_closed": True})
    result = analyze(data, gold)
    result.update(source_closure=str(args.closure), record_count=5376, cell_count=12, gold_read=True)
    write(args.output_dir / "analysis.json", result)
    for name in ("cells", "primary_contrasts", "auxiliary_contrasts", "length_contrasts", "cross_prompt_descriptive"):
        write_csv(args.output_dir / (name + ".csv"), result[name])
    write_report(args.output_dir / "CoT与decode衰减分析报告.md", result)
    print(json.dumps({"status": result["status"], "record_count": 5376, "cell_count": 12,
                      "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close", action="store_true")
    for name in ("plan", "closure", "gold"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--gold-format", choices=("p11_jsonl", "p11_json"), default="p11_jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
