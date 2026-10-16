#!/usr/bin/env python3
"""Close D7 raw/S2 predictions before gold; analyze the complete five-cell panel."""
import argparse
import csv
import datetime
import json
from pathlib import Path

from tflt.loopscope.phase11_analysis import gold_indices
from tflt.loopscope.phase11_d7_analysis import CELLS, FIRST_BATCH, analyze
from tflt.loopscope.phase11_mmlupro_s2 import PARSER_VERSION, extract_mmlupro_s2

COUNT = 12032
BINDING = ("cell_key", "cell_id", "seed", "source_commit", "prompt_mode", "damping_scope")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return list(iter_lines(path))


def iter_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_cells(value, *, closure=False, first=False):
    expected = FIRST_BATCH if first else CELLS
    schema = "loopscope.phase11.d7.raw_prediction_closure.v1" if closure else "loopscope.phase11.d7.plan.v1"
    require(value.get("schema") == schema, "D7 schema differs")
    # A full plan can close its first batch; a first-only plan has 36096 rows.
    require(value.get("record_count") in ((COUNT * 3, COUNT * 5) if first else (COUNT * 5,)) and
            value.get("target_gold_loaded") is False, "all 60160 raw predictions must close before gold")
    cells = [c for c in value["cells"] if c["cell_key"] in expected]
    require(len(cells) == len(expected) and {c["cell_key"] for c in cells} == set(expected),
            "D7 cell membership differs")
    if not first:
        require(len(value["cells"]) == 5, "D7 requires exactly five cells")
    if closure:
        require(value.get("status") == "GOLD_UNREAD" and value.get("prediction_closed") is True and
                value.get("cell_count") == 5 and value.get("parser_version") == PARSER_VERSION,
                "D7 raw/S2 prediction closure incomplete")
    for cell in cells:
        expected_scope = "full" if cell["cell_key"] in FIRST_BATCH[1:] else "none"
        require(cell.get("population_count") == COUNT and cell.get("seed") == 20261016 and
                cell.get("synthetic") is False and cell.get("engineering") is False and
                cell.get("prompt_mode") == "cot" and cell.get("damping_scope") == expected_scope and
                isinstance(cell.get("source_commit"), str) and bool(cell["source_commit"]),
                "D7 formal population/seed/prompt/scope/source differs")
        require(not closure or cell.get("record_count") == COUNT, "D7 closed cell incomplete")
    return cells


def canonical_pool(path):
    # Discard prompts/input token arrays immediately; raw outputs are streamed below.
    rows = [{k: row[k] for k in ("identity", "subject", "choices", "choice_labels")}
            for row in read(path)["rows"]]
    identities = [row["identity"] for row in rows]
    require(len(rows) == COUNT and len(set(identities)) == COUNT, "D7 canonical identity population differs")
    require(all(isinstance(r["subject"], str) and 1 <= len(r["choices"]) <= 26 and
                r["choice_labels"] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(r["choices"])]) for r in rows),
            "D7 canonical choice labels/categories differ")
    require(len({r["subject"] for r in rows}) == 14, "D7 requires all 14 canonical categories")
    return rows, identities


def align(rows, identities):
    indexed = {}
    indices = {identity: index for index, identity in enumerate(identities)}
    for row in rows:
        identity = row["identity"]
        require(identity in indices and identity not in indexed and
                type(row.get("canonical_index")) is int and row["canonical_index"] == indices[identity],
                "D7 identity/canonical index missing, unknown or duplicated")
        indexed[identity] = row
    require(set(indexed) == set(identities), "D7 cell incomplete")
    return [indexed[identity] for identity in identities]


def load_raw_cells(cells, *, parse=True):
    """Read only canonical inputs and raw outputs; never consult a gold path."""
    data, pool_cache, canonical, common_rows = {}, {}, None, None
    for cell in cells:
        path = cell["pool"]
        if path not in pool_cache:
            pool_cache[path] = canonical_pool(path)
        pool_rows, identities = pool_cache[path]
        if canonical is None:
            canonical, common_rows = identities, pool_rows
        else:
            require(identities == canonical and all(all(a[k] == b[k] for k in
                    ("choices", "choice_labels", "subject")) for a, b in zip(common_rows, pool_rows)),
                    "D7 canonical choices/categories/order differ across cells")
        indices = {identity: index for index, identity in enumerate(identities)}
        raw_by_index, parsed_by_index = {}, {}
        # At 60160 x 8192 tokens the raw token/text panel can exceed RAM. Keep
        # only identity, length and termination scalars after validating each line.
        for r in (r for root in cell["raw_roots"] for r in iter_lines(root["records_path"])):
            identity, index = r.get("identity"), r.get("canonical_index")
            require(identity in indices and type(index) is int and index == indices[identity] and
                    index not in raw_by_index, "D7 identity/canonical index missing, unknown or duplicated")
            original = pool_rows[index]
            require(all(r.get("d7_binding", {}).get(k) == cell[k] for k in BINDING),
                    "D7 raw config/source/runtime binding differs")
            require(r.get("status") == "OK" and r.get("category") == original["subject"] and
                    r.get("candidate_labels") == original["choice_labels"] and
                    isinstance(r.get("generated_text"), str) and isinstance(r.get("generated_token_ids"), list) and
                    type(r.get("generated_token_count")) is int and 0 < r["generated_token_count"] <= 8192 and
                    len(r["generated_token_ids"]) == r["generated_token_count"] and
                    r.get("max_new_tokens") == 8192 and r.get("end_reason") in ("eos", "length_limit"),
                    "D7 raw status/category/options/length/termination differs")
            raw_by_index[index] = {k: r[k] for k in
                                  ("identity", "canonical_index", "category", "generated_token_count", "end_reason")}
            if parse:
                parsed_by_index[index] = extract_mmlupro_s2(r["generated_text"], len(original["choices"]))
        require(len(raw_by_index) == COUNT, "D7 cell incomplete")
        raw = [raw_by_index[index] for index in range(COUNT)]
        parsed = [parsed_by_index[index] for index in range(COUNT)] if parse else []
        data[cell["cell_key"]] = {"raw": raw, "auxiliary": parsed,
                                  "predictions": [p["prediction_index"] for p in parsed]}
    return data, common_rows, canonical


def close(plan, *, first=False):
    cells = validate_cells(plan, first=first)
    data, _, identities = load_raw_cells(cells, parse=not first)
    if first:
        return {"schema": "loopscope.phase11.d7.first_batch_raw_closure.v1",
            "status": "FIRST_BATCH_RAW_CLOSED_GOLD_UNREAD", "target_gold_loaded": False,
            "record_count": COUNT * 3, "cell_count": 3, "cell_keys": list(FIRST_BATCH),
            "cells": [{**cell, "record_count": COUNT} for cell in cells], "closed_at_utc": now()}, []
    predictions = []
    for cell in cells:
        content = data[cell["cell_key"]]
        for index, (identity, raw, parsed) in enumerate(zip(identities, content["raw"], content["auxiliary"])):
            predictions.append({"cell_key": cell["cell_key"], "identity": identity, "canonical_index": index,
                "seed": cell["seed"], **parsed, "generated_token_count": raw["generated_token_count"],
                "end_reason": raw["end_reason"]})
    return {"schema": "loopscope.phase11.d7.raw_prediction_closure.v1", "status": "GOLD_UNREAD",
        "target_gold_loaded": False, "prediction_closed": True, "record_count": COUNT * 5, "cell_count": 5,
        "parser_version": PARSER_VERSION, "closed_at_utc": now(),
        "cells": [{**cell, "record_count": COUNT} for cell in cells]}, predictions


def load_before_gold(closure):
    cells = validate_cells(closure, closure=True)
    data, pool_rows, identities = load_raw_cells(cells)
    saved = lines(closure["predictions_path"])
    require(len(saved) == COUNT * 5, "D7 frozen prediction population differs")
    by_cell = {}
    for row in saved:
        by_cell.setdefault(row["cell_key"], []).append(row)
    require(set(by_cell) == set(CELLS), "D7 frozen prediction cell membership differs")
    for cid, value in data.items():
        for p, raw, parsed in zip(align(by_cell[cid], identities), value["raw"], value["auxiliary"]):
            require(p["seed"] == 20261016 and all(p.get(k) == v for k, v in parsed.items()) and
                    p["generated_token_count"] == raw["generated_token_count"] and p["end_reason"] == raw["end_reason"],
                    "D7 frozen S2 prediction differs from raw")
    return data, pool_rows, identities


def load_gold(path, format, pool_rows, identities):
    require(format in ("p11_jsonl", "p11_json"), "unknown D7 gold format")
    rows = lines(path) if format == "p11_jsonl" else read(path)
    return gold_indices(rows, identities, {r["identity"]: r["choice_labels"] for r in pool_rows},
                        {r["identity"]: r["subject"] for r in pool_rows})


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def write_report(path, result):
    text = ["# Gate D-7：MMLU-Pro 固定配置迁移评测", "",
        "五配置均为完整12032题、5-shot CoT、8192生成上限、运行seed20261016。方法配置由GPQA事后选定；本次为固定配置迁移评测，不称MMLU-Pro任务内最优或完全未见数据的独立验证。", "",
        "| 配置 | 正确数 | micro accuracy | ΔNative pp | Δ同KLoop pp | 类别macro | token均值/中位/p90 | EOS/上限 | S2成功/无匹配/冲突/越界 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["cells"]:
        loop = "—" if r["delta_loop_pp"] is None else f"{r['delta_loop_pp']:+.4f}"
        text.append(f"| {r['cell']} | {r['correct']}/{r['n']} | {r['accuracy_percent']:.4f}% | {r['delta_native_pp']:+.4f} | {loop} | {100*r['category_macro_accuracy']:.4f}% | {r['generated_tokens_mean']:.2f}/{r['generated_tokens_median']:.2f}/{r['generated_tokens_p90']:.2f} | {r['eos_count']}/{r['length_limit_count']} | {r['S2_success_count']}/{r['S2_no_match_count']}/{r['S2_conflict_count']}/{r['S2_out_of_range_count']} |")
    text += ["", "主family四项与基础family两项分别Holm校正；均使用exact双侧McNemar。CI为按14类别分层、保持原类别样本量与micro权重的10000次paired bootstrap名义95%CI，seed20261008。", "",
             "| family | 比较 | Δpp [名义95%CI] | 原始p | Holm p |", "|---|---|---:|---:|---:|"]
    for r in result["contrasts"]:
        ci = r["bootstrap"]
        text.append(f"| {r['family']} | {r['treatment']} − {r['reference']} | {r['delta_pp']:+.4f} [{ci['ci_low_pp']:+.4f},{ci['ci_high_pp']:+.4f}] | {r['p_value']:.6g} | {r['holm_adjusted_p']:.6g} |")
    text += ["", "S2-MMLU-Pro只解析生成正文整行Final answer，按每题原选项数判断合法字母。重复一致合法行接受；越界、冲突、无匹配均计错并保持完整分母。14类别n/accuracy见categories.csv，macro仅描述，不挑选最佳类别或作未校正显著声明。无Matched-norm不能证明方向特异性。调度墙钟/GPUh/峰值/packing及失败成本另见executor资源结算证据。", ""]
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write("\n".join(text))


def run(args):
    if args.close or args.close_first_batch:
        require(args.plan is not None and args.closure is None and args.gold is None,
                "gold-free closure requires --plan only, without --gold")
        closure, predictions = close(read(args.plan), first=args.close_first_batch)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        if not args.close_first_batch:
            path = args.output_dir / "predictions.jsonl"
            with path.open("x", encoding="utf-8") as handle:
                for row in predictions:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            closure["predictions_path"] = str(path)
        closure["source_plan"] = str(args.plan)
        filename = "first_batch_closure.json" if args.close_first_batch else "closure.json"
        write(args.output_dir / filename, closure)
        print(json.dumps({k: closure[k] for k in ("status", "record_count", "cell_count")}))
        return 0
    require(args.closure is not None and args.gold is not None and args.plan is None,
            "analysis requires --closure and --gold")
    closure = read(args.closure)
    data, pool_rows, identities = load_before_gold(closure)
    require(Path(args.gold).is_file(), "D7 gold absent; no gold read")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir / "analysis_spec.json", {"schema": "loopscope.phase11.d7.analysis_spec.v1",
        "gold_read": False, "source_closure": str(args.closure), "parser_version": PARSER_VERSION,
        "record_count": 60160, "primary_family_size": 4, "baseline_family_size": 2,
        "bootstrap_replicates": 10000, "statistical_seed": 20261008,
        "bootstrap_sampling": "within original categories with fixed sample counts; micro weights"})
    gold = load_gold(args.gold, args.gold_format, pool_rows, identities)
    write(args.output_dir / "gold_read_receipt.json", {"timestamp_utc": now(), "gold_path": str(args.gold),
        "source_closure": str(args.closure), "prior_record_count": 60160, "prior_prediction_closed": True,
        "parser_version": PARSER_VERSION, "n": len(gold)})
    result = analyze(data, gold, [row["subject"] for row in pool_rows])
    result.update(source_closure=str(args.closure), record_count=60160, cell_count=5, gold_read=True)
    write(args.output_dir / "analysis.json", result)
    for name in ("cells", "categories", "contrasts"):
        write_csv(args.output_dir / (name + ".csv"), result[name])
    write_report(args.output_dir / "MMLU-Pro迁移评测报告.md", result)
    print(json.dumps({"status": result["status"], "record_count": 60160, "cell_count": 5}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--close", action="store_true")
    mode.add_argument("--close-first-batch", action="store_true")
    for name in ("plan", "closure", "gold"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--gold-format", choices=("p11_jsonl", "p11_json"), default="p11_jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
