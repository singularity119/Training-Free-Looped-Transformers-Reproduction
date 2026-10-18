#!/usr/bin/env python3
"""Close all D8 raw/predictions before the separately invoked gold analysis."""
import argparse
import csv
import datetime
import json
from pathlib import Path

from phase11_d8 import read, metadata, validate_bundle
from tflt.loopscope.phase11_d8 import MODEL, REVISION, CELLS, COUNT, validate_job, validate_pool
from tflt.loopscope.phase11_d8_analysis import analyze
from tflt.loopscope.phase11_panel import require, write_json_once
from tflt.loopscope.phase11_analysis import gold_indices


def lines(path):
    with Path(path).open(encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def load_all(plan):
    from transformers import AutoTokenizer
    require(plan.get("schema") == "loopscope.phase11.d8.full_registry.v1", "D8 full registry differs")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    data, pools, seen, canonical, common_rows = {k: {} for k in CELLS}, {}, set(), None, None
    commits = set()
    for job in plan["jobs"]:
        cell = validate_job(job)
        require(job["synthetic"] is False and job["engineering"] is False, "D8 formal closure contains debug")
        if job["pool"] not in pools:
            pools[job["pool"]] = validate_pool(read(job["pool"]))
        pool = pools[job["pool"]]
        if canonical is None:
            canonical, common_rows = pool["test_identities"], pool["rows"]
        require(pool["test_identities"] == canonical and pool["rows"] == common_rows,
                "D8 formal cells use different prompts/options")
        meta = metadata(job, cell, pool)
        root = Path(job["run_root"])
        require(str(root) not in seen, "D8 duplicate raw root")
        seen.add(str(root))
        summary = read(root / "summary.json")
        require(summary["status"] == "RAW_COMPLETE" and summary["metadata"] == meta and
                summary["count"] == len(job["indices"]), "D8 raw root lacks complete summary")
        commits.add(job["source_commit"])
        for index in job["indices"]:
            require(index not in data[job["cell_key"]], "D8 duplicate successful cell/item")
            bundle = read(root / "complete" / f"{index:06d}.json")
            r, t = bundle["record"], bundle["telemetry"]
            validate_bundle(index, r, t, meta, pool, tok)
            compact = {k: r[k] for k in ("identity", "canonical_index", "generated_token_count", "end_reason", "actual_seed")}
            compact.update({k: v for k, v in r["final_span"].items() if k != "final_text"})
            compact.update({k: t[k] for k in ("elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes")})
            compact.update({k: t[k] for k in ("prefill_elapsed_seconds", "decode_elapsed_seconds") if k in t})
            data[job["cell_key"]][index] = compact
    require(len(commits) == 1, "D8 producer source revisions differ")
    for key in CELLS:
        require(set(data[key]) == set(range(COUNT)), "D8 full 448 identities missing in " + key)
        data[key] = [data[key][i] for i in range(COUNT)]
    return data, common_rows, canonical


def flattened(data):
    return [{"cell_key": key, **r} for key in CELLS for r in data[key]]


def report(path, result):
    text = ["# Gate D-8：GPQA 原版 Qwen3 thinking 三组迁移", "",
        "三组各448题，共1344条fresh；32768生成tokens，temperature 0.6/top_p 0.95/top_k 20/min_p 0。每题seed=20261018+canonical_index，三组同题同seed。", "",
        "仅在首个生成</think>之后应用既定GPQA S2；未闭合、最终回答为空、无匹配或冲突均计错。完整448题为分母，截断不排除。", "",
        "| 配置 | 正确数 | Accuracy | ΔNative pp | EOS/上限 | 未闭合thinking | final为空 | S2无匹配/冲突/一致重复 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["cells"]:
        text.append(f"| {r['cell']} | {r['correct']}/448 | {r['accuracy_percent']:.4f}% | {r['delta_native_pp']:+.4f} | {r['EOS_count']}/{r['length_limit_count']} | {r['unclosed_thinking_count']} | {r['empty_final_count']} | {r['S2_no_match']}/{r['S2_conflict']}/{r['S2_consistent_multiple']} |")
    text += ["", "| 方法−Native | Δpp [名义95%CI] | exact双侧McNemar p | Holm p（两项） |", "|---|---:|---:|---:|"]
    for r in result["contrasts"]:
        ci = r["bootstrap"]
        text.append(f"| {r['treatment']} | {r['delta_pp']:+.4f} [{ci['ci_low_pp']:+.4f}, {ci['ci_high_pp']:+.4f}] | {r['mcnemar_exact_p']:.6g} | {r['holm_adjusted_p']:.6g} |")
    text += ["", "题级配对bootstrap 10000次，seed20261018，百分位名义95%CI；两项比较组成一个Holm family。", "",
        "| 配置 | 全生成长度 均值/中位/p90 | thinking长度 均值/中位/p90 | final长度 均值/中位/p90 | 题耗时合计秒 | peak reserved GiB |", "|---|---:|---:|---:|---:|---:|"]
    for r in result["cells"]:
        values = ["/".join(f"{r[f][k]:.2f}" for k in ("mean", "median", "p90")) for f in ("generated_token_count", "thinking_token_count", "final_token_count")]
        text.append(f"| {r['cell']} | {values[0]} | {values[1]} | {values[2]} | {r['question_wall_seconds']:.2f} | {r['peak_reserved_bytes']/2**30:.3f} |")
    text += ["", "长度区间采用生成token的半开索引；thinking排除首个开启标记和首个闭合标记，final排除闭合标记和末端EOS。异常重复边界保留诊断，不选择有利片段。", "",
        "配置来自同GPQA旧结果选择；本Gate仅是跨checkpoint/thinking配方的单次采样探索性迁移，不是独立题集确认，不能把差异单独归因为thinking或checkpoint，也未估计采样方差。原题、生成、gold和逐题预测只保留在HPC。"]
    with Path(path).open("x", encoding="utf-8") as h:
        h.write("\n".join(text) + "\n")


def run(args):
    if args.close:
        require(args.registry and not args.gold and not args.closure, "D8 closure is gold-free")
        plan = read(args.registry)
        data, _, _ = load_all(plan)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / "predictions.jsonl").open("x") as h:
            for row in flattened(data):
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
            h.flush()
            import os
            os.fsync(h.fileno())
        closure = {"schema": "loopscope.phase11.d8.closure.v1", "status": "GOLD_UNREAD",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "record_count": 1344, "prediction_closed": True, "target_gold_loaded": False,
            "registry": plan, "predictions_path": str(args.output_dir / "predictions.jsonl")}
        write_json_once(args.output_dir / "closure.json", closure)
        print(json.dumps({k: closure[k] for k in ("status", "timestamp_utc", "record_count", "prediction_closed")}))
        return 0
    require(args.closure and args.gold and not args.registry, "D8 scoring requires prior closure")
    closure = read(args.closure)
    require(closure["status"] == "GOLD_UNREAD" and closure["prediction_closed"] and
            closure["record_count"] == 1344 and closure["target_gold_loaded"] is False, "D8 gold barrier incomplete")
    data, rows, identities = load_all(closure["registry"])
    require(flattened(data) == lines(closure["predictions_path"]), "D8 frozen predictions differ before gold")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_once(args.output_dir / "analysis_spec.json", {"gold_read": False, "record_count": 1344,
        "source_closure": str(args.closure), "family_size": 2, "bootstrap_replicates": 10000, "seed": 20261018})
    gold = gold_indices(lines(args.gold), identities, {r["identity"]: r["choice_labels"] for r in rows},
                        {r["identity"]: r["subject"] for r in rows})
    write_json_once(args.output_dir / "gold_read_receipt.json", {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gold_path": str(args.gold), "source_closure": str(args.closure), "prior_prediction_closed": True, "n": len(gold)})
    result = analyze(data, gold)
    result.update(source_closure=str(args.closure), gold_read=True)
    write_json_once(args.output_dir / "analysis.json", result)
    for name in ("cells", "contrasts"):
        with (args.output_dir / (name + ".csv")).open("x", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=list(result[name][0]))
            writer.writeheader()
            writer.writerows(result[name])
    report(args.output_dir / "GPQA-thinking迁移结果.md", result)
    print(json.dumps({"status": result["status"], "record_count": 1344, "gold_read": True}))
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--close", action="store_true")
    for key in ("registry", "closure", "gold", "output-dir"):
        p.add_argument("--" + key, type=Path, required=key == "output-dir")
    raise SystemExit(run(p.parse_args()))
