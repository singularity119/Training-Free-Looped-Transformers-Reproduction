#!/usr/bin/env python3
"""One aggregate D5 analysis, only after all raw records/predictions close."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import datetime
import json
import math
from pathlib import Path

from tflt.loopscope.phase8_accuracy_stats import paired_contrast, holm_adjust
from tflt.loopscope.phase11_analysis import gold_indices
from tflt.loopscope.phase11_s2 import extract_s2

COUNTS = {"p10-base-mmlu": 14042, "p10-base-arc": 1172,
          "p11-instruct-arc": 1172, "p11-instruct-gpqa": 448}
MEMBERS = {
    "p10-base-mmlu": ("q4-native", "q4-w15-18-k2-loop", "q4-w15-18-k2-online-current-t-lambda0.9"),
    "p10-base-arc": ("q4-native", "q4-w15-18-k2-loop", "q4-w15-18-k2-online-fixed-t0-lambda0.4"),
    "p11-instruct-arc": tuple("arc_challenge-q4-" + s for s in
        ("native", "w15-18-k2-loop", "w15-18-k2-online-fixed-t0-lambda0.9")),
    "p11-instruct-gpqa": tuple("gpqa_main-q4-" + s for s in
        ("native", "w15-18-k2-loop", "w15-18-k3-loop", "w15-18-k2-online-current-t-lambda0.5", "w15-18-k3-online-fixed-t0-lambda0.9")),
}
ATOL, RTOL = 1e-4, 1e-5


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


def key(cell):
    return cell["family"], cell["repeat"], cell["cell_id"]


def validate_closure(closure):
    require(closure.get("schema") == "loopscope.phase11.d5.raw_prediction_closure.v1" and
            closure.get("status") == "GOLD_UNREAD" and closure.get("target_gold_loaded") is False and
            closure.get("prediction_closed") is True and closure.get("record_count") == 102796 and
            closure.get("cell_count") == 28, "all 102796 raw records and predictions must close before gold")
    expected = {(f, r, cid) for f, ids in MEMBERS.items() for r in (1, 2) for cid in ids}
    cells = closure.get("cells", [])
    require(len(cells) == 28 and {key(c) for c in cells} == expected,
            "closure differs from 28 independent repeat cells")
    for c in cells:
        require(type(c["repeat"]) is int and c.get("record_count") == COUNTS[c["family"]] and
                c.get("population_count") == COUNTS[c["family"]] and c.get("synthetic") is False and
                c.get("engineering", False) is False and
                c.get("seed") == {1: 20261014, 2: 20261015}[c["repeat"]],
                "closure cell count/seed/formal scope differs")
    require(sum(c["record_count"] for c in cells) == 102796, "closure cell totals differ")


def canonical_rows(pool, family):
    rows = pool["rows"]
    require(len(rows) == COUNTS[family], "canonical pool count differs")
    ids = [r["sample_id"] if family == "p10-base-arc" else r["identity"] for r in rows]
    require(len(set(ids)) == len(ids), "canonical input identities repeat")
    return rows, ids


def align(raw, identities):
    by_identity = {}
    order = {identity: i for i, identity in enumerate(identities)}
    for row in raw:
        ident = row["identity"]
        require(ident in order and ident not in by_identity, "raw identity missing/unknown/duplicated")
        if "canonical_index" in row:
            require(type(row["canonical_index"]) is int and row["canonical_index"] == order[ident],
                    "raw canonical index differs")
        by_identity[ident] = row
    require(set(by_identity) == set(identities), "raw cell is incomplete")
    return [by_identity[identity] for identity in identities]


def predict(family, raw, pool_rows):
    predictions, auxiliary = [], []
    for row, source in zip(raw, pool_rows):
        if family.endswith("gpqa"):
            parsed = extract_s2(row["generated_text"])
            predictions.append(parsed["prediction_index"])
            auxiliary.append(parsed)
            require(isinstance(row.get("generated_token_ids"), list) and
                    len(row["generated_token_ids"]) == row["generated_token_count"] and
                    row["end_reason"] in ("eos", "length_limit"), "GPQA token/termination metadata differs")
        else:
            scores = row["scores"]
            choices = list("ABCD") if family.endswith("mmlu") else source["choices"]
            lengths = [len(c) for c in choices]
            require(len(scores) == len(lengths) and all(math.isfinite(float(v)) for v in scores),
                    "raw scores must include all finite candidates")
            require("choice_text_lengths" not in row or row["choice_text_lengths"] == lengths,
                    "raw normalization lengths differ from original choices")
            raw_pred = max(range(len(scores)), key=scores.__getitem__)
            predictions.append(max(range(len(scores)), key=lambda i: scores[i] / lengths[i]) if family.endswith("arc") else raw_pred)
            auxiliary.append({"raw_prediction_index": raw_pred})
    return predictions, auxiliary


def load_gold(binding, pool_rows, identities):
    # This function is called exactly once/family, after all raw/predictions verify.
    require(Path(binding["gold"]).is_file(), "canonical gold path is absent")
    if binding["format"] == "base_aligned_json":
        value = read(binding["gold"])
        require(value.get("identities") == identities and len(value.get("gold", [])) == len(identities),
                "aligned Base gold identity/order/count differs")
        gold = value["gold"]
    else:
        require(binding["format"] in ("p11_jsonl", "p11_json"), "unknown canonical gold format")
        rows = lines(binding["gold"]) if binding["format"] == "p11_jsonl" else read(binding["gold"])
        gold = gold_indices(rows, identities,
            {identity: row["choice_labels"] for identity, row in zip(identities, pool_rows)},
            {identity: row["subject"] for identity, row in zip(identities, pool_rows)})
    for value, row in zip(gold, pool_rows):
        require(type(value) is int and 0 <= value < len(row["choices"]), "gold outside candidate indices")
    return gold


def load_before_gold(closure, binding):
    require(binding.get("schema") == "loopscope.phase11.d5.reference_binding.v1" and
            set(binding["families"]) == set(MEMBERS), "reference binding family/schema differs")
    predictions = lines(closure["predictions_path"])
    require(len(predictions) == 102796, "closed prediction file count differs")
    by_key = {}
    for row in predictions:
        by_key.setdefault(key(row), []).append(row)
    specs = {key(c): c for c in closure["cells"]}
    require(set(by_key) == set(specs), "closed predictions differ from cells")
    data = {}
    for family, ids in MEMBERS.items():
        bound = binding["families"][family]
        require(set(bound["refs"]) == set(ids), "reference0 cell membership differs")
        pool_rows, identities = canonical_rows(read(bound["pool"]), family)
        versions = {}
        for version in (0, 1, 2):
            cells = {}
            for cid in ids:
                if version == 0:
                    ref = bound["refs"][cid]
                    raw = []
                    for root in ref["roots"]:
                        path = root["records_path"] if isinstance(root, dict) else str(Path(root) / ref["filename"])
                        records = lines(path)
                        raw.extend(r[ref["record_key"]] if ref.get("record_key") else r for r in records)
                else:
                    spec = specs[(family, version, cid)]
                    raw = [r for root in spec["raw_roots"] for r in lines(root["records_path"])]
                raw = align(raw, identities)
                if version:
                    for row in raw:
                        actual = row.get("d5_binding", row)
                        require(all(actual.get(field) == spec[field] for field in
                            ("family", "repeat", "cell_id", "seed", "source_commit")),
                            "new raw repeat/cell/seed/source binding differs")
                predicted, aux = predict(family, raw, pool_rows)
                if version:
                    frozen = align(by_key[(family, version, cid)], identities)
                    require([p["prediction_index"] for p in frozen] == predicted and
                            all(p["seed"] == {1: 20261014, 2: 20261015}[version] for p in frozen),
                            "saved predictions differ from closed raw/seed")
                    if family.endswith("gpqa"):
                        require(all(all(p.get(k) == a[k] for k in a) for p, a in zip(frozen, aux)),
                                "saved GPQA S2 status differs from raw")
                        require(all(p["generated_token_count"] == r["generated_token_count"] and
                                    p["end_reason"] == r["end_reason"] for p, r in zip(frozen, raw)),
                                "saved GPQA length/termination differs from raw")
                cells[cid] = {"raw": raw, "predictions": predicted, "auxiliary": aux}
            versions[version] = cells
        data[family] = {"pool_rows": pool_rows, "identities": identities, "versions": versions}
    return data


def reproduce(family, cid, a, b, gold, versions):
    n = len(gold)
    before, after = a["predictions"], b["predictions"]
    correct_a = [int(p is not None and p == g) for p, g in zip(before, gold)]
    correct_b = [int(p is not None and p == g) for p, g in zip(after, gold)]
    same = sum(x == y for x, y in zip(before, after))
    result = {"family": family, "cell_id": cid, "from": versions[0], "to": versions[1], "n": n,
        "prediction_same_count": same, "prediction_agreement": same / n, "prediction_changed_count": n - same,
        "wrong_to_right": sum(x == 0 and y == 1 for x, y in zip(correct_a, correct_b)),
        "right_to_wrong": sum(x == 1 and y == 0 for x, y in zip(correct_a, correct_b)),
        "same_correct_total": sum(correct_a) == sum(correct_b), "strict_prediction_identity": same == n}
    if family.endswith("gpqa"):
        for name, field in (("token_sequence", "generated_token_ids"), ("length", "generated_token_count"),
                            ("end_reason", "end_reason")):
            count = sum(x[field] == y[field] for x, y in zip(a["raw"], b["raw"]))
            result[name + "_same_count"] = count
            result[name + "_agreement"] = count / n
        for name in ("extraction_status", "answer_mode"):
            count = sum(x[name] == y[name] for x, y in zip(a["auxiliary"], b["auxiliary"]))
            result["S2_" + name + "_same_count"] = count
            result["S2_" + name + "_agreement"] = count / n
    else:
        strict, close, largest, compared = 0, 0, 0., 0
        for x, y in zip(a["raw"], b["raw"]):
            xs, ys = x["scores"], y["scores"]
            strict += xs == ys
            close += all(abs(u - v) <= ATOL + RTOL * abs(v) for u, v in zip(xs, ys))
            largest = max(largest, max(abs(u - v) for u, v in zip(xs, ys)))
            compared += len(xs)
        result.update(raw_scores_strict_same_count=strict, raw_scores_strict_agreement=strict / n,
                      raw_scores_close_count=close, raw_scores_close_agreement=close / n,
                      raw_scores_max_abs_difference=largest, compared_score_count=compared, atol=ATOL, rtol=RTOL)
    return result


def analyze(data, gold_by_family, *, replicates=10000):
    cells, contrasts, reproduction, correctness = [], [], [], {}
    for family, content in data.items():
        gold = gold_by_family[family]
        n = len(gold)
        require(n == len(content["pool_rows"]) and n > 0, "gold population differs")
        native = MEMBERS[family][0]
        loops = {int(cid.split("-k")[1].split("-")[0]): cid for cid in MEMBERS[family] if cid.endswith("-loop")}
        for version, variants in content["versions"].items():
            for cid, value in variants.items():
                right = [int(p is not None and p == g) for p, g in zip(value["predictions"], gold)]
                require(len(right) == n, "cell denominator differs")
                correctness[(family, version, cid)] = right
                stat = {"family": family, "version": "reference0" if version == 0 else "repeat" + str(version),
                        "repeat": version, "cell_id": cid, "n": n, "correct": sum(right),
                        "primary_accuracy": sum(right) / n,
                        "primary_metric": "S2" if family.endswith("gpqa") else "acc_norm" if family.endswith("arc") else "acc"}
                if family.endswith("arc"):
                    stat["acc_correct"] = sum(a["raw_prediction_index"] == g for a, g in zip(value["auxiliary"], gold))
                    stat["acc"] = stat["acc_correct"] / n
                if family.endswith("gpqa"):
                    stat["S2_mode_counts"] = dict(Counter(a["answer_mode"] for a in value["auxiliary"]))
                    stat["end_reason_counts"] = dict(Counter(r["end_reason"] for r in value["raw"]))
                cells.append(stat)
        for version in (0, 1, 2):
            for cid in MEMBERS[family]:
                stat = next(c for c in cells if (c["family"], c["repeat"], c["cell_id"]) == (family, version, cid))
                right = correctness[(family, version, cid)]
                stat["delta_native_pp"] = 100 * (sum(right) - sum(correctness[(family, version, native)])) / n
                kval = None if cid == native else int(cid.split("-k")[1].split("-")[0])
                stat["delta_loop_pp"] = None if kval is None else 100 * (sum(right) - sum(correctness[(family, version, loops[kval])])) / n
                if version and cid != native:
                    refs = [(native, "MAIN_20")] if "-online-" in cid else [(native, "SECONDARY_10")]
                    if "-online-" in cid:
                        refs.append((loops[kval], "MAIN_20"))
                    strata = [r["subject"] for r in content["pool_rows"]] if family.endswith("mmlu") else ["items"] * n
                    seed = 20261002 if family.startswith("p10-") else 20261008
                    for ref, group in refs:
                        contrasts.append({"family": family, "repeat": version, "statistical_family": group,
                            "reference": ref, "treatment": cid,
                            **paired_contrast(correctness[(family, version, ref)], right, strata,
                                              bootstrap_replicates=replicates, seed=seed)})
        for cid in MEMBERS[family]:
            for before, after in ((0, 1), (0, 2), (1, 2)):
                reproduction.append(reproduce(family, cid, content["versions"][before][cid],
                                              content["versions"][after][cid], gold, (before, after)))
    for group, expected in (("MAIN_20", 20), ("SECONDARY_10", 10)):
        selected = [c for c in contrasts if c["statistical_family"] == group]
        require(len(selected) == expected, "D5 contrast family count differs")
        for row, adjusted in zip(selected, holm_adjust([r["mcnemar_exact_p"] for r in selected])):
            row.update(holm_family_size=expected, holm_adjusted_p=adjusted, holm_reject_alpha_0_05=adjusted <= .05)
    for row in cells:
        original = next(r for r in cells if r["family"] == row["family"] and r["cell_id"] == row["cell_id"] and r["repeat"] == 0)
        row["delta_native_minus_reference0_pp"] = row["delta_native_pp"] - original["delta_native_pp"]
        row["delta_loop_minus_reference0_pp"] = None if row["delta_loop_pp"] is None else row["delta_loop_pp"] - original["delta_loop_pp"]
    return {"schema": "loopscope.phase11.d5.analysis.v1", "status": "D5_ANALYZED_COMPLETE_REPEATS",
        "cells": cells, "contrasts": contrasts, "reproduction": reproduction,
        "statistical_families": {"MAIN_20": 20, "SECONDARY_10": 10},
        "numeric_comparison": {"atol": ATOL, "rtol": RTOL, "strict_equality_separate": True},
        "interpretation": {"new_compute_seeds": {"repeat1": 20261014, "repeat2": 20261015},
            "scope": "same fixed questions/prompt/options under two new computation seeds; no independent dataset confirmation",
            "selection": "historically selected best configurations remain exploratory even if rerun p is significant",
            "base_arc": "reference0 damping-minus-Loop gain is zero; do not claim reproduced extra gain",
            "ci": "nominal paired bootstrap 95 percent intervals; each repeat retains its own n, never pooled into 2N",
            "engineering": "negative gain and numerical differences are reported, not treated as automatic Gate failure"}}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def historical_contrasts(family, value):
    """Select only the original full-grid primary parser, using its actual schema."""
    if family.endswith("gpqa"):
        if isinstance(value.get("s2"), dict) and isinstance(value["s2"].get("contrasts"), list):
            return value["s2"]["contrasts"]
        require(value.get("primary_parser") == "S2" and isinstance(value.get("contrasts"), list),
                "historical GPQA analysis must explicitly identify primary S2")
    require(isinstance(value.get("contrasts"), list), "historical full-grid contrast list is absent")
    return value["contrasts"]


def attach_historical(result, binding, analyses):
    """Attach descriptive old p columns; never change a new p or Holm family."""
    for row in result["cells"]:
        refs = binding["families"][row["family"]]["refs"]
        seed = refs[row["cell_id"]].get("reference0_runtime_seed")
        row["reference0_runtime_seed"] = seed
        row["runtime_seed"] = seed if row["repeat"] == 0 else {1: 20261014, 2: 20261015}[row["repeat"]]
        row["reference0_seed_note"] = None if seed is not None else "reference0 runtime seed absent from supplied binding"
    for row in result["contrasts"]:
        family = row["family"]
        bound = binding["families"][family]
        row.update(old_exploratory_family=None, old_holm_adjusted_p=None,
                   old_holm_family_size=None, old_exploratory_source=bound.get("historical_analysis"),
                   old_reference_runtime_seed=bound["refs"][row["reference"]].get("reference0_runtime_seed"),
                   old_treatment_runtime_seed=bound["refs"][row["treatment"]].get("reference0_runtime_seed"),
                   old_exploratory_match_reason=None)
        if family not in analyses:
            row["old_exploratory_match_reason"] = "historical analysis was not supplied"
            continue
        if family.startswith("p10-") and row["statistical_family"] == "SECONDARY_10":
            row["old_exploratory_match_reason"] = "original Phase10 full-grid families did not include ordinary Loop-Native contrasts"
            continue
        expected_family = ("LAMBDA_SCAN_126" if family.startswith("p10-") else
            "ONLINE_VS_NATIVE_AND_LOOP_EXPLORATORY" if row["statistical_family"] == "MAIN_20" else "LOOP_VS_NATIVE_BASE")
        old_rows = historical_contrasts(family, analyses[family])
        matches = [r for r in old_rows if r.get("reference") == row["reference"] and
                   r.get("treatment") == row["treatment"] and r.get("family") == expected_family]
        if len(matches) != 1:
            row["old_exploratory_match_reason"] = ("no exact directed reference/treatment match in " + expected_family
                if not matches else "multiple exact historical matches; no value selected")
            continue
        old = matches[0]
        old_p = old.get("holm_adjusted_p")
        require(isinstance(old_p, (int, float)) and not isinstance(old_p, bool) and
                math.isfinite(old_p) and 0 <= old_p <= 1, "historical Holm p is invalid")
        row.update(old_exploratory_family=old["family"], old_holm_adjusted_p=old_p,
                   old_holm_family_size=old.get("holm_family_size"),
                   old_exploratory_match_reason="exact directed reference/treatment and original full-grid family match")
    result["interpretation"]["historical_p"] = "old full-grid adjusted p values are descriptive side columns; they never replace the new 20/10 Holm families"
    return result


def write_report(path, result):
    text = ["# D5 两轮新计算 seed 复现分析", "",
        "固定题集、顺序、prompt与选项，新计算 seed 分别为20261014和20261015。重复同一题集不能作为独立数据验证；原最优配置是事后选择，当前结果继续按探索性解释。", "",
        "| 比较 | 汇总正确数相同 cells | 逐题预测全同 cells | 似然分数严格全同 cells | GPQA token序列全同 cells |",
        "|---|---:|---:|---:|---:|"]
    for before, after in ((0, 1), (0, 2), (1, 2)):
        selected = [r for r in result["reproduction"] if (r["from"], r["to"]) == (before, after)]
        same_total = sum(r["same_correct_total"] for r in selected)
        same_pred = sum(r["strict_prediction_identity"] for r in selected)
        same_raw = sum(r.get("raw_scores_strict_same_count") == r["n"] for r in selected)
        same_tokens = sum(r.get("token_sequence_same_count") == r["n"] for r in selected)
        text.append(f"| {before}→{after}（0为reference0） | {same_total}/14 | {same_pred}/14 | {same_raw}/9 | {same_tokens}/5 |")
    text += ["", "| family | 配置 | 轮次 | correct/N | 主准确率 | ΔNative pp | Δ同KLoop pp |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in result["cells"]:
        loop = "—" if row["delta_loop_pp"] is None else f"{row['delta_loop_pp']:+.4f}"
        text.append(f"| {row['family']} | {row['cell_id']} | {row['version']} | {row['correct']}/{row['n']} | {100*row['primary_accuracy']:.4f}% | {row['delta_native_pp']:+.4f} | {loop} |")
    text += ["", "主比较20项在两个repeat间组成一个Holm family；普通Loop–Native共10项另成一个family。双侧exact McNemar、alpha=.05；CI为10000次名义95%配对bootstrap，Base seed20261002、Instruct seed20261008。两轮分别保留分母，不拼成2N。", "",
        "Base ARC原衰减相对普通Loop额外收益为0，不能表述为复现了额外增益。负增益、同分但预测不同、严格相同与数值容差内相同均保留。", "",
        "逐cell三组复现比较见reproduction.csv（reference0→repeat1、reference0→repeat2、repeat1→repeat2）：含预测一致/改变、正确转错/错误转对，Base/ARC分数严格一致与rtol1e-5/atol1e-4容差；GPQA另含token序列、长度、结束原因及S2状态一致率。ARC附acc见cells.csv。", "",
        "逐题raw与预测留在HPC原路径，本报告仅包含聚合统计。"]
    text += ["", "reference0各cell原运行seed与本次新seed见cells.csv。旧全网格探索性调整p仅作历史旁列，不替代本次20/10项新family；缺少精确对应项时列为缺失并在contrasts.csv说明原因。", "",
             "| family | repeat | 衰减候选 | 对照 | Δpp及名义95%CI | 新Holm p（20项） | 旧全网格探索p |",
             "|---|---|---|---|---:|---:|---:|"]
    for row in result["contrasts"]:
        if row["statistical_family"] == "MAIN_20":
            ci = row["bootstrap"]
            old_p = "—" if row.get("old_holm_adjusted_p") is None else f"{row['old_holm_adjusted_p']:.6g}"
            text.append(f"| {row['family']} | {row['repeat']} | {row['treatment']} | {row['reference']} | {row['delta_pp']:+.4f} [{ci['ci_low_pp']:+.4f},{ci['ci_high_pp']:+.4f}] | {row['holm_adjusted_p']:.6g} | {old_p} |")
    Path(path).write_text("\n".join(text) + "\n", encoding="utf-8")


def run(args):
    closure = read(args.closure)
    validate_closure(closure)
    binding = read(args.reference_binding)
    # All sources/predictions are checked without target gold first.
    data = load_before_gold(closure, binding)
    require(all(Path(item["gold"]).is_file() for item in binding["families"].values()),
            "one or more canonical gold paths are absent; no gold has been read")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "analysis_spec.json", {"closure": str(args.closure), "reference_binding": str(args.reference_binding),
        "record_count": 102796, "cell_count": 28, "prediction_closed": True, "gold_read": False,
        "main_holm_size": 20, "secondary_holm_size": 10, "replicates": 10000,
        "numeric_comparison": {"atol": ATOL, "rtol": RTOL}})
    gold = {family: load_gold(binding["families"][family], item["pool_rows"], item["identities"])
            for family, item in data.items()}
    write(output / "gold_read_receipt.json", {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prior_prediction_closure": str(args.closure), "read_once_per_family": True,
        "sources": {family: {"gold": binding["families"][family]["gold"], "n": len(values)} for family, values in gold.items()}})
    result = analyze(data, gold)
    # Historical statistics are read only after the full D5 barrier and canonical
    # gold stage. Paths are supplied explicitly; never discover a different run.
    historical = {family: read(bound["historical_analysis"]) for family, bound in binding["families"].items()
                  if bound.get("historical_analysis")}
    attach_historical(result, binding, historical)
    result.update(source_closure=str(args.closure), source_reference_binding=str(args.reference_binding),
                  record_count=102796, cell_count=28, gold_read=True)
    write(output / "analysis.json", result)
    for name in ("cells", "contrasts", "reproduction"):
        write_csv(output / (name + ".csv"), result[name])
    write_report(output / "复现分析报告.md", result)
    print(json.dumps({"status": result["status"], "cells": len(result["cells"]),
                      "contrasts": len(result["contrasts"]), "output": str(output)}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("closure", "reference-binding", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
