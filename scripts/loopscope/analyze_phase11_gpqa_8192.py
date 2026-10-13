#!/usr/bin/env python3
"""D-4 CPU closure and user-authorized S2-primary analysis; S1/S0 preserved."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path

from tflt.loopscope.phase11_accuracy import extract_final_answer, load_pool, validate_record
from tflt.loopscope.phase11_analysis import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FAMILY_COUNTS, align_records, frozen_contrasts, gold_indices,
)
from tflt.loopscope.phase11_panel import build_panel, load_config, require, write_json_once
from tflt.loopscope.phase11_posthoc import S1_PATTERN, extract_s1, score_s1
from tflt.loopscope.phase11_s2 import extract_s2, analyze_s2_snapshot
from tflt.loopscope.phase8_accuracy_stats import holm_adjust, paired_contrast, stratified_paired_bootstrap

DATASET = "gpqa_main"
LIMIT = 8192
SCHEMA = "loopscope.phase11.gpqa_8192.v1"
POPULATION = 448


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def close_raw_panel(membership_path, cell_roots, approved_source_commits=None):
    """Fresh complete-population closure. No gold path is accepted or opened here."""
    frozen = read_json(membership_path)
    panel = build_panel(DATASET)
    indices = list(range(POPULATION))
    require(frozen.get("schema") == SCHEMA and frozen.get("population_count") == POPULATION
            and frozen.get("record_count") == POPULATION * 18 and frozen.get("indices") == indices
            and frozen.get("max_new_tokens") == LIMIT and frozen.get("target_gold_loaded") is False,
            "membership differs from frozen D-4 full population")
    require(frozen.get("cells") == [{"cell": cell, "indices": indices} for cell in panel["cells"]],
            "membership differs from frozen 18 independent cells")
    pool = load_pool(Path(frozen["pool"]), DATASET, "FORMAL_TEST")
    rows = pool["rows"]
    identities = [row["identity"] for row in rows]
    require(len(identities) == len(set(identities)) == POPULATION, "canonical population differs")
    require(frozen.get("samples") == [{"canonical_index": i, "identity": row["identity"],
            "prompt_token_length": len(row["input_ids"])} for i, row in enumerate(rows)],
            "membership identities differ from exact input pool")
    require(all(row["choice_labels"] == list("ABCD") for row in rows), "GPQA candidate order differs")
    capacity = load_config()["max_length"]
    max_prompt = max(len(row["input_ids"]) for row in rows)
    require(max_prompt + LIMIT <= capacity and frozen.get("context_admission") == {
        "all_population_inputs_checked": POPULATION, "max_prompt_tokens": max_prompt,
        "required_capacity": max_prompt + LIMIT, "configured_native_capacity": capacity},
        "full input/context admission differs")
    require(set(cell_roots) == {c["cell_id"] for c in panel["cells"]}, "closure lacks exact 18 cell roots")
    aligned, raw_roots, commits, checked_memberships = {}, {}, set(), set()
    for cell in panel["cells"]:
        cid = cell["cell_id"]
        roots = cell_roots[cid]
        if isinstance(roots, (str, Path)):
            roots = [roots]
        require(bool(roots), "cell has no attempt roots")
        records, seen, facts = [], set(), []
        for directory in roots:
            root = Path(directory)
            summary = read_json(root / "summary.json")
            meta = summary["metadata"]
            require(summary.get("schema") == SCHEMA and summary.get("status") == "GPQA_8192_ATTEMPT_COMPLETE"
                    and summary.get("target_gold_loaded") is False and meta.get("schema") == SCHEMA
                    and meta.get("scope") == "FORMAL_TEST" and meta.get("target_gold_loaded") is False
                    and meta.get("max_new_tokens") == LIMIT and meta.get("synthetic_min_new_tokens") is None,
                    "root is not a complete gold-unread formal D-4 attempt")
            require(meta.get("cell") == cell and meta.get("pool") == frozen["pool"],
                    "attempt scientific cell/input path differs")
            source = meta.get("source_commit")
            require(isinstance(source, str) and bool(source), "attempt lacks source commit")
            require(read_json(root / "command_args.json") == meta, "command args differ from attempt metadata")
            member_path = meta["membership"]
            if member_path not in checked_memberships:
                require(read_json(member_path) == frozen, "attempt membership differs")
                checked_memberships.add(member_path)
            declared = meta["indices"]
            require(declared and len(set(declared)) == len(declared)
                    and all(type(i) is int and 0 <= i < POPULATION for i in declared),
                    "attempt indices outside full population or duplicated")
            require(not seen.intersection(declared), "cell shards overlap")
            shard, telemetry = read_rows(root / "records.jsonl"), read_rows(root / "telemetry.jsonl")
            require([r["canonical_index"] for r in shard] == declared
                    and len(shard) == summary["count"] == len(telemetry), "attempt records/telemetry incomplete")
            for record, fact in zip(shard, telemetry):
                validate_record(record, DATASET)
                i = record["canonical_index"]
                row = rows[i]
                require(record["identity"] == row["identity"] and record["category"] == row["subject"]
                        and record["candidate_labels"] == row["choice_labels"] and record["max_new_tokens"] == LIMIT,
                        "raw identity/category/candidate order/length differs")
                require(fact["canonical_index"] == i and fact["identity"] == record["identity"]
                        and fact["prefill_count"] == 1 and fact["decode_count"] == record["generated_token_count"] - 1
                        and fact["cache_growth_closed"] is True, "prefill/decode/cache lifecycle differs")
                if cell["arm"] == "Online":
                    online = fact.get("online", {})
                    require(online.get("decode_fit_count") == 0 and online.get("directions_released") is True,
                            "frozen prefill direction lifecycle differs")
            require(summary["generated_token_count"] == sum(r["generated_token_count"] for r in shard),
                    "summary token count differs")
            # Token IDs have been checked directly against native EOS/8192 above.
            # Keep only scoring inputs across the full panel, freeing large token lists.
            scoring_keys = ("identity", "canonical_index", "status", "category", "candidate_labels",
                            "generated_text", "generated_token_count", "end_reason")
            records.extend({key: record[key] for key in scoring_keys} for record in shard)
            seen.update(declared)
            commits.add(source)
            facts.append({"run_root": str(root.resolve()), "cell_id": cid, "indices": declared,
                          "records_count": len(shard), "source_commit": source})
        aligned[cid] = align_records(DATASET, records, identities)
        raw_roots[cid] = facts
    if approved_source_commits is None:
        require(len(commits) == 1, "mixed source revisions require explicit approved source set")
    else:
        require(not isinstance(approved_source_commits, str) and bool(approved_source_commits)
                and commits.issubset(set(approved_source_commits)), "source revision outside approved set")
    closure = {"schema": SCHEMA, "gate": "D-4", "status": "PANEL_CLOSED_GOLD_UNREAD", "timestamp": now(),
        "dataset": DATASET, "target_gold_loaded": False, "gold_read": False, "max_new_tokens": LIMIT,
        "sample_count": POPULATION, "independent_cell_count": 18, "record_count": POPULATION * 18,
        "identities": identities, "cells": panel["cells"], "raw_roots": raw_roots,
        "record_counts": {cid: len(records) for cid, records in aligned.items()},
        "source_commits": sorted(commits), "approved_source_commits":
            None if approved_source_commits is None else sorted(set(approved_source_commits))}
    return closure, panel, aligned


def extract_population(panel, aligned, output):
    """Persist all parser predictions remotely before opening gold; no response text copied."""
    by_cell, status, transition = {}, {"S2": Counter(), "S1": Counter(), "S0": Counter()}, Counter()
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for cell in panel["cells"]:
            cid = cell["cell_id"]
            by_cell[cid] = []
            for record in aligned[cid]:
                s0 = extract_final_answer(record["generated_text"], record["candidate_labels"])
                s1 = extract_s1(record["generated_text"])
                s2 = extract_s2(record["generated_text"])
                row = {"cell_id": cid, "identity": record["identity"], "canonical_index": record["canonical_index"],
                       "end_reason": record["end_reason"], "S2": s2, "S1": s1, "S0": s0}
                for name in status:
                    status[name][row[name]["extraction_status"]] += 1
                transition[s0["extraction_status"] + " -> " + s1["extraction_status"]] += 1
                by_cell[cid].append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    require(all(sum(counts.values()) == POPULATION * 18 for counts in status.values()), "extraction population incomplete")
    barrier = {"status": "ALL_S2_S1_S0_PREDICTIONS_SAVED_GOLD_UNREAD", "timestamp": now(), "gold_read": False,
        "target_gold_loaded": False, "record_count": POPULATION * 18, "independent_cell_count": 18,
        "samples_per_cell": POPULATION, "pattern": S1_PATTERN, "flags": "re.IGNORECASE",
        "line_processing": "splitlines then strip then fullmatch",
        "ambiguity": {"S1": "exactly one matching line; duplicates fail",
                      "S2": "at least one match, exactly one distinct normalized letter; conflicts fail"},
        "status_counts": {name: dict(counts) for name, counts in status.items()}, "status_transitions": dict(transition)}
    write_json_once(output / "extraction_barrier.json", barrier)
    return by_cell, barrier


def length_summary(values):
    ordered = sorted(values)
    return {"min": ordered[0], "max": ordered[-1], "mean": sum(values) / len(values),
            "quantiles_including_right_censoring": {str(q): ordered[round((len(values) - 1) * q)]
                                                   for q in (0, .25, .5, .75, .9, .95, 1)}}


def parser_analysis(panel, predictions, aligned, gold, name):
    cells, correctness = [], {}
    for cell in panel["cells"]:
        cid = cell["cell_id"]
        rows = [{**row[name], "end_reason": row["end_reason"]} for row in predictions[cid]]
        stats, correctness[cid] = score_s1(rows, gold)
        ci = stratified_paired_bootstrap(correctness[cid], ["all_items"] * POPULATION,
                                        BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        cells.append({**cell, **stats, "accuracy_ci": {**ci, "ci_low": ci["ci_low_pp"] / 100,
                                                       "ci_high": ci["ci_high_pp"] / 100},
                      "generation_length": length_summary([r["generated_token_count"] for r in aligned[cid]])})
    contrasts = [{**spec, **paired_contrast(correctness[spec["reference"]], correctness[spec["treatment"]],
                  ["all_items"] * POPULATION, bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)}
                 for spec in frozen_contrasts(panel)]
    for family, count in FAMILY_COUNTS.items():
        selected = [row for row in contrasts if row["family"] == family]
        require(len(selected) == count, "contrast family count differs")
        for row, p in zip(selected, holm_adjust([r["mcnemar_exact_p"] for r in selected])):
            row.update(holm_adjusted_p=p, holm_family_size=count, holm_reject_alpha_0_05=p <= .05)
    native = next(c for c in cells if c["arm"] == "Native")
    loops = {c["k"]: c for c in cells if c["arm"] == "Loop"}
    for cell in cells:
        cell["delta_native_pp"] = 100 * (cell["accuracy"] - native["accuracy"])
        cell["delta_loop_pp"] = None if cell["k"] is None else 100 * (cell["accuracy"] - loops[cell["k"]]["accuracy"])
    groups = {reason: {key: sum(c["groups"][reason][key] for c in cells)
                       for key in ("n", "extraction_success_count", "correct")}
              for reason in ("eos", "length_limit")}
    return {"parser": name, "cells": cells, "contrasts": contrasts, "groups": groups}


def run(args):
    output = Path(args.output_dir)
    require(not output.exists(), "analysis output root already exists")
    roots = read_json(args.cell_roots)
    closure, panel, aligned = close_raw_panel(args.membership, roots, args.approved_source_commit)
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(output / "raw_closure.json", closure)
    print(json.dumps({"status": closure["status"], "record_count": closure["record_count"], "target_gold_loaded": False}), flush=True)
    write_json_once(output / "s2_spec.json", {"schema": "loopscope.phase11.s2.v1", "parser": "S2",
        "frozen_at_utc": now(), "pattern": S1_PATTERN, "normalization": "uppercase A-D",
        "valid": "at least one exact matching line and exactly one distinct letter",
        "consistent_multiple": "shared letter scored against gold", "conflicting_multiple": "wrong",
        "no_match": "wrong; no prose inference", "gold_read": False,
        "interpretation": "user-selected posthoc rule; S1/S0 auxiliary artifacts retained"})
    predictions, barrier = extract_population(panel, aligned, output)
    # Both complete-population artifacts are closed on disk before the first gold open.
    gold_rows = read_rows(args.gold)
    gold_read_time = now()
    identities = closure["identities"]
    first = aligned[panel["cells"][0]["cell_id"]]
    gold = gold_indices(gold_rows, identities, {r["identity"]: r["candidate_labels"] for r in first},
                        {r["identity"]: r["category"] for r in first})
    s1 = parser_analysis(panel, predictions, aligned, gold, "S1")
    s0 = parser_analysis(panel, predictions, aligned, gold, "S0")
    s2_snapshot = analyze_s2_snapshot(panel, aligned, dict(zip(identities, gold)))
    s2_common = s2_snapshot["s2"]["common_intersection"]
    require(s2_common["n"] == POPULATION, "S2 full common population differs")
    s2 = {"parser": "S2", "cells": s2_common["cells"], "contrasts": s2_common["contrasts"]}
    s2["groups"] = {reason: {key: sum(c["groups"][reason][key] for c in s2["cells"])
                             for key in ("n", "extraction_success_count", "correct")}
                    for reason in ("eos", "length_limit")}
    report = {"schema": SCHEMA, "gate": "D-4", "dataset": DATASET,
        "status": "GPQA_8192_ANALYSIS_COMPLETE_AUDIT_REQUESTED", "executor_commit": args.executor_commit,
        "source_commits": closure["source_commits"], "gold_read_timestamp": gold_read_time,
        "raw_closure": str(output / "raw_closure.json"), "extraction_barrier": barrier,
        "sample_count": POPULATION, "independent_cell_count": 18, "record_count": POPULATION * 18,
        "max_new_tokens": LIMIT, "primary_parser": "S2", "auxiliary_parsers": ["S1", "S0"],
        "cells": s2["cells"], "contrasts": s2["contrasts"], "groups": s2["groups"],
        "s2": s2, "s1": s1, "s0": s0,
        "families": FAMILY_COUNTS, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "paths": {key: str(getattr(args, key)) for key in ("membership", "cell_roots", "gold", "output_dir")},
        "interpretation": {"scope": "same previously seen GPQA population; 8192 follow-up exploratory experiment",
            "selection_limit": "S2 consistent-answer merge and length limit chosen after output diagnostics; posthoc exploratory, Holm does not undo selection",
            "denominator": "448 per cell; extraction failures count as wrong",
            "ci": "nominal 95 percent percentile bootstrap; Holm within original 30/12/2 families per parser",
            "groups": "EOS and length groups are descriptive selected subsets; repeated cell outputs are not independent questions",
            "length": "8192 length-limit observations are right-censored",
            "limits": "No matched-norm control; no direction-specific causal or transfer claim"},
        "mmlu_accessed": False, "gpu_jobs_submitted": False, "gate_pass_claimed": False}
    write_json_once(output / "analysis.json", report)
    print(json.dumps({"status": report["status"], "record_count": report["record_count"], "output_dir": str(output),
                      "s2_holm_rejections": sum(r["holm_reject_alpha_0_05"] for r in report["contrasts"])}), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("membership", "cell-roots", "gold", "output-dir", "executor-commit"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--approved-source-commit", action="append",
                   help="Externally approved compatible producer source revision; repeat for each allowed revision")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
