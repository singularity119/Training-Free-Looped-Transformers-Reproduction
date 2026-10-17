#!/usr/bin/env python3
"""User-authorized complete K3 snapshot, reusing the earlier exact-bound gold."""
import argparse
from collections import Counter
import json
from pathlib import Path
from analyze_phase11_d7 import (COUNT, align, lines, load_gold, load_raw_cells,
                               now, read, require, validate_cells, write)
from prepare_phase11_d7_gold import SOURCE, REVISION
from tflt.loopscope.phase11_mmlupro_s2 import PARSER_VERSION


def run(args):
    key = "fixed_t0_k3_l09"
    cells = [c for c in validate_cells(read(args.plan), first=True) if c["cell_key"] == key]
    require(len(cells) == 1, "exact K3 cell absent")
    data, pool, identities = load_raw_cells(cells)
    value = data[key]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "predictions.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        for index, (raw, parsed) in enumerate(zip(value["raw"], value["auxiliary"])):
            row = {"cell_key": key, "identity": identities[index], "canonical_index": index,
                   "seed": 20261016, **parsed, "generated_token_count": raw["generated_token_count"],
                   "end_reason": raw["end_reason"]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    saved = align(lines(path), identities)
    require(len(saved) == COUNT and all(p["cell_key"] == key and p["seed"] == 20261016 and
            all(p.get(k) == v for k, v in parsed.items()) and
            p["generated_token_count"] == raw["generated_token_count"] and p["end_reason"] == raw["end_reason"]
            for p, raw, parsed in zip(saved, value["raw"], value["auxiliary"])), "K3 saved snapshot differs")
    common = {"authorization": "2026-10-18 complete K3 interim extension",
              "source_plan": str(args.plan), "cell_key": key, "record_count": COUNT,
              "predictions_path": str(path), "parser_version": PARSER_VERSION,
              "prior_interim": str(args.prior_interim), "prior_gold_access": True}
    closure_path = args.output_dir / "interim_closure.json"
    closure = {**common, "status": "K3_PREDICTION_CLOSED_BEFORE_REUSED_GOLD_READ",
               "closed_at_utc": now(), "prediction_closed": True, "cells": cells}
    write(closure_path, closure)
    require(read(closure_path) == closure, "K3 closure readback differs")
    receipt = read(args.prior_interim / "gold_adapter_receipt.json")
    require(receipt["status"] == "GOLD_PROJECTED_EXACT_JOIN_SEALED" and receipt["joined_count"] == COUNT
            and receipt["source_path"] == str(SOURCE) and receipt["source_revision"] == REVISION
            and receipt["gold_path"] == str(args.prior_interim / "sealed/gold.jsonl"), "prior gold binding differs")
    started = now()
    write(args.output_dir / "pre_reused_gold_access.json", {**common, "timestamp_utc": started,
          "source_closure": str(closure_path), "closure_closed_at_utc": closure["closed_at_utc"],
          "first_gold_access": False, "gold_path": receipt["gold_path"]})
    gold = load_gold(receipt["gold_path"], "p11_jsonl", pool, identities)
    ended = now()
    write(args.output_dir / "post_reused_gold_access.json", {**common, "timestamp_utc": ended,
          "first_gold_access": False, "joined_count": len(gold), "gold_path": receipt["gold_path"]})
    correct = sum(p["extraction_status"] == "ok" and p["prediction_index"] == g for p, g in zip(saved, gold))
    counts = Counter(p["extraction_status"] for p in saved)
    prior = read(args.prior_interim / "aggregate.json")
    require(prior["cell_keys"] == ["native", "current_t_k2_l05"] and
            all(r["n"] == COUNT for r in prior["cells"]), "prior two-cell aggregation differs")
    row = {"cell_key": key, "correct": correct, "n": COUNT, "accuracy_percent": 100 * correct / COUNT,
           "delta_native_pp": 100 * correct / COUNT - prior["cells"][0]["accuracy_percent"],
           "format_failure_count": COUNT - counts["ok"],
           "format_counts": {k: counts[k] for k in ("ok", "no_match", "conflict", "out_of_range")}}
    result = {**common, "status": "K3_INTERIM_SCORED_NOT_GATE_TERMINAL", "source_closure": str(closure_path),
              "reused_gold_read_started_at_utc": started, "reused_gold_read_completed_at_utc": ended,
              "first_gold_projection_started_after_utc": prior["first_gold_projection_started_after_utc"],
              "scored_at_utc": now(), "cells": prior["cells"] + [row], "loop_outcomes_read": False}
    write(args.output_dir / "aggregate.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "prior-interim", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
