#!/usr/bin/env python3
"""One authorized two-cell interim snapshot; never relax the full-panel CLI."""
import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_phase11_d7 import (COUNT, align, lines, load_raw_cells, now, read,
                               require, validate_cells, write)
from prepare_phase11_d7_gold import SOURCE, PROJECTION, REVISION, read_projection, join_gold
from tflt.loopscope.phase11_mmlupro_s2 import PARSER_VERSION

CELLS = ("native", "current_t_k2_l05")
AUTHORIZATION = "2026-10-18 user single interim exception: Native and current_t K2 lambda0.5 only"


def verify_saved(saved, data, identities):
    require(len(saved) == COUNT * 2 and {r["cell_key"] for r in saved} == set(CELLS),
            "interim snapshot requires exactly two complete cells")
    for cell in CELLS:
        rows = align([r for r in saved if r["cell_key"] == cell], identities)
        for row, raw, parsed in zip(rows, data[cell]["raw"], data[cell]["auxiliary"]):
            require(row["seed"] == 20261016 and all(row.get(k) == v for k, v in parsed.items())
                    and row["generated_token_count"] == raw["generated_token_count"]
                    and row["end_reason"] == raw["end_reason"], "interim frozen prediction differs")


def score_saved(saved, gold, identities):
    labels = {r["identity"]: r["label_index"] for r in gold}
    require(len(labels) == COUNT and set(labels) == set(identities), "interim gold identity join differs")
    result = []
    for cell in CELLS:
        rows = align([r for r in saved if r["cell_key"] == cell], identities)
        counts = Counter(r["extraction_status"] for r in rows)
        correct = sum(r["extraction_status"] == "ok" and r["prediction_index"] == labels[r["identity"]]
                      for r in rows)
        result.append({"cell_key": cell, "correct": correct, "n": COUNT,
                       "accuracy_percent": 100 * correct / COUNT,
                       "format_counts": {key: counts[key] for key in ("ok", "no_match", "conflict", "out_of_range")},
                       "format_failure_count": COUNT - counts["ok"]})
    for row in result:
        row["delta_native_pp"] = row["accuracy_percent"] - result[0]["accuracy_percent"]
    return result


def run(args):
    plan = read(args.plan)
    cells = [c for c in validate_cells(plan, first=True) if c["cell_key"] in CELLS]
    require(len(cells) == 2, "interim exact two cells absent")
    # This reads/parses only the authorized two cells, including retained canary.
    data, pool, identities = load_raw_cells(cells)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = args.output_dir / "predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8") as handle:
        for cell in cells:
            value = data[cell["cell_key"]]
            for index, (raw, parsed) in enumerate(zip(value["raw"], value["auxiliary"])):
                row = {"cell_key": cell["cell_key"], "identity": identities[index],
                       "canonical_index": index, "seed": cell["seed"], **parsed,
                       "generated_token_count": raw["generated_token_count"], "end_reason": raw["end_reason"]}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    saved = lines(prediction_path)
    verify_saved(saved, data, identities)
    del data
    common = {"authorization": AUTHORIZATION, "source_plan": str(args.plan),
              "record_count": COUNT * 2, "cell_count": 2, "cell_keys": list(CELLS),
              "parser_version": PARSER_VERSION, "predictions_path": str(prediction_path),
              "producer_sources": {c["cell_key"]: c["source_commit"] for c in cells}}
    closure = {**common, "schema": "loopscope.phase11.d7.interim_prediction_closure.v1",
               "status": "INTERIM_TWO_CELLS_CLOSED_GOLD_UNREAD", "target_gold_loaded": False,
               "prediction_closed": True, "closed_at_utc": now(),
               "cells": [{**c, "record_count": COUNT} for c in cells]}
    closure_path = args.output_dir / "interim_closure.json"
    write(closure_path, closure)
    require(read(closure_path) == closure, "interim closure readback differs")
    access = {**common, "source_closure": str(closure_path), "source_path": str(SOURCE),
              "source_revision": REVISION, "projection_columns": list(PROJECTION),
              "closure_closed_at_utc": closure["closed_at_utc"],
              "first_access_context": "USER_AUTHORIZED_INTERIM_BEFORE_FULL_PANEL"}
    started = now()
    write(args.output_dir / "pre_gold_access.json", {**access, "timestamp_utc": started,
          "target_gold_read": False, "status": "INTERIM_CLOSURE_VERIFIED_BEFORE_FIRST_GOLD_ACCESS"})
    try:
        projected = read_projection(SOURCE)
        completed = now()
        write(args.output_dir / "post_gold_access.json", {**access, "target_gold_read": True,
              "status": "FIRST_GOLD_PROJECTION_READ_INTERIM_EXCEPTION",
              "projection_started_after_utc": started, "projection_completed_at_utc": completed,
              "projected_row_count": len(projected)})
        gold = join_gold(projected, pool, identities)
        sealed = args.output_dir / "sealed"
        sealed.mkdir(exist_ok=False)
        gold_path = sealed / "gold.jsonl"
        with gold_path.open("x", encoding="utf-8") as handle:
            for row in gold:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write(args.output_dir / "gold_adapter_receipt.json", {**access, "target_gold_read": True,
              "status": "GOLD_PROJECTED_EXACT_JOIN_SEALED", "joined_count": len(gold),
              "missing_count": 0, "duplicate_count": 0, "category_match": True,
              "option_range_valid": True, "gold_path": str(gold_path), "joined_at_utc": now()})
        result = {**common, "status": "INTERIM_TWO_CELL_SCORED_NOT_GATE_TERMINAL",
                  "target_gold_loaded": True, "source_closure": str(closure_path),
                  "first_gold_projection_started_after_utc": started,
                  "first_gold_projection_completed_at_utc": completed, "scored_at_utc": now(),
                  "cells": score_saved(saved, gold, identities),
                  "remaining_cell_outcomes_read": False,
                  "full_panel_first_gold_claim_permitted": False}
        write(args.output_dir / "aggregate.json", result)
    except Exception as error:
        write(args.output_dir / "failure.json", {**access, "timestamp_utc": now(),
              "gold_read_attempted": True, "error_type": type(error).__name__})
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
