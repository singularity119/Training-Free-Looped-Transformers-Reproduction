#!/usr/bin/env python3
"""CPU-only D-2: persist all frozen S1 predictions before reading GPQA gold."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path

from tflt.loopscope.phase11_accuracy import extract_final_answer, termination
from tflt.loopscope.phase11_analysis import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, FAMILY_COUNTS, frozen_contrasts, gold_indices,
)
from tflt.loopscope.phase8_accuracy_stats import holm_adjust, paired_contrast
from tflt.loopscope.phase11_posthoc import S1_PATTERN, extract_s1, score_s1


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("closure", "manifest", "gold", "s0-analysis", "output-dir", "executor-commit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    closure_bytes = Path(args.closure).read_bytes()
    s0_bytes = Path(args.s0_analysis).read_bytes()
    closure, panel = json.loads(closure_bytes), read_json(args.manifest)
    assert closure["dataset"] == panel["dataset"] == "gpqa_main"
    assert closure["status"] == "PANEL_CLOSED_GOLD_UNREAD"
    assert closure["sample_count"] == 448 and closure["independent_cell_count"] == 18
    assert closure["cells"] == panel["cells"]
    specs = frozen_contrasts(panel)
    identities = closure["identities"]
    assert len(identities) == len(set(identities)) == 448
    assert set(closure["raw_roots"]) == {c["cell_id"] for c in panel["cells"]}
    allowed = Path(args.closure).parents[1]
    s0_status, s1_status, transition = Counter(), Counter(), Counter()
    common = agreed = retained_c = retained_d = 0
    labels, categories = {}, {}
    path = output / "predictions.jsonl"
    with path.open("x", encoding="utf-8") as saved:
        for cell in panel["cells"]:
            cell_id = cell["cell_id"]
            by_index = {}
            for facts in closure["raw_roots"][cell_id]:
                root = Path(facts["run_root"])
                assert root.is_relative_to(allowed / "runs")
                assert root.name.startswith(("phase11-gate-c-", "phase11-gate-d-"))
                assert facts["cell_id"] == cell_id
                indices = []
                with (root / "records.jsonl").open(encoding="utf-8") as handle:
                    for line in handle:
                        rec = json.loads(line)
                        i = rec["canonical_index"]
                        assert i not in by_index and 0 <= i < 448
                        assert rec["identity"] == identities[i] and rec["status"] == "OK"
                        assert rec["candidate_labels"] == list("ABCD")
                        ident = rec["identity"]
                        if ident in categories:
                            assert categories[ident] == rec["category"]
                        labels[ident] = list("ABCD")
                        categories[ident] = rec["category"]
                        s0 = extract_final_answer(rec["generated_text"], rec["candidate_labels"])
                        assert all(rec[k] == v for k, v in s0.items())
                        stop = termination(rec["generated_token_ids"], rec["eos_token_ids"], rec["max_new_tokens"])
                        assert all(rec[k] == v for k, v in stop.items())
                        s1 = extract_s1(rec["generated_text"])
                        s0_status[s0["extraction_status"]] += 1
                        s1_status[s1["extraction_status"]] += 1
                        transition[s0["extraction_status"] + " -> " + s1["extraction_status"]] += 1
                        if s0["prediction_index"] is not None and s1["prediction_index"] is not None:
                            common += 1
                            agreed += s0["prediction_index"] == s1["prediction_index"]
                        by_index[i] = {"cell_id": cell_id, "identity": ident, "canonical_index": i,
                                       "end_reason": rec["end_reason"], "s0": s0, **s1}
                        indices.append(i)
                        if root.name.startswith("phase11-gate-c-"):
                            retained_c += 1
                        else:
                            retained_d += 1
                assert sorted(indices) == facts["indices"] and len(indices) == facts["records_count"]
            assert set(by_index) == set(range(448))
            for i in range(448):
                saved.write(json.dumps(by_index[i], ensure_ascii=False) + "\n")
    assert sum(s0_status.values()) == sum(s1_status.values()) == 8064
    assert retained_c == 72 and retained_d == 7992
    barrier = {"status": "ALL_S1_PREDICTIONS_SAVED_GOLD_UNREAD", "timestamp": now(),
               "record_count": 8064, "independent_cells": 18, "samples_per_cell": 448,
               "pattern": S1_PATTERN, "flags": "re.IGNORECASE", "line_processing": "splitlines then strip then fullmatch",
               "ambiguity": "exactly one matching line; duplicates always fail", "gold_read": False,
               "s0_status_counts": dict(s0_status), "s1_status_counts": dict(s1_status),
               "status_transitions": dict(transition), "common_valid_count": common,
               "common_valid_prediction_agreement_count": agreed,
               "retained_c_records": retained_c, "retained_d_records": retained_d}
    write_json(output / "extraction_barrier.json", barrier)
    print(json.dumps({"status": barrier["status"], "record_count": 8064}), flush=True)
    # No gold access occurs above this persisted, complete-population barrier.
    with Path(args.gold).open(encoding="utf-8") as handle:
        gold_rows = [json.loads(line) for line in handle if line.strip()]
    gold_read_time = now()
    gold = gold_indices(gold_rows, identities, labels, categories)
    by_cell = {c["cell_id"]: [] for c in panel["cells"]}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_cell[row["cell_id"]].append(row)
    old = json.loads(s0_bytes)
    old_cells = {c["cell_id"]: c for c in old["cells"]}
    cells, correctness = [], {}
    for cell in panel["cells"]:
        cid, rows = cell["cell_id"], by_cell[cell["cell_id"]]
        assert [r["identity"] for r in rows] == identities
        stats, correctness[cid] = score_s1(rows, gold)
        s0_correct = sum(r["s0"]["prediction_index"] is not None and r["s0"]["prediction_index"] == g
                         for r, g in zip(rows, gold))
        assert old_cells[cid]["correct"] == s0_correct and old_cells[cid]["n"] == 448
        assert old_cells[cid]["extraction_failure_count"] == sum(r["s0"]["prediction_index"] is None for r in rows)
        assert old_cells[cid]["truncated_count"] == stats["truncated_count"]
        cells.append({**cell, **stats, "s0_correct": s0_correct, "s0_accuracy": s0_correct / 448})
    contrasts = [{**spec, **paired_contrast(correctness[spec["reference"]], correctness[spec["treatment"]],
                  ["all_items"] * 448, bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)}
                 for spec in specs]
    for family, count in FAMILY_COUNTS.items():
        rows = [r for r in contrasts if r["family"] == family]
        assert len(rows) == count
        for row, p in zip(rows, holm_adjust([r["mcnemar_exact_p"] for r in rows])):
            row.update(holm_adjusted_p=p, holm_family_size=count, holm_reject_alpha_0_05=p <= 0.05)
    native = next(c for c in cells if c["arm"] == "Native")
    loops = {c["k"]: c for c in cells if c["arm"] == "Loop"}
    for cell in cells:
        cell["delta_native_pp"] = 100 * (cell["accuracy"] - native["accuracy"])
        cell["delta_loop_pp"] = None if cell["k"] is None else 100 * (cell["accuracy"] - loops[cell["k"]]["accuracy"])
    groups = {reason: {key: sum(c["groups"][reason][key] for c in cells)
                       for key in ("n", "extraction_success_count", "correct")}
              for reason in ("eos", "length_limit")}
    assert Path(args.closure).read_bytes() == closure_bytes
    assert Path(args.s0_analysis).read_bytes() == s0_bytes
    report = {"schema": "loopscope.phase11.gpqa_posthoc.v1", "gate": "D-2",
              "status": "POSTHOC_ANALYSIS_COMPLETE_AUDIT_REQUESTED", "executor": "01a124db-33fa-7322-9317-12e53d3d3861/local",
              "executor_commit": args.executor_commit, "source_commits": closure["source_commits"],
              "paths": vars(args), "extraction_barrier": barrier, "gold_read_timestamp": gold_read_time,
              "sample_count": 448, "independent_cell_count": 18, "record_count": 8064,
              "cells": cells, "contrasts": contrasts, "groups": groups,
              "families": FAMILY_COUNTS, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
              "s0_unchanged_direct_reread": True, "s0_scores_reproduced": True,
              "interpretation": {"scope": "post-hoc exploratory extraction sensitivity",
                  "selection_limit": "S1 was proposed after seeing format diagnostics; Holm does not undo post-hoc parser selection",
                  "ci": "nominal paired bootstrap 95 percent; Holm within original 30/12/2 families",
                  "denominator": "448 per cell; extraction failures count as wrong",
                  "groups": "EOS and length groups are descriptive selected subsets; counts are repeated outputs, not independent questions",
                  "limits": "No matched-norm control, no direction-specific causal or transfer claim; no new model generation"},
              "gpu_jobs_submitted": False, "mmlu_accessed": False, "gate_pass_claimed": False}
    write_json(output / "analysis.json", report)
    print(json.dumps({"status": report["status"], "output_dir": str(output),
                      "holm_rejections": sum(r["holm_reject_alpha_0_05"] for r in contrasts)}))


if __name__ == "__main__":
    main()
