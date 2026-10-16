import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tflt.loopscope.phase11_d7_analysis import CELLS, FIRST_BATCH, analyze
from tflt.loopscope.phase11_mmlupro_s2 import extract_mmlupro_s2

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("d7_analysis_test", ROOT / "scripts/loopscope/analyze_phase11_d7.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def data_for(n=4):
    data = {}
    for cid in CELLS:
        raw = [{"generated_text": "Final answer: E", "generated_token_count": i + 2,
                "end_reason": "eos" if i % 2 else "length_limit"} for i in range(n)]
        aux = [extract_mmlupro_s2(r["generated_text"], 5) for r in raw]
        data[cid] = {"raw": raw, "predictions": [p["prediction_index"] for p in aux], "auxiliary": aux}
    return data


def fixture(directory, n=14):
    root = Path(directory)
    pool = root / "pool.json"
    rows = [{"identity": f"synthetic{i}", "subject": f"cat{i % 14}", "choices": ["a"] * (4 + i % 7),
             "choice_labels": list("ABCDEFGHIJ"[:4 + i % 7])} for i in range(n)]
    pool.write_text(json.dumps({"rows": rows}))
    cells = []
    for cid in CELLS:
        cell = {"cell_key": cid, "cell_id": "mmlu-pro-" + cid, "seed": 20261016,
                "source_commit": "immutable", "prompt_mode": "cot",
                "damping_scope": "full" if cid in FIRST_BATCH[1:] else "none",
                "population_count": n, "synthetic": False, "engineering": False, "pool": str(pool)}
        binding = {k: cell[k] for k in cli.BINDING}
        records = [{"identity": row["identity"], "canonical_index": i, "category": row["subject"],
                    "candidate_labels": row["choice_labels"], "d7_binding": binding, "status": "OK",
                    "generated_text": "Final answer: A", "generated_token_ids": [1, 2],
                    "generated_token_count": 2, "max_new_tokens": 8192, "end_reason": "eos"}
                   for i, row in enumerate(rows)]
        path = root / (cid + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        cell["raw_roots"] = [{"records_path": str(path)}]
        cells.append(cell)
    return {"schema": "loopscope.phase11.d7.plan.v1", "record_count": n * 5,
            "target_gold_loaded": False, "cells": cells}


class D7AnalysisTests(unittest.TestCase):
    def test_frozen_six_contrasts_separate_holm_and_category_strata(self):
        result = analyze(data_for(), [4] * 4, ["small", "large", "large", "large"], replicates=100)
        self.assertEqual((len(result["cells"]), len(result["contrasts"])), (5, 6))
        self.assertEqual(len(result["categories"]), 10)
        for group, size in (("primary_contrasts", 4), ("baseline_contrasts", 2)):
            for r in result[group]:
                self.assertEqual((r["n"], r["holm_family_size"], r["holm_adjusted_p"]), (4, size, 1))
                self.assertEqual((r["bootstrap"]["subject_count"], r["bootstrap"]["seed"]), (2, 20261008))
        native = result["cells"][0]
        self.assertEqual((native["generated_tokens_mean"], native["generated_tokens_median"]), (3.5, 3.5))
        self.assertAlmostEqual(native["generated_tokens_p90"], 4.7)

    def test_micro_and_ci_retain_category_weights(self):
        data = data_for()
        data["native"]["predictions"] = [None] * 4
        data["current_t_k2_l05"]["predictions"] = [4, None, None, None]
        result = analyze(data, [4] * 4, ["small", "large", "large", "large"], replicates=200)
        r = result["primary_contrasts"][0]
        self.assertEqual(r["delta_pp"], 25)
        self.assertEqual(r["subject_macro_delta_pp"], 50)
        self.assertEqual((r["bootstrap"]["ci_low_pp"], r["bootstrap"]["ci_high_pp"]), (25, 25))

    def test_all_extraction_failures_remain_in_full_denominator(self):
        data = data_for()
        value = data["native"]
        texts = ["Final answer: E", "Final answer: F", "Final answer: A\nFinal answer: E", "E"]
        value["auxiliary"] = [extract_mmlupro_s2(t, 5) for t in texts]
        value["predictions"] = [p["prediction_index"] for p in value["auxiliary"]]
        result = analyze(data, [4] * 4, ["c"] * 4, replicates=100)
        r = result["cells"][0]
        self.assertEqual((r["correct"], r["n"]), (1, 4))
        self.assertEqual([r[k] for k in ("S2_success_count", "S2_out_of_range_count", "S2_conflict_count", "S2_no_match_count")], [1]*4)

    def test_incomplete_closure_stops_before_gold(self):
        args = SimpleNamespace(close=False, close_first_batch=False, closure="closure", gold="never",
                               plan=None, output_dir=Path("unused"))
        with patch.object(cli, "read", return_value={"schema": "loopscope.phase11.d7.raw_prediction_closure.v1",
                                                     "record_count": 60159}), \
             patch.object(cli, "load_gold", side_effect=AssertionError("gold read")) as gold:
            with self.assertRaisesRegex(ValueError, "60160"):
                cli.run(args)
            gold.assert_not_called()

    def test_synthetic_gold_free_closure_saved_prediction_readback_and_first_batch(self):
        # Tiny invented inputs exercise the identical closure path, with no real data.
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, "COUNT", 14):
            plan = fixture(directory)
            with patch.object(cli, "load_gold", side_effect=AssertionError("gold read")) as gold:
                receipt, _ = cli.close(plan, first=True)
                closure, predictions = cli.close(plan)
                gold.assert_not_called()
            self.assertEqual(receipt["cell_keys"], list(FIRST_BATCH))
            self.assertEqual((receipt["record_count"], len(receipt["cells"])), (42, 3))
            self.assertEqual(len(predictions), 70)
            self.assertNotIn("generated_text", predictions[0])
            path = Path(directory) / "predictions.jsonl"
            path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
            closure["predictions_path"] = str(path)
            data, _, ids = cli.load_before_gold(closure)
            self.assertEqual((len(data), len(ids)), (5, 14))
            predictions[0]["prediction_index"] = 1
            path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
            with self.assertRaisesRegex(ValueError, "frozen S2"):
                cli.load_before_gold(closure)

    def test_first_batch_does_not_parse_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, "COUNT", 14):
            plan = fixture(directory)
            with patch.object(cli, "extract_mmlupro_s2", side_effect=AssertionError("parsed")):
                cli.close(plan, first=True)
            bad = copy.deepcopy(plan)
            bad["cells"][1]["seed"] = 0
            with self.assertRaisesRegex(ValueError, "seed"):
                cli.close(bad)
            path = Path(plan["cells"][0]["raw_roots"][0]["records_path"])
            records = cli.lines(path)
            records[-1] = records[0]
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            with self.assertRaisesRegex(ValueError, "duplicated"):
                cli.close(plan)


if __name__ == "__main__":
    unittest.main()
