import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tflt.loopscope.phase11_d6_analysis import CELLS, analyze, paired_mean_bootstrap, sign_flip
from tflt.loopscope.phase11_s2 import extract_s2

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("d6_analysis_test", ROOT / "scripts/loopscope/analyze_phase11_d6.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def data_for(n=4):
    data = {}
    for cid in CELLS:
        raw = [{"generated_text": "Final answer: (A)", "generated_token_count": 2 + i,
                "end_reason": "eos" if i % 2 else "length_limit"} for i in range(n)]
        aux = [extract_s2(row["generated_text"]) for row in raw]
        data[cid] = {"raw": raw, "predictions": [p["prediction_index"] for p in aux], "auxiliary": aux}
    return data


def fixture(directory):
    root = Path(directory)
    rows = [{"identity": f"id{i}", "subject": "gpqa_main", "choices": ["a", "b", "c", "d"],
             "choice_labels": list("ABCD")} for i in range(448)]
    pool = root / "pool.json"
    pool.write_text(json.dumps({"rows": rows}))
    cells = []
    for cid in CELLS:
        cell = {"cell_key": cid, "cell_id": "base-native" if cid.endswith("native") else "base-loop",
                "prompt_mode": cid.split("_")[0], "seed": 20261015, "source_commit": "immutable-source",
                "damping_scope": "prefill_only" if cid.endswith("prefill") else "full" if cid.endswith("full") else "none",
                "population_count": 448, "synthetic": False, "engineering": False, "pool": str(pool)}
        path = root / (cid + ".jsonl")
        binding = {k: cell[k] for k in cli.BINDING}
        records = [{"identity": row["identity"], "canonical_index": i, "candidate_labels": list("ABCD"),
                    "d6_binding": binding, "generated_text": "Final answer: (A)", "generated_token_ids": [1, 2],
                    "generated_token_count": 2, "end_reason": "eos"} for i, row in enumerate(rows)]
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        cell["raw_roots"] = [{"records_path": str(path)}]
        cells.append(cell)
    return {"schema": "loopscope.phase11.d6.plan.v1", "record_count": 5376,
            "target_gold_loaded": False, "cells": cells}


class D6AnalysisTests(unittest.TestCase):
    def test_frozen_families_denominators_and_length_summaries(self):
        result = analyze(data_for(), [0] * 4, replicates=100, sign_flip_draws=100)
        self.assertEqual(len(result["cells"]), 12)
        self.assertEqual(len(result["primary_contrasts"]), 4)
        self.assertEqual(len(result["auxiliary_contrasts"]), 10)
        self.assertEqual(len(result["length_contrasts"]), 6)
        self.assertEqual(len(result["cross_prompt_descriptive"]), 5)
        for group, size in (("primary_contrasts", 4), ("auxiliary_contrasts", 10)):
            for row in result[group]:
                self.assertEqual(row["holm_family_size"], size)
                self.assertEqual(row["n"], 4)
                self.assertEqual(row["bootstrap"]["seed"], 20261008)
                self.assertEqual(row["holm_adjusted_p"], 1)
        row = result["cells"][0]
        self.assertEqual(row["generated_tokens_mean"], 3.5)
        self.assertEqual(row["generated_tokens_median"], 3.5)
        self.assertAlmostEqual(row["generated_tokens_p90"], 4.7)
        self.assertEqual((row["eos_count"], row["length_limit_count"]), (2, 2))
        self.assertEqual(row["eos_only_tokens_mean"], 4)

    def test_interaction_is_four_arm_difference_and_ci_allows_two(self):
        data = data_for()
        data["cot_loop_k2"]["predictions"] = [1] * 4
        data["direct_current_k2_full"]["predictions"] = [1] * 4
        result = analyze(data, [0] * 4, replicates=100, sign_flip_draws=100)
        row = result["primary_contrasts"][0]
        self.assertEqual(row["delta_pp"], 200)
        self.assertEqual(row["bootstrap"]["ci_low_pp"], 200)
        self.assertEqual(row["bootstrap"]["ci_high_pp"], 200)
        self.assertNotIn("mcnemar_exact_p", row)
        self.assertEqual(result["primary_contrasts"][2]["reference"], "cot_current_k2_prefill")
        self.assertEqual(result["primary_contrasts"][2]["treatment"], "cot_current_k2_full")

    def test_sign_flip_plus_one_two_sided_tail_and_fixed_draws(self):
        import numpy as np
        draws = 100000
        expected_tail = 0
        rng = np.random.default_rng(20261008)
        for _ in range(100):
            signs = 2 * rng.integers(0, 2, size=(1000, 2), dtype=np.int8) - 1
            expected_tail += int((np.abs(signs.sum(axis=1)) >= 2).sum())
        result = sign_flip([1, 1])
        self.assertEqual(result["draws"], draws)
        self.assertEqual(result["tail_count"], expected_tail)
        self.assertEqual(result["p_value"], (1 + expected_tail) / 100001)
        self.assertEqual(sign_flip([0, 0], draws=20)["p_value"], 1)

    def test_length_ci_preserves_paired_difference(self):
        ci = paired_mean_bootstrap([-100, 100], scale=1., unit="tokens", replicates=1000)
        self.assertEqual(ci["ci_low_tokens"], -100)
        self.assertEqual(ci["ci_high_tokens"], 100)

    def test_conflict_and_no_match_remain_wrong_on_full_denominator(self):
        data = data_for()
        value = data["cot_native"]
        value["raw"][0]["generated_text"] = "Final answer: (A)\nFinal answer: (B)"
        value["raw"][1]["generated_text"] = "A"
        value["auxiliary"] = [extract_s2(r["generated_text"]) for r in value["raw"]]
        value["predictions"] = [p["prediction_index"] for p in value["auxiliary"]]
        result = analyze(data, [0] * 4, replicates=100, sign_flip_draws=100)
        row = result["cells"][0]
        self.assertEqual((row["correct"], row["n"]), (2, 4))
        self.assertEqual((row["S2_success_count"], row["S2_no_match_count"], row["S2_conflict_count"]), (2, 1, 1))

    def test_incomplete_closure_stops_before_gold(self):
        args = SimpleNamespace(close=False, closure="closure", gold="never", plan=None, output_dir="unused")
        with patch.object(cli, "read", return_value={"schema": "loopscope.phase11.d6.raw_prediction_closure.v1",
                                                    "record_count": 5375}), \
             patch.object(cli, "load_gold", side_effect=AssertionError("gold read")) as gold:
            with self.assertRaisesRegex(ValueError, "5376"):
                cli.run(args)
            gold.assert_not_called()

    def test_complete_gold_free_closure_then_raw_prediction_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = fixture(directory)
            with patch.object(cli, "load_gold", side_effect=AssertionError("gold read")) as gold:
                closure, predictions = cli.close(plan)
                gold.assert_not_called()
            self.assertEqual(len(predictions), 5376)
            self.assertEqual(closure["status"], "GOLD_UNREAD")
            self.assertNotIn("generated_text", predictions[0])
            path = Path(directory) / "predictions.jsonl"
            path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
            closure["predictions_path"] = str(path)
            data, _, ids = cli.load_before_gold(closure)
            self.assertEqual(len(ids), 448)
            self.assertEqual(len(data), 12)
            predictions[0]["prediction_index"] = 1
            path.write_text("".join(json.dumps(p) + "\n" for p in predictions))
            with self.assertRaisesRegex(ValueError, "frozen S2"):
                cli.load_before_gold(closure)

    def test_duplicate_identity_and_wrong_scope_fail_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = fixture(directory)
            wrong = copy.deepcopy(plan)
            wrong["cells"][-1]["damping_scope"] = "full"
            with self.assertRaisesRegex(ValueError, "prompt/scope"):
                cli.close(wrong)
            path = Path(plan["cells"][0]["raw_roots"][0]["records_path"])
            records = cli.lines(path)
            records[-1] = records[0]
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            with self.assertRaisesRegex(ValueError, "duplicated"):
                cli.close(plan)


if __name__ == "__main__":
    unittest.main()
