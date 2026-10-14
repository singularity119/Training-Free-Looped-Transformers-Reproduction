import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("d5_analysis_test", ROOT / "scripts/loopscope/analyze_phase11_d5.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def synthetic():
    data, gold = {}, {}
    for family, ids in analysis.MEMBERS.items():
        pool = [{"subject": "subject-a", "choices": ["aa", "bbbb", "c", "d"]},
                {"subject": "subject-b", "choices": ["a", "b", "c", "d"]}]
        versions = {}
        for repeat in (0, 1, 2):
            cells = {}
            for cid in ids:
                if family.endswith("gpqa"):
                    raw = [{"generated_text": "Final answer: (A)", "generated_token_ids": [10, 11],
                            "generated_token_count": 2, "end_reason": "eos"},
                           {"generated_text": "Final answer: (B)", "generated_token_ids": [12, 13],
                            "generated_token_count": 2, "end_reason": "eos"}]
                else:
                    raw = [{"scores": [-1., -4., -3., -4.]}, {"scores": [-2., -1., -3., -4.]}]
                predictions, aux = analysis.predict(family, raw, pool)
                cells[cid] = {"raw": raw, "predictions": predictions, "auxiliary": aux}
            versions[repeat] = cells
        data[family] = {"pool_rows": pool, "identities": ["i0", "i1"], "versions": versions}
        gold[family] = [0, 1]
    return data, gold


class D5AnalysisTests(unittest.TestCase):
    def test_strict_gold_barrier_is_checked_before_any_other_read(self):
        with patch.object(analysis, "read", return_value={"status": "GOLD_UNREAD"}) as read, \
             patch.object(analysis, "load_gold", side_effect=AssertionError("gold opened")) as gold:
            with self.assertRaisesRegex(ValueError, "102796"):
                analysis.run(SimpleNamespace(closure="closure.json", reference_binding="binding.json", output_dir="unused"))
            self.assertEqual(read.call_count, 1)
            gold.assert_not_called()

    def test_all_20_and_10_contrasts_keep_repeat_denominators_and_explicit_seeds(self):
        data, gold = synthetic()
        result = analysis.analyze(data, gold, replicates=100)
        self.assertEqual(len(result["cells"]), 42)
        self.assertEqual(len(result["reproduction"]), 42)
        self.assertEqual(len(result["contrasts"]), 30)
        for row in result["contrasts"]:
            self.assertEqual(row["n"], 2)
            self.assertEqual(row["holm_family_size"], 20 if row["statistical_family"] == "MAIN_20" else 10)
            self.assertEqual(row["bootstrap"]["seed"], 20261002 if row["family"].startswith("p10-") else 20261008)
        base_arc = [r for r in result["cells"] if r["family"] == "p10-base-arc" and "online" in r["cell_id"]]
        self.assertTrue(all(r["delta_loop_pp"] == 0 for r in base_arc))

    def test_arc_normalizes_character_lengths_and_preserves_raw_acc(self):
        pool = [{"choices": ["a", "long text"]}]
        raw = [{"scores": [-1., -2.], "choice_text_lengths": [1, 9]}]
        pred, aux = analysis.predict("p10-base-arc", raw, pool)
        self.assertEqual(pred, [1])
        self.assertEqual(aux, [{"raw_prediction_index": 0}])
        with self.assertRaisesRegex(ValueError, "normalization"):
            analysis.predict("p10-base-arc", [{"scores": [-1., -2.], "choice_text_lengths": [1, 2]}], pool)

    def test_same_accuracy_does_not_imply_same_predictions_or_scores(self):
        data, gold = synthetic()
        f, cid = "p10-base-mmlu", analysis.MEMBERS["p10-base-mmlu"][0]
        a, b = copy.deepcopy(data[f]["versions"][0][cid]), copy.deepcopy(data[f]["versions"][1][cid])
        a["predictions"], b["predictions"] = [0, 0], [1, 1]
        b["raw"][0]["scores"][0] += 1e-6
        row = analysis.reproduce(f, cid, a, b, gold[f], (0, 1))
        self.assertTrue(row["same_correct_total"])
        self.assertFalse(row["strict_prediction_identity"])
        self.assertEqual(row["prediction_changed_count"], 2)
        self.assertEqual(row["raw_scores_strict_same_count"], 1)
        self.assertEqual(row["raw_scores_close_count"], 2)
        self.assertEqual((row["wrong_to_right"], row["right_to_wrong"]), (1, 1))

    def test_gpqa_conflicting_s2_is_wrong_and_token_status_changes_reported(self):
        data, gold = synthetic()
        f, cid = "p11-instruct-gpqa", analysis.MEMBERS["p11-instruct-gpqa"][0]
        b = data[f]["versions"][1][cid]
        b["raw"][0].update(generated_text="Final answer: (A)\nFinal answer: (B)",
                           generated_token_ids=[99], generated_token_count=1, end_reason="length_limit")
        b["predictions"], b["auxiliary"] = analysis.predict(f, b["raw"], data[f]["pool_rows"])
        self.assertIsNone(b["predictions"][0])
        result = analysis.analyze(data, gold, replicates=100)
        row = next(r for r in result["reproduction"] if r["family"] == f and r["cell_id"] == cid and (r["from"], r["to"]) == (0, 1))
        self.assertEqual(row["token_sequence_same_count"], 1)
        self.assertEqual(row["length_same_count"], 1)
        self.assertEqual(row["end_reason_same_count"], 1)
        self.assertEqual(row["S2_answer_mode_same_count"], 1)
        self.assertEqual(row["right_to_wrong"], 1)

    def test_gold_join_order_and_identity_are_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            pool = [{"choices": ["x", "y"], "choice_labels": ["A", "B"], "subject": "s"}]
            base_path = Path(temp) / "gold.json"
            base_path.write_text('{"identities":["id"],"gold":[1]}')
            self.assertEqual(analysis.load_gold({"format": "base_aligned_json", "gold": str(base_path)}, pool, ["id"]), [1])
            with self.assertRaisesRegex(ValueError, "identity/order"):
                analysis.load_gold({"format": "base_aligned_json", "gold": str(base_path)}, pool, ["other"])
            p11_path = Path(temp) / "gold.jsonl"
            p11_path.write_text('{"identity":"id","gold_letter":"B","category":"s"}\n')
            self.assertEqual(analysis.load_gold({"format": "p11_jsonl", "gold": str(p11_path)}, pool, ["id"]), [1])

    def test_old_full_grid_p_is_a_side_column_not_the_new_adjustment(self):
        data, gold = synthetic()
        result = analysis.analyze(data, gold, replicates=100)
        binding = {"families": {f: {"historical_analysis": f + ".json", "refs": {
            cid: {"reference0_runtime_seed": 20260905 if f.endswith("mmlu") and cid in analysis.MEMBERS[f][:2]
                  else 20261002 if f.startswith("p10-") else 20261008} for cid in ids}}
            for f, ids in analysis.MEMBERS.items()}}
        original = [r["holm_adjusted_p"] for r in result["contrasts"]]
        old = {}
        for family in analysis.MEMBERS:
            rows = []
            for row in result["contrasts"]:
                if row["family"] == family and row["repeat"] == 1:
                    group = ("LAMBDA_SCAN_126" if family.startswith("p10-") else
                        "ONLINE_VS_NATIVE_AND_LOOP_EXPLORATORY" if row["statistical_family"] == "MAIN_20" else "LOOP_VS_NATIVE_BASE")
                    rows.append({"reference": row["reference"], "treatment": row["treatment"],
                                 "family": group, "holm_adjusted_p": .123, "holm_family_size": 126})
                    if family == "p10-base-arc":
                        rows.append({**rows[-1], "family": "ARC_DIRECT_TRANSFER_14", "holm_adjusted_p": .456})
            old[family] = {"s2": {"contrasts": rows}, "contrasts": [{"wrong_parser": True}]} if family.endswith("gpqa") else {"contrasts": rows}
        analysis.attach_historical(result, binding, old)
        self.assertEqual([r["holm_adjusted_p"] for r in result["contrasts"]], original)
        for row in result["contrasts"]:
            if row["family"].startswith("p10-") and row["statistical_family"] == "SECONDARY_10":
                self.assertIsNone(row["old_holm_adjusted_p"])
                self.assertIn("did not include", row["old_exploratory_match_reason"])
            else:
                self.assertEqual(row["old_holm_adjusted_p"], .123)
        native = next(r for r in result["cells"] if r["family"] == "p10-base-mmlu" and r["cell_id"] == "q4-native" and r["repeat"] == 0)
        self.assertEqual(native["runtime_seed"], 20260905)

    def test_historical_match_missing_or_reversed_is_explicit(self):
        data, gold = synthetic()
        result = analysis.analyze(data, gold, replicates=100)
        binding = {"families": {f: {"refs": {cid: {} for cid in ids}} for f, ids in analysis.MEMBERS.items()}}
        first = next(r for r in result["contrasts"] if r["statistical_family"] == "MAIN_20")
        old = {first["family"]: {"contrasts": [{"reference": first["treatment"], "treatment": first["reference"],
            "family": "LAMBDA_SCAN_126", "holm_adjusted_p": .02}]}}
        analysis.attach_historical(result, binding, old)
        self.assertIsNone(first["old_holm_adjusted_p"])
        self.assertIn("no exact directed", first["old_exploratory_match_reason"])


if __name__ == "__main__":
    unittest.main()
