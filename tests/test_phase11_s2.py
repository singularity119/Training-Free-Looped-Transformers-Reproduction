from collections import Counter
import json
import unittest
from unittest.mock import patch

from tflt.loopscope import phase11_s2 as s2
from tflt.loopscope.phase11_panel import build_panel
from tflt.loopscope.phase11_posthoc import extract_s1


def row(index, text):
    return {"identity": f"synthetic:{index}", "canonical_index": index, "generated_text": text,
            "generated_token_count": 7, "end_reason": "eos", "candidate_labels": list("ABCD")}


def bootstrap(correct, subjects, replicates, seed):
    assert replicates == 10000 and seed == 20261008
    assert subjects == ["all_items"] * len(correct)
    point = 100 * sum(correct) / len(correct)
    return {"ci_low_pp": point, "ci_high_pp": point, "replicates": replicates, "seed": seed}


def contrast(reference, treatment, subjects, bootstrap_replicates, seed):
    assert bootstrap_replicates == 10000 and seed == 20261008
    assert len(reference) == len(treatment) == len(subjects) == 2
    assert subjects == ["all_items", "all_items"]
    return {"n": 2, "delta_pp": 50 * (sum(treatment) - sum(reference)), "mcnemar_exact_p": 1}


class S2Tests(unittest.TestCase):
    def test_single_zero_and_whole_line_syntax(self):
        for text in ("Final answer: A", "reasoning\nFinal answer: (A)\n", " FINAL\tAnSwEr ： ( a )。 "):
            result = s2.extract_s2(text)
            self.assertEqual(result["answer_mode"], "single")
            self.assertEqual(result["extraction_status"], "unique_valid")
            self.assertEqual(result["prediction_index"], 0)
        for text in ("the final answer is A", "**Final answer: A**", "Final answer: A because ...",
                     "\\boxed{A}", "Final answer: E", "Final answer: A，", "Final\nanswer: A"):
            result = s2.extract_s2(text)
            self.assertEqual(result["answer_mode"], "no_match")
            self.assertIsNone(result["prediction_index"])

    def test_consistent_multiple_normalized_without_changing_s1(self):
        text = "Final answer: (C)\nfinal answer ： c！\nFINAL ANSWER: ( c )."
        result = s2.extract_s2(text)
        self.assertEqual(result["answer_mode"], "consistent_multiple")
        self.assertEqual(result["extraction_status"], "unique_valid")
        self.assertEqual(result["predicted_label"], "C")
        self.assertEqual(result["prediction_index"], 2)
        self.assertEqual(result["valid_final_line_count"], 3)
        self.assertEqual(result["distinct_valid_label_count"], 1)
        self.assertIsNone(extract_s1(text)["prediction_index"])

    def test_conflicting_multiple_fails_regardless_of_majority(self):
        result = s2.extract_s2("Final answer: A\nFinal answer: (a)\nFinal answer: B")
        self.assertEqual(result["answer_mode"], "conflicting_multiple")
        self.assertEqual(result["extraction_status"], "multiple_valid_final_lines")
        self.assertEqual(result["valid_final_line_count"], 3)
        self.assertEqual(result["distinct_valid_label_count"], 2)
        self.assertIsNone(result["prediction_index"])

    def test_unequal_snapshot_counts_consistent_correct_incorrect_and_common_pairing(self):
        panel = build_panel("gpqa_main")
        aligned = {cell["cell_id"]: [row(0, "Final answer: A\nFinal answer: (a)"),
                                     row(1, "Final answer: B\nFinal answer: (B)")]
                   for cell in panel["cells"]}
        native = panel["cells"][0]["cell_id"]
        loop = panel["cells"][1]["cell_id"]
        aligned[native] += [row(2, "Final answer: A\nFinal answer: B"),
                            row(3, "unfinished"), row(4, "Final answer: (A)")]
        aligned[loop][0] = row(0, "Final answer: B\nFinal answer: (B)")
        gold = {f"synthetic:{i}": 0 for i in range(5)}
        with patch.object(s2, "stratified_paired_bootstrap", side_effect=bootstrap) as boot, patch.object(
                s2, "paired_contrast", side_effect=contrast) as paired:
            result = s2.analyze_s2_snapshot(panel, aligned, gold)
        self.assertEqual(boot.call_count, 36)
        self.assertEqual(paired.call_count, 44)
        self.assertEqual(result["primary_parser"], "S2")
        own = {cell["cell_id"]: cell for cell in result["s2"]["own_completion"]}
        cell = own[native]
        self.assertEqual((cell["n"], cell["correct"], cell["accuracy"]), (5, 2, .4))
        self.assertEqual(cell["consistent_multiple_count"], 2)
        self.assertEqual(cell["consistent_multiple_correct"], 1)
        self.assertEqual(cell["consistent_multiple_incorrect"], 1)
        self.assertEqual(cell["conflicting_multiple_count"], 1)
        self.assertEqual(cell["single_count"], 1)
        self.assertEqual(cell["no_match_count"], 1)
        self.assertEqual(cell["extraction_success_count"], 3)
        self.assertEqual(own[loop]["n"], 2)
        self.assertTrue(all("delta_native_pp" not in cell and "delta_loop_pp" not in cell for cell in own.values()))
        common = result["s2"]["common_intersection"]
        self.assertEqual(common["n"], 2)
        fair = {cell["cell_id"]: cell for cell in common["cells"]}
        self.assertEqual(fair[native]["accuracy"], .5)
        self.assertEqual(fair[loop]["delta_native_pp"], -50)
        self.assertEqual(Counter(c["family"] for c in common["contrasts"]), Counter(s2.FAMILY_COUNTS))
        self.assertTrue(all(c["n"] == 2 and c["holm_family_size"] == s2.FAMILY_COUNTS[c["family"]]
                            for c in common["contrasts"]))
        for forbidden in ("synthetic:", "prediction_index", "generated_text", "Final answer:"):
            self.assertNotIn(forbidden, json.dumps(result))

    def test_empty_intersection_no_paired_test_missing_gold_raises(self):
        panel = build_panel("gpqa_main")
        aligned = {cell["cell_id"]: [row(0, "Final answer: A")] for cell in panel["cells"]}
        aligned[panel["cells"][0]["cell_id"]] = []
        with patch.object(s2, "stratified_paired_bootstrap", side_effect=bootstrap), patch.object(s2, "paired_contrast") as paired:
            result = s2.analyze_s2_snapshot(panel, aligned, {"synthetic:0": 0})
        self.assertEqual(paired.call_count, 0)
        self.assertIsNone(result["s2"]["own_completion"][0]["accuracy"])
        self.assertEqual(result["s2"]["common_intersection"]["n"], 0)
        self.assertEqual(len(result["s2"]["common_intersection"]["contrasts"]), 44)
        with self.assertRaisesRegex(ValueError, "gold join"):
            s2.analyze_s2_snapshot(panel, aligned, {})


if __name__ == "__main__":
    unittest.main()
