import copy
import json
from collections import Counter
import unittest
from unittest.mock import patch

from tflt.loopscope import phase11_interim_stats as stats
from tflt.loopscope.phase11_panel import build_panel
from tflt.loopscope.phase8_accuracy_stats import exact_mcnemar_p


def row(index, text="Final answer: (A)", reason="eos"):
    return {"identity": f"synthetic:{index}", "canonical_index": index, "generated_text": text,
            "generated_token_count": 8192 if reason == "length_limit" else 7,
            "end_reason": reason, "candidate_labels": list("ABCD")}


def bootstrap(differences, subjects, replicates=10000, seed=20261008):
    assert replicates == 10000 and seed == 20261008
    assert len(differences) == len(subjects) and subjects == ["all_items"] * len(differences)
    mean = sum(differences) / len(differences) * 100
    return {"ci_low_pp": mean, "ci_high_pp": mean, "replicates": replicates, "seed": seed}


def contrast(reference, treatment, subjects, bootstrap_replicates=10000, seed=20261008):
    assert bootstrap_replicates == 10000 and seed == 20261008
    assert len(reference) == len(treatment) == len(subjects) == 2
    assert subjects == ["all_items", "all_items"]
    gains = sum(r == 0 and t == 1 for r, t in zip(reference, treatment))
    losses = sum(r == 1 and t == 0 for r, t in zip(reference, treatment))
    return {"n": len(subjects), "wrong_to_right": gains, "right_to_wrong": losses,
            "delta_pp": 100 * (sum(treatment) - sum(reference)) / len(subjects),
            "mcnemar_exact_p": exact_mcnemar_p(gains, losses),
            "bootstrap": {"replicates": bootstrap_replicates, "seed": seed}}


class InterimTests(unittest.TestCase):
    def test_unequal_completion_uses_own_n_and_only_common_pairing(self):
        panel = build_panel("gpqa_main")
        aligned = {cell["cell_id"]: [row(0), row(1)] for cell in panel["cells"]}
        native, loop = panel["cells"][0]["cell_id"], panel["cells"][1]["cell_id"]
        aligned[native].append(row(2, "Final answer: (B)"))
        aligned[loop][0] = row(0, "Final answer: (B)")
        gold = {f"synthetic:{i}": 0 for i in range(3)}
        with patch.object(stats, "stratified_paired_bootstrap", side_effect=bootstrap) as boot, patch.object(
                stats, "paired_contrast", side_effect=contrast) as paired:
            result = stats.analyze_snapshot(panel, aligned, gold)
        self.assertEqual(paired.call_count, 88)
        self.assertEqual(boot.call_count, 72)
        for parser in ("s1", "s0"):
            own = {cell["cell_id"]: cell for cell in result[parser]["own_completion"]}
            self.assertEqual(own[native]["n"], 3)
            self.assertEqual(own[native]["correct"], 2)
            self.assertEqual(own[native]["accuracy"], 2 / 3)
            self.assertEqual(own[loop]["n"], 2)
            self.assertEqual(own[loop]["accuracy"], .5)
            self.assertTrue(all("delta_native_pp" not in cell and "delta_loop_pp" not in cell for cell in own.values()))
            common = result[parser]["common_intersection"]
            self.assertEqual(common["n"], 2)
            common_cells = {cell["cell_id"]: cell for cell in common["cells"]}
            self.assertEqual(common_cells[native]["accuracy"], 1)
            self.assertEqual(common_cells[loop]["delta_native_pp"], -50)
            self.assertEqual(common_cells[loop]["delta_loop_pp"], 0)
            self.assertEqual(Counter(c["family"] for c in common["contrasts"]), Counter(stats.FAMILY_COUNTS))
            for c in common["contrasts"]:
                self.assertEqual(c["n"], 2)
                self.assertEqual(c["holm_family_size"], stats.FAMILY_COUNTS[c["family"]])
        serialized = json.dumps(result)
        for forbidden in ("synthetic:", "prediction_index", "generated_text", "Final answer:"):
            self.assertNotIn(forbidden, serialized)

    def test_consistent_conflicting_duplicates_stay_wrong_and_truncation_descriptive(self):
        panel = build_panel("gpqa_main")
        records = [row(0, "Final answer: (A)\nfinal ANSWER: A."),
                   row(1, "Final answer: A\nFinal answer: (B)"), row(2, "unfinished", "length_limit"),
                   row(3, "Final answer: A")]
        aligned = {cell["cell_id"]: copy.deepcopy(records) for cell in panel["cells"]}
        with patch.object(stats, "stratified_paired_bootstrap", side_effect=bootstrap), patch.object(
                stats, "paired_contrast", return_value={"n": 4, "delta_pp": 0, "mcnemar_exact_p": 1}):
            result = stats.analyze_snapshot(panel, aligned, {f"synthetic:{i}": 0 for i in range(4)})
        cell = result["s1"]["own_completion"][0]
        self.assertEqual(cell["n"], 4)
        self.assertEqual(cell["correct"], 1)
        self.assertEqual(cell["accuracy"], .25)
        self.assertEqual(cell["extraction_failure_count"], 3)
        self.assertEqual(cell["truncated_count"], 1)
        self.assertEqual(cell["extraction_status_counts"], {"multiple_valid_final_lines": 2,
                         "no_valid_final_line": 1, "unique_valid": 1})
        self.assertEqual(cell["s1_duplicate_line_counts"], {"consistent_multiple": 1, "conflicting_multiple": 1})
        self.assertEqual(result["s0"]["own_completion"][0]["correct"], 1)

    def test_empty_cell_yields_no_common_test_and_does_not_count_missing_wrong(self):
        panel = build_panel("gpqa_main")
        aligned = {cell["cell_id"]: [row(0)] for cell in panel["cells"]}
        empty = panel["cells"][0]["cell_id"]
        aligned[empty] = []
        with patch.object(stats, "stratified_paired_bootstrap", side_effect=bootstrap), patch.object(stats, "paired_contrast") as paired:
            result = stats.analyze_snapshot(panel, aligned, {"synthetic:0": 0})
        self.assertEqual(paired.call_count, 0)
        self.assertEqual(result["s1"]["own_completion"][0]["n"], 0)
        self.assertIsNone(result["s1"]["own_completion"][0]["accuracy"])
        common = result["s1"]["common_intersection"]
        self.assertEqual(common["n"], 0)
        self.assertEqual(len(common["contrasts"]), 44)
        self.assertTrue(all(c["holm_adjusted_p"] is None for c in common["contrasts"]))

    def test_identity_pairing_and_missing_gold_errors_propagate(self):
        panel = build_panel("gpqa_main")
        aligned = {cell["cell_id"]: [row(0), row(1)] for cell in panel["cells"]}
        bad = copy.deepcopy(aligned)
        bad[panel["cells"][1]["cell_id"]][0]["identity"] = "different-identity"
        with self.assertRaisesRegex(ValueError, "index has inconsistent identity"):
            stats.analyze_snapshot(panel, bad, {"synthetic:0": 0, "synthetic:1": 0, "different-identity": 0})
        with self.assertRaisesRegex(ValueError, "gold join"):
            stats.analyze_snapshot(panel, aligned, {"synthetic:0": 0})


if __name__ == "__main__":
    unittest.main()
