from __future__ import annotations

import copy
import unittest

from tflt.loopscope.phase4_schema import Phase4ContractError
from tflt.loopscope.phase4_selector import (
    analyze_selector,
    build_outcome_panel,
    verify_selector_report,
)
from tflt.loopscope.phase4_verifier import _synthetic_scalar_record


class Phase4SelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = [_synthetic_scalar_record(index) for index in range(24)]
        cls.report = analyze_selector(cls.records, replicates=64, seed=20260717)

    def test_width4_and_variable_width_counts(self) -> None:
        self.assertEqual(self.report["width4_primary"]["candidate_start_count"], 33)
        self.assertEqual(self.report["width4_primary"]["consensus_start_count"], 25)
        self.assertEqual(self.report["variable_width_strict"]["row_count"], 154)
        self.assertEqual(self.report["variable_width_edge"]["row_count"], 224)
        self.assertEqual(len(self.report["width4_primary"]["all_window_metrics"]), 33)

    def test_width4_strict_subset_is_identical_to_primary(self) -> None:
        verify_selector_report(self.report)
        primary = {row["start"]: row for row in self.report["width4_primary"]["rows"]}
        strict = {
            row["start"]: row
            for row in self.report["variable_width_strict"]["rows"]
            if row["width"] == 4
        }
        self.assertEqual(set(primary), set(range(4, 29)))
        for start in primary:
            self.assertEqual(primary[start]["score"], strict[start]["score"])
            self.assertEqual(primary[start]["eligible"], strict[start]["eligible"])
            self.assertEqual(primary[start]["primary_rank"], strict[start]["width4_primary_rank"])

    def test_min_z_eligibility_ranking_and_selection(self) -> None:
        rows = self.report["width4_primary"]["rows"]
        selected = self.report["width4_primary"]["selected_window"]
        self.assertEqual(selected, "12:15")
        selected_row = next(row for row in rows if row["window"] == selected)
        z_values = [
            value["point"] / max(value["se"], 1e-12)
            for value in selected_row["contrasts"].values()
        ]
        self.assertAlmostEqual(selected_row["score"], min(z_values), places=10)
        self.assertTrue(selected_row["eligible"])
        self.assertGreaterEqual(selected_row["selection_frequency"], 0.80)

    def test_no_eligible_produces_abstain(self) -> None:
        flat = []
        for index in range(12):
            record = copy.deepcopy(_synthetic_scalar_record(index))
            for boundary in record["boundaries"]:
                boundary["full_vocabulary_entropy"] = 5.0
                boundary["normalized_entropy"] = 0.5
                boundary["kl_to_final"] = 0.0
            flat.append(record)
        report = analyze_selector(flat, replicates=16, seed=20260717)
        self.assertTrue(report["width4_primary"]["abstain"])
        self.assertEqual(report["width4_primary"]["abstain_reason"], "NO_ELIGIBLE_WINDOW")

    def test_known_outcome_exclusion_and_no_variable_width_panel(self) -> None:
        panel = build_outcome_panel(self.report, ["15:18", "4:7"])
        blind = panel["blind_high3"] + panel["blind_low3"]
        self.assertNotIn("15:18", blind)
        self.assertNotIn("4:7", blind)
        self.assertFalse(panel["variable_width_outcomes_authorized"])

    def test_selected_outside_panel_is_fail_fast(self) -> None:
        tampered = copy.deepcopy(self.report)
        tampered["width4_primary"]["selected_window"] = "20:23"
        with self.assertRaises(Phase4ContractError):
            build_outcome_panel(tampered, ["15:18"])

    def test_verifier_rejects_truncated_or_misranked_secondary_tables(self) -> None:
        cases = []
        empty_edge = copy.deepcopy(self.report)
        empty_edge["variable_width_edge"]["rows"] = []
        cases.append(empty_edge)
        truncated_strict = copy.deepcopy(self.report)
        truncated_strict["variable_width_strict"]["rows"] = truncated_strict[
            "variable_width_strict"
        ]["rows"][:25]
        cases.append(truncated_strict)
        wrong_order = copy.deepcopy(self.report)
        wrong_order["variable_width_edge"]["rows"][:2] = reversed(
            wrong_order["variable_width_edge"]["rows"][:2]
        )
        cases.append(wrong_order)
        for report in cases:
            with self.subTest():
                with self.assertRaises(Phase4ContractError):
                    verify_selector_report(report)


if __name__ == "__main__":
    unittest.main()
