import copy
import math
import unittest
from pathlib import Path

from tflt.loopscope.schema import SchemaError
from tflt.loopscope.phase3_analysis import (
    analyze_window_signals,
    iter_subject_bootstrap_indices,
    linear_percentile,
    rank_variant_scope,
    sample_standard_deviation,
    variant_comparison_starts,
    variant_support,
    verify_analysis_report,
)
from tflt.loopscope.phase3_schema import load_phase3_card
from tflt.loopscope.phase3_verifier import synthetic_signal_records


ROOT = Path(__file__).resolve().parents[1]


class Phase3AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_phase3_card(ROOT / "configs/loopscope/phase3_card.json")

    def test_bootstrap_rng_order_exact_and_joint(self):
        records = [
            {"subject": "a"},
            {"subject": "a"},
            {"subject": "b"},
            {"subject": "b"},
            {"subject": "b"},
        ]
        plan = list(
            iter_subject_bootstrap_indices(records, replicates=3, seed=20260716)
        )
        self.assertEqual(
            plan,
            [
                (1, 0, 2, 3, 3),
                (0, 1, 3, 2, 4),
                (1, 1, 3, 3, 3),
            ],
        )

    def test_percentile_interpolation_and_ddof_one(self):
        values = [0.0, 10.0, 20.0, 30.0, 40.0]
        self.assertEqual(linear_percentile(values, 0.025), 1.0)
        self.assertEqual(linear_percentile(values, 0.975), 39.0)
        self.assertEqual(sample_standard_deviation([1.0, 2.0, 3.0]), 1.0)

    def test_variant_supports_and_comparisons_are_frozen(self):
        self.assertEqual(variant_support(self.card, "SHIFT"), tuple(range(1, 24)))
        self.assertEqual(variant_support(self.card, "FLANK"), tuple(range(4, 21)))
        self.assertEqual(variant_support(self.card, "CONSENSUS"), tuple(range(4, 21)))
        self.assertEqual(variant_comparison_starts("SHIFT", 1), (0, 2))
        self.assertEqual(variant_comparison_starts("SHIFT", 23), (22, 24))
        self.assertEqual(variant_comparison_starts("FLANK", 4), (0, 8))
        self.assertEqual(variant_comparison_starts("FLANK", 20), (16, 24))
        self.assertEqual(
            variant_comparison_starts("CONSENSUS", 12), (11, 13, 8, 16)
        )
        with self.assertRaises(SchemaError):
            variant_comparison_starts("SHIFT", 0)

    def test_scorable_eligible_ranked_selected_and_abstain_are_separate(self):
        signals = synthetic_signal_records()
        report = analyze_window_signals(
            signals,
            self.card,
            replicates=31,
            seed=20260716,
            enforce_population=False,
        )
        expected_counts = {
            "SHIFT": (23, 13, 10),
            "FLANK": (17, 12, 5),
            "CONSENSUS": (17, 12, 5),
        }
        for variant, counts in expected_counts.items():
            payload = report["variants"][variant]
            self.assertEqual(
                [row["start"] for row in payload["windows"] if row["point_eligible"]],
                [12],
            )
            self.assertEqual(payload["outside_support_result"], "UNSCORABLE_ABSTAIN")
            self.assertEqual(payload["scopes"]["deployment"]["scorable_count"], counts[0])
            self.assertEqual(payload["scopes"]["retrospective"]["scorable_count"], counts[1])
            blind = payload["scopes"]["blind_enrichment"]
            self.assertEqual(blind["scorable_count"], counts[2])
            self.assertEqual(len(blind["published_ranking"]), counts[2])
            self.assertEqual(blind["window_decision"], "ABSTAIN_NO_POINT_ELIGIBLE")
            self.assertEqual(payload["scopes"]["deployment"]["point_top1_start"], 12)
            self.assertEqual(payload["scopes"]["deployment"]["selection_frequency"], 1.0)
        self.assertFalse(report["variant_selection_performed"])
        self.assertFalse(report["outcome_fields_consumed"])
        for row in report["window_metrics"]:
            self.assertAlmostEqual(
                row["ce_drop"]["point"],
                row["entropy_drop"]["point"] - row["kl_rise"]["point"],
            )

    def test_strict_center_and_comparison_zero_are_ineligible(self):
        center_zero = synthetic_signal_records()
        for record in center_zero:
            record["windows"][12]["entropy_drop"] = 0.0
            record["windows"][12]["ce_drop"] = -record["windows"][12]["kl_rise"]
        report = analyze_window_signals(
            center_zero,
            self.card,
            replicates=7,
            seed=9,
            enforce_population=False,
        )
        row = next(row for row in report["variants"]["SHIFT"]["windows"] if row["start"] == 12)
        self.assertFalse(row["point_eligible"])
        self.assertIn("center_entropy_drop_not_strictly_positive", row["eligibility_failures"])

        comparison_zero = synthetic_signal_records()
        for record in comparison_zero:
            record["windows"][11]["entropy_drop"] = 0.0
            record["windows"][11]["ce_drop"] = -record["windows"][11]["kl_rise"]
        report = analyze_window_signals(
            comparison_zero,
            self.card,
            replicates=7,
            seed=9,
            enforce_population=False,
        )
        row = next(row for row in report["variants"]["SHIFT"]["windows"] if row["start"] == 12)
        self.assertFalse(row["point_eligible"])
        self.assertIn(
            "comparison_11_entropy_drop_not_strictly_negative", row["eligibility_failures"]
        )

    def test_point_tie_and_selection_frequency_080_boundary(self):
        def row(start, score):
            return {
                "start": start,
                "score": score,
                "point_eligible": True,
                "contrasts": {
                    "O_H": {"standard_error": 1.0},
                    "O_K": {"standard_error": 1.0},
                },
            }

        tied = [row(1, 2.0), row(2, 2.0 - 5e-13)]
        bootstrap = {
            1: {"O_H": [2.0] * 5, "O_K": [2.0] * 5},
            2: {"O_H": [1.0] * 5, "O_K": [1.0] * 5},
        }
        result = rank_variant_scope("SHIFT", tied, bootstrap, (1, 2), self.card)
        self.assertEqual(result["window_decision"], "ABSTAIN_NO_UNIQUE_TOP1")
        self.assertIsNone(result["point_top1_start"])

        rows = [row(1, 2.0), row(2, 1.0)]
        bootstrap[1] = {"O_H": [2, 2, 2, 2, 0], "O_K": [2, 2, 2, 2, 0]}
        result = rank_variant_scope("SHIFT", rows, bootstrap, (1, 2), self.card)
        self.assertEqual(result["selection_frequency"], 0.8)
        self.assertEqual(result["window_decision"], "SELECTED")

        bootstrap[1] = {"O_H": [2, 2, 2, 0, 0], "O_K": [2, 2, 2, 0, 0]}
        result = rank_variant_scope("SHIFT", rows, bootstrap, (1, 2), self.card)
        self.assertEqual(result["selection_frequency"], 0.6)
        self.assertEqual(result["window_decision"], "ABSTAIN_LOW_SELECTION_FREQUENCY")

    def test_report_verification_recomputes_and_rejects_score_tamper(self):
        signals = synthetic_signal_records()
        report = analyze_window_signals(
            signals,
            self.card,
            replicates=7,
            seed=3,
            enforce_population=False,
        )
        verify_analysis_report(
            signals,
            report,
            self.card,
            replicates=7,
            seed=3,
            enforce_population=False,
        )
        changed = copy.deepcopy(report)
        changed["variants"]["SHIFT"]["windows"][0]["score"] += 1.0
        with self.assertRaisesRegex(SchemaError, "raw deterministic recomputation"):
            verify_analysis_report(
                signals,
                changed,
                self.card,
                replicates=7,
                seed=3,
                enforce_population=False,
            )

    def test_production_population_and_bootstrap_overrides_fail(self):
        with self.assertRaisesRegex(SchemaError, "1,531"):
            analyze_window_signals(synthetic_signal_records(), self.card)


if __name__ == "__main__":
    unittest.main()
