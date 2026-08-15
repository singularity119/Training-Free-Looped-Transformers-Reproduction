import copy
import unittest

from tflt.loopscope.phase7_schema import Phase7ContractError
from tflt.loopscope.phase7_v3 import (
    analyze_records,
    fit_turn,
    macro_curves,
    rates,
    validate_v3_payload,
)

from test_phase7_schema import make_record


def make_trajectory_record(layer_count, identity, subject, category_offset):
    record = make_record(layer_count=layer_count, identity=identity, subject=subject)
    for index, boundary in enumerate(record["boundaries"]):
        # The small category offset makes the category-macro assertion
        # observable while keeping all values finite and non-negative.
        boundary["choice_entropy"] = 6.0 - 0.25 * index + category_offset
        if index < layer_count:
            boundary["kl_to_final"] = 0.125 * index + category_offset
        else:
            boundary["kl_to_final"] = 0.0
    return record


class Phase7V3Tests(unittest.TestCase):
    def test_rates_and_turn_fit_follow_frozen_scalar_formulas(self):
        r_h, r_k = rates([3.0, 2.0, 1.0], [0.0, 1.0, 2.0])
        self.assertEqual(r_h, [1.0, 1.0])
        self.assertEqual(r_k, [1.0, 1.0])
        fit = fit_turn([1.0, 1.0, 1.0], start=0, width=3)
        self.assertEqual(fit["failure"], "NO_MEASURABLE_TURN")
        self.assertIsNone(fit["tau"])

    def test_category_macro_is_equal_weighted_and_v3_is_depth_general(self):
        records = [
            make_trajectory_record(20, "a-0", "alpha", 0.0),
            make_trajectory_record(20, "a-1", "alpha", 0.5),
            make_trajectory_record(20, "b-0", "beta", 10.0),
        ]
        categories, h_bar, d_bar = macro_curves(
            [
                {
                    "identity": row["canonical_identity"],
                    "category": row["subject"],
                    "H": [item["choice_entropy"] for item in row["boundaries"]],
                    "D": [item["kl_to_final"] for item in row["boundaries"]],
                }
                for row in records
            ]
        )
        self.assertEqual(categories, ["alpha", "beta"])
        self.assertAlmostEqual(h_bar[0], (6.25 + 16.0) / 2.0)
        self.assertAlmostEqual(d_bar[-1], 0.0)

        payload = analyze_records(records, layer_count=20, replicates=8, seed=17)
        validate_v3_payload(payload, expected_layer_count=20)
        self.assertEqual(payload["candidate_count"], 18)
        self.assertIn(
            payload["selector_decision"],
            {"SELECTED_WINDOW", "ABSTAIN_NO_V3_ELIGIBLE", "ABSTAIN_COMBINED_RANK_UNSTABLE"},
        )
        self.assertEqual(payload["selector_projection_fields"], ["identity", "category", "H", "D"])
        self.assertFalse(payload["hidden_diagnostics_in_selector"])

    def test_v3_rejects_forbidden_input_and_payload_fields(self):
        bad = make_trajectory_record(20, "bad", "alpha", 0.0)
        bad["correctness"] = False
        with self.assertRaisesRegex(Phase7ContractError, "BLOCK_INFORMATION_BARRIER_VIOLATION"):
            analyze_records([bad], layer_count=20, replicates=4, seed=3)

        records = [make_trajectory_record(20, "a", "alpha", 0.0), make_trajectory_record(20, "b", "beta", 0.0)]
        payload = analyze_records(records, layer_count=20, replicates=4, seed=3)
        payload["gold_label"] = "D"
        with self.assertRaisesRegex(Phase7ContractError, "BLOCK_INFORMATION_BARRIER_VIOLATION"):
            validate_v3_payload(payload, expected_layer_count=20)

    def test_safe_aggregate_source_data_is_complete_and_diagnostics_do_not_change_selector(self):
        records = [
            make_trajectory_record(20, "a-0", "alpha", 0.0),
            make_trajectory_record(20, "a-1", "alpha", 0.5),
            make_trajectory_record(20, "b-0", "beta", 1.0),
        ]
        payload = analyze_records(records, layer_count=20, replicates=8, seed=17)
        validate_v3_payload(payload, expected_layer_count=20, require_source_data=True)
        source = payload["aggregate_source_data"]
        self.assertEqual(source["record_count"], 3)
        self.assertEqual(source["subject_count"], 2)
        self.assertEqual(source["boundary_count"], 21)
        self.assertEqual(source["transition_count"], 20)
        self.assertEqual(
            set(source["boundary_metrics"]),
            {
                "choice_entropy",
                "kl_to_final",
                "hidden_rms_l2_to_final",
                "hidden_cosine_to_final",
                "hidden_cosine_distance_to_final",
            },
        )
        self.assertEqual(set(source["transition_metrics"]), {"adjacent_angular_distance"})
        self.assertEqual(
            len(source["boundary_metrics"]["choice_entropy"]["point_mean"]),
            21,
        )
        self.assertEqual(
            len(source["transition_metrics"]["adjacent_angular_distance"]["point_mean"]),
            20,
        )

        changed_records = copy.deepcopy(records)
        changed_records[0]["boundaries"][0]["hidden_rms_l2_to_final"] += 3.0
        changed = analyze_records(changed_records, layer_count=20, replicates=8, seed=17)
        self.assertEqual(payload["candidates"], changed["candidates"])
        self.assertEqual(payload["selector_decision"], changed["selector_decision"])
        self.assertEqual(payload["selected_window"], changed["selected_window"])
        self.assertNotEqual(
            payload["aggregate_source_data"]["boundary_metrics"]["hidden_rms_l2_to_final"]["point_mean"],
            changed["aggregate_source_data"]["boundary_metrics"]["hidden_rms_l2_to_final"]["point_mean"],
        )


if __name__ == "__main__":
    unittest.main()
