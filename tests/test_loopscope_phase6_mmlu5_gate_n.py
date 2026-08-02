import copy
import json
import unittest

from tflt.loopscope import phase6_mmlu5_v3_selector as v3


def _record(index, category, h=None, d=None):
    h = list(h if h is not None else [1.0] * v3.BOUNDARY_COUNT)
    d = list(d if d is not None else [0.0] * v3.BOUNDARY_COUNT)
    return {
        "schema_version": "loopscope.phase6.mmlu5-prefix-trajectory.v1",
        "identity": {"task": "mmlu", "doc_id": "%s:%03d" % (category, index), "doc_hash": "%064d" % (index + 1)},
        "subject": category,
        "split": "validation",
        "prompt_sha256": "%064d" % (index + 101),
        "renderer_provenance": {"contract_version": "test", "semantic_contract_sha256": "0" * 64, "task_source_manifest_sha256": "1" * 64, "choice_surface_manifest_sha256": "2" * 64},
        "boundaries": [
            {
                "boundary_id": "B_%d" % layer,
                "choice_entropy": h[layer],
                "kl_to_final": d[layer],
                "hidden_rms_l2_to_final": float(index + layer),
                "hidden_cosine_to_final": 0.1,
                "hidden_cosine_distance_to_final": 0.9,
                "choice_probabilities": [0.25, 0.25, 0.25, 0.25],
            }
            for layer in range(v3.BOUNDARY_COUNT)
        ],
        "adjacent_angular_distance": [
            {"transition_id": "B_%d->B_%d" % (layer, layer + 1), "angle_over_pi": 0.1}
            for layer in range(v3.LAYERS)
        ],
    }


class GateNV3Tests(unittest.TestCase):
    def test_candidate_domain_and_15_18_mapping(self):
        candidates = v3.enumerate_candidates()
        self.assertEqual(len(candidates), 42)
        row = next(item for item in candidates if (item["width"], item["start"]) == (4, 15))
        self.assertEqual(row["window_half_open"], "15:19")
        self.assertEqual(row["window_layers_inclusive"], "15:18")
        self.assertEqual(row["boundary_entry"], "B_15")
        self.assertEqual(row["boundary_exit"], "B_19")
        self.assertEqual(row["stop_exclusive"], 19)

    def test_projection_is_exactly_identity_category_h_d_and_hidden_changes_do_not_matter(self):
        records = [_record(0, "a"), _record(1, "b")]
        projected = v3.project_records(records)
        self.assertEqual(set(projected[0]), {"identity", "category", "H", "D"})
        changed = copy.deepcopy(records)
        changed[0]["boundaries"][0]["hidden_rms_l2_to_final"] = 999999.0
        changed[0]["boundaries"][0]["hidden_cosine_to_final"] = -1.0
        changed_projected = v3.project_records(changed)
        self.assertEqual(projected, changed_projected)

    def test_equal_category_macro_is_not_micro_mean(self):
        first = _record(0, "small", h=[0.0] * v3.BOUNDARY_COUNT)
        second = _record(1, "large", h=[100.0] * v3.BOUNDARY_COUNT)
        third = _record(2, "large", h=[100.0] * v3.BOUNDARY_COUNT)
        fourth = _record(3, "large", h=[100.0] * v3.BOUNDARY_COUNT)
        projected = v3.project_records([first, second, third, fourth])
        _categories, h_bar, _d_bar = v3._macro_curves(projected)
        self.assertEqual(h_bar[0], 50.0)
        self.assertNotEqual(h_bar[0], 75.0)

    def test_bootstrap_digest_is_invariant_to_record_order(self):
        records = [_record(index, category) for category in ("a", "b") for index in (0, 1)]
        projected = v3.project_records(records)
        reordered = v3.project_records(list(reversed(records)))
        left = v3._bootstrap_macro_curves(projected, replicates=12, seed=v3.FORMAL_SEED)
        right = v3._bootstrap_macro_curves(reordered, replicates=12, seed=v3.FORMAL_SEED)
        self.assertEqual(left["draw_index_sha256"], right["draw_index_sha256"])
        self.assertEqual(left["H"], right["H"])
        self.assertEqual(left["D"], right["D"])

    def test_constant_rates_are_not_a_measurable_turn(self):
        result = v3._fit_turn([1.0, 1.0, 1.0], 0, 3)
        self.assertEqual(result["failure"], "NO_MEASURABLE_TURN")
        self.assertIsNone(result["tau"])

    def test_same_sign_fast_to_slow_turn_is_accepted(self):
        h = v3._fit_turn([4.0, 4.0, 1.0], 0, 3)
        k = v3._fit_turn([3.0, 3.0, 0.5], 0, 3)
        self.assertEqual(h["tau"], 2)
        self.assertEqual(k["tau"], 2)
        self.assertTrue(h["turn_strength_pass"])
        self.assertTrue(k["turn_strength_pass"])
        self.assertEqual(h["tau"], k["tau"])

    def test_direction_reversal_is_not_required(self):
        result = v3._fit_turn([4.0, -1.0, -1.0], 0, 3)
        self.assertTrue(result["turn_strength_pass"])
        self.assertEqual(result["tau"], 1)

    def test_asynchronous_tau_is_rejected(self):
        h = v3._fit_turn([4.0, 4.0, 1.0], 0, 3)
        k = v3._fit_turn([4.0, 1.0, 1.0], 0, 3)
        self.assertNotEqual(h["tau"], k["tau"])

    def test_tau_tie_is_not_broken_by_position(self):
        result = v3._fit_turn([1.0, 2.0, 1.0], 0, 3)
        self.assertEqual(result["failure"], "TURN_TAU_TIE")
        self.assertIsNone(result["tau"])
        self.assertEqual(result["ties"], [1, 2])

    def test_q_below_half_and_q_close_to_half_fail(self):
        below = v3._fit_turn([0.0, 0.0, 1.0, 0.0], 0, 4)
        close = v3._fit_turn([0.0, 0.0, 0.0, 1.0, 0.0, 1.0], 0, 6)
        self.assertEqual(below["failure"], "TURN_NOT_DOMINANT")
        self.assertEqual(close["failure"], "TURN_NOT_DOMINANT")
        self.assertLess(max(entry["Q"] for entry in below["entries"]), 0.5)
        self.assertTrue(math_isclose(max(entry["Q"] for entry in close["entries"]), 0.5))

    def test_terminal_states_include_both_v3_abstains_and_selected_threshold(self):
        self.assertEqual(v3.decide_terminal(0, None)[0], "ABSTAIN_NO_V3_ELIGIBLE")
        self.assertEqual(v3.decide_terminal(1, 0.7999)[0], "ABSTAIN_COMBINED_RANK_UNSTABLE")
        self.assertEqual(v3.decide_terminal(1, 0.80)[0], "SELECTED_WINDOW")

    def test_no_eligible_synthetic_analysis_is_a_legal_v3_abstain(self):
        records = [_record(index, category) for category in ("a", "b") for index in (0, 1)]
        projected = v3.project_records(records)
        payload = v3.analyze_projected(projected, replicates=12, seed=v3.FORMAL_SEED, formal=False)
        self.assertEqual(payload["candidate_count"], 42)
        self.assertEqual(payload["selector_decision"], "ABSTAIN_NO_V3_ELIGIBLE")
        self.assertIsNone(payload["selected_key"])
        self.assertFalse(payload["outcome_fields_consumed"])

    def test_schema_document_is_closed(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "configs/loopscope/phase6_mmlu5_v3_selector_freeze_schema.json").read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["$id"], "loopscope.phase6.mmlu5-gate-n-v3-selector-freeze.v1")


def math_isclose(left, right):
    return abs(left - right) <= 1e-12


if __name__ == "__main__":
    unittest.main()
