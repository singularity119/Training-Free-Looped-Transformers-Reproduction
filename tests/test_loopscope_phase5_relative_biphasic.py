import copy
import math
import unittest
from pathlib import Path

from tflt.loopscope.phase5_relative_biphasic import (
    FORBIDDEN_OUTPUT_KEY_TOKENS,
    MODEL_SPECS,
    RelativeBiphasicError,
    analyze_model,
    analyze_shape_samples,
    assert_no_forbidden_output_keys,
    load_strict_json,
    normalize_scalar_samples,
    validate_card,
)


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/phase5_relative_biphasic_reversal_card.json"


def _boundary_samples(model_key, count=18):
    boundary_count = MODEL_SPECS[model_key]["boundary_count"]
    rows = []
    for index in range(count):
        h_values = [
            2.0
            - 0.012 * boundary
            + 0.0002 * index * math.sin(boundary + 1)
            for boundary in range(boundary_count)
        ]
        d_values = [
            0.7
            - 0.007 * boundary
            + 0.00015 * index * math.cos(boundary + 1)
            for boundary in range(boundary_count)
        ]
        rows.append(
            {
                "identity": "id-%04d" % index,
                "subject": "subject-%d" % (index % 3),
                "H": h_values,
                "D": d_values,
            }
        )
    return rows


def _shape_rows(pattern, *, layers=12, count=24, perturb=False):
    rows = []
    for index in range(count):
        values = [-1.0] * layers
        values[4:8] = pattern
        if perturb:
            values = [
                value + 0.0001 * math.sin(index + transition + 1)
                for transition, value in enumerate(values)
            ]
        rows.append(values)
    return rows


class RelativeBiphasicGateGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_strict_json(CARD_PATH)

    def test_card_identity_and_candidate_domains(self):
        validate_card(self.card)
        self.assertEqual(
            {width: len(MODEL_SPECS["qwen17"]["starts"][width]) for width in (3, 4, 5, 6)},
            {3: 8, 4: 7, 5: 6, 6: 5},
        )
        self.assertEqual(
            {width: len(MODEL_SPECS["qwen4"]["starts"][width]) for width in (3, 4, 5, 6)},
            {3: 12, 4: 11, 5: 10, 6: 9},
        )

    def test_synthetic_positive_control_passes_biphasic_stable(self):
        h_rows = _shape_rows([2.0, 2.0, -1.0, -1.0], perturb=True)
        k_rows = copy.deepcopy(h_rows)
        result = analyze_shape_samples(
            h_rows, k_rows, start=4, width=4, replicates=101, seed=20260722
        )
        self.assertTrue(result["Scorable"])
        self.assertTrue(result["BiphasicStable"])
        self.assertFalse(result["tau_ambiguous"])
        self.assertEqual(
            (result["best_shape_pair"]["q"], result["best_shape_pair"]["tau"]),
            (1, 6),
        )
        self.assertGreaterEqual(result["best_shape_pair"]["tau_pair_frequency"], 0.80)
        best = next(
            pair
            for pair in result["shape_pairs"]
            if pair["q"] == 1 and pair["tau"] == 6
        )
        self.assertTrue(best["segment_direction_pass"])
        self.assertTrue(best["core_direction_pass"])
        self.assertEqual(best["external_side"], "none")
        self.assertTrue(
            all(
                best["margins"][name]["simultaneous_lower95"] > 0.0
                for name in best["margins"]
            )
        )

    def test_synthetic_negative_single_step_zigzag_fails(self):
        h_rows = _shape_rows([0.0, 2.0, -2.0, 0.0])
        k_rows = copy.deepcopy(h_rows)
        result = analyze_shape_samples(
            h_rows, k_rows, start=4, width=4, replicates=101, seed=20260722
        )
        self.assertTrue(result["Scorable"])
        self.assertFalse(result["BiphasicStable"])
        self.assertTrue(result["biphasic_failures"])

    def test_external_support_is_explicit_and_never_padded(self):
        h_rows = _shape_rows([2.0, 2.0, -1.0, -1.0])
        result = analyze_shape_samples(
            h_rows, copy.deepcopy(h_rows), start=4, width=3, replicates=31, seed=7
        )
        left = next(
            pair for pair in result["shape_pairs"] if pair["q"] == 1 and pair["tau"] == 5
        )
        right = next(
            pair for pair in result["shape_pairs"] if pair["q"] == 1 and pair["tau"] == 6
        )
        self.assertEqual(left["external_side"], "left")
        self.assertEqual(left["support_transition_ids"]["core_left"][0], "B_3->B_4")
        self.assertEqual(right["external_side"], "right")
        self.assertEqual(right["support_transition_ids"]["core_right"][-1], "B_7->B_8")

    def test_candidate_ranking_formula_and_joint_draw_reuse(self):
        samples = normalize_scalar_samples(
            _boundary_samples("qwen17"), "qwen17", enforce_population=False
        )
        analysis = analyze_model(
            samples,
            "qwen17",
            replicates=37,
            seed=20260716,
            enforce_population=False,
        )
        reversed_analysis = analyze_model(
            list(reversed(samples)),
            "qwen17",
            replicates=37,
            seed=20260716,
            enforce_population=False,
        )
        self.assertEqual(analysis["candidate_count"], 26)
        self.assertEqual(
            analysis["bootstrap"]["draw_index_sha256"],
            reversed_analysis["bootstrap"]["draw_index_sha256"],
        )
        self.assertEqual(
            [row["global_rank"] for row in analysis["rows"]], list(range(1, 27))
        )
        for row in analysis["rows"]:
            self.assertAlmostEqual(
                row["S_REL"],
                min(row["z_OH"], row["z_OK"], row["z_FH"], row["z_FK"]),
                places=14,
            )
            self.assertEqual(
                row["NewEligible"],
                row["Scorable"]
                and row["NetPositive"]
                and row["SoftRelativeStable"]
                and row["BiphasicStable"],
            )

    def test_scalar_input_and_output_information_boundary_fail_closed(self):
        contaminated = _boundary_samples("qwen4")
        contaminated[0]["result_payload"] = {"value": 1}
        with self.assertRaisesRegex(RelativeBiphasicError, "keys differ"):
            normalize_scalar_samples(contaminated, "qwen4", enforce_population=False)
        for token in FORBIDDEN_OUTPUT_KEY_TOKENS:
            with self.assertRaisesRegex(RelativeBiphasicError, "forbidden"):
                assert_no_forbidden_output_keys({token + "_field": 1})


if __name__ == "__main__":
    unittest.main()
