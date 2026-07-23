import copy
import math
import unittest
from pathlib import Path

from tflt.loopscope.phase5_relative_biphasic import load_strict_json
from tflt.loopscope.phase5_relative_biphasic_v2_absolute_rate import (
    AbsoluteRateError,
    compare_v1_rows,
    compute_rate_statistics,
    rate_score,
    select_rate_decision,
    validate_card,
)


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = (
    ROOT
    / "configs/loopscope/phase5_relative_biphasic_v2_absolute_rate_card.json"
)


def _row(width, start, score, *, eligible=True, stable=True):
    return {
        "width": width,
        "start": start,
        "window": "%d:%d" % (start, start + width - 1),
        "S_RATE": score,
        "NewEligible": eligible,
        "RateStable": stable,
    }


class RelativeBiphasicV2AbsoluteRateTests(unittest.TestCase):
    def test_01_zero_rate_has_no_main_score(self):
        self.assertIsNone(rate_score(0.0, 1.0))
        self.assertIsNone(rate_score(1.0, 0.0))

    def test_02_negative_rate_fails_main_score(self):
        self.assertIsNone(rate_score(-0.1, 1.0))
        self.assertIsNone(rate_score(1.0, -0.1))

    def test_03_point_top_tolerance_tie_abstains(self):
        rows = [_row(3, 9, 1.0), _row(4, 9, 1.0 + 5e-13)]
        boot = {
            (3, 9): {"H": [1.0, 1.0], "K": [1.0, 1.0]},
            (4, 9): {"H": [1.0, 1.0], "K": [1.0, 1.0]},
        }
        result = select_rate_decision(rows, boot, 2)
        self.assertEqual(result["decision"], "ABSTAIN_NO_UNIQUE_TOP1")
        self.assertTrue(all(row["point_top_tie"] for row in rows))

    def test_04_score_is_strictly_monotone_in_each_positive_rate(self):
        baseline = rate_score(1.0, 2.0)
        self.assertGreater(rate_score(1.1, 2.0), baseline)
        self.assertGreater(rate_score(1.0, 2.1), baseline)

    def test_05_per_layer_rate_not_total_change_controls_ranking(self):
        short_score = rate_score(0.9, 0.9)
        wide_score = rate_score(0.6, 0.6)
        self.assertGreater(short_score, wide_score)
        self.assertLess(3 * 0.9, 6 * 0.6)

    def test_06_nonpositive_simultaneous_lcb_is_not_rate_stable(self):
        result = compute_rate_statistics(
            0.1,
            0.1,
            [-0.1] * 6,
            [-0.1] * 6,
        )
        self.assertFalse(result["RateStable"])
        self.assertTrue(
            result["rate_LCB_H"] <= 0.0 or result["rate_LCB_K"] <= 0.0
        )

    def test_07_unstable_bootstrap_top1_abstains(self):
        rows = [_row(3, 9, 2.0), _row(4, 9, 1.0)]
        boot = {
            (3, 9): {"H": [4.0, 1.0, 4.0, 1.0], "K": [4.0, 1.0, 4.0, 1.0]},
            (4, 9): {"H": [1.0, 4.0, 1.0, 4.0], "K": [1.0, 4.0, 1.0, 4.0]},
        }
        result = select_rate_decision(rows, boot, 4)
        self.assertEqual(result["decision"], "ABSTAIN_RATE_RANK_UNSTABLE")
        self.assertEqual(result["selection_frequency"], 0.5)

    def test_08_nonpositive_replicate_rate_cannot_win_and_stays_in_denominator(self):
        rows = [_row(3, 9, 2.0), _row(4, 9, 1.0)]
        boot = {
            (3, 9): {"H": [4.0, -1.0], "K": [4.0, 1.0]},
            (4, 9): {"H": [1.0, -1.0], "K": [1.0, 1.0]},
        }
        result = select_rate_decision(rows, boot, 2)
        self.assertEqual(result["replicate_no_winner_count"], 1)
        self.assertEqual(result["selection_frequency"], 0.5)

    def test_09_reference_changes_do_not_change_score_after_admission(self):
        admitted = {"NewEligible": True, "G_H": 0.4, "G_K": 0.9, "O_H": 2.0}
        changed = copy.deepcopy(admitted)
        changed["O_H"] = -999.0
        self.assertEqual(
            rate_score(admitted["G_H"], admitted["G_K"]),
            rate_score(changed["G_H"], changed["G_K"]),
        )

    def test_10_card_and_full_v1_backward_compatibility(self):
        validate_card(load_strict_json(CARD_PATH))
        expected = [
            {
                "width": 3,
                "start": 9,
                "flags": {"NewEligible": True, "failure_reasons": []},
                "values": [1.0, 2.0, {"x": 3.0}],
            }
        ]
        receipt = compare_v1_rows(copy.deepcopy(expected), expected)
        self.assertEqual(receipt["status"], "PASS")
        altered = copy.deepcopy(expected)
        altered[0]["values"][2]["x"] += 1e-4
        with self.assertRaisesRegex(
            AbsoluteRateError, "BLOCK_V1_BACKWARD_COMPATIBILITY_MISMATCH"
        ):
            compare_v1_rows(altered, expected)


if __name__ == "__main__":
    unittest.main()
