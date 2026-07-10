import math
import unittest

from tflt.loopscope.metrics import (
    MetricError,
    activity_contraction_score,
    choice_entropy,
    contraction_ratio,
    effective_rank_from_singular_values,
    kl_to_reference,
    relative_activity,
    summarize,
    top1_agreement,
)


class LoopScopeMetricsTest(unittest.TestCase):
    def test_choice_metrics(self):
        uniform = [0.25, 0.25, 0.25, 0.25]
        peaked = [0.7, 0.1, 0.1, 0.1]
        self.assertAlmostEqual(choice_entropy(uniform), math.log(4.0))
        self.assertGreater(kl_to_reference(peaked, uniform), 0.0)
        self.assertEqual(top1_agreement(peaked, [0.8, 0.1, 0.05, 0.05]), 1.0)
        self.assertEqual(top1_agreement(peaked, [0.1, 0.8, 0.05, 0.05]), 0.0)

    def test_activity_and_contraction_do_not_fill_zero(self):
        self.assertAlmostEqual(relative_activity(2.0, 4.0), 0.5)
        self.assertAlmostEqual(contraction_ratio(1.0, 2.0), 0.5)
        self.assertAlmostEqual(activity_contraction_score(0.5, 0.5), 0.25)
        with self.assertRaises(MetricError):
            relative_activity(0.0, 4.0)
        with self.assertRaises(MetricError):
            contraction_ratio(1.0, 0.0)

    def test_effective_rank_and_summary(self):
        self.assertAlmostEqual(effective_rank_from_singular_values([1.0, 1.0]), 2.0)
        stats = summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["count"], 4.0)
        self.assertEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["p90"], 3.7)

    def test_effective_rank_uses_squared_singular_values(self):
        self.assertAlmostEqual(
            effective_rank_from_singular_values([2.0, 1.0]),
            1.6493848884661177,
        )


if __name__ == "__main__":
    unittest.main()
