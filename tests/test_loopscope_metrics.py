import math
import unittest

from tflt.loopscope.metrics import (
    MetricError,
    activity_contraction_score,
    choice_entropy,
    contraction_ratio,
    effective_rank_measurement_from_singular_values,
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

    def test_effective_rank_zero_spectrum_is_explicit_zero(self):
        self.assertEqual(effective_rank_from_singular_values([0.0, 0.0]), 0.0)
        self.assertEqual(
            effective_rank_measurement_from_singular_values([0.0, 0.0]),
            {
                "effective_rank": 0.0,
                "zero_centered_spectrum": True,
                "centered_spectrum_mass": 0.0,
            },
        )

    def test_effective_rank_positive_spectrum_has_no_zero_threshold(self):
        measurement = effective_rank_measurement_from_singular_values([2.0, 1.0])
        self.assertAlmostEqual(measurement["effective_rank"], 1.6493848884661177)
        self.assertEqual(measurement["centered_spectrum_mass"], 5.0)
        self.assertIs(measurement["zero_centered_spectrum"], False)

        tiny = effective_rank_measurement_from_singular_values([1e-150])
        self.assertEqual(tiny["effective_rank"], 1.0)
        self.assertGreater(tiny["centered_spectrum_mass"], 0.0)
        self.assertIs(tiny["zero_centered_spectrum"], False)

    def test_effective_rank_rejects_invalid_singular_values(self):
        for values in ([-1.0], [math.nan], [math.inf], [1e308]):
            with self.subTest(values=values), self.assertRaises(MetricError):
                effective_rank_measurement_from_singular_values(values)


if __name__ == "__main__":
    unittest.main()
