import math
import unittest

from tflt.loopscope.phase9_spectral_core import (
    MATCHED_NORM,
    ONLINE_T0,
    canonicalize_direction,
    matched_norm_uniform,
    prefix_positions,
    rank1_projection,
    soft_damping,
)


class Phase9SpectralCoreTests(unittest.TestCase):
    delta = (3.0, 4.0, 2.0)
    direction = (0.6, 0.8, 0.0)

    def test_online_and_matched_norm_contracts(self):
        online = soft_damping(self.delta, self.direction, 0.5)
        self.assertEqual(online, (1.5, 2.0, 2.0))
        matched = matched_norm_uniform(self.delta, self.direction, 0.5)
        self.assertAlmostEqual(math.hypot(*matched), math.hypot(*online))
        self.assertAlmostEqual(matched[0] / 3.0, matched[1] / 4.0)
        self.assertAlmostEqual(matched[1] / 4.0, matched[2] / 2.0)
        self.assertEqual(ONLINE_T0, "Online-t0")
        self.assertEqual(MATCHED_NORM, "Matched-norm")

    def test_rank_one_projection_and_zero_path(self):
        projection = rank1_projection(self.delta, self.direction)
        self.assertEqual(projection, (3.0, 4.0, 0.0))
        self.assertEqual(soft_damping(self.delta, self.direction, 0.0), self.delta)
        self.assertEqual(matched_norm_uniform((0.0, 0.0, 0.0), self.direction), (0.0, 0.0, 0.0))

    def test_direction_canonicalization(self):
        observed = canonicalize_direction((-3.0, 4.0))
        self.assertEqual(observed, (-0.6, 0.8))
        self.assertEqual(canonicalize_direction(observed), observed)

    def test_prefix_mask_excludes_special_padding_and_continuation(self):
        # The last two IDs are a multi-token continuation.  Only positions <=p
        # can enter S, and padding/special IDs are removed from that prefix.
        token_ids = (0, 101, 11, 12, 13, 99, 100)
        valid = prefix_positions(token_ids, 4, special_token_ids=(101,), pad_token_id=0)
        self.assertEqual(valid, (2, 3, 4))
        self.assertEqual(
            prefix_positions(token_ids, 4, special_token_ids=(101,), pad_token_id=0,
                             attention_mask=(0, 1, 1, 1, 1, 1, 1)),
            (2, 3, 4),
        )

    def test_invalid_inputs_fail_fast(self):
        with self.assertRaises(ValueError):
            soft_damping(self.delta, (1.0, 1.0, 0.0), 0.5)
        with self.assertRaises(ValueError):
            prefix_positions((1, 2), 1, special_token_ids=(1, 2))
        with self.assertRaises(ValueError):
            prefix_positions((1, 2), 3)


if __name__ == "__main__":
    unittest.main()
