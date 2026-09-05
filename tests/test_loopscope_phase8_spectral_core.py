import json
import math
from pathlib import Path
import unittest

from tflt.loopscope.phase8_spectral_core import (
    rank1_projection, soft_damping, matched_norm_uniform,
)


class SpectralCoreTests(unittest.TestCase):
    delta = (3.0, 4.0, 2.0)
    direction = (0.6, 0.8, 0.0)

    def test_identity_and_full_removal(self):
        self.assertEqual(soft_damping(self.delta, self.direction, 0), self.delta)
        self.assertEqual(soft_damping(self.delta, self.direction, 1), (0, 0, 2))

    def test_half_energy_norm_and_orthogonal_component(self):
        output = soft_damping(self.delta, self.direction, 0.5)
        self.assertEqual(output, (1.5, 2, 2))
        self.assertAlmostEqual(math.hypot(*rank1_projection(output, self.direction)) ** 2, 25 * 0.25)
        self.assertAlmostEqual(math.hypot(*output) ** 2, 29 - (2 * 0.5 - 0.5**2) * 25)
        self.assertEqual(output[2], self.delta[2])

    def test_sign_invariance(self):
        self.assertEqual(soft_damping(self.delta, self.direction, 0.5),
                         soft_damping(self.delta, tuple(-v for v in self.direction), 0.5))

    def test_uniform_norm_and_direction(self):
        for strength in (0, 0.5, 1):
            output = matched_norm_uniform(self.delta, self.direction, strength)
            self.assertAlmostEqual(math.hypot(*output), math.hypot(*soft_damping(self.delta, self.direction, strength)))
            self.assertAlmostEqual(output[0] / 3, output[1] / 4)
            self.assertAlmostEqual(output[1] / 4, output[2] / 2)
        self.assertEqual(matched_norm_uniform((0, 0, 0), self.direction, 0.5), (0, 0, 0))
        self.assertEqual(matched_norm_uniform((3, 4, 0), self.direction, 1), (0, 0, 0))

    def test_invalid_scientific_inputs_fail(self):
        for delta, direction, strength in [
            (self.delta, (1, 1, 0), 0.5), ((math.nan, 0, 0), self.direction, 0.5),
            (self.delta, (math.inf, 0, 0), 0.5), (self.delta, self.direction, math.nan),
            (self.delta, self.direction, -0.1), (self.delta, self.direction, 1.1),
        ]:
            with self.subTest(delta=delta, direction=direction, strength=strength):
                with self.assertRaises(ValueError):
                    soft_damping(delta, direction, strength)

    def test_four_confirmed_windows(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'configs/loopscope/phase8_model_windows.json').read_text())
        expected = [('Qwen/Qwen3-4B-Base', 'first', [12, 15], [12, 16]),
                    ('Qwen/Qwen3-4B-Base', 'first', [13, 16], [13, 17]),
                    ('Qwen/Qwen3-1.7B-Base', 'last', [12, 15], [12, 16]),
                    ('Qwen/Qwen3-1.7B-Base', 'last', [6, 9], [6, 10])]
        actual = []
        for model in config['models']:
            for window in model['windows']:
                start, end = window['layers']
                self.assertEqual(window['boundaries'], [start, end + 1])
                self.assertEqual(end - start + 1, 4)
                self.assertTrue(0 <= start <= end < model['observed_num_hidden_layers'])
                actual.append((model['model'], model['cache_strategy'], window['layers'], window['boundaries']))
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
