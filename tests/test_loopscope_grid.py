import unittest

from tflt.loopscope.grid import candidate_windows, generate_window_grid, validate_window_grid
from tflt.loopscope.schema import SchemaError


class LoopScopeGridTest(unittest.TestCase):
    def test_qwen17_grid_is_deterministic_and_contains_anchors(self):
        first = generate_window_grid(28)
        second = generate_window_grid(28)
        self.assertEqual(first, second)
        windows = candidate_windows(first)
        self.assertIn((12, 15), windows)
        fixed = tuple(int(value) for value in first["anchors"]["fixed_depth"].split(":"))
        self.assertIn(fixed, windows)
        self.assertTrue(all(end - start + 1 == 4 for start, end in windows))
        self.assertEqual(len(first["comparison_windows"]["random_in_band"]), 5)
        self.assertEqual(len(set(first["comparison_windows"]["random_in_band"])), 5)

    def test_grid_hash_freezes_exact_windows(self):
        manifest = generate_window_grid(28)
        validate_window_grid(manifest)
        manifest["windows"][0]["start"] += 1
        with self.assertRaises(SchemaError):
            validate_window_grid(manifest)

    def test_anchor_must_fit_model(self):
        with self.assertRaises(ValueError):
            generate_window_grid(12)


if __name__ == "__main__":
    unittest.main()
