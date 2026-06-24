import unittest

from tflt.config import LoopConfig


class LoopConfigTest(unittest.TestCase):
    def test_window_string(self):
        cfg = LoopConfig.from_window_string("qwen3-1.7b-base", "12:15")
        self.assertEqual(cfg.window, (12, 15))
        self.assertEqual(cfg.window_width, 4)

    def test_first_n_required(self):
        with self.assertRaises(ValueError):
            LoopConfig("m", (1, 2), decode_mode="first_n")

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            LoopConfig("m", (3, 2))


if __name__ == "__main__":
    unittest.main()
