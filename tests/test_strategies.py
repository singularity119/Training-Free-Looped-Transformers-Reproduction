import unittest

from tflt.config import LoopConfig
from tflt.strategies import run_loop


class StrategyTest(unittest.TestCase):
    def test_naive_repeats_operator(self):
        cfg = LoopConfig("m", (0, 0), k=3, strategy="naive")
        self.assertEqual(run_loop(lambda x: x + 1.0, 0.0, cfg), 3.0)

    def test_damped_euler_integrates_constant_residual(self):
        cfg = LoopConfig("m", (0, 0), k=4, strategy="damped_euler")
        self.assertAlmostEqual(run_loop(lambda x: x + 2.0, 0.0, cfg), 2.0)

    def test_rk4_constant_residual(self):
        cfg = LoopConfig("m", (0, 0), k=4, strategy="rk4")
        self.assertAlmostEqual(run_loop(lambda x: x + 2.0, 0.0, cfg), 2.0)

    def test_paper_rk_beta_blends_one_step(self):
        cfg = LoopConfig("m", (0, 0), k=2, strategy="rk", beta=0.5)
        self.assertAlmostEqual(run_loop(lambda x: 2.0 * x + 2.0, 0.0, cfg), 2.25)

    def test_heavy_ball_runs(self):
        cfg = LoopConfig("m", (0, 0), k=3, strategy="heavy_ball", beta=0.1)
        self.assertGreater(run_loop(lambda x: x + 1.0, 0.0, cfg), 0.0)


if __name__ == "__main__":
    unittest.main()
