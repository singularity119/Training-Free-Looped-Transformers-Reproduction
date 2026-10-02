import importlib.util
from pathlib import Path
import sys
import unittest


DIRECTORY = Path(__file__).resolve().parents[1]/"scripts"/"loopscope"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("check_phase10_gpu", DIRECTORY/"check_phase10_gpu.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class GPUCheckTests(unittest.TestCase):
    def test_complete_scores_and_greedy_flags_require_exact_equivalence(self):
        self.assertTrue(checker.score_comparison([(-4., False), (-2., True)], [(-4., False), (-2., True)])["exact"])
        changed = checker.score_comparison([(-4., False)], [(-4.00001, False)])
        self.assertFalse(changed["exact"])
        self.assertGreater(changed["max_abs_error"], 0)
        self.assertGreater(changed["max_relative_error"], 0)
        self.assertFalse(checker.score_comparison([(-4., True)], [(-4., False)])["exact"])
        for a, b in (([], []), ([(-1., True)], []), ([(float("nan"), True)], [(-1., True)])):
            with self.assertRaises(ValueError):
                checker.score_comparison(a, b)

    def test_only_fixed_bounded_validation_targets(self):
        cell = {"dataset": "mmlu", "model": "model"}
        rows = [{"split": "validation", "prompt_token_lengths": {"model": n}} for n in (3, 9, 9)]
        self.assertEqual(checker.validation_selection(rows, cell), [0, 1])
        self.assertEqual(checker.validation_selection(rows, cell, [1, 2]), [1, 2])
        with self.assertRaises(ValueError):
            checker.validation_selection([{**rows[0], "split": "test"}], cell)
        with self.assertRaises(ValueError):
            checker.validation_selection(rows*6, cell, list(range(17)))

    def test_capture_releases_context_and_retains_only_first_candidate(self):
        class FakeTensor:
            def detach(self):
                return self
            def clone(self):
                return "in_memory_raw"
        capture = checker.CaptureResidual()
        delta = FakeTensor()
        for candidate in range(2):
            with capture.context("same", 1, (0, 1), (2, 3, candidate)):
                self.assertIs(capture(delta, 0), delta)
                self.assertIs(capture(delta, 1), delta)
            self.assertIsNone(capture.active)
        self.assertEqual(capture.raw, ["in_memory_raw", "in_memory_raw"])


if __name__ == "__main__":
    unittest.main()
