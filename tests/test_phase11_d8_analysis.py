import copy
import unittest

try:
    import numpy
except ImportError:
    numpy = None

from tflt.loopscope.phase11_d8 import CELLS
from tflt.loopscope.phase11_d8_analysis import analyze


@unittest.skipIf(numpy is None, "NumPy statistics checked in the pinned HPC environment")
class D8AnalysisTests(unittest.TestCase):
    def data(self):
        data = {}
        for key, ncorrect in zip(CELLS, (200, 220, 180)):
            data[key] = [{"prediction_index": 0 if i < ncorrect else None,
                "answer_mode": "single" if i < ncorrect else "no_match", "end_reason": "eos",
                "unclosed_thinking": i >= ncorrect, "final_empty": i >= ncorrect,
                "anomalous_thinking_boundaries": False, "elapsed_seconds": 1,
                "peak_allocated_bytes": 10, "peak_reserved_bytes": 12,
                "generated_token_count": 20, "thinking_token_count": 15, "final_token_count": 3}
                for i in range(448)]
        return data

    def test_full_denominator_and_single_two_comparison_family(self):
        result = analyze(self.data(), [0] * 448, replicates=100)
        self.assertEqual([c["correct"] for c in result["cells"]], [200, 220, 180])
        self.assertEqual([c["n"] for c in result["cells"]], [448] * 3)
        self.assertEqual(len(result["contrasts"]), 2)
        a, b = result["contrasts"]
        self.assertEqual((a["wrong_to_right"], a["right_to_wrong"]), (20, 0))
        self.assertEqual((b["wrong_to_right"], b["right_to_wrong"]), (0, 20))
        self.assertAlmostEqual(a["mcnemar_exact_p"], 2 ** -19)
        self.assertEqual(a["holm_family_size"], 2)
        self.assertAlmostEqual(a["delta_pp"], 100 * 20 / 448)
        self.assertEqual(a["bootstrap"]["seed"], 20261018)
        self.assertEqual(result, analyze(self.data(), [0] * 448, replicates=100))

    def test_incomplete_population_rejected(self):
        data = copy.deepcopy(self.data())
        data[CELLS[1]].pop()
        with self.assertRaises(ValueError):
            analyze(data, [0] * 448, replicates=10)


if __name__ == "__main__":
    unittest.main()
