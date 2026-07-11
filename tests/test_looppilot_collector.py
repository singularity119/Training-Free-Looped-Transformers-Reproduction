import unittest

try:
    import torch
except Exception:  # pragma: no cover - python -S contract
    torch = None

from tflt.looppilot.collector import TensorSignalCollector


@unittest.skipIf(torch is None, "torch is intentionally optional for local no-site tests")
class TensorSignalCollectorTest(unittest.TestCase):
    def test_collects_first_and_second_call_without_mutating_inputs(self):
        x0 = torch.tensor([[[3.0, 4.0], [100.0, 0.0]]])
        y0 = torch.tensor([[[6.0, 8.0], [-100.0, 0.0]]])
        first_updates = [
            torch.tensor([[[1.0, 0.0], [999.0, 0.0]]]),
            torch.tensor([[[1.0, 0.0], [999.0, 0.0]]]),
        ]
        x0_before = x0.clone()
        y0_before = y0.clone()
        collector = TensorSignalCollector(torch.tensor([[True, False]]), epsilon=1e-12)

        collector.observe_operator_call(x0, y0, first_updates)
        probe = collector.controller_probe(x0, y0)
        self.assertAlmostEqual(probe["r0_median"], 1.0)
        self.assertAlmostEqual(probe["c0"], 1.0)
        self.assertAlmostEqual(probe["n0_median"], 1.0)

        x1 = torch.tensor([[[4.5, 6.0], [0.0, 0.0]]])
        y1 = torch.tensor([[[5.5, 6.0], [999.0, 0.0]]])
        collector.observe_operator_call(x1, y1, [])
        result = collector.finalize(expected_calls=2)

        self.assertEqual(result["diagnostics_status"], "paid")
        self.assertIsNotNone(result["q1"])
        self.assertIsNotNone(result["residual_cosine"])
        self.assertEqual(result["operator_body_calls"], 2)
        self.assertTrue(torch.equal(x0, x0_before))
        self.assertTrue(torch.equal(y0, y0_before))

    def test_never_loop_second_call_is_explicitly_not_paid(self):
        collector = TensorSignalCollector(torch.tensor([[True]]))
        x0 = torch.tensor([[[1.0, 0.0]]])
        y0 = torch.tensor([[[2.0, 0.0]]])
        collector.observe_operator_call(x0, y0, [y0 - x0])

        result = collector.finalize(expected_calls=1)

        self.assertEqual(result["diagnostics_status"], "not_paid")
        self.assertIsNone(result["q1"])
        self.assertIsNone(result["residual_cosine"])

    def test_rejects_bad_mask_non_finite_and_extra_body_call(self):
        x0 = torch.tensor([[[1.0]]])
        y0 = torch.tensor([[[2.0]]])
        with self.assertRaises(ValueError):
            TensorSignalCollector(torch.tensor([[False]])).observe_operator_call(x0, y0, [y0 - x0])
        with self.assertRaises(ValueError):
            TensorSignalCollector(torch.tensor([[True]])).observe_operator_call(
                x0, torch.tensor([[[float("nan")]]]), [y0 - x0]
            )
        collector = TensorSignalCollector(torch.tensor([[True]]))
        collector.observe_operator_call(x0, y0, [y0 - x0])
        collector.observe_operator_call(x0, y0, [])
        with self.assertRaises(RuntimeError):
            collector.observe_operator_call(x0, y0, [])


if __name__ == "__main__":
    unittest.main()
