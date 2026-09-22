import unittest

from tflt.loopscope.phase9_runtime import (
    MATCHED_NORM,
    ONLINE_T0,
    Phase9DiagnosticCollector,
    Phase9Runtime,
)


class Phase9RuntimeLifecycleTests(unittest.TestCase):
    def test_context_validates_and_releases_prompt_state(self):
        runtime = Phase9Runtime(ONLINE_T0)
        with runtime.context("sample-a", 3, (1, 2, 3), token_ids=(7, 8, 9, 10)):
            self.assertIsNotNone(runtime._active)
            self.assertIsNone(runtime.direction)
            with self.assertRaises(RuntimeError):
                with runtime.context("nested", 1, (1,), token_ids=(1,)):
                    pass
        self.assertIsNone(runtime._active)

    def test_new_prompt_releases_cached_direction_key(self):
        runtime = Phase9Runtime(MATCHED_NORM)
        with runtime.context("sample-a", 1, (0, 1), token_ids=(10, 11)):
            first_key = runtime._cache_key
        with runtime.context("sample-b", 1, (0, 1), token_ids=(10, 11)):
            second_key = runtime._cache_key
        self.assertNotEqual(first_key, second_key)
        self.assertIsNone(runtime.direction)

    def test_invalid_contexts_fail_before_tensor_import(self):
        runtime = Phase9Runtime(ONLINE_T0)
        for args in (
            ("sample", 0, (), (1,)),
            ("sample", 0, (1, 0), (1, 2)),
            ("sample", 2, (0, 1), (1, 2)),
        ):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    with runtime.context(args[0], args[1], args[2], token_ids=args[3]):
                        pass

    def test_label_free_collector_records_cpu_reference_and_no_intervention(self):
        import torch

        collector = Phase9DiagnosticCollector(["sample-a"], 2)
        runtime = Phase9Runtime(ONLINE_T0, collector=collector, intervene=False)
        delta = torch.tensor(
            [[[1.0, 2.0], [2.0, 4.0], [3.0, 1.0]]], dtype=torch.float32
        )
        with runtime.context("sample-a", 2, (0, 1, 2), token_ids=(10, 11, 12)):
            self.assertIs(runtime(delta, 0), delta)
            self.assertIs(runtime(delta, 1), delta)
        self.assertEqual(runtime.context_summaries[0]["call_count"], 2)
        finished = collector.finish("sample-a", runtime.direction, runtime._fit_metadata)
        self.assertEqual([row["t"] for row in finished["records"]], [0, 1])
        self.assertIn("E0", finished["records"][0])
        self.assertIn("C0t", finished["records"][1])
        self.assertEqual(finished["records"][1]["mutation"]["nonanswer_max_abs_change"], 0.0)
        self.assertLessEqual(
            finished["fit_comparison"]["projection_relative_error"], 1e-3
        )

    def test_zero_strength_keeps_native_residual_object(self):
        import torch

        runtime = Phase9Runtime(ONLINE_T0, strength=0.0)
        delta = torch.tensor([[[1.0, 2.0], [2.0, 4.0]]], dtype=torch.float32)
        with runtime.context("sample", 1, (0, 1), token_ids=(10, 11)):
            runtime(delta, 0)
            updated = runtime(delta, 1)
        self.assertIs(updated, delta)
        self.assertFalse(runtime.calls[-1]["applied"])


if __name__ == "__main__":
    unittest.main()
