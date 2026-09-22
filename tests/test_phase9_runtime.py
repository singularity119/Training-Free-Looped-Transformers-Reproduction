import unittest

from tflt.loopscope.phase9_runtime import MATCHED_NORM, ONLINE_T0, Phase9Runtime


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


if __name__ == "__main__":
    unittest.main()
