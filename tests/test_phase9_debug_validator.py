import unittest

from scripts.loopscope.run_phase9_debug import validate_context_calls


class Runtime:
    def __init__(self, contexts):
        self.context_summaries = contexts


class Phase9DebugValidatorTests(unittest.TestCase):
    def test_context_count_is_scoped_to_the_current_score_call(self):
        context = {"call_count": 2, "timesteps": [0, 1]}
        runtime = Runtime([context, context])

        with self.assertRaisesRegex(RuntimeError, "expected candidate contexts"):
            validate_context_calls(runtime, 2, 1, start_index=0)
        self.assertEqual(validate_context_calls(runtime, 2, 1, start_index=1), [context])


if __name__ == "__main__":
    unittest.main()
