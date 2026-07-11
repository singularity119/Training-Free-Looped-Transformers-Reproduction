import unittest
from contextlib import redirect_stdout
from io import StringIO

from tflt.cli import main


class LoopPilotCliTest(unittest.TestCase):
    def test_dry_run_passes_controller_and_signal_arguments(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            code = main(
                [
                    "eval",
                    "--model",
                    "qwen3-1.7b-base",
                    "--tasks",
                    "mmlu",
                    "--num-fewshot",
                    "5",
                    "--batch-size",
                    "4",
                    "--dtype",
                    "float16",
                    "--loop",
                    "--looppilot-controller",
                    "always_loop",
                    "--looppilot-signal-jsonl",
                    "/tmp/signals.jsonl",
                    "--dry-run",
                ]
            )
        rendered = stdout.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("--looppilot-controller always_loop", rendered)
        self.assertIn("--looppilot-signal-jsonl /tmp/signals.jsonl", rendered)

    def test_invalid_controller_fails_before_execution(self):
        with self.assertRaises(ValueError):
            main(
                [
                    "eval",
                    "--model",
                    "qwen3-1.7b-base",
                    "--loop",
                    "--looppilot-controller",
                    "silent_fallback",
                    "--dry-run",
                ]
            )

    def test_controller_requires_loop(self):
        with self.assertRaises(ValueError):
            main(
                [
                    "eval",
                    "--model",
                    "qwen3-1.7b-base",
                    "--looppilot-controller",
                    "never_loop",
                    "--dry-run",
                ]
            )


if __name__ == "__main__":
    unittest.main()
