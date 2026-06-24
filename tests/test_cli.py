import unittest
from contextlib import redirect_stdout
from io import StringIO

from tflt.cli import main


class CliTest(unittest.TestCase):
    def test_eval_dry_run(self):
        out = StringIO()
        with redirect_stdout(out):
            code = main(
                [
                    "eval",
                    "--model",
                    "qwen3-1.7b-base",
                    "--tasks",
                    "sciq",
                    "--limit",
                    "1",
                    "--dry-run",
                ]
            )
        self.assertEqual(code, 0)
        self.assertIn("tflt.eval_runner", out.getvalue())


if __name__ == "__main__":
    unittest.main()
