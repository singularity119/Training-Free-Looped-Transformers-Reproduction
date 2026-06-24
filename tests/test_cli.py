import unittest
import json
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

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

    def test_report_quotes_metric_headers_with_commas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results.json").write_text(
                json.dumps({"results": {"sciq": {"acc,none": 1.0}}}),
                encoding="utf-8",
            )
            out_path = root / "summary.csv"
            code = main(["report", "--results", str(root), "--output", str(out_path)])
            self.assertEqual(code, 0)
            self.assertIn('"acc,none"', out_path.read_text(encoding="utf-8").splitlines()[0])


if __name__ == "__main__":
    unittest.main()
