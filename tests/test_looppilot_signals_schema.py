import json
import math
import tempfile
import unittest
from pathlib import Path

from tflt.looppilot.schema import DocKey, SignalRecord, write_json_once, write_jsonl_once
from tflt.looppilot.signals import compute_signal_row


class SignalSchemaTest(unittest.TestCase):
    def test_signal_row_uses_only_masked_question_tokens(self):
        row = compute_signal_row(
            x0=[[3.0, 4.0], [100.0, 0.0]],
            y0=[[6.0, 8.0], [-100.0, 0.0]],
            question_mask=[True, False],
            layer_updates=[
                [[1.0, 0.0], [999.0, 0.0]],
                [[1.0, 0.0], [999.0, 0.0]],
            ],
            x1=[[4.5, 6.0], [0.0, 0.0]],
            y1=[[5.5, 6.0], [999.0, 0.0]],
            epsilon=1e-12,
        )
        self.assertEqual(row["valid_token_count"], 1)
        self.assertAlmostEqual(row["r0_median"], 1.0)
        self.assertAlmostEqual(row["c0"], 1.0)
        self.assertAlmostEqual(row["n0_median"], 1.0)
        self.assertIsNotNone(row["q1"])

    def test_empty_mask_and_non_finite_inputs_fail_fast(self):
        with self.assertRaises(ValueError):
            compute_signal_row([[1.0]], [[2.0]], [False])
        with self.assertRaises(ValueError):
            compute_signal_row([[1.0]], [[math.inf]], [True])

    def test_signal_record_rejects_nan(self):
        key = DocKey("mmlu", "1", "a" * 64, "b" * 64, "0")
        with self.assertRaises(ValueError):
            SignalRecord(
                key=key,
                row_id="row-0",
                choice_index=0,
                question_token_count=1,
                r0_median=float("nan"),
                r0_p90=1.0,
                r0_max=1.0,
                c0=1.0,
                n0_median=1.0,
                n0_p90=1.0,
                n0_max=1.0,
            )

    def test_write_once_json_and_jsonl_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path = root / "summary.json"
            jsonl_path = root / "signals.jsonl"
            write_json_once(json_path, {"ok": True})
            write_jsonl_once(jsonl_path, [{"row": 1}, {"row": 2}])
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["ok"], True)
            self.assertEqual(len(jsonl_path.read_text(encoding="utf-8").splitlines()), 2)
            with self.assertRaises(FileExistsError):
                write_json_once(json_path, {"ok": False})
            with self.assertRaises(FileExistsError):
                write_jsonl_once(jsonl_path, [])


if __name__ == "__main__":
    unittest.main()
