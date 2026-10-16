import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
# The CLI imports its adjacent closure CLI, as it does on remote execution.
sys.path.insert(0, str(ROOT / "scripts/loopscope"))
spec = importlib.util.spec_from_file_location("d7_gold_test", ROOT / "scripts/loopscope/prepare_phase11_d7_gold.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def fixture(n=3):
    projected = [{"question_id": 80 + i, "answer_index": i, "category": f"cat{i}"} for i in range(n)]
    pool = [{"identity": f"TIGER-Lab/MMLU-Pro@{cli.REVISION}:test:{i}:{80+i}",
             "subject": f"cat{i}", "choices": ["synthetic"] * (i+2),
             "choice_labels": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:i+2])} for i in range(n)]
    identities = [r["identity"] for r in pool]
    closure = {"status": "GOLD_UNREAD", "record_count": n*5, "cell_count": 5,
               "prediction_closed": True, "closed_at_utc": "synthetic-before-access",
               "cells": [{"cell_key": "synthetic", "source_commit": "original-producer"}]}
    return projected, pool, identities, closure


class D7GoldTests(unittest.TestCase):
    def test_parquet_reader_projects_only_authorized_columns(self):
        projected, _, _, _ = fixture()
        calls = []
        def read_columns(*, columns):
            calls.append(columns)
            return SimpleNamespace(to_pylist=lambda: projected)
        source = SimpleNamespace(schema_arrow=SimpleNamespace(names=list(cli.PROJECTION) + ["cot_content"]),
                                 metadata=SimpleNamespace(num_rows=3), read=read_columns)
        arrow = ModuleType("pyarrow")
        parquet = ModuleType("pyarrow.parquet")
        parquet.ParquetFile = lambda path: source
        arrow.parquet = parquet
        with patch.dict(sys.modules, {"pyarrow": arrow, "pyarrow.parquet": parquet}), patch.object(cli, "COUNT", 3):
            self.assertEqual(cli.read_projection(cli.SOURCE), projected)
            self.assertEqual(calls, [list(cli.PROJECTION)])
            source.schema_arrow.names = ["question_id", "answer", "category"]
            with self.assertRaisesRegex(ValueError, "schema differs"):
                cli.read_projection(cli.SOURCE)
            self.assertEqual(len(calls), 1)

    def test_source_order_identity_category_and_actual_option_range(self):
        projected, pool, identities, _ = fixture()
        with patch.object(cli, "COUNT", 3):
            result = cli.join_gold(projected, pool, identities)
            self.assertEqual([r["label_index"] for r in result], [0, 1, 2])
            self.assertEqual(set(result[0]), {"identity", "label_index", "category"})
            for field, value, error in (("question_id", 999, "identity"),
                                        ("category", "wrong", "category"),
                                        ("answer_index", 2, "option range"),
                                        ("answer_index", True, "option range")):
                wrong = copy.deepcopy(projected)
                wrong[0][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    cli.join_gold(wrong, pool, identities)
            with self.assertRaisesRegex(ValueError, "identity"):
                cli.join_gold(list(reversed(projected)), pool, identities)
            with self.assertRaisesRegex(ValueError, "population"):
                cli.join_gold(projected[:-1], pool, identities)

    def test_incomplete_barrier_prevents_any_projection_or_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempt"
            args = SimpleNamespace(source=cli.SOURCE, closure="closure", output_dir=output)
            with patch.object(cli, "read", return_value={}), \
                 patch.object(cli, "load_before_gold", side_effect=ValueError("incomplete 60160 closure")), \
                 patch.object(cli, "read_projection") as reader:
                with self.assertRaisesRegex(ValueError, "60160"):
                    cli.run(args)
                reader.assert_not_called()
            self.assertFalse(output.exists())

    def test_before_after_receipts_and_no_labels_on_stdout(self):
        projected, pool, identities, closure = fixture()
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, "COUNT", 3):
            output = Path(directory) / "attempt"
            args = SimpleNamespace(source=cli.SOURCE, closure="closure", output_dir=output)
            events = []
            def barrier(value):
                self.assertFalse(output.exists())
                events.append("barrier")
                return {}, pool, identities
            def reader(source):
                events.append("projection")
                pre = json.loads((output / "pre_gold_access.json").read_text())
                self.assertEqual(pre["status"], "CLOSURE_VERIFIED_BEFORE_FIRST_GOLD_ACCESS")
                self.assertFalse(pre["target_gold_read"])
                self.assertEqual(pre["prior_record_count"], 15)
                self.assertEqual(tuple(pre["projection_columns"]), cli.PROJECTION)
                self.assertFalse((output / "post_gold_access.json").exists())
                return projected
            capture = io.StringIO()
            with patch.object(cli, "read", return_value=closure), \
                 patch.object(cli, "load_before_gold", side_effect=barrier), \
                 patch.object(cli, "read_projection", side_effect=reader), contextlib.redirect_stdout(capture):
                self.assertEqual(cli.run(args), 0)
            self.assertEqual(events, ["barrier", "projection"])
            post = json.loads((output / "post_gold_access.json").read_text())
            self.assertEqual(post["status"], "FIRST_GOLD_PROJECTION_READ")
            receipt = json.loads((output / "gold_adapter_receipt.json").read_text())
            self.assertEqual((receipt["joined_count"], receipt["missing_count"]), (3, 0))
            self.assertEqual(receipt["producer_sources"], {"synthetic": "original-producer"})
            self.assertNotIn("label_index", capture.getvalue())
            self.assertNotIn(identities[0], capture.getvalue())
            rows = [json.loads(line) for line in (output / "sealed/gold.jsonl").read_text().splitlines()]
            self.assertEqual([r["identity"] for r in rows], identities)
            with patch.object(cli, "read", return_value=closure), \
                 patch.object(cli, "load_before_gold", return_value=({}, pool, identities)), \
                 patch.object(cli, "read_projection") as reader:
                with self.assertRaises(FileExistsError):
                    cli.run(args)
                reader.assert_not_called()

    def test_fixed_source_revision_required_before_answer_access(self):
        args = SimpleNamespace(source=Path("/another/revision.parquet"), closure="closure", output_dir=Path("unused"))
        with patch.object(cli, "load_before_gold") as barrier, patch.object(cli, "read_projection") as reader:
            with self.assertRaisesRegex(ValueError, "path/revision"):
                cli.run(args)
            barrier.assert_not_called()
            reader.assert_not_called()

    def test_failed_join_preserves_actual_first_access_receipts(self):
        projected, pool, identities, closure = fixture()
        projected[0]["category"] = "mismatch"
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, "COUNT", 3):
            output = Path(directory) / "attempt"
            args = SimpleNamespace(source=cli.SOURCE, closure="closure", output_dir=output)
            with patch.object(cli, "read", return_value=closure), \
                 patch.object(cli, "load_before_gold", return_value=({}, pool, identities)), \
                 patch.object(cli, "read_projection", return_value=projected):
                with self.assertRaisesRegex(ValueError, "category"):
                    cli.run(args)
            self.assertTrue((output / "pre_gold_access.json").exists())
            self.assertTrue((output / "post_gold_access.json").exists())
            failure = json.loads((output / "gold_access_failure.json").read_text())
            self.assertTrue(failure["first_gold_read_attempted"])
            self.assertFalse((output / "sealed/gold.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
