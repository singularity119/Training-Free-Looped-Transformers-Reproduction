import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace

from tflt.loopscope.phase11_s2 import extract_s2

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "loopscope"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("d5_closure", SCRIPTS / "close_phase11_d5.py")
d5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d5)


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class D5ClosureTests(unittest.TestCase):
    def test_plan_missing_or_duplicate_indices_cannot_close(self):
        cell = {"family": "p10-base-mmlu", "repeat": 1, "cell_id": "q4-native", "population_count": 2}
        recipe = {"cells": [cell]}
        jobs = [{**cell, "indices": [0], "run_root": "/root/a"},
                {**cell, "indices": [1], "run_root": "/root/b"}]
        self.assertEqual(list(d5.membership(recipe, jobs)), [d5.key(cell)])
        with self.assertRaisesRegex(ValueError, "missing"):
            d5.membership(recipe, jobs[:1])
        jobs[1]["indices"] = [0]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            d5.membership(recipe, jobs)

    def test_predictions_keep_native_normalization_and_original_s2(self):
        job = {"family": "p10-base-arc", "repeat": 1, "cell_id": "q4-native", "seed": 20261014}
        record = {"identity": "sample:0", "canonical_index": 0, "scores": [-2., -3.],
                  "choice_text_lengths": [1, 10]}
        arc = d5.prediction(job, record)
        self.assertEqual((arc["prediction_index"], arc["raw_prediction_index"]), (1, 0))
        job["family"] = "p10-base-mmlu"
        record.update(subject="subject", doc_index=0)
        self.assertEqual(d5.prediction(job, record)["prediction_index"], 0)
        job["family"] = "p11-instruct-gpqa"
        self.assertIs(d5.extract_s2, extract_s2)
        for text in ("Final answer: (A)\nFinal Answer: A", "Final answer: (A)\nFinal answer: (B)", "none"):
            record.update(generated_text=text, generated_token_count=10, end_reason="eos")
            result = d5.prediction(job, record)
            self.assertEqual({k: result[k] for k in extract_s2(text)}, extract_s2(text))

    def test_materialized_records_need_complete_membership_and_finite_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = {"family": "p10-base-mmlu", "repeat": 1, "cell_id": "q4-native", "seed": 20261014,
                   "indices": [0], "run_root": tmp, "source_commit": "source", "producer_commit": "old",
                   "producer_source": "/immutable", "synthetic": False, "engineering": False}
            row = {"identity": "sample:0", "subject": "subject", "doc_index": 0}
            cell = {"model": "model"}
            record = {**d5.base.binding(job), "canonical_index": 0, "identity": row["identity"],
                      "scores": [-1., -2., -3., -4.], "choice_text_lengths": [1]*4,
                      "continuation_token_lengths": [1]*4, "subject": "subject", "doc_index": 0}
            env = root / "env.json"
            write(env, {**d5.base.binding(job), "effective_rng_seeds":
                        {k: job["seed"] for k in ("python", "numpy", "torch", "cuda_all_devices")}})
            telemetry = {**d5.base.binding(job), "canonical_index": 0, "identity": row["identity"],
                         "peak_allocated_bytes": 10, "peak_reserved_bytes": 20, "environment_file": str(env),
                         "position_metadata": [{"continuation_length": 1}]*4}
            command = {"schema": "loopscope.phase11.d5.base_attempt.v1", "job": job, "cell": cell,
                       "dataset": "mmlu", "indices": [0], "target_gold_loaded": False}
            write(root / "command_args.json", command)
            write(root / "summary.json", {"metadata": command, "status": "SCORES_COMPLETE",
                                          "count": 1, "target_gold_loaded": False})
            (root / "complete").mkdir()
            write(root / "complete/000000.json", {"record": record, "telemetry": telemetry})
            for name, value in (("records.jsonl", record), ("telemetry.jsonl", telemetry)):
                (root / name).write_text(json.dumps(value) + "\n")
            context = {"dataset": "mmlu", "rows": [row], "cell": cell,
                       "validator": d5.base.record_validator(job, [row], "mmlu")}
            self.assertEqual(d5.inspect_attempt(job, context), [record])
            (root / "records.jsonl").write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "missing, duplicated"):
                d5.inspect_attempt(job, context)
            (root / "records.jsonl").write_text(json.dumps(record) + "\n")
            bad = copy.deepcopy(record)
            bad["scores"][0] = float("nan")
            (root / "records.jsonl").write_text(json.dumps(bad) + "\n")
            with self.assertRaisesRegex(ValueError, "finite"):
                d5.inspect_attempt(job, context)

    def test_failed_closure_writes_no_partial_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            write(path, {})
            output = Path(tmp) / "output"
            args = Namespace(plan=path, recipe=path, output_dir=output)
            with patch.object(d5, "close", side_effect=ValueError("missing canonical raw record")):
                with self.assertRaises(ValueError):
                    d5.run(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
