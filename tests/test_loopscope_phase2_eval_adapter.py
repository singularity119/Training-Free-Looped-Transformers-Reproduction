import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tflt import eval_runner


def identity(index=7):
    return {"task": "mmlu_x", "doc_id": str(index), "doc_hash": "%064x" % (index + 1)}


def logged_sample(sample_identity, scores, gold):
    top = max(range(4), key=lambda index: scores[index])
    return {
        "doc_id": sample_identity["doc_id"],
        "doc_hash": sample_identity["doc_hash"],
        "filtered_resps": list(scores),
        "target": gold,
        "metrics": {"acc,none": int(top == gold)},
    }


class Phase2EvalAdapterTests(unittest.TestCase):
    def test_default_loop_config_has_no_collector(self):
        args = argparse.Namespace(
            model="qwen3-1.7b-base", window="12:15", k=2, iteration_mode="block",
            strategy="damped_euler", alpha=1.0, beta=0.0, cache_strategy="last",
            decode_mode="bypass", first_n=None,
        )
        self.assertIsNone(eval_runner._loop_config(args).audit_collector)

    def test_default_command_args_and_results_schema_have_no_phase2_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "eval"
            with mock.patch.object(eval_runner, "run_lm_eval", return_value={"results": {}}):
                rc = eval_runner.main(
                    ["--model", "qwen3-1.7b-base", "--tasks", "sciq", "--output-dir", str(output)]
                )
            self.assertEqual(rc, 0)
            args = json.loads((output / "command_args.json").read_text(encoding="utf-8"))
            self.assertNotIn("phase2_final_output_manifest", args)
            self.assertFalse((output / "phase2_final_outputs.json").exists())
            self.assertEqual(json.loads((output / "results.json").read_text())["results"], {})

    def test_opt_in_auto_batch_is_allowed_and_sidecar_remains_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "request.json"
            request.write_text("{}", encoding="utf-8")
            output = Path(tmp) / "out"
            context = {"request": {}, "card": {}, "identity_manifest": {}}
            result = {"results": {"mmlu_x": {"acc,none": 1.0}}}
            with mock.patch.object(
                eval_runner, "_load_phase2_final_output_manifest", return_value=context
            ) as load, mock.patch.object(
                eval_runner, "_build_phase2_final_output_sidecar", return_value={"artifact": "final-only"}
            ), mock.patch.object(eval_runner, "run_lm_eval", return_value=result):
                rc = eval_runner.main(
                    [
                        "--model", "qwen3-1.7b-base", "--tasks", "mmlu_x",
                        "--output-dir", str(output),
                        "--phase2-final-output-manifest", str(request),
                    ]
                )
            self.assertEqual(rc, 0)
            load.assert_called_once()
            self.assertEqual(json.loads((output / "phase2_final_outputs.json").read_text()), {"artifact": "final-only"})
            self.assertEqual(json.loads((output / "results.json").read_text()), result)

    def test_exact_join_restores_canonical_order_and_rejects_extra(self):
        first, second = identity(1), identity(2)
        result = {
            "samples": {
                "mmlu_x": [
                    logged_sample(second, [1, 4, 0, -1], 1),
                    logged_sample(first, [4, 1, 0, -1], 0),
                ]
            }
        }
        rows = eval_runner._join_phase2_logged_samples(result, [first, second])
        self.assertEqual([row["sample_identity"] for row in rows], [first, second])
        extra = json.loads(json.dumps(result))
        extra["samples"]["mmlu_x"].append(logged_sample(identity(3), [4, 1, 0, -1], 0))
        with self.assertRaisesRegex(ValueError, "extra=1"):
            eval_runner._join_phase2_logged_samples(extra, [first, second])

    def test_exact_join_rejects_duplicate_and_missing_four_scores(self):
        expected = [identity(1)]
        sample = logged_sample(expected[0], [4, 1, 0, -1], 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            eval_runner._join_phase2_logged_samples(
                {"samples": {"mmlu_x": [sample, dict(sample)]}}, expected
            )
        broken = dict(sample)
        broken["filtered_resps"] = [4, 1, 0]
        broken["resps"] = []
        with self.assertRaisesRegex(ValueError, "four raw choice"):
            eval_runner._join_phase2_logged_samples({"samples": {"mmlu_x": [broken]}}, expected)


if __name__ == "__main__":
    unittest.main()
