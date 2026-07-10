import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from tflt.cli import main
from tflt.loopscope.schema import PROBE_SCHEMA_VERSION, manifest_sha256
try:
    from loopscope_fixtures import finalize_probe_pool
except ModuleNotFoundError:
    from tests.loopscope_fixtures import finalize_probe_pool


def _probe_report(window=False):
    summary = {"count": 2, "mean": 0.5, "median": 0.5, "p90": 0.5}
    report = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "git": {"root": "/repo", "branch": "loopscope", "commit": "abc", "dirty": False},
        "model": {
            "alias": "m",
            "repo_id": "org/m",
            "revision": "r",
            "layer_count": 28 if window else 1,
        },
        "tokenizer": {
            "revision": "r",
            "choice_token_ids": {} if window else {"A": {}, "B": {}},
        },
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "versions": {
                "python": "3.11",
                "torch": "2.3.1",
                "transformers": "4.51.3",
                "lm_eval": "0.4.11",
                "tflt": "source-tree",
            },
        },
        "probe_pool": {
            "source": "fixture",
            "split": "dev",
            "count": 2,
            "seed": 1,
            "manifest_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "selected_subset_sha256": "a" * 64,
            "source_manifest_count": 2,
            "sample_ids": ["x", "y"],
            "records": [
                {"id": "x", "prompt_sha256": "e" * 64},
                {"id": "y", "prompt_sha256": "f" * 64},
            ],
        },
        "position_rule": "last_non_padding",
        "warnings": [],
    }
    finalize_probe_pool(report["probe_pool"])
    if window:
        examples = [
            {"sample_id": sample_id, "valid": True, "errors": []}
            for sample_id in ("x", "y")
        ]
        report["boundary_metrics"] = []
        report["window_grid"] = {
            "manifest_sha256": "c" * 64,
            "layer_count": 28,
            "candidate_windows": ["12:15"],
        }
        report["window_metrics"] = [
            {
                "window": "12:15",
                "valid": True,
                "sample_count": 2,
                "valid_sample_count": 2,
                "answer_position": {"r": summary, "q": summary},
                "all_non_padding_tokens": {"r": summary, "q": summary},
                "examples": examples,
                "errors": [],
            }
        ]
    else:
        report["boundary_contract"] = {
            "version": "loopscope.boundary.v1",
            "definition": "B_j is the state after decoder layers 0 through j-1",
            "boundary_count": 2,
            "window_entry": "B_a",
            "window_exit": "B_(b+1)",
        }
        report["boundary_metrics"] = [
            {
                "boundary_index": boundary_index,
                "after_layer": boundary_index - 1 if boundary_index else None,
                "before_layer": boundary_index if boundary_index < 1 else None,
                "choice_entropy": summary,
                "kl_to_final": summary,
                "top1_to_final_agreement": summary,
                "effective_rank": 1.5,
                "effective_rank_sampling": {
                    "representation_space": "final_norm raw-logit-lens space",
                    "estimator": "gram_spectrum_shannon_effective_rank",
                    "estimator_version": "1",
                    "spectrum": "squared_singular_values",
                    "unit_normalized": True,
                    "centered_across_vectors": True,
                    "count": 2,
                    "sample_ids": ["x", "y"],
                },
            }
            for boundary_index in range(2)
        ]
        report["window_metrics"] = []
        report["examples"] = [
            {
                "id": sample_id,
                "boundary_metrics": [
                    {"boundary_index": boundary_index} for boundary_index in range(2)
                ],
            }
            for sample_id in ("x", "y")
        ]
    return report


class CliTest(unittest.TestCase):
    def test_probe_layers_dispatch_writes_serializable_command_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "layer-probe"
            with patch("tflt.loopscope.probe.run_layer_probe", return_value=_probe_report()):
                code = main(
                    [
                        "probe-layers",
                        "--model",
                        "m",
                        "--input-jsonl",
                        "unused.jsonl",
                        "--input-manifest",
                        "unused.manifest.json",
                        "--output-dir",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0)
            command_args = json.loads((output / "command_args.json").read_text())
            self.assertNotIn("func", command_args)
            self.assertTrue((output / "probe_report.json").exists())

    def test_probe_window_dispatch_writes_serializable_command_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "window-probe"
            with patch(
                "tflt.loopscope.window_probe.run_window_probe",
                return_value=_probe_report(window=True),
            ):
                code = main(
                    [
                        "probe-window",
                        "--model",
                        "m",
                        "--input-jsonl",
                        "unused.jsonl",
                        "--input-manifest",
                        "unused.manifest.json",
                        "--window-grid",
                        "unused-grid.json",
                        "--output-dir",
                        str(output),
                        "--window",
                        "12:15",
                    ]
                )
            self.assertEqual(code, 0)
            command_args = json.loads((output / "command_args.json").read_text())
            self.assertNotIn("func", command_args)
            self.assertTrue((output / "probe_report.json").exists())

    def test_probe_failure_preserves_command_and_error_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed-probe"
            with patch(
                "tflt.loopscope.probe.run_layer_probe",
                side_effect=RuntimeError("fixture failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    main(
                        [
                            "probe-layers",
                            "--model",
                            "m",
                            "--input-jsonl",
                            "unused.jsonl",
                            "--input-manifest",
                            "unused.manifest.json",
                            "--output-dir",
                            str(output),
                        ]
                    )
            self.assertTrue((output / "command_args.json").exists())
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["message"], "fixture failure")

    def test_loopscope_help_commands_do_not_import_model_stack(self):
        commands = (
            "probe-layers",
            "probe-window",
            "make-window-grid",
            "score-windows",
            "analyze-window-grid",
        )
        for command in commands:
            out = StringIO()
            with self.assertRaises(SystemExit) as caught, redirect_stdout(out):
                main([command, "--help"])
            self.assertEqual(caught.exception.code, 0)
            self.assertIn("usage:", out.getvalue())

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
