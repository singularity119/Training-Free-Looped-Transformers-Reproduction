import math
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.schema import (
    PROBE_SCHEMA_VERSION,
    SchemaError,
    attach_manifest_sha256,
    ensure_new_directory,
    manifest_sha256,
    probe_summary,
    validate_probe_report,
    verify_manifest_sha256,
)


def _report():
    summary = {"count": 2, "mean": 0.5, "median": 0.5, "p90": 0.5}
    report = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "git": {"root": "/repo", "branch": "loopscope", "commit": "abc", "dirty": False},
        "model": {"alias": "m", "repo_id": "org/m", "revision": "r", "layer_count": 1},
        "tokenizer": {"revision": "r", "choice_token_ids": {"A": {}, "B": {}}},
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
        "layer_metrics": [
            {
                "layer_index": 0,
                "choice_entropy": summary,
                "kl_to_final": summary,
                "top1_to_final_agreement": summary,
                "effective_rank": 1.5,
                "effective_rank_sampling": {
                    "representation_space": "final_norm raw-logit-lens space",
                    "count": 2,
                    "sample_ids": ["x", "y"],
                },
            }
        ],
        "window_metrics": [],
        "examples": [
            {"id": "x", "layer_metrics": [{"layer_index": 0}]},
            {"id": "y", "layer_metrics": [{"layer_index": 0}]},
        ],
        "warnings": [],
    }
    pool = report["probe_pool"]
    selected = {
        "schema_version": "loopscope.probe-pool-selection.v1",
        "source": pool["source"],
        "split": pool["split"],
        "count": pool["count"],
        "seed": pool["seed"],
        "sample_ids": pool["sample_ids"],
        "records": pool["records"],
        "source_manifest_sha256": pool["source_manifest_sha256"],
    }
    pool["manifest_sha256"] = manifest_sha256(selected)
    pool["selected_subset_sha256"] = pool["manifest_sha256"]
    return report


class LoopScopeSchemaTest(unittest.TestCase):
    def test_probe_summary_is_scalar_only(self):
        summary = probe_summary(_report())["results"]["loopscope_probe"]
        self.assertEqual(summary["probe_count"], 2)
        self.assertEqual(summary["valid_window_count"], 0)
        scalar_types = (str, int, float, bool)
        self.assertTrue(all(isinstance(value, scalar_types) for value in summary.values()))

    def test_non_finite_report_is_rejected(self):
        report = _report()
        report["layer_metrics"][0]["choice_entropy"]["mean"] = math.nan
        with self.assertRaises(SchemaError):
            validate_probe_report(report)

    def test_missing_revision_is_rejected(self):
        report = _report()
        report["model"]["revision"] = None
        with self.assertRaises(SchemaError):
            validate_probe_report(report)

    def test_duplicate_or_missing_layer_is_rejected(self):
        report = _report()
        report["model"]["layer_count"] = 2
        report["layer_metrics"].append(dict(report["layer_metrics"][0]))
        with self.assertRaises(SchemaError):
            validate_probe_report(report)

    def test_manifest_hash_detects_change(self):
        payload = attach_manifest_sha256({"schema_version": "x", "value": 1})
        verify_manifest_sha256(payload)
        payload["value"] = 2
        with self.assertRaises(SchemaError):
            verify_manifest_sha256(payload)

    def test_output_directory_is_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new"
            ensure_new_directory(path)
            with self.assertRaises(FileExistsError):
                ensure_new_directory(path)


if __name__ == "__main__":
    unittest.main()
