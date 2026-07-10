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
try:
    from loopscope_fixtures import finalize_probe_pool
except ModuleNotFoundError:
    from tests.loopscope_fixtures import finalize_probe_pool


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
        "boundary_contract": {
            "version": "loopscope.boundary.v1",
            "definition": "B_j is the state after decoder layers 0 through j-1",
            "boundary_count": 2,
            "window_entry": "B_a",
            "window_exit": "B_(b+1)",
        },
        "boundary_metrics": [
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
                    "estimator_version": "2",
                    "spectrum": "squared_singular_values",
                    "unit_normalized": True,
                    "centered_across_vectors": True,
                    "zero_centered_spectrum": False,
                    "centered_spectrum_mass": 2.0,
                    "count": 2,
                    "sample_ids": ["x", "y"],
                },
            }
            for boundary_index in range(2)
        ],
        "window_metrics": [],
        "examples": [
            {
                "id": sample_id,
                "boundary_metrics": [
                    {"boundary_index": boundary_index} for boundary_index in range(2)
                ],
            }
            for sample_id in ("x", "y")
        ],
        "warnings": [],
    }
    finalize_probe_pool(report["probe_pool"])
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
        report["boundary_metrics"][0]["choice_entropy"]["mean"] = math.nan
        with self.assertRaises(SchemaError):
            validate_probe_report(report)

    def test_missing_revision_is_rejected(self):
        report = _report()
        report["model"]["revision"] = None
        with self.assertRaises(SchemaError):
            validate_probe_report(report)

    def test_duplicate_or_missing_boundary_is_rejected(self):
        report = _report()
        report["model"]["layer_count"] = 2
        report["boundary_metrics"].append(dict(report["boundary_metrics"][0]))
        with self.assertRaises(SchemaError):
            validate_probe_report(report)

    def test_probe_schema_accepts_explicit_zero_centered_spectrum(self):
        report = _report()
        for metric in report["boundary_metrics"]:
            metric["effective_rank_sampling"].update(
                {
                    "estimator_version": "2",
                    "zero_centered_spectrum": False,
                    "centered_spectrum_mass": 2.0,
                }
            )
        report["boundary_metrics"][0]["effective_rank"] = 0.0
        report["boundary_metrics"][0]["effective_rank_sampling"].update(
            {
                "zero_centered_spectrum": True,
                "centered_spectrum_mass": 0.0,
            }
        )
        validate_probe_report(report)

    def test_probe_schema_rejects_inconsistent_effective_rank_contract(self):
        mutations = (
            (0.0, False, 0.0),
            (0.0, True, 1.0),
            (1.0, True, 1.0),
            (1.0, False, 0.0),
            (-1.0, False, 1.0),
            (math.nan, False, 1.0),
            (1.0, False, math.inf),
        )
        for value, flag, mass in mutations:
            with self.subTest(value=value, flag=flag, mass=mass):
                report = _report()
                metric = report["boundary_metrics"][0]
                metric["effective_rank"] = value
                metric["effective_rank_sampling"]["zero_centered_spectrum"] = flag
                metric["effective_rank_sampling"]["centered_spectrum_mass"] = mass
                with self.assertRaises(SchemaError):
                    validate_probe_report(report)

    def test_probe_schema_rejects_v2_and_estimator_v1(self):
        report = _report()
        report["schema_version"] = "loopscope.probe.v2"
        with self.assertRaises(SchemaError):
            validate_probe_report(report)
        report = _report()
        report["boundary_metrics"][0]["effective_rank_sampling"][
            "estimator_version"
        ] = "1"
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
