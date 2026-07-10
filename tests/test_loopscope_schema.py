import math
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.schema import (
    PROBE_SCHEMA_VERSION,
    SchemaError,
    attach_manifest_sha256,
    ensure_new_directory,
    probe_summary,
    validate_probe_report,
    verify_manifest_sha256,
)


def _report():
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "git": {"branch": "loopscope", "commit": "abc", "dirty": False},
        "model": {"alias": "m", "revision": "r", "layer_count": 2},
        "tokenizer": {"revision": "r", "choice_token_ids": {}},
        "runtime": {"device": "cpu", "dtype": "float32", "versions": {}},
        "probe_pool": {"source": "fixture", "split": "dev", "count": 2},
        "position_rule": "last_non_padding",
        "layer_metrics": [{"layer_index": 0}],
        "window_metrics": [{"window": "0:1", "valid": True}],
        "warnings": [],
    }


class LoopScopeSchemaTest(unittest.TestCase):
    def test_probe_summary_is_scalar_only(self):
        summary = probe_summary(_report())["results"]["loopscope_probe"]
        self.assertEqual(summary["probe_count"], 2)
        self.assertEqual(summary["valid_window_count"], 1)
        self.assertTrue(all(isinstance(value, (str, int, float, bool)) for value in summary.values()))

    def test_non_finite_report_is_rejected(self):
        report = _report()
        report["layer_metrics"][0]["choice_entropy"] = math.nan
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
