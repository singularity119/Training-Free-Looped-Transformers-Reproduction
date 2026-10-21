"""Network-free checks for the D9 mounted-model source comparator."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    "phase11_d9_model_source",
    Path(__file__).resolve().parents[1] / "scripts/loopscope/phase11_d9_model_source.py",
)
source = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(source)


class Response(io.BytesIO):
    status = 200
    headers = {}


class ModelSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / source.FILES[0]).write_bytes(b"abc")
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def test_equal_full_file(self):
        with patch.object(source.urllib.request, "urlopen", return_value=Response(b"abc")):
            result = source.compare_file(self.root, source.FILES[0], time.monotonic()+100)
        self.assertTrue(result["equal"])
        self.assertEqual(result["bytes_compared"], 3)
        self.assertIn(source.REVISION, result["reference_url"])

    def test_mismatch_and_lengths_fail(self):
        for body in (b"axc", b"ab", b"abcd"):
            with self.subTest(body=body):
                with patch.object(source.urllib.request, "urlopen", return_value=Response(body)):
                    with self.assertRaises(ValueError):
                        source.compare_file(self.root, source.FILES[0], time.monotonic()+100)

    def test_http_operation_timeout_is_capped(self):
        for remaining, expected in ((100, 20), (3, 3)):
            with self.subTest(remaining=remaining):
                with patch.object(source.time, "monotonic", return_value=100), patch.object(
                    source.urllib.request, "urlopen", return_value=Response(b"abc")
                ) as opened:
                    source.compare_file(self.root, source.FILES[0], 100+remaining)
                self.assertEqual(opened.call_args.kwargs["timeout"], expected)

    def test_prefix_diagnostic_never_admits_source(self):
        output = self.root / "diagnostic.json"
        argv = ["model-source", "--model-dir", str(self.root), "--output", str(output),
                "--timeout-seconds", "60", "--probe-only"]
        with patch.object(source.sys, "argv", argv), patch.object(
            source.urllib.request, "urlopen", side_effect=lambda *a, **kw: Response(b"abc")
        ), patch.object(source, "compare_file") as compare:
            self.assertEqual(source.main(), 0)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "NETWORK_PROBE_COMPLETE_NOT_SOURCE_ADMITTED")
        self.assertEqual(report["files"], [])
        self.assertEqual(len(report["endpoint_probes"]), 2)
        self.assertTrue(all(p["status"] == "PREFIX_ONLY_EQUAL" for p in report["endpoint_probes"]))
        compare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
