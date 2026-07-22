from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tflt.loopscope import phase5_outcome as outcome


class Phase5GateCContractTests(unittest.TestCase):
    def manifest(self, mode="formal"):
        with mock.patch.object(outcome, "implementation_hashes", return_value={"x": "a" * 64}):
            return outcome.build_launch_manifest(
                mode=mode,
                run_root=outcome.AUTHORIZED_RUN_ROOT,
                expected_commit="b" * 40,
                partition="gpu-a800",
                gres="gpu:a800:1",
                qos=None,
                account=None,
                array_throttle=8,
                time_limit="24:00:00",
                batch_size="16",
                created_at_utc="2026-07-22T00:00:00Z",
            )

    def test_exact_panel_and_baseline_loop_contract(self):
        manifest = self.manifest()
        self.assertEqual(
            [(row["role"], row["window"]) for row in manifest["cells"]],
            [(role, window) for _, role, window in outcome.CELLS],
        )
        self.assertNotIn("--loop", manifest["cells"][0]["eval_argv"])
        for cell in manifest["cells"][1:]:
            argv = cell["eval_argv"]
            self.assertIn("--loop", argv)
            self.assertEqual(argv[argv.index("--k") + 1], "3")
            self.assertEqual(argv[argv.index("--strategy") + 1], "euler")
            self.assertEqual(argv[argv.index("--cache-strategy") + 1], "first")
            self.assertEqual(argv[argv.index("--decode-mode") + 1], "bypass")

    def test_panel_drift_fails_closed(self):
        manifest = self.manifest()
        changed = copy.deepcopy(manifest)
        changed["cells"][2]["window"] = "3:6"
        changed.pop("manifest_sha256")
        outcome._attach_manifest(changed)
        with self.assertRaisesRegex(outcome.Phase5OutcomeError, "membership/order"):
            outcome.validate_launch_manifest(changed, mode="formal")

    def test_smoke_is_exact_four_identity_subset(self):
        manifest = self.manifest("smoke")
        self.assertEqual(manifest["population"]["required_count"], 4)
        self.assertEqual(len(manifest["population"]["smoke_identities"]), 4)
        self.assertEqual(manifest["recipe"]["limit"], 2)
        self.assertEqual(
            manifest["cells"][0]["eval_argv"][manifest["cells"][0]["eval_argv"].index("--tasks") + 1],
            "mmlu_abstract_algebra,mmlu_anatomy",
        )

    def test_identity_projection_accesses_only_stable_identity(self):
        records = [
            {"identity": {"task": "mmlu_a", "doc_id": "0", "doc_hash": "a" * 64},
             "subject": "a", "split": "test"},
            {"identity": {"task": "mmlu_b", "doc_id": "1", "doc_hash": "b" * 64},
             "subject": "b", "split": "test"},
        ]
        result = {"samples": {
            "mmlu_b": [{"doc_id": 1, "doc_hash": "b" * 64, "forbidden_outcome": object()}],
            "mmlu_a": [{"doc_id": 0, "doc_hash": "a" * 64, "accuracy": object()}],
        }}
        projected = outcome.project_identity_only(result, records)
        self.assertEqual([row["identity"] for row in projected], [row["identity"] for row in records])
        self.assertNotIn("accuracy", projected[0])

    def test_partial_or_failed_scheduler_refused(self):
        rows = [
            {"JobID": "123_%d" % index, "State": "COMPLETED", "ExitCode": "0:0",
             "ElapsedRaw": "1", "AllocTRES": "gres/gpu=1", "Partition": "p", "NodeList": "n"}
            for index in range(7)
        ]
        with self.assertRaisesRegex(outcome.Phase5OutcomeError, "exact eight"):
            outcome.validate_scheduler_rows(rows, "123")
        rows.append({"JobID": "123_7", "State": "FAILED", "ExitCode": "1:0",
                     "ElapsedRaw": "1", "AllocTRES": "gres/gpu=1", "Partition": "p", "NodeList": "n"})
        with self.assertRaisesRegex(outcome.Phase5OutcomeError, "not COMPLETED"):
            outcome.validate_scheduler_rows(rows, "123")

    def test_write_new_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            outcome._write_new_json(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                outcome._write_new_json(path, {"x": 2})


if __name__ == "__main__":
    unittest.main()
