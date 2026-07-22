from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tflt.loopscope import phase5_outcome as outcome
from tflt.loopscope.phase3_p3c import safe_test_doc_sha256
from tflt.loopscope.phase3_schema import sanitized_content_sha256


class OutcomeTrap(dict):
    """Fail if identity projection touches lm-eval internal/outcome fields."""

    forbidden = frozenset((
        "doc_hash", "target", "answer", "gold", "prediction", "correctness",
        "resps", "filtered_resps", "metrics", "accuracy",
    ))

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError("forbidden field accessed: %s" % key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError("forbidden field accessed: %s" % key)
        return super().get(key, default)


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

    @staticmethod
    def _record(task, doc_id, doc):
        safe_hash = safe_test_doc_sha256(doc)
        return {
            "identity": {"task": task, "doc_id": str(doc_id), "doc_hash": safe_hash},
            "subject": doc["subject"],
            "split": "test",
            "sanitized_content_sha256": sanitized_content_sha256(
                doc["question"], doc["choices"]
            ),
        }

    def test_identity_projection_recomputes_safe_hash_without_outcomes(self):
        doc_a = {
            "question": "Question A?", "choices": ["A0", "A1", "A2", "A3"],
            "subject": "a",
        }
        doc_b = {
            "question": "Question B?", "choices": ["B0", "B1", "B2", "B3"],
            "subject": "b",
        }
        records = [
            self._record("mmlu_a", 0, doc_a),
            self._record("mmlu_b", 1, doc_b),
        ]
        result = {"samples": {
            "mmlu_b": [OutcomeTrap({
                "doc_id": 1, "doc_hash": "1" * 64, "doc": doc_b,
                "target": object(), "resps": object(), "accuracy": object(),
            })],
            "mmlu_a": [OutcomeTrap({
                "doc_id": 0, "doc_hash": "2" * 64, "doc": doc_a,
                "gold": object(), "prediction": object(), "correctness": object(),
            })],
        }}
        projected = outcome.project_identity_only(result, records)
        self.assertEqual([row["identity"] for row in projected], [row["identity"] for row in records])
        self.assertNotIn("accuracy", projected[0])

    def test_identity_projection_fails_on_safe_doc_or_membership_drift(self):
        doc = {
            "question": "Question?", "choices": ["A", "B", "C", "D"],
            "subject": "subject",
        }
        record = self._record("mmlu_subject", 0, doc)
        changed = dict(doc)
        changed["question"] = "Changed?"
        with self.assertRaisesRegex(outcome.Phase5OutcomeError, "safe canonical doc hash"):
            outcome.project_identity_only(
                {"samples": {"mmlu_subject": [{"doc_id": 0, "doc": changed}]}},
                [record],
            )
        with self.assertRaisesRegex(outcome.Phase5OutcomeError, "missing=1 extra=1"):
            outcome.project_identity_only(
                {"samples": {"mmlu_subject": [{"doc_id": 1, "doc": doc}]}},
                [record],
            )

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
