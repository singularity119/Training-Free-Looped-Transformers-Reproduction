import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tflt.loopscope.phase6_runtime import (
    GateCRuntimeError,
    assemble_raw_boundaries,
    assert_native_no_loop_runtime,
    normalize_raw_boundary_vectors,
    semantic_sha256,
)
from tflt.wrapper import LoopIdentityLayer


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/loopscope/run_qwen4_phase6_gate_c.py"
SPEC = importlib.util.spec_from_file_location("phase6_gate_c", SCRIPT)
gate_c = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate_c)


def _duplicate_inputs():
    records = []
    evidence = []
    for index in range(4):
        identity = "identity-%d" % index
        record = {
            "canonical_identity": identity,
            "H": [float(index), 1.0],
            "D": [1.0, 0.0],
        }
        digest = semantic_sha256(record)
        closure = {
            "canonical_identity": identity,
            "prompt_sha256": "a" * 64,
            "generated_completion_sha256": "b" * 64,
            "generation_ids_sha256": "c" * 64,
            "replay_ids_sha256": "c" * 64,
            "generation_length": 9,
            "replay_length": 12,
            "anchor_token_index": 4,
            "answer_span_start_offset": 20,
            "answer_span_end_offset": 21,
            "answer_match_count": 2,
            "selected_match_ordinal": 0,
            "answer_first_token_index": 5,
            "record_semantic_sha256": digest,
            "generation_count": 1,
            "replay_count": 1,
            "loop_insertions": 0,
            "anchor_resolved": True,
            "unique_token_mapping": True,
            "final_norm_pre_hook_count": 1,
            "raw_boundary_count": 37,
            "final_norm_postnorm_allclose": True,
            "final_norm_postnorm_max_abs": 0.0,
            "replay_next_token_closure": True,
        }
        for replicate in (1, 2):
            records.append({"replicate": replicate, "record": dict(record)})
            evidence.append({"replicate": replicate, **closure})
    return records, evidence


class GateCRuntimePureTests(unittest.TestCase):
    def test_native_runtime_checks_model_application_not_harmless_imports(self):
        class NativeLayer:
            pass

        class NativeModel:
            def __init__(self):
                self.layer = NativeLayer()

            def named_modules(self):
                return iter((("", self), ("model.layers.0", self.layer)))

        self.assertIn("tflt.wrapper", sys.modules)
        assert_native_no_loop_runtime(NativeModel())

    def test_native_runtime_rejects_registered_loop_wrapper(self):
        class WrappedModel:
            def __init__(self):
                self.layer = LoopIdentityLayer()

            def named_modules(self):
                return iter((("", self), ("model.layers.0", self.layer)))

        with self.assertRaisesRegex(GateCRuntimeError, "model.layers.0"):
            assert_native_no_loop_runtime(WrappedModel())

    def test_raw_b36_replaces_postnorm_endpoint_and_requires_one_hook(self):
        hidden = tuple("post-%d" % index for index in range(37))
        raw = assemble_raw_boundaries(hidden, "raw-36", 1)
        self.assertEqual(len(raw), 37)
        self.assertEqual(raw[:-1], hidden[:-1])
        self.assertEqual(raw[-1], "raw-36")
        with self.assertRaisesRegex(GateCRuntimeError, "count"):
            assemble_raw_boundaries(hidden, "raw-36", 2)
        with self.assertRaisesRegex(GateCRuntimeError, "hidden states"):
            assemble_raw_boundaries(hidden[:-1], "raw-36", 1)

    def test_raw_boundary_vectors_normalize_exact_3d_2d_mix(self):
        hidden = tuple(
            np.full((1, 3, 4), float(index), dtype=np.float32)
            for index in range(36)
        )
        raw_b36 = np.full((1, 4), 36.0, dtype=np.float32)
        vectors = normalize_raw_boundary_vectors(hidden + (raw_b36,))
        self.assertEqual(len(vectors), 37)
        self.assertTrue(all(vector.shape == (1, 4) for vector in vectors))
        np.testing.assert_array_equal(vectors[0], hidden[0][:, -1, :])
        np.testing.assert_array_equal(vectors[-1], raw_b36)

        with self.assertRaisesRegex(GateCRuntimeError, "B0.*\\[1,S,H\\]"):
            normalize_raw_boundary_vectors((hidden[0][:, -1, :],) + hidden[1:] + (raw_b36,))
        with self.assertRaisesRegex(GateCRuntimeError, "raw B36"):
            normalize_raw_boundary_vectors(hidden + (raw_b36[:, None, :],))

    def test_duplicate_closure_requires_exact_two_equal_replicates(self):
        records, evidence = _duplicate_inputs()
        receipt = gate_c._duplicate_closure(records, evidence)
        self.assertTrue(receipt["deterministic_duplicate_pass"])
        self.assertEqual(receipt["identity_count"], 4)
        self.assertEqual(len(receipt["rows"]), 4)
        self.assertTrue(
            all(row["answer_match_count"] == 2 for row in receipt["rows"])
        )
        self.assertTrue(
            all(row["selected_match_ordinal"] == 0 for row in receipt["rows"])
        )

        altered_records, altered_evidence = _duplicate_inputs()
        altered_evidence[1]["generation_ids_sha256"] = "d" * 64
        with self.assertRaisesRegex(gate_c.GateCError, "mismatch"):
            gate_c._duplicate_closure(altered_records, altered_evidence)

    def test_write_new_json_is_exclusive_and_canonicalizable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            digest = gate_c._write_new_json(path, {"status": "PASS"})
            self.assertEqual(len(digest), 64)
            self.assertEqual(json.loads(path.read_text()), {"status": "PASS"})
            with self.assertRaises(FileExistsError):
                gate_c._write_new_json(path, {"status": "CHANGED"})


class GateCRunnerContractTests(unittest.TestCase):
    def test_parser_exposes_only_dry_run_smoke_and_verify(self):
        parser = gate_c.build_parser()
        self.assertEqual(parser.parse_args(["dry-run"]).command, "dry-run")
        common = [
            "--run-root",
            "/tmp/fresh",
            "--gate-b-manifest",
            "/tmp/gate_b_manifest.json",
            "--expected-commit",
            "a" * 40,
        ]
        for command in ("smoke", "verify"):
            self.assertEqual(parser.parse_args([command, *common]).command, command)

    def test_dry_run_loads_no_model_data_or_gpu(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(gate_c.main(["dry-run"]), 0)
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["status"], "GATE_C_DRY_RUN_PASS")
        self.assertEqual(payload["identity_count"], 4)
        self.assertEqual(payload["replicates_per_identity"], 2)
        self.assertEqual(payload["generation_count_each"], 1)
        self.assertEqual(payload["replay_count_each"], 1)
        self.assertEqual(payload["loop_insertions"], 0)
        self.assertFalse(payload["model_or_data_loaded"])
        self.assertFalse(payload["gpu_or_slurm_used"])
        self.assertFalse(payload["gate_d_entered"])

    def test_exact_gate_and_provenance_constants_are_bound(self):
        self.assertEqual(
            gate_c.EXECUTOR_THREAD_ID,
            "019fb452-e2af-7bf2-a411-612c81971245",
        )
        self.assertEqual(
            gate_c.PLANNING_THREAD_ID,
            "019fb3de-2298-75f2-a083-0dca453ea79c",
        )
        self.assertEqual(
            gate_c.GATE_B_MANIFEST_SHA256,
            "ccb148bd61971d24ad81b240cbe1b5f0810e06e620fb568473a61b2147b44a1d",
        )
        self.assertEqual(
            gate_c.ORDERED_IDENTITY_SHA256,
            "ac52d6e43c693bd0b47b567c2955dae0c6be895ce3230e854d4e1538f05503a5",
        )


if __name__ == "__main__":
    unittest.main()
