import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import tflt.loopscope.phase6_runtime as phase6_runtime

from tflt.loopscope.phase6_runtime import (
    GateCRuntimeError,
    assemble_raw_boundaries,
    assert_native_no_loop_runtime,
    normalize_raw_boundary_vectors,
    project_boundaries_in_model_dtype,
    semantic_sha256,
    native_replay_argmax_matches_generated,
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
            "replay_mode": "cache_aligned_incremental_use_cache_true",
            "replay_use_cache": True,
            "replay_prompt_length": 7,
            "replay_generated_prefix_length": 5,
            "replay_incremental_step_count": 5,
            "replay_forward_count": 6,
            "replay_step_trace_sha256": "d" * 64,
            "replay_cache_position_closure": True,
            "replay_attention_mask_closure": True,
            "replay_argmax_matches_generated": True,
            "replay_sequence_length": 12,
        }
        for replicate in (1, 2):
            records.append({"replicate": replicate, "record": dict(record)})
            evidence.append({"replicate": replicate, **closure})
    return records, evidence


class GateCRuntimePureTests(unittest.TestCase):
    def test_next_token_closure_uses_native_replay_logits(self):
        class FakeScalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class FakeFinite:
            def all(self):
                return FakeScalar(True)

        class FakeTorch:
            @staticmethod
            def isfinite(_values):
                return FakeFinite()

            @staticmethod
            def argmax(values):
                return FakeScalar(int(np.argmax(values)))

        native_logits = np.asarray([[[0.0, 3.0, 1.0]]], dtype=np.float32)
        self.assertTrue(
            native_replay_argmax_matches_generated(FakeTorch, native_logits, 1)
        )
        self.assertFalse(
            native_replay_argmax_matches_generated(FakeTorch, native_logits, 2)
        )

    def test_cache_aligned_incremental_replay_closes_steps_and_retains_mismatch(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest(
                "local CPU environment does not include torch; remote focused test does"
            )

        class CpuTorch:
            long = torch.long
            inference_mode = staticmethod(torch.inference_mode)
            cat = staticmethod(torch.cat)
            isfinite = staticmethod(torch.isfinite)
            allclose = staticmethod(torch.allclose)
            argmax = staticmethod(torch.argmax)
            ones_like = staticmethod(torch.ones_like)

            @staticmethod
            def tensor(value, *, dtype, device):
                del device
                return torch.tensor(value, dtype=dtype)

            @staticmethod
            def arange(*args, dtype, device):
                del device
                return torch.arange(*args, dtype=dtype)

            @staticmethod
            def ones(shape, *, dtype, device):
                del device
                return torch.ones(shape, dtype=dtype)

        class Cache:
            def __init__(self, length):
                self.length = length

            def get_seq_length(self):
                return self.length

        final_norm = torch.nn.Identity()
        lm_head = torch.nn.Linear(phase6_runtime.HIDDEN_SIZE, 5, bias=False)

        class Model:
            def __init__(self, wrong_cache=False):
                self.calls = []
                self.wrong_cache = wrong_cache

            def named_modules(self):
                return iter((("", self), ("model.norm", final_norm)))

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                input_length = int(kwargs["input_ids"].shape[1])
                previous = kwargs["past_key_values"]
                previous_length = 0 if previous is None else previous.get_seq_length()
                cache_length = previous_length + input_length
                if self.wrong_cache and previous is not None:
                    cache_length += 1
                hidden_states = None
                logits = torch.tensor([[[0.0, 4.0, 1.0, 0.0, 0.0]]])
                if kwargs["output_hidden_states"]:
                    raw = torch.ones((1, 1, phase6_runtime.HIDDEN_SIZE))
                    post = final_norm(raw)
                    hidden_states = tuple(raw.clone() for _ in range(36)) + (post,)
                return SimpleNamespace(
                    past_key_values=Cache(cache_length),
                    hidden_states=hidden_states,
                    logits=logits,
                )

        model = Model()
        runtime = phase6_runtime.GateCRuntime(
            torch=CpuTorch,
            tokenizer=SimpleNamespace(),
            model=model,
            final_norm=final_norm,
            lm_head=lm_head,
            final_norm_path="model.norm",
            lm_head_path="lm_head",
            revision_closure={},
        )
        _logits, _normalized, _raw, evidence = phase6_runtime._replay_once(
            runtime,
            prompt_ids=(10, 11),
            generated_prefix_ids=(20, 21, 22),
            expected_next_token_id=2,
        )
        self.assertEqual(len(model.calls), 4)
        self.assertTrue(all(call["use_cache"] is True for call in model.calls))
        self.assertEqual(
            [int(call["attention_mask"].shape[1]) for call in model.calls],
            [2, 3, 4, 5],
        )
        self.assertEqual(evidence["replay_incremental_step_count"], 3)
        self.assertEqual(evidence["replay_forward_count"], 4)
        self.assertTrue(evidence["replay_cache_position_closure"])
        self.assertTrue(evidence["replay_attention_mask_closure"])
        self.assertFalse(evidence["replay_argmax_matches_generated"])

        failing_model = Model(wrong_cache=True)
        failing_runtime = phase6_runtime.GateCRuntime(
            torch=CpuTorch,
            tokenizer=SimpleNamespace(),
            model=failing_model,
            final_norm=final_norm,
            lm_head=lm_head,
            final_norm_path="model.norm",
            lm_head_path="lm_head",
            revision_closure={},
        )
        with self.assertRaisesRegex(GateCRuntimeError, "KV-cache length"):
            phase6_runtime._replay_once(
                failing_runtime,
                prompt_ids=(10, 11),
                generated_prefix_ids=(20,),
                expected_next_token_id=1,
            )

    def test_zero_match_masks_replay_and_trajectory_but_keeps_sealed_payload(self):
        class Tokenizer:
            all_special_ids = []

            def encode(self, _text):
                return [10]

            def decode(
                self,
                token_ids,
                *,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ):
                del token_ids, skip_special_tokens, clean_up_tokenization_spaces
                return "No exact answer cue."

        runtime = SimpleNamespace(tokenizer=Tokenizer())
        identity = {
            "ordinal": 0,
            "canonical_identity": "identity-not-expressed",
            "category": "biology",
            "rendered_prefix_sha256": "a" * 64,
            "rendered_token_ids_sha256": "b" * 64,
            "sequence_length": 1,
        }
        prompt_metadata = {
            "rendered_prefix_sha256": "a" * 64,
            "rendered_token_ids_sha256": "b" * 64,
            "sequence_length": 1,
        }
        with mock.patch.object(
            phase6_runtime, "tokenization_metadata", return_value=prompt_metadata
        ), mock.patch.object(
            phase6_runtime, "_generate_once", return_value=(20,)
        ), mock.patch.object(phase6_runtime, "_replay_once") as replay:
            record, eligibility, evidence, payload = (
                phase6_runtime.acquire_two_pass_record_and_payload(
                    runtime,
                    prefix="prompt",
                    identity=identity,
                    gate_b_manifest_sha256="c" * 64,
                    card_sha256="d" * 64,
                )
            )
        self.assertIsNone(record)
        self.assertEqual(eligibility["eligibility_state"], "ANCHOR_NOT_EXPRESSED")
        self.assertEqual(evidence["replay_count"], 0)
        self.assertEqual(evidence["trajectory_count"], 0)
        self.assertEqual(payload, b"No exact answer cue.")
        replay.assert_not_called()

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

    def test_boundary_projection_casts_only_after_native_lm_head(self):
        class FakeTensor:
            def __init__(self, shape, dtype):
                self.shape = shape
                self.ndim = len(shape)
                self.dtype = dtype

            def float(self):
                return FakeTensor(self.shape, "float32")

        raw = FakeTensor((37, 4), "bfloat16")
        observed = []

        def final_norm(tensor):
            observed.append(("final_norm", tensor.dtype))
            return FakeTensor(tensor.shape, tensor.dtype)

        def lm_head(tensor):
            observed.append(("lm_head", tensor.dtype))
            return FakeTensor((37, 11), tensor.dtype)

        normalized, logits = project_boundaries_in_model_dtype(
            final_norm, lm_head, raw
        )
        self.assertEqual(
            observed,
            [("final_norm", "bfloat16"), ("lm_head", "bfloat16")],
        )
        self.assertEqual(normalized.dtype, "float32")
        self.assertEqual(logits.dtype, "float32")

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
