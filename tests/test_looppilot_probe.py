import hashlib
import json
import unittest
from pathlib import Path

from tflt.looppilot.probe import (
    GATE_C_TASKS,
    assert_no_forbidden_fields,
    build_doc_identity,
    build_question_mask,
    validate_gate_c_config,
    validate_action_record,
)


ROOT = Path(__file__).resolve().parents[1]


class GateCProbeHelperTest(unittest.TestCase):
    def test_question_mask_uses_only_current_doc_offsets(self):
        context = "fewshot answer\nCurrent question: Q?\nAnswer:"
        current = "Current question: Q?\nAnswer:"
        offsets = [(0, 0), (0, 7), (8, 14), (15, 22), (23, 31), (32, 34), (35, 42)]
        attention = [1] * len(offsets)

        mask = build_question_mask(context, current, offsets, attention)

        self.assertEqual(mask[:3], [False, False, False])
        self.assertTrue(any(mask[3:]))
        self.assertFalse(mask[0])

    def test_doc_identity_is_renderer_bound_and_label_free(self):
        doc = {"question": "Q?", "choices": ["A", "B"], "answer": 1, "subject": "x"}
        first = build_doc_identity(
            task="mmlu_anatomy",
            task_version="1.0",
            renderer_hash="a" * 64,
            occurrence_id="test:0",
            doc=doc,
        )
        second = build_doc_identity(
            task="mmlu_anatomy",
            task_version="1.0",
            renderer_hash="b" * 64,
            occurrence_id="test:0",
            doc=doc,
        )
        self.assertNotEqual(first["doc_key"], second["doc_key"])
        self.assertNotIn("answer", first["manifest_doc"])
        self.assertEqual(len(first["doc_hash"]), 64)
        self.assertEqual(
            first["manifest_doc_hash"],
            hashlib.sha256(first["manifest_doc_json"].encode("utf-8")).hexdigest(),
        )

    def test_config_is_exactly_frozen(self):
        valid = {
            "schema_version": 1,
            "model": "Qwen/Qwen3-1.7B-Base",
            "model_revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
            "tokenizer_revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
            "dtype": "float16",
            "task": "mmlu",
            "num_fewshot": 5,
            "batch_size": 1,
            "window": [12, 15],
            "k": 2,
            "iteration_mode": "block",
            "strategy": "damped_euler",
            "alpha": 1.0,
            "beta": 0.0,
            "cache_strategy": "last",
            "decode_mode": "bypass",
            "tasks": list(GATE_C_TASKS),
            "endpoint_allclose": {"rtol": 0.001, "atol": 0.001},
            "restore_allclose": {"rtol": 0.001, "atol": 0.001},
            "fewshot_seed": 1234,
            "split": "test",
            "doc_index": 0,
            "epsilon": 1e-12,
            "use_cache": True,
        }
        validate_gate_c_config(valid)
        invalid = dict(valid, batch_size="auto")
        with self.assertRaises(ValueError):
            validate_gate_c_config(invalid)

    def test_checked_in_gate_c_config_is_frozen(self):
        payload = json.loads(
            (ROOT / "configs/looppilot/gate_c_probe.json").read_text(encoding="utf-8")
        )
        validate_gate_c_config(payload)

    def test_action_record_enforces_body_call_closure(self):
        row = {
            "schema_version": 1,
            "doc_key": "key",
            "task": "mmlu_anatomy",
            "action": "never_loop",
            "decision_reason": "never_loop_engineering_control",
            "k_used": 1,
            "operator_body_calls": 1,
            "endpoint": {"sha256": "a" * 64},
            "audit": {},
        }
        validate_action_record(row)
        row["operator_body_calls"] = 2
        with self.assertRaises(ValueError):
            validate_action_record(row)

    def test_artifacts_reject_gold_and_correctness_keys_recursively(self):
        assert_no_forbidden_fields({"doc": {"question": "Q", "choices": ["A", "B"]}})
        for key in ("gold", "answer", "correctness", "is_correct"):
            with self.assertRaises(ValueError):
                assert_no_forbidden_fields({"nested": [{key: 0}]})


if __name__ == "__main__":
    unittest.main()
