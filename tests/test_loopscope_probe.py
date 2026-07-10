import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope.probe import (
    ProbeInputError,
    choice_tokenization,
    decoder_boundary_states,
    lens_space_hidden,
    load_probe_records,
    parse_choice_labels,
    probe_pool_metadata,
)
from tflt.loopscope.schema import attach_manifest_sha256
from tflt.loopscope.window_probe import WindowProbeCollector, build_window_sample_metrics
try:
    from loopscope_fixtures import contract_record, source_pool_manifest
except ModuleNotFoundError:
    from tests.loopscope_fixtures import contract_record, source_pool_manifest


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == " B":
            return [20, 21]
        return [len(text)]


class LoopScopeProbeTest(unittest.TestCase):
    def _pool_records(self):
        return [
            contract_record(record_id, text, source="mmlu")
            for record_id, text in (("x", "Question X\nAnswer:"), ("y", "Question Y\nAnswer:"))
        ]

    def _pool_manifest(self, records):
        return source_pool_manifest(records)

    def test_choice_tokenization_reports_all_ids_and_never_truncates(self):
        with self.assertRaises(ProbeInputError) as caught:
            choice_tokenization(FakeTokenizer(), ["A", "B"])
        details = caught.exception.details
        self.assertEqual(details["choice_token_ids"]["B"]["token_ids"], [20, 21])
        self.assertFalse(details["choice_token_ids"]["B"]["is_single_token"])

    def test_probe_records_reject_test_split_and_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "x",
                        "text": "Question\nAnswer:",
                        "source": "mmlu",
                        "split": "test",
                        "subject": "math",
                        "answer": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ProbeInputError):
                load_probe_records(path)

    def test_choice_labels_must_be_unique(self):
        with self.assertRaises(ProbeInputError):
            parse_choice_labels("A,A")

    def test_manifest_rejects_changed_prompt_with_same_id(self):
        records = self._pool_records()
        manifest = self._pool_manifest(records)
        changed = [dict(records[0])]
        changed[0]["prompt_sha256"] = "f" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ProbeInputError):
                probe_pool_metadata(changed, str(path))

    def test_manifest_requires_sample_ids(self):
        records = self._pool_records()
        manifest = self._pool_manifest(records)
        del manifest["sample_ids"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ProbeInputError):
                probe_pool_metadata(records, str(path))

    def test_max_example_prefix_has_distinct_source_and_subset_hashes(self):
        records = self._pool_records()
        manifest = self._pool_manifest(records)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            metadata, warnings = probe_pool_metadata(records[:1], str(path))
        self.assertEqual(warnings, [])
        self.assertEqual(metadata["count"], 1)
        self.assertEqual(metadata["source_manifest_count"], 2)
        self.assertEqual(metadata["source_manifest_sha256"], manifest["manifest_sha256"])
        self.assertEqual(metadata["manifest_sha256"], metadata["selected_subset_sha256"])
        self.assertNotEqual(metadata["manifest_sha256"], metadata["source_manifest_sha256"])

    def test_boundary_states_and_lens_space_are_explicit(self):
        calls = []

        def final_norm(value):
            calls.append(value)
            return "norm(%s)" % value

        self.assertEqual(
            decoder_boundary_states(("B0", "B1", "B2"), 2),
            ("B0", "B1", "B2"),
        )
        with self.assertRaisesRegex(RuntimeError, "boundary-state"):
            decoder_boundary_states(("B1", "B2"), 2)
        self.assertEqual(lens_space_hidden(final_norm, "B0", 0, 2), "norm(B0)")
        self.assertEqual(lens_space_hidden(final_norm, "B1", 1, 2), "norm(B1)")
        self.assertEqual(
            lens_space_hidden(final_norm, "B2-final-normed", 2, 2),
            "B2-final-normed",
        )
        self.assertEqual(calls, ["B0", "B1"])

    def test_window_metrics_use_answer_and_all_valid_tokens(self):
        result = build_window_sample_metrics(
            first_residual_norms=[2.0, 4.0, 8.0],
            second_residual_norms=[1.0, 2.0, 4.0],
            first_state_norms=[4.0, 8.0, 16.0],
            valid_positions=[0, 1, 2],
            answer_position=2,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["answer_position_metrics"], {"r": 0.5, "q": 0.5})
        self.assertEqual(result["all_non_padding_tokens"]["r"]["count"], 3.0)

    def test_window_metrics_flag_zero_norm(self):
        result = build_window_sample_metrics([0.0], [1.0], [2.0], [0], 0)
        self.assertFalse(result["valid"])
        self.assertIn("first residual norm", result["errors"][0])

    def test_window_collector_counts_one_wrapper_and_two_body_calls(self):
        collector = WindowProbeCollector()
        collector.begin_forward("x", [0], 0)
        collector.record("wrapper_forward", {"bypass": False})
        collector.record("body_call", {})
        collector._active["counts"]["g_minus_x"] += 1
        collector._active["timeline"].append("g_minus_x")
        collector.record("body_call", {})
        collector._active["counts"]["g_minus_x"] += 1
        collector._active["timeline"].append("g_minus_x")
        collector._active["counts"]["looped_hidden_vs_input"] += 1
        collector._active["timeline"].append("looped_hidden_vs_input")
        collector._active["looped_hidden_max_abs"] = 0.5
        collector.record("stash_pass", {})
        for _ in range(3):
            collector.record("identity_forward", {})
        collector._active["g_records"] = [
            {"residual_norms": [2.0], "state_norms": [4.0]},
            {"residual_norms": [1.0], "state_norms": [5.0]},
        ]
        result = collector.end_forward()
        self.assertTrue(result["valid"])
        self.assertEqual(result["event_counts"]["wrapper_forward"], 1)
        self.assertEqual(result["event_counts"]["operator_body_calls"], 2)


if __name__ == "__main__":
    unittest.main()
