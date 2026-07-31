import math
import unittest

import numpy as np

from tflt.loopscope.phase6_acquisition import (
    PROVENANCE_KEYS,
    SCHEMA_VERSION,
    TOP_LEVEL_KEYS,
    build_sanitized_trajectory_record,
    full_vocabulary_entropy_and_kl,
    hidden_diagnostics,
    raw_adjacent_angular_distance,
    stable_float32_log_softmax,
    validate_generation_replay_id_closure,
)
from tflt.loopscope.phase6_schema import Phase6ContractError


SHA = "a" * 64


def _provenance():
    return {
        "producer_version": "phase6-gate-a",
        "card_sha256": SHA,
        "renderer_manifest_sha256": SHA,
        "tokenizer_manifest_sha256": SHA,
        "boundary_capture": "raw_B0_through_B36_final_norm_pre_hook",
        "final_norm_path": "model.norm",
        "lm_head_path": "lm_head",
    }


def _record(**changes):
    logits = np.zeros((37, 5), dtype=np.float32)
    logits[:, 0] = np.arange(37, dtype=np.float32)
    normalized = np.stack(
        [np.asarray([1.0 + index / 10.0, 0.5, -0.25], dtype=np.float32)
         for index in range(37)]
    )
    raw = np.stack(
        [np.asarray([1.0, index / 20.0, 0.25], dtype=np.float32)
         for index in range(37)]
    )
    kwargs = {
        "model_repo": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "dataset_repo": "TIGER-Lab/MMLU-Pro",
        "dataset_revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "split": "test",
        "canonical_identity": "mmlu_pro_test:0",
        "category": "biology",
        "prompt_sha256": SHA,
        "generated_completion_sha256": SHA,
        "generation_length": 8,
        "replay_length": 3,
        "anchor_token_index": 2,
        "answer_span_start_offset": 10,
        "answer_span_end_offset": 11,
        "answer_match_count": 2,
        "selected_match_ordinal": 0,
        "answer_first_token_index": 3,
        "answer_span_extractor_sha256": SHA,
        "generated_id_text_aligner_sha256": SHA,
        "generation_prefix_ids": [11, 12, 13],
        "replay_ids": [11, 12, 13],
        "boundary_logits": logits,
        "final_normalized_vectors": normalized,
        "raw_boundaries": raw,
        "provenance": _provenance(),
    }
    kwargs.update(changes)
    return build_sanitized_trajectory_record(**kwargs)


class Phase6AcquisitionTests(unittest.TestCase):
    def test_extreme_float32_log_softmax_entropy_and_kl_are_stable(self):
        log_probabilities = stable_float32_log_softmax([10000.0, 0.0, -10000.0])
        self.assertEqual(log_probabilities.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(log_probabilities)))
        self.assertAlmostEqual(float(log_probabilities[0]), 0.0)

        rows = np.zeros((37, 4), dtype=np.float32)
        rows[:, 0] = np.linspace(-10000.0, 10000.0, 37, dtype=np.float32)
        H, D = full_vocabulary_entropy_and_kl(rows)
        self.assertEqual((len(H), len(D)), (37, 37))
        self.assertTrue(all(math.isfinite(value) for value in H + D))
        self.assertEqual(D[-1], 0.0)

    def test_lengths_endpoints_and_exact_flat_record(self):
        record = _record()
        self.assertEqual(record["schema_version"], SCHEMA_VERSION)
        self.assertEqual(set(record), set(TOP_LEVEL_KEYS))
        self.assertEqual(set(record["provenance"]), set(PROVENANCE_KEYS))
        self.assertEqual(len(record["H"]), 37)
        self.assertEqual(len(record["D"]), 37)
        self.assertEqual(len(record["hidden_rms_l2_to_final"]), 37)
        self.assertEqual(len(record["hidden_cosine_to_final"]), 37)
        self.assertEqual(len(record["hidden_cosine_distance_to_final"]), 37)
        self.assertEqual(len(record["adjacent_angular_distance"]), 36)
        self.assertEqual(record["D"][-1], 0.0)
        self.assertEqual(record["hidden_rms_l2_to_final"][-1], 0.0)
        self.assertEqual(record["hidden_cosine_to_final"][-1], 1.0)
        self.assertEqual(record["hidden_cosine_distance_to_final"][-1], 0.0)
        self.assertEqual(record["answer_match_count"], 2)
        self.assertEqual(record["selected_match_ordinal"], 0)

    def test_angular_uses_raw_not_final_normalized_vectors(self):
        raw = np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (37, 1))
        raw[1] = [0.0, 1.0]
        normalized = np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (37, 1))
        angles = raw_adjacent_angular_distance(raw)
        _, cosines, _ = hidden_diagnostics(normalized)
        self.assertAlmostEqual(angles[0], 0.5)
        self.assertEqual(cosines[0], 1.0)

    def test_zero_norm_nonfinite_and_wrong_lengths_fail_closed(self):
        rows = np.zeros((37, 3), dtype=np.float32)
        rows[4, 1] = np.nan
        with self.assertRaises(Phase6ContractError):
            full_vocabulary_entropy_and_kl(rows)
        with self.assertRaises(Phase6ContractError):
            hidden_diagnostics(np.zeros((37, 3), dtype=np.float32))
        raw = np.ones((37, 3), dtype=np.float32)
        raw[5] = 0.0
        with self.assertRaises(Phase6ContractError):
            raw_adjacent_angular_distance(raw)
        with self.assertRaises(Phase6ContractError):
            full_vocabulary_entropy_and_kl(np.zeros((36, 3), dtype=np.float32))

    def test_generation_replay_mismatch_fails_closed(self):
        self.assertEqual(
            validate_generation_replay_id_closure([1, 2, 3], [1, 2, 3]),
            (1, 2, 3),
        )
        with self.assertRaisesRegex(Phase6ContractError, "mismatch"):
            validate_generation_replay_id_closure([1, 2, 3], [1, 2, 4])
        with self.assertRaisesRegex(Phase6ContractError, "replay_length"):
            _record(replay_length=4)
        with self.assertRaisesRegex(Phase6ContractError, "ordinal"):
            _record(selected_match_ordinal=1)

    def test_sanitizer_has_no_forbidden_payload_fields(self):
        record = _record()
        forbidden = {
            "text",
            "input_ids",
            "answer",
            "prediction",
            "gold",
            "label",
            "correctness",
            "accuracy",
            "gain",
            "flip",
            "outcome",
            "logits",
            "probabilities",
            "hidden",
        }
        lowered_keys = {str(key).lower() for key in record}
        lowered_keys.update(str(key).lower() for key in record["provenance"])
        self.assertTrue(forbidden.isdisjoint(lowered_keys))
        self.assertNotIn("generation_prefix_ids", record)
        self.assertNotIn("replay_ids", record)


if __name__ == "__main__":
    unittest.main()
