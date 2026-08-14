import unittest

from tflt.loopscope.phase7_schema import (
    BOUNDARY_KEYS,
    CHOICE_SURFACES,
    RECORD_KEYS,
    TRANSITION_KEYS,
    TRAJECTORY_SCHEMA_VERSION,
    Phase7ContractError,
    candidate_domain,
    choice_surface_summary,
    validate_candidate_domain,
    validate_sanitized_record,
)


def make_record(layer_count=36, model_key="qwen25_3b", identity="id-0", subject="subject-0"):
    model_repo = {
        "qwen25_3b": "Qwen/Qwen2.5-3B",
        "llama32_3b": "meta-llama/Llama-3.2-3B",
        "gemma2_2b": "google/gemma-2-2b",
    }[model_key]
    boundaries = []
    for index in range(layer_count + 1):
        boundaries.append(
            {
                "boundary_id": "B_%d" % index,
                "choice_entropy": 1.0 + 0.01 * index,
                "kl_to_final": 0.02 * (layer_count - index),
                "hidden_rms_l2_to_final": 0.1 + 0.001 * index,
                "hidden_cosine_to_final": 0.9,
                "hidden_cosine_distance_to_final": 0.1,
            }
        )
    boundaries[-1]["kl_to_final"] = 0.0
    transitions = [
        {"transition_id": "T_%d" % index, "adjacent_angular_distance": 0.2}
        for index in range(layer_count)
    ]
    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "model_key": model_key,
        "model_repo": model_repo,
        "model_revision": "local-revision",
        "tokenizer_revision": "local-tokenizer-revision",
        "dataset_repo": "cais/mmlu",
        "split": "validation",
        "canonical_identity": identity,
        "subject": subject,
        "sequence_length": 8,
        "probe_index": 7,
        "boundary_count": layer_count + 1,
        "transition_count": layer_count,
        "forward_count": 1,
        "loop_insertions": 0,
        "batch_size": 1,
        "use_cache": False,
        "generation": False,
        "probe_rule": "final_non_padding_token_of_rendered_Answer_prefix",
        "boundary_capture": "raw_B0_through_BL",
        "raw_final_boundary_capture": "temporary_final_norm_forward_pre_hook",
        "final_norm_path": "model.norm",
        "lm_head_path": "lm_head",
        "final_norm_prehook_calls": 1,
        "final_norm_closure": True,
        "double_norm_applied": False,
        "choice_surface_count": 4,
        "choice_surface_single_token": True,
        "choice_surface_distinct": True,
        "runtime_dtype": "bfloat16",
        "boundaries": boundaries,
        "transitions": transitions,
    }


class FakeTokenizer:
    def __init__(self, values):
        self.values = dict(values)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self.values[text]]


class Phase7SchemaTests(unittest.TestCase):
    def test_sanitized_record_is_closed_and_scalar_only(self):
        record = make_record()
        validate_sanitized_record(record, expected_model_key="qwen25_3b", expected_layer_count=36)
        self.assertEqual(set(record), set(RECORD_KEYS))
        self.assertEqual(set(record["boundaries"][0]), set(BOUNDARY_KEYS))
        self.assertEqual(set(record["transitions"][0]), set(TRANSITION_KEYS))

    def test_forbidden_field_is_rejected(self):
        for field, value in (("gold", "D"), ("probs", [0.25] * 4), ("hidden_states", [])):
            record = make_record()
            record[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(Phase7ContractError, "BLOCK_INFORMATION_BARRIER_VIOLATION"):
                    validate_sanitized_record(record)

    def test_candidate_domain_is_formula_generated_for_arbitrary_depth(self):
        domain = candidate_domain(36)
        self.assertEqual(domain["trim_per_side"], 11)
        self.assertEqual(domain["central_blocks"], [11, 24])
        self.assertEqual(domain["candidate_count"], 42)
        self.assertEqual(
            [(row["width"], row["start"]) for row in domain["candidates"][:4]],
            [(3, 11), (3, 12), (3, 13), (3, 14)],
        )
        self.assertEqual(
            [(row["width"], row["start"]) for row in domain["candidates"][-3:]],
            [(6, 17), (6, 18), (6, 19)],
        )
        validate_candidate_domain(domain)
        other = candidate_domain(28)
        self.assertEqual(other["trim_per_side"], 9)
        self.assertNotEqual(other["candidate_count"], domain["candidate_count"])

    def test_choice_surface_summary_never_returns_token_ids(self):
        summary = choice_surface_summary(
            FakeTokenizer({surface: index + 101 for index, surface in enumerate(CHOICE_SURFACES)})
        )
        self.assertEqual(summary["single_token_lengths"], [1, 1, 1, 1])
        self.assertTrue(summary["all_single_token"])
        self.assertTrue(summary["all_distinct"])
        self.assertFalse(any("id" in key.lower() for key in summary))

        repeated = choice_surface_summary(FakeTokenizer({surface: 101 for surface in CHOICE_SURFACES}))
        self.assertFalse(repeated["all_distinct"])


if __name__ == "__main__":
    unittest.main()
