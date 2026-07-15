import copy
import hashlib
import json
import math
import unittest
from pathlib import Path

from tflt.loopscope.schema import SchemaError
from tflt.loopscope.phase3_analysis import (
    choice_probabilities_from_logits,
    trajectory_record_from_logits,
    window_signals_from_trajectory,
)
from tflt.loopscope.phase3_schema import (
    BOUNDARY_IDS,
    PHASE3_CARD_BYTE_SHA256,
    file_sha256,
    load_phase3_card,
    pool_record_json_schema,
    sanitized_content_sha256,
    trajectory_record_json_schema,
    validate_phase3_card,
    validate_trajectory_record,
)


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/phase3_card.json"


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _renderer(card):
    return {
        "dataset_repo": card["task"]["dataset"],
        "dataset_revision": card["task"]["dataset_revision"],
        "model_repo": card["model"]["repo"],
        "model_revision": card["model"]["revision"],
        "tokenizer_revision": card["model"]["revision"],
        "source_manifest_sha256": _sha("source"),
        "pool_manifest_sha256": _sha("pool"),
        "renderer_manifest_sha256": _sha("renderer"),
        "forward_type": card["trajectory"]["forward_type"],
        "formal_forward_count_per_identity": 1,
        "loop_insertions": 0,
        "answer_position": card["trajectory"]["answer_position"],
        "projection": card["trajectory"]["projection"],
        "choice_order": ["A", "B", "C", "D"],
    }


def _trajectory(card, tied=False):
    logits = {}
    for index, boundary_id in enumerate(BOUNDARY_IDS):
        logits[boundary_id] = (
            [0.0, 0.0, 0.0, 0.0]
            if tied
            else [0.02 * index, -0.2, -0.4, -0.6]
        )
    return trajectory_record_from_logits(
        {
            "identity": {"task": "mmlu_x", "doc_id": "1", "doc_hash": _sha("doc")},
            "subject": "x",
            "split": "validation",
            "prompt_sha256": _sha("prompt"),
        },
        logits,
        _renderer(card),
        card,
    )


class Phase3SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_phase3_card(CARD_PATH)

    def test_card_is_byte_and_semantically_frozen(self):
        self.assertEqual(file_sha256(CARD_PATH), PHASE3_CARD_BYTE_SHA256)
        validate_phase3_card(self.card)
        changed = copy.deepcopy(self.card)
        changed["ranking"]["selection_frequency"]["threshold"] = 0.81
        with self.assertRaisesRegex(SchemaError, "semantic SHA256"):
            validate_phase3_card(changed)

    def test_json_schemas_are_closed_world(self):
        pool = pool_record_json_schema()
        trajectory = trajectory_record_json_schema()
        self.assertFalse(pool["additionalProperties"])
        self.assertFalse(pool["properties"]["identity"]["additionalProperties"])
        self.assertFalse(trajectory["additionalProperties"])
        self.assertFalse(
            trajectory["properties"]["boundaries"]["items"]["additionalProperties"]
        )

    def test_sanitized_content_hash_is_nfc_whitespace_stable_and_order_sensitive(self):
        left = sanitized_content_sha256(
            "  Cafe\u0301\tquestion\n", [" A ", "B\u00a0value", "C", "D"]
        )
        right = sanitized_content_sha256(
            "Caf\u00e9 question", ["A", "B value", "C", "D"]
        )
        self.assertEqual(left, right)
        self.assertNotEqual(
            left,
            sanitized_content_sha256("Caf\u00e9 question", ["B value", "A", "C", "D"]),
        )

    def test_raw_logits_recompute_exact_b0_b28_and_ce_identity(self):
        record = _trajectory(self.card)
        self.assertEqual([row["boundary_id"] for row in record["boundaries"]], list(BOUNDARY_IDS))
        self.assertNotIn("choice_logits", json.dumps(record))
        final = record["boundaries"][-1]
        self.assertAlmostEqual(final["kl_to_final"], 0.0, places=12)
        self.assertAlmostEqual(
            final["cross_entropy_to_final"], final["choice_entropy"], places=12
        )
        signals = window_signals_from_trajectory(record, self.card)
        row = signals["windows"][12]
        self.assertEqual(row["window"], "12:15")
        self.assertAlmostEqual(row["ce_drop"], row["entropy_drop"] - row["kl_rise"])

    def test_top1_tie_uses_a_first(self):
        record = _trajectory(self.card, tied=True)
        self.assertTrue(all(row["top1_to_final_agreement"] for row in record["boundaries"]))
        self.assertTrue(math.isclose(record["boundaries"][0]["choice_entropy"], math.log(4.0)))

    def test_boundary_probability_metric_and_unknown_tampers_fail(self):
        record = _trajectory(self.card)
        changed = copy.deepcopy(record)
        changed["boundaries"][12]["choice_entropy"] += 1e-4
        with self.assertRaises(SchemaError):
            validate_trajectory_record(changed, self.card)

        changed = copy.deepcopy(record)
        changed["boundaries"][0]["choice_distribution"][0] = 0.0
        with self.assertRaisesRegex(SchemaError, "strictly greater"):
            validate_trajectory_record(changed, self.card)

        changed = copy.deepcopy(record)
        changed["boundaries"][0]["choice_distribution"][0] += 0.1
        with self.assertRaisesRegex(SchemaError, "normalized"):
            validate_trajectory_record(changed, self.card)

        changed = copy.deepcopy(record)
        changed["boundaries"].pop()
        with self.assertRaisesRegex(SchemaError, "B_0 through B_28"):
            validate_trajectory_record(changed, self.card)

        changed = copy.deepcopy(record)
        changed["accuracy"] = 1.0
        with self.assertRaisesRegex(SchemaError, "forbidden selector field"):
            validate_trajectory_record(changed, self.card)

        changed = copy.deepcopy(record)
        changed["mystery"] = 1
        with self.assertRaisesRegex(SchemaError, "extra"):
            validate_trajectory_record(changed, self.card)

    def test_nonfinite_and_softmax_underflow_fail_without_clamping(self):
        with self.assertRaises(SchemaError):
            choice_probabilities_from_logits([0.0, float("nan"), 1.0, 2.0])
        with self.assertRaisesRegex(SchemaError, "zero probability"):
            choice_probabilities_from_logits([0.0, -1000.0, -2000.0, -3000.0])


if __name__ == "__main__":
    unittest.main()
