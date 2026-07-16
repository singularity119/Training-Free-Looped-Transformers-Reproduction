from __future__ import annotations

import copy
import unittest
from pathlib import Path

from tflt.loopscope.phase4_schema import (
    Phase4ContractError,
    load_phase4_card,
    validate_phase4_card,
    validate_shared_identity_contract,
    validate_source_record,
)
from tflt.loopscope.phase4_verifier import _synthetic_source_record


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "configs/loopscope/phase4_pv_ek_trs_card.json"


class Phase4ContractTests(unittest.TestCase):
    def test_card_freezes_science_and_unresolved_copy_still_fails_closed(self) -> None:
        card = load_phase4_card(CARD)
        self.assertEqual(card["selector"]["width4_start_count"], 33)
        self.assertEqual(card["selector"]["width4_consensus_count"], 25)
        self.assertEqual(card["selector"]["secondary_strict_count"], 154)
        self.assertEqual(card["selector"]["secondary_edge_count"], 224)
        self.assertEqual(
            card["provenance_closure"]["closure_mode"],
            "NOT_REUSABLE_FRESH_REQUIRED",
        )
        unresolved = copy.deepcopy(card)
        unresolved["status"] = "BLOCKED_PROVENANCE"
        unresolved["provenance_closure"]["state"] = "UNRESOLVED_LOCAL_EVIDENCE"
        unresolved["provenance_closure"]["values"]["model_revision"] = (
            "UNRESOLVED_LOCAL_EVIDENCE"
        )
        validate_phase4_card(unresolved, require_provenance_closed=False)
        with self.assertRaisesRegex(Phase4ContractError, "exact immutable provenance"):
            validate_phase4_card(unresolved)

    def test_shared_identity_missing_extra_reordered_duplicate_fail_fast(self) -> None:
        expected = [_synthetic_source_record(index) for index in range(8)]
        validate_shared_identity_contract(expected, expected, require_full_population=False)
        cases = [
            expected[:-1],
            expected + [_synthetic_source_record(99)],
            list(reversed(expected)),
            expected[:-1] + [expected[0]],
        ]
        for observed in cases:
            with self.subTest(count=len(observed)):
                with self.assertRaises(Phase4ContractError):
                    validate_shared_identity_contract(
                        expected, observed, require_full_population=False
                    )

    def test_canonical_manifest_itself_must_be_ordered_and_question_id_consistent(self) -> None:
        expected = [_synthetic_source_record(index) for index in range(4)]
        with self.assertRaisesRegex(Phase4ContractError, "canonical shared manifest order"):
            validate_shared_identity_contract(
                list(reversed(expected)), list(reversed(expected)), require_full_population=False
            )
        conflicting = copy.deepcopy(expected)
        conflicting[1]["question_id"] = conflicting[0]["question_id"]
        conflicting[1]["canonical_identity"] = conflicting[1]["canonical_identity"].replace(
            "1:", "0:", 1
        )
        with self.assertRaises(Phase4ContractError):
            validate_shared_identity_contract(
                conflicting, conflicting, require_full_population=False
            )

    def test_forbidden_fields_rejected_on_normal_source_path(self) -> None:
        for key in (
            "cot_content",
            "answer",
            "gold",
            "correctness",
            "generated_answer",
            "teacher_forcing",
            "baseline_outcome",
            "loop_outcome",
            "logits",
            "probabilities",
            "hidden_states",
        ):
            record = copy.deepcopy(_synthetic_source_record(0))
            record[key] = "forbidden"
            with self.subTest(key=key):
                with self.assertRaisesRegex(Phase4ContractError, "forbidden"):
                    validate_source_record(record)

    def test_nested_alias_and_non_json_payload_rejected(self) -> None:
        aliases = ("target_cot_content", "target_answer", "persisted_full_logits", "tensor_payload")
        for key in aliases:
            record = copy.deepcopy(_synthetic_source_record(0))
            record["renderer_provenance"][key] = "forbidden"
            with self.subTest(key=key):
                with self.assertRaisesRegex(Phase4ContractError, "forbidden"):
                    validate_source_record(record)
        record = copy.deepcopy(_synthetic_source_record(0))
        record["renderer_provenance"]["opaque_payload"] = object()
        with self.assertRaisesRegex(Phase4ContractError, "non-JSON"):
            validate_source_record(record)

    def test_prompt_hash_mismatch_rejected_by_shared_closure(self) -> None:
        expected = [_synthetic_source_record(index) for index in range(4)]
        observed = copy.deepcopy(expected)
        observed[2]["rendered_prefix_sha256"] = "f" * 64
        with self.assertRaisesRegex(Phase4ContractError, "hash closure"):
            validate_shared_identity_contract(expected, observed, require_full_population=False)

    def test_unknown_source_field_rejected(self) -> None:
        record = copy.deepcopy(_synthetic_source_record(0))
        record["harmless_but_unknown"] = 1
        with self.assertRaisesRegex(Phase4ContractError, "closed-world"):
            validate_source_record(record)


if __name__ == "__main__":
    unittest.main()
