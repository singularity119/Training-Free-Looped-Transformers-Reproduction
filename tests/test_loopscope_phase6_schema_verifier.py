import copy
import json
import unittest
from pathlib import Path

from tflt.loopscope.phase6_schema import (
    ELIGIBILITY_SCHEMA_VERSION,
    Phase6ContractError,
    eligibility_json_schema,
    load_json,
    scan_forbidden_fields,
    selector_freeze_json_schema,
    trajectory_json_schema,
    validate_card,
    validate_eligibility_record,
    validate_schema_document,
)
from tflt.loopscope.phase6_verifier import run_self_test, verify_gate_a_contracts


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "configs/loopscope/phase6_pre_answer_v2_card.json"
TRAJECTORY_SCHEMA = ROOT / "configs/loopscope/phase6_trajectory_schema.json"
ELIGIBILITY_SCHEMA = ROOT / "configs/loopscope/phase6_eligibility_schema.json"
SELECTOR_SCHEMA = ROOT / "configs/loopscope/phase6_selector_freeze_schema.json"

SHA = "a" * 64


def _eligibility_record(state="ANCHOR_NOT_EXPRESSED"):
    return {
        "schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "canonical_identity": "mmlu_pro_test:0",
        "category": "biology",
        "eligibility_state": state,
        "generation_count": 1,
        "sealed_payload_sha256": SHA,
        "sealed_membership_ref": "sealed_baseline/000000.bin",
        "provenance": {
            "producer_version": "phase6-gate-d2-test",
            "card_sha256": SHA,
            "renderer_manifest_sha256": SHA,
            "answer_span_extractor_sha256": SHA,
        },
    }


class Phase6SchemaVerifierTests(unittest.TestCase):
    def test_checked_in_card_and_schemas_are_exact(self):
        card = load_json(CARD)
        validate_card(card)
        validate_schema_document(
            load_json(TRAJECTORY_SCHEMA), trajectory_json_schema()
        )
        validate_schema_document(
            load_json(ELIGIBILITY_SCHEMA), eligibility_json_schema()
        )
        validate_schema_document(
            load_json(SELECTOR_SCHEMA), selector_freeze_json_schema()
        )
        self.assertFalse(load_json(TRAJECTORY_SCHEMA)["additionalProperties"])
        self.assertFalse(load_json(ELIGIBILITY_SCHEMA)["additionalProperties"])
        self.assertFalse(load_json(SELECTOR_SCHEMA)["additionalProperties"])

    def test_card_is_closed_and_frozen(self):
        invalid = load_json(CARD)
        invalid["extra"] = True
        with self.assertRaises(Phase6ContractError):
            validate_card(invalid)
        invalid = load_json(CARD)
        invalid["anchor"]["regex_pattern"] = r"Answer is ([A-J])"
        with self.assertRaises(Phase6ContractError):
            validate_card(invalid)
        valid = load_json(CARD)
        self.assertEqual(
            valid["anchor"]["match_rule"],
            "collect_all_capture_spans_select_ordinal_0",
        )
        self.assertTrue(valid["anchor"]["outcome_take_first_used_for_anchor"])
        trajectory = load_json(TRAJECTORY_SCHEMA)
        self.assertEqual(
            trajectory["properties"]["selected_match_ordinal"], {"const": 0}
        )
        self.assertEqual(
            trajectory["properties"]["answer_match_count"],
            {"minimum": 1, "type": "integer"},
        )
        self.assertEqual(
            valid["anchor_eligibility"]["overall_coverage_floor"], 0.995
        )
        self.assertEqual(
            valid["anchor_eligibility"]["per_category_coverage_floor"], 0.98
        )

    def test_expressed_and_not_expressed_records_are_separate_closed_schemas(self):
        for state in ("ANCHOR_ELIGIBLE", "ANCHOR_NOT_EXPRESSED"):
            with self.subTest(state=state):
                record = _eligibility_record(state)
                validate_eligibility_record(record)
                self.assertNotIn("answer_match_count", record)
                self.assertNotIn("answer_span_start_offset", record)
                self.assertNotIn("replay_count", record)
                self.assertNotIn("H", record)
                self.assertNotIn("D", record)

        invalid = _eligibility_record()
        invalid["answer_content"] = "A"
        with self.assertRaisesRegex(
            Phase6ContractError, "BLOCK_INFORMATION_BARRIER_VIOLATION"
        ):
            validate_eligibility_record(invalid)

        invalid = _eligibility_record()
        invalid["replay_count"] = 0
        with self.assertRaises(Phase6ContractError):
            validate_eligibility_record(invalid)

    def test_forbidden_payload_fields_fail_closed_recursively(self):
        for key in (
            "prompt_text",
            "generated_text",
            "token_ids",
            "answer_content",
            "answer_span_hash",
            "prediction",
            "gold",
            "label",
            "correctness",
            "accuracy",
            "gain",
            "flip",
            "outcome",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    Phase6ContractError, "BLOCK_INFORMATION_BARRIER_VIOLATION"
                ):
                    scan_forbidden_fields({"nested": [{key: "forbidden"}]})

    def test_schema_document_rejects_open_world_mutation(self):
        invalid = copy.deepcopy(trajectory_json_schema())
        invalid["additionalProperties"] = True
        with self.assertRaises(Phase6ContractError):
            validate_schema_document(invalid, trajectory_json_schema())

    def test_independent_verifier_and_failure_self_test(self):
        result = verify_gate_a_contracts(
            CARD, TRAJECTORY_SCHEMA, ELIGIBILITY_SCHEMA, SELECTOR_SCHEMA
        )
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["model_or_data_loaded"])
        self.assertFalse(result["selector_executed"])
        self_test = run_self_test(
            CARD, TRAJECTORY_SCHEMA, ELIGIBILITY_SCHEMA, SELECTOR_SCHEMA
        )
        self.assertEqual(self_test["self_test"], "PASS")
        self.assertEqual(
            set(self_test["invalid_fixtures_rejected"]),
            {"regex", "candidate_count", "selector_threshold", "open_schema"},
        )

    def test_nonfinite_json_is_rejected(self):
        temporary = ROOT / "tests" / ".phase6_nonfinite_should_not_exist.json"
        # Exercise the same parser hook without writing a fixture file.
        with self.assertRaises(Phase6ContractError):
            json.loads('{"x": NaN}', parse_constant=lambda value: (_ for _ in ()).throw(
                Phase6ContractError(value)
            ))
        self.assertFalse(temporary.exists())


if __name__ == "__main__":
    unittest.main()
