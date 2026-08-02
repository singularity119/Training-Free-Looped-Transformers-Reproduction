import copy
import unittest
from pathlib import Path

from tflt.loopscope import phase6_mmlu5_gate_k as gate_k
from tflt.loopscope import phase6_mmlu5_gate_l as gate_l
from tflt.loopscope.phase6_mmlu5_schema import (
    BOUNDARY_COUNT,
    CHOICE_SURFACE_MANIFEST_SHA256,
    DATASET_REVISION,
    DEMONSTRATION_COUNT,
    PROMPT_PROJECTION_SCHEMA_VERSION,
    TASK_SOURCE_MANIFEST_SHA256,
    TRAJECTORY_SCHEMA_VERSION,
    build_trajectory_record,
    choice_logits_to_probabilities,
    enumerate_candidates,
    load_json,
    selector_sample,
    validate_card,
    validate_prompt_projection_record,
    validate_schema_document,
    validate_selector_sample,
    validate_trajectory_record,
)


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/phase6_mmlu5_prefix_card.json"
PROMPT_SCHEMA_PATH = ROOT / "configs/loopscope/phase6_mmlu5_prompt_projection_schema.json"
TRAJECTORY_SCHEMA_PATH = ROOT / "configs/loopscope/phase6_mmlu5_trajectory_schema.json"


class GateKMMLU5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_json(CARD_PATH)
        cls.prompt_schema = load_json(PROMPT_SCHEMA_PATH)
        cls.trajectory_schema = load_json(TRAJECTORY_SCHEMA_PATH)

    def test_card_schema_and_gate_l_import_only_plan_close(self):
        validate_card(self.card)
        validate_schema_document(self.prompt_schema)
        validate_schema_document(self.trajectory_schema)
        plan = gate_l.import_only_dry_plan(CARD_PATH, TRAJECTORY_SCHEMA_PATH)
        self.assertEqual(plan["status"], "IMPORT_ONLY_DRY_PLAN")
        self.assertEqual(plan["partition"], "debug")
        self.assertFalse(plan["model_weights_loaded"])
        self.assertFalse(plan["forward_executed"])
        dry_run = gate_k.dry_run()
        self.assertEqual(dry_run["status"], "DRY_RUN_VALID")
        self.assertEqual(dry_run["demonstration_count"], 5)
        self.assertFalse(dry_run["model_forward_executed"])

    def test_five_shot_projection_is_closed_and_last_valid_token(self):
        record = gate_k._synthetic_projection_record()
        validate_prompt_projection_record(record, self.card)
        self.assertEqual(record["schema_version"], PROMPT_PROJECTION_SCHEMA_VERSION)
        self.assertEqual(record["demonstration_count"], DEMONSTRATION_COUNT)
        self.assertEqual(record["probe_position"], record["sequence_length"] - 1)
        self.assertNotIn("prompt", record)
        self.assertNotIn("token_ids", record)

    def test_standard_renderer_target_sentinel_is_empty_and_non_gold(self):
        target = {"question": "q", "choices": ["a", "b", "c", "d"], "subject": "math"}
        rendered = gate_k._renderer_target_doc(target)
        self.assertEqual(rendered["answer"], "")
        self.assertEqual(target, {"question": "q", "choices": ["a", "b", "c", "d"], "subject": "math"})
        with self.assertRaises(gate_k.GateKMMLU5Error):
            gate_k._renderer_target_doc({**target, "answer": 0})

    def test_standard_dev_demo_order_count_and_target_exclusion_are_fail_closed(self):
        record = gate_k._synthetic_projection_record()
        invalid = copy.deepcopy(record)
        invalid["demonstration_ids"] = invalid["demonstration_ids"][:4]
        with self.assertRaises(ValueError):
            validate_prompt_projection_record(invalid, self.card)

        invalid = copy.deepcopy(record)
        invalid["demonstration_ids"][0], invalid["demonstration_ids"][1] = invalid["demonstration_ids"][1], invalid["demonstration_ids"][0]
        with self.assertRaises(ValueError):
            validate_prompt_projection_record(invalid, self.card)

        invalid = copy.deepcopy(record)
        invalid["demonstration_ids"][0] = "mmlu_math:validation:0"
        with self.assertRaises(ValueError):
            validate_prompt_projection_record(invalid, self.card)

        invalid = copy.deepcopy(record)
        invalid["identity"]["target"] = "forbidden"
        with self.assertRaises(ValueError):
            validate_prompt_projection_record(invalid, self.card)

    def test_zero_shot_and_five_shot_contracts_cannot_mix(self):
        invalid = copy.deepcopy(self.card)
        invalid["evaluator"]["num_fewshot"] = 0
        with self.assertRaises(ValueError):
            validate_card(invalid)
        invalid = copy.deepcopy(self.card)
        invalid["renderer"]["semantic_contract"]["fewshot_split"] = None
        with self.assertRaises(ValueError):
            validate_card(invalid)
        invalid = copy.deepcopy(gate_k._synthetic_projection_record())
        invalid["num_fewshot"] = 0
        with self.assertRaises(ValueError):
            validate_prompt_projection_record(invalid, self.card)

    def test_selector_candidate_domain_and_projection_surface_are_frozen(self):
        candidates = enumerate_candidates()
        self.assertEqual(len(candidates), 42)
        self.assertEqual(candidates[0]["window"], "11:14")
        self.assertEqual(candidates[0]["boundary_entry"], "B_11")
        self.assertEqual(candidates[-1]["window"], "19:25")
        self.assertEqual(len({(row["width"], row["start"]) for row in candidates}), 42)

        probabilities = [
            choice_logits_to_probabilities([1.0 + 0.01 * index, 0.2, -0.1, -0.4])
            for index in range(BOUNDARY_COUNT)
        ]
        hidden = [[1.0 + 0.01 * index, 0.2 + 0.005 * index, 0.5] for index in range(BOUNDARY_COUNT)]
        provenance = {
            "contract_version": self.card["renderer"]["contract_version"],
            "semantic_contract_sha256": self.card["renderer"]["semantic_contract_sha256"],
            "task_source_manifest_sha256": TASK_SOURCE_MANIFEST_SHA256,
            "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
        }
        trajectory = build_trajectory_record(
            {"task": "mmlu", "doc_id": "synthetic-0", "doc_hash": "a" * 64},
            "math",
            "b" * 64,
            provenance,
            probabilities,
            hidden,
            self.card,
        )
        validate_trajectory_record(trajectory, self.card)
        projected = selector_sample(trajectory)
        validate_selector_sample(projected)
        self.assertEqual(set(projected), {"identity", "category", "H", "D"})
        self.assertEqual(len(projected["H"]), 37)
        self.assertEqual(len(projected["D"]), 37)

    def test_independent_gate_k_self_test_is_no_execution(self):
        result = gate_k.run_self_test()
        self.assertEqual(result["status"], "SELF_TEST_PASS")
        self.assertEqual(result["negative_case_count"], 10)
        self.assertFalse(result["model_weights_loaded"])
        self.assertFalse(result["model_forward_executed"])
        self.assertFalse(result["validation_target_gold_read"])
        self.assertFalse(result["outcome_read"])


if __name__ == "__main__":
    unittest.main()
