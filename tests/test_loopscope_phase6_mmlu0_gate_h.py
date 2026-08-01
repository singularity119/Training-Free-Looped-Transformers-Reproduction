import copy
import math
import unittest
from pathlib import Path

from tflt.loopscope.phase6_mmlu0_schema import (
    BOUNDARY_COUNT,
    CHOICE_SURFACES,
    CHOICE_TOKEN_IDS,
    MMLU0ContractError,
    adjacent_angular_distance,
    build_trajectory_record,
    choice_entropy,
    choice_logits_to_probabilities,
    choice_surface_manifest,
    choice_trajectory_metrics,
    freeze_choice_surface_manifest,
    hidden_geometry_to_final,
    load_json,
    selector_sample,
    validate_card,
    validate_choice_surface_manifest,
    validate_schema_document,
    validate_selector_sample,
    validate_trajectory_record,
)
from tflt.loopscope.phase6_mmlu0_selector import (
    analyze_selector,
    enumerate_candidates,
    project_selector_records,
)
from tflt.loopscope.phase6_mmlu0_verifier import (
    dry_run_payload,
    run_self_test,
    verify_gate_h_contracts,
)


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/phase6_mmlu0_prefix_card.json"
SCHEMA_PATH = ROOT / "configs/loopscope/phase6_mmlu0_trajectory_schema.json"


def _synthetic_record():
    card = load_json(CARD_PATH)
    probabilities = [
        choice_logits_to_probabilities([1.0 + 0.01 * i, 0.2, -0.1, -0.4])
        for i in range(BOUNDARY_COUNT)
    ]
    hidden = [[1.0 + 0.01 * i, 0.2 + 0.005 * i, 0.5] for i in range(BOUNDARY_COUNT)]
    provenance = {
        "contract_version": card["renderer"]["contract_version"],
        "semantic_contract_sha256": card["renderer"]["semantic_contract_sha256"],
        "task_source_manifest_sha256": card["task"]["lm_eval_task_source"]["default_source_manifest_sha256"],
        "choice_surface_manifest_sha256": "ef77986db0381810d98f7fa20e4996895cf6fdbfe506cc4adcd356184179c8ed",
    }
    return build_trajectory_record(
        {"task": "mmlu", "doc_id": "synthetic-0", "doc_hash": "a" * 64},
        "synthetic_subject",
        "b" * 64,
        provenance,
        probabilities,
        hidden,
        card,
    )


class MMLU0GateHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_json(CARD_PATH)
        cls.schema = load_json(SCHEMA_PATH)

    def test_card_schema_and_verifier_close(self):
        validate_card(self.card)
        validate_schema_document(self.schema)
        result = verify_gate_h_contracts(CARD_PATH, SCHEMA_PATH)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["candidate_count"], 42)
        self.assertFalse(result["formal_selector_executed"])
        dry_run = dry_run_payload(CARD_PATH, SCHEMA_PATH)
        self.assertEqual(dry_run["status"], "DRY_RUN_VALID")
        self.assertFalse(dry_run["test_split_read"])

    def test_choice_surface_is_exact_unique_one_token(self):
        manifest = choice_surface_manifest()
        validate_choice_surface_manifest(manifest)
        self.assertEqual(
            {key: value["surface"] for key, value in manifest.items()}, CHOICE_SURFACES
        )
        self.assertEqual(
            {key: value["token_ids"][0] for key, value in manifest.items()}, CHOICE_TOKEN_IDS
        )
        invalid = copy.deepcopy(manifest)
        invalid["A"]["token_ids"] = [362, 999]
        with self.assertRaises(MMLU0ContractError):
            validate_choice_surface_manifest(invalid)

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                self.assert_false = add_special_tokens
                return [CHOICE_TOKEN_IDS[text.strip()]]

            def decode(self, token_ids, skip_special_tokens=False):
                inverse = {value: key for key, value in CHOICE_TOKEN_IDS.items()}
                return " " + inverse[token_ids[0]]

        self.assertEqual(freeze_choice_surface_manifest(FakeTokenizer()), manifest)
        invalid = copy.deepcopy(manifest)
        invalid["B"]["surface"] = " A"
        invalid["B"]["decoded"] = " A"
        with self.assertRaises(MMLU0ContractError):
            validate_choice_surface_manifest(invalid)

    def test_choice_entropy_kl_and_endpoint_recompute(self):
        uniform = [0.25, 0.25, 0.25, 0.25]
        self.assertAlmostEqual(choice_entropy(uniform), math.log(4.0), places=14)
        probabilities = [
            choice_logits_to_probabilities([1.0 + i * 0.01, 0.2, -0.1, -0.4])
            for i in range(BOUNDARY_COUNT)
        ]
        metrics = choice_trajectory_metrics(probabilities)
        self.assertEqual(len(metrics["H"]), 37)
        self.assertEqual(len(metrics["D"]), 37)
        self.assertAlmostEqual(metrics["D"][-1], 0.0, places=14)
        self.assertTrue(all(value >= 0.0 for value in metrics["D"]))

    def test_hidden_geometry_and_raw_adjacent_angle(self):
        hidden = [[1.0, 0.0], [0.0, 1.0]] + [[1.0, 1.0]] * 35
        geometry = hidden_geometry_to_final(hidden)
        self.assertAlmostEqual(geometry["hidden_rms_l2_to_final"][-1], 0.0, places=14)
        self.assertAlmostEqual(geometry["hidden_cosine_to_final"][-1], 1.0, places=14)
        self.assertAlmostEqual(geometry["hidden_cosine_distance_to_final"][-1], 0.0, places=14)
        angles = adjacent_angular_distance(hidden)
        self.assertEqual(len(angles), 36)
        self.assertAlmostEqual(angles[0]["angle_over_pi"], 0.5, places=14)
        self.assertTrue(all(0.0 <= row["angle_over_pi"] <= 1.0 for row in angles))

    def test_closed_trajectory_and_information_barrier(self):
        record = _synthetic_record()
        validate_trajectory_record(record, self.card)
        projected = selector_sample(record)
        self.assertEqual(set(projected), {"identity", "category", "H", "D"})
        validate_selector_sample(projected)
        self.assertEqual(set(project_selector_records([record])[0]), {"identity", "category", "H", "D"})

        invalid = copy.deepcopy(record)
        invalid["target"] = 0
        with self.assertRaises(MMLU0ContractError):
            validate_trajectory_record(invalid, self.card)
        invalid = copy.deepcopy(projected)
        invalid["hidden_cosine_to_final"] = [0.0] * BOUNDARY_COUNT
        with self.assertRaises(MMLU0ContractError):
            validate_selector_sample(invalid)

    def test_candidate_domain_is_exact_half_open_42(self):
        candidates = enumerate_candidates()
        self.assertEqual(len(candidates), 42)
        self.assertEqual(candidates[0]["window"], "11:14")
        self.assertEqual(candidates[0]["boundary_entry"], "B_11")
        self.assertEqual(candidates[0]["boundary_exit"], "B_14")
        self.assertEqual(candidates[-1]["window"], "19:25")
        self.assertEqual(len({(row["width"], row["start"]) for row in candidates}), 42)

    def test_selector_adapter_stays_outcome_blind_and_returns_legal_abstain(self):
        result = analyze_selector([_synthetic_record()], replicates=17, seed=7, formal=False)
        self.assertEqual(result["candidate_count"], 42)
        self.assertEqual(result["record_count"], 1)
        self.assertIn(
            result["selector_decision"],
            {
                "SELECTED_WINDOW",
                "ABSTAIN_NO_RATE_STABLE_ELIGIBLE",
                "ABSTAIN_NO_UNIQUE_TOP1",
                "ABSTAIN_RATE_RANK_UNSTABLE",
            },
        )
        self.assertNotIn("outcome", result)
        self.assertNotIn("hidden_cosine_to_final", result)

    def test_independent_self_test_has_no_execution(self):
        result = run_self_test(CARD_PATH, SCHEMA_PATH)
        self.assertEqual(result["status"], "SELF_TEST_PASS")
        self.assertEqual(len(result["invalid_cases_passed"]), 11)
        self.assertFalse(result["model_or_data_forward_executed"])
        self.assertFalse(result["formal_selector_executed"])
        self.assertFalse(result["outcomes_read"])


if __name__ == "__main__":
    unittest.main()
