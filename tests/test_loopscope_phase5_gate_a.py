import copy
import hashlib
import json
import math
import unittest
from pathlib import Path

from tflt.config import LoopConfig
from tflt.loopscope.phase5_schema import (
    ABSTAIN_LABELS,
    BOUNDARY_IDS,
    CONSENSUS_STARTS,
    Phase5ContractError,
    SHIFT_STARTS,
    WINDOW_STARTS,
    analyze_signal_samples,
    boundary_contract,
    construct_panel,
    hidden_geometry,
    load_json,
    stable_choice_probabilities,
    trajectory_record_from_inputs,
    trajectory_window_signals,
    validate_card,
    validate_card_schema,
    validate_gate_a_receipt,
    validate_registry,
    validate_token_ids,
    validate_trajectory_record,
)
from tflt.strategies import run_loop


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs" / "loopscope" / "phase5_card.json"
SCHEMA_PATH = ROOT / "configs" / "loopscope" / "phase5_card_schema.json"
REGISTRY_PATH = ROOT / "configs" / "loopscope" / "phase5_known_outcome_registry.json"
RECEIPT_PATH = ROOT / "configs" / "loopscope" / "phase5_gate_a_receipt.json"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _renderer(card):
    return {
        "model_repo": card["model"]["repo"],
        "model_revision": card["model"]["revision"],
        "tokenizer_revision": card["model"]["tokenizer_revision"],
        "dataset_repo": card["task"]["dataset"],
        "dataset_revision": card["task"]["dataset_revision"],
        "renderer_manifest_sha256": card["task"][
            "validation_renderer_manifest_internal_sha256"
        ],
        "forward_type": "native_no_loop",
        "formal_forward_count_per_identity": 1,
        "loop_insertions": 0,
        "position": "final_non_padding_prompt_token",
        "projection": "frozen_final_norm_then_tied_output_weight_exact_choice_rows",
        "choice_order": ["A", "B", "C", "D"],
        "spaced_choice_token_ids": card["model"]["spaced_choice_token_ids"],
        "final_norm_path": "model.norm",
        "output_head_path": "lm_head_tied_to_model.embed_tokens.weight",
    }


def _trajectory(card):
    logits = [[0.03 * index, -0.2, -0.4, -0.6] for index in range(37)]
    hidden = [[1.0 + index / 100.0, 0.5, -0.25, 0.75] for index in range(37)]
    return trajectory_record_from_inputs(
        identity={"task": "mmlu_test", "doc_id": "0", "doc_hash": _sha("doc")},
        subject="subject",
        prompt_sha256=_sha("prompt"),
        boundary_choice_logits=logits,
        final_normalized_hidden=hidden,
        renderer_provenance=_renderer(card),
    )


def _sample(identity, centers):
    e_values = [-1.0] * 33
    k_values = [-1.0] * 33
    for start, value in centers.items():
        e_values[start] = value
        k_values[start] = value
    return {"identity": identity, "subject": "one", "E": e_values, "K": k_values}


class Phase5GateATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_json(CARD_PATH)
        cls.schema = load_json(SCHEMA_PATH)
        cls.registry = load_json(REGISTRY_PATH)

    def test_card_schema_registry_are_hash_closed_and_exact(self):
        validate_card(self.card)
        validate_card_schema(self.schema)
        validate_registry(self.registry)

        changed = copy.deepcopy(self.card)
        changed["selector"]["selection_frequency_threshold"] = 0.81
        with self.assertRaisesRegex(Phase5ContractError, "manifest_sha256"):
            validate_card(changed)

        changed = copy.deepcopy(self.registry)
        changed["entries"][1]["gain_percentage_points"] = 0.35
        with self.assertRaisesRegex(Phase5ContractError, "manifest_sha256"):
            validate_registry(changed)

    def test_gate_receipt_recomputes_artifact_hashes(self):
        receipt = load_json(RECEIPT_PATH)
        validate_gate_a_receipt(receipt, ROOT)
        changed = copy.deepcopy(receipt)
        changed["artifacts"]["card"]["file_sha256"] = "0" * 64
        with self.assertRaisesRegex(Phase5ContractError, "manifest_sha256"):
            validate_gate_a_receipt(changed, ROOT)

    def test_boundary_universe_and_inclusive_mapping(self):
        contract = boundary_contract()
        self.assertEqual(contract["boundaries"], list(BOUNDARY_IDS))
        self.assertEqual(contract["window_starts"], list(WINDOW_STARTS))
        self.assertEqual(len(WINDOW_STARTS), 33)
        self.assertEqual(contract["shift_starts"], list(SHIFT_STARTS))
        self.assertEqual((SHIFT_STARTS[0], SHIFT_STARTS[-1]), (1, 31))
        self.assertEqual(contract["consensus_starts"], list(CONSENSUS_STARTS))
        self.assertEqual((CONSENSUS_STARTS[0], CONSENSUS_STARTS[-1]), (4, 28))
        self.assertEqual(contract["inclusive_15_18"], {"entry": "B_15", "exit": "B_19"})

    def test_exact_choice_token_ids_are_unique_single_tokens_and_fail_closed(self):
        spaced = {key: [value] for key, value in self.card["model"]["spaced_choice_token_ids"].items()}
        plain = {key: [value] for key, value in self.card["model"]["plain_choice_token_ids"].items()}
        validate_token_ids(spaced, plain)
        changed = copy.deepcopy(spaced)
        changed["A"] = [362, 32]
        with self.assertRaisesRegex(Phase5ContractError, "exactly one token"):
            validate_token_ids(changed, plain)
        changed = copy.deepcopy(spaced)
        changed["A"] = [999]
        with self.assertRaisesRegex(Phase5ContractError, "token id differs"):
            validate_token_ids(changed, plain)

    def test_five_trajectory_scalars_and_final_boundary_recompute(self):
        record = _trajectory(self.card)
        self.assertEqual([row["boundary_id"] for row in record["boundaries"]], list(BOUNDARY_IDS))
        final = record["boundaries"][-1]
        self.assertAlmostEqual(final["kl_to_final"], 0.0, places=12)
        self.assertAlmostEqual(final["hidden_l2_to_final"], 0.0, places=12)
        self.assertAlmostEqual(final["hidden_cosine_to_final"], 1.0, places=12)
        self.assertAlmostEqual(final["hidden_cosine_distance_to_final"], 0.0, places=12)
        signals = trajectory_window_signals(record)
        self.assertEqual(len(signals["windows"]), 33)
        row = signals["windows"][15]
        self.assertEqual(row["window"], "15:18")
        self.assertAlmostEqual(row["E"], -row["delta_choice_entropy"])
        self.assertAlmostEqual(row["K"], row["delta_kl_to_final"])
        self.assertAlmostEqual(
            row["delta_hidden_cosine_distance_to_final"],
            -row["delta_hidden_cosine_to_final"],
        )

    def test_trajectory_nonfinite_zero_norm_forbidden_and_tamper_fail(self):
        with self.assertRaises(Phase5ContractError):
            stable_choice_probabilities([0.0, float("nan"), 1.0, 2.0])
        with self.assertRaisesRegex(Phase5ContractError, "zero-norm"):
            hidden_geometry([0.0, 0.0], [1.0, 0.0])
        record = _trajectory(self.card)
        changed = copy.deepcopy(record)
        changed["gold"] = "A"
        with self.assertRaises(Phase5ContractError):
            validate_trajectory_record(changed)
        changed = copy.deepcopy(record)
        changed["boundaries"][2]["choice_entropy"] += 1e-3
        with self.assertRaisesRegex(Phase5ContractError, "entropy recomputation"):
            validate_trajectory_record(changed)
        changed = copy.deepcopy(record)
        changed["boundaries"][2]["hidden_cosine_distance_to_final"] += 1e-3
        with self.assertRaisesRegex(Phase5ContractError, "derived exactly once"):
            validate_trajectory_record(changed)

    def test_consensus_selection_signs_score_and_geometry_independence(self):
        samples = [_sample("id-%d" % index, {12: 1.0 + index / 100.0}) for index in range(4)]
        report = analyze_signal_samples(samples, replicates=31, seed=20260722)
        self.assertEqual(report["window_decision"], "SELECTED")
        self.assertEqual(report["selected_window"], "12:15")
        self.assertEqual(report["selection_frequency"], 1.0)
        row = next(row for row in report["published_ranking"] if row["start"] == 12)
        self.assertTrue(row["point_eligible"])
        self.assertTrue(all(payload["lower95"] > 0.0 for payload in row["contrasts"].values()))
        self.assertAlmostEqual(
            row["score"],
            min(
                payload["point"] / max(payload["standard_error"], 1e-12)
                for payload in row["contrasts"].values()
            ),
        )
        with_geometry = copy.deepcopy(samples)
        for sample in with_geometry:
            sample["geometry_diagnostic"] = [999.0]
        changed = analyze_signal_samples(with_geometry, replicates=31, seed=20260722)
        self.assertEqual(
            json.dumps(report, sort_keys=True, allow_nan=False),
            json.dumps(changed, sort_keys=True, allow_nan=False),
        )
        self.assertFalse(report["geometry_fields_consumed"])

    def test_three_abstain_boundaries(self):
        no_eligible = analyze_signal_samples(
            [_sample("none-%d" % index, {}) for index in range(4)],
            replicates=7,
            seed=1,
        )
        self.assertEqual(no_eligible["window_decision"], ABSTAIN_LABELS[0])

        tied = analyze_signal_samples(
            [_sample("tie-%d" % index, {8: 1.0, 20: 1.0}) for index in range(4)],
            replicates=7,
            seed=1,
        )
        self.assertEqual(tied["window_decision"], ABSTAIN_LABELS[1])

        values_8 = [5.1, 5.1, 5.1, -0.9, -0.9]
        values_20 = [5.0, 5.0, -1.0, 5.0, -1.0]
        unstable = [
            _sample("unstable-%d" % index, {8: values_8[index], 20: values_20[index]})
            for index in range(5)
        ]
        low_frequency = analyze_signal_samples(unstable, replicates=401, seed=20260722)
        self.assertEqual(low_frequency["point_top1_window"], "8:11")
        self.assertLess(low_frequency["selection_frequency"], 0.80)
        self.assertEqual(low_frequency["window_decision"], ABSTAIN_LABELS[2])

    def test_known_exclusion_selected_inclusive_high3_low3_and_dedup(self):
        ranking = [
            {"window": "%d:%d" % (start, start + 3), "start": start, "score": 100.0 - start}
            for start in CONSENSUS_STARTS
        ]
        panel = construct_panel(ranking, selected_window="10:13", known_windows=["15:18"])
        self.assertEqual(panel["raw_high3"], ["4:7", "5:8", "6:9"])
        self.assertEqual(panel["panel_high3"], ["4:7", "5:8", "10:13"])
        self.assertEqual(panel["selected_inclusive_replacement"]["removed"], "6:9")
        self.assertTrue(set(panel["panel_high3"]).isdisjoint(panel["blind_low3"]))
        self.assertNotIn("15:18", panel["panel_high3"] + panel["blind_low3"])
        self.assertLessEqual(panel["unique_cell_count"], 8)
        self.assertEqual(len(panel["unique_cells"]), len(set(panel["unique_cells"])))
        abstain_panel = construct_panel(ranking, selected_window=None, known_windows=["15:18"])
        self.assertIsNone(abstain_panel["selected_inclusive_replacement"])
        self.assertEqual(abstain_panel["panel_high3"], abstain_panel["raw_high3"])

    def test_current_euler_alias_k3_alpha1_has_three_calls_step_third_horizon_one(self):
        inputs = []

        def body(value):
            inputs.append(value)
            return value + 1.0

        config = LoopConfig("synthetic", (15, 18), k=3, strategy="euler", alpha=1.0)
        output = run_loop(body, 0.0, config)
        self.assertEqual(len(inputs), 3)
        self.assertAlmostEqual(inputs[0], 0.0)
        self.assertAlmostEqual(inputs[1], 1.0 / 3.0)
        self.assertAlmostEqual(inputs[2], 2.0 / 3.0)
        self.assertAlmostEqual(config.alpha / config.k, 1.0 / 3.0)
        self.assertAlmostEqual(config.k * (config.alpha / config.k), 1.0)
        self.assertAlmostEqual(output, 1.0)


if __name__ == "__main__":
    unittest.main()
