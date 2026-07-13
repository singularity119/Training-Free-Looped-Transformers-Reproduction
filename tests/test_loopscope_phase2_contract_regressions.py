import copy
import json
import unittest
from pathlib import Path

from tflt.loopscope.phase2_analysis import choice_output
from tflt.loopscope.phase2_schema import (
    CALIBRATION_IDENTITY_NAMESPACE,
    DIRECT_PROBE_SCORE_SOURCE,
    FULL_IDENTITY_NAMESPACE,
    FULL_LM_EVAL_SCORE_SOURCE,
    SchemaError,
    make_calibration_cell_envelope,
    make_full_final_output_envelope,
    make_identity_manifest,
    validate_calibration_cell_envelope,
    validate_full_final_output_envelope,
    validate_identity_manifest,
    validate_phase2_card,
)
from tflt.loopscope.schema import attach_manifest_sha256


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/qwen17_mmlu_phase2_h1_v2.json"


def identities(count):
    return [
        {"task": "mmlu_subject", "doc_id": str(index), "doc_hash": "%064x" % (index + 1)}
        for index in range(count)
    ]


def producer(kind):
    return {
        "producer_kind": kind,
        "attempt_manifest_sha256": "1" * 64,
        "receipt_manifest_sha256": "2" * 64,
        "command_sha256": "3" * 64,
        "environment_sha256": "4" * 64,
        "revision_report_sha256": "5" * 64,
    }


def direct_choice(identity):
    return choice_output(
        [4.0, 3.0, 2.0, 1.0],
        identity=identity,
        score_source=DIRECT_PROBE_SCORE_SOURCE,
    )


def trajectory_sample(identity):
    return {
        "sample_identity": dict(identity),
        "answer_position": 7,
        "position_rule": "final_pre_answer_prompt_token",
        "boundary_identity": "native_continuation=B_N-B_(b+1)",
        "body_call_indexing": "zero_based_t=0..K-1",
        "steps": [
            {
                "body_call_t": 0,
                "residual_norm": 1.0,
                "state_norm": 2.0,
                "relative_activity": 0.5,
                "residual_ratio_to_previous": None,
                "adjacent_residual_cosine": None,
                "adjacent_residual_cosine_valid": None,
                "nca": None,
                "valid": True,
            },
            {
                "body_call_t": 1,
                "residual_norm": 0.9,
                "state_norm": 2.1,
                "relative_activity": 0.9 / 2.1,
                "residual_ratio_to_previous": 0.9,
                "adjacent_residual_cosine": 0.8,
                "adjacent_residual_cosine_valid": True,
                "nca": {
                    "value": 0.4,
                    "residual_norm": 0.9,
                    "native_continuation_norm": 1.2,
                    "valid": True,
                    "invalid_reason": None,
                    "dtype": "float32",
                    "provenance": "actual_loop_repeated_step",
                    "body_call_t": 1,
                },
                "valid": True,
            },
        ],
        "repeated_step_nca_all_valid": True,
        "valid": True,
        "errors": [],
        "event_counts": {
            "wrapper_forward": 1,
            "bypass_false": 1,
            "body_call": 2,
            "operator_body_calls": 2,
            "g_minus_x": 2,
            "looped_hidden_vs_input": 1,
            "stash_pass": 1,
            "identity_forward": 3,
        },
        "final_output": direct_choice(identity),
    }


class Phase2ClosedWorldRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = json.loads(CARD_PATH.read_text(encoding="utf-8"))

    def _rehash(self, payload):
        payload.pop("manifest_sha256", None)
        attach_manifest_sha256(payload)
        return payload

    def test_card_is_closed_world_and_keeps_seal_scale_and_statistics(self):
        validate_phase2_card(self.card)
        cases = []
        unknown = copy.deepcopy(self.card)
        unknown["unknown_science_escape"] = True
        cases.append(unknown)
        unsealed = copy.deepcopy(self.card)
        unsealed["science"]["calibration_gold_sealed"] = False
        cases.append(unsealed)
        wrong_count = copy.deepcopy(self.card)
        wrong_count["science"]["calibration_count"] = 511
        cases.append(wrong_count)
        wrong_seed = copy.deepcopy(self.card)
        wrong_seed["statistics"]["nca_bootstrap"]["seed"] = 1
        cases.append(wrong_seed)
        wrong_replicates = copy.deepcopy(self.card)
        wrong_replicates["statistics"]["full_paired_bootstrap"]["replicates"] = 1999
        cases.append(wrong_replicates)
        reversed_interval = copy.deepcopy(self.card)
        reversed_interval["statistics"]["nca_bootstrap"]["interval"] = [97.5, 2.5]
        cases.append(reversed_interval)
        nonpositive_k = copy.deepcopy(self.card)
        nonpositive_k["trajectories"]["fixed_step"][0]["k"] = 0
        cases.append(nonpositive_k)
        full_vectors = copy.deepcopy(self.card)
        full_vectors["write_once_contract"]["full_vectors_persisted"] = True
        cases.append(full_vectors)
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(SchemaError):
                    validate_phase2_card(self._rehash(payload))

    def test_identity_manifest_rejects_duplicates_and_sidecar_reordering(self):
        ordered = identities(512)
        manifest = make_identity_manifest(
            self.card,
            identity_namespace=CALIBRATION_IDENTITY_NAMESPACE,
            split="phase1_frozen_validation",
            identities=ordered,
        )
        validate_identity_manifest(manifest, self.card)

        duplicate = copy.deepcopy(manifest)
        duplicate["ordered_sample_identity"][-1] = duplicate["ordered_sample_identity"][0]
        duplicate["ordered_identity_sha256"] = duplicate["ordered_identity_sha256"]
        with self.assertRaises(SchemaError):
            validate_identity_manifest(self._rehash(duplicate), self.card)

        cell = {
            "cell_id": "shared_k2_anchor_12_15_k2_a1",
            "protocol": "shared_k2_anchor",
            "window": "12:15",
            "k": 2,
            "alpha": 1.0,
            "step_size": 0.5,
        }
        samples = [trajectory_sample(identity) for identity in ordered]
        envelope = make_calibration_cell_envelope(
            self.card, manifest, cell=cell, samples=samples,
            producer=producer("gate_b_remote_calibration_probe"),
        )
        validate_calibration_cell_envelope(envelope, self.card, manifest)
        reordered = copy.deepcopy(envelope)
        reordered["samples"] = list(reversed(reordered["samples"]))
        with self.assertRaises(SchemaError):
            validate_calibration_cell_envelope(self._rehash(reordered), self.card, manifest)

    def test_sealed_calibration_rejects_gold_malformed_choice_and_out_of_range_nca(self):
        ordered = identities(512)
        manifest = make_identity_manifest(
            self.card,
            identity_namespace=CALIBRATION_IDENTITY_NAMESPACE,
            split="phase1_frozen_validation",
            identities=ordered,
        )
        cell = {
            "cell_id": "shared_k2_anchor_12_15_k2_a1",
            "protocol": "shared_k2_anchor",
            "window": "12:15",
            "k": 2,
            "alpha": 1.0,
            "step_size": 0.5,
        }
        base = make_calibration_cell_envelope(
            self.card,
            manifest,
            cell=cell,
            samples=[trajectory_sample(identity) for identity in ordered],
            producer=producer("gate_b_remote_calibration_probe"),
        )

        gold = copy.deepcopy(base)
        gold["samples"][0]["final_output"]["correctness"] = True
        malformed = copy.deepcopy(base)
        malformed["samples"][0]["final_output"]["choice_probabilities"][0] = -0.1
        cosine = copy.deepcopy(base)
        cosine["samples"][0]["steps"][1]["nca"]["value"] = 1.01
        for payload in (gold, malformed, cosine):
            with self.assertRaises(SchemaError):
                validate_calibration_cell_envelope(self._rehash(payload), self.card, manifest)

    def test_full_sidecar_forbids_residual_and_cross_namespace_join(self):
        ordered = identities(14042)
        full_manifest = make_identity_manifest(
            self.card,
            identity_namespace=FULL_IDENTITY_NAMESPACE,
            split="mmlu_test_full",
            identities=ordered,
        )
        cell = {
            "cell_id": "fixed_step_12_15_k3_a1p5",
            "protocol": "fixed_step",
            "window": "12:15",
            "k": 3,
            "alpha": 1.5,
            "step_size": 0.5,
        }
        samples = [
            choice_output(
                [4.0, 3.0, 2.0, 1.0],
                identity=identity,
                score_source=FULL_LM_EVAL_SCORE_SOURCE,
                gold_index=0,
                evaluator_acc=True,
            )
            for identity in ordered
        ]
        envelope = make_full_final_output_envelope(
            self.card,
            full_manifest,
            cell=cell,
            samples=samples,
            producer=producer("lm_eval_logged_samples_adapter"),
        )
        validate_full_final_output_envelope(envelope, self.card, full_manifest)

        residual = copy.deepcopy(envelope)
        residual["samples"][0]["residual_norm"] = 1.0
        with self.assertRaises(SchemaError):
            validate_full_final_output_envelope(self._rehash(residual), self.card, full_manifest)

        calibration_manifest = make_identity_manifest(
            self.card,
            identity_namespace=CALIBRATION_IDENTITY_NAMESPACE,
            split="phase1_frozen_validation",
            identities=identities(512),
        )
        with self.assertRaises(SchemaError):
            validate_full_final_output_envelope(envelope, self.card, calibration_manifest)


if __name__ == "__main__":
    unittest.main()
