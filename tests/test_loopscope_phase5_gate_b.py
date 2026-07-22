import copy
import hashlib
import random
import tempfile
import unittest

from tflt.loopscope.phase5_acquisition import (
    Phase5AcquisitionError,
    _smoke_ordinals,
    authoritative_b36_choice_logits,
    formal_retry_allowed,
    normalized_boundary_views,
    write_new_json,
)

from tflt.loopscope.phase5_schema import (
    Phase5ContractError,
    canonical_json_bytes,
    load_json,
    stable_choice_probabilities,
    trajectory_record_from_inputs,
)
from tflt.loopscope.phase5_selector import (
    bootstrap_index_digest,
    build_selector_artifacts,
    canonical_identity_string,
    selector_inputs_from_trajectories,
    selector_ranking_csv,
    verify_selector_artifacts,
)


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs" / "loopscope" / "phase5_card.json"
REGISTRY_PATH = ROOT / "configs" / "loopscope" / "phase5_known_outcome_registry.json"


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


def _trajectory(card, index, subject=None):
    logits = [
        [0.01 * boundary + 0.001 * index, -0.2, -0.4, -0.6]
        for boundary in range(37)
    ]
    hidden = [
        [1.0 + boundary / 100.0, 0.5 + index / 1000.0, -0.25, 0.75]
        for boundary in range(37)
    ]
    return trajectory_record_from_inputs(
        identity={
            "task": "mmlu_subject_%d" % (index % 2),
            "doc_id": "validation:%d" % index,
            "doc_hash": _sha("doc-%d" % index),
        },
        subject=subject or "subject-%d" % (index % 2),
        prompt_sha256=_sha("prompt-%d" % index),
        boundary_choice_logits=logits,
        final_normalized_hidden=hidden,
        renderer_provenance=_renderer(card),
    )


def _membership(records):
    return [
        {
            "ordinal": index,
            "identity": copy.deepcopy(record["identity"]),
            "subject": record["subject"],
            "split": record["split"],
            "prompt_sha256": record["prompt_sha256"],
        }
        for index, record in enumerate(records)
    ]


def _signals():
    rows = []
    for index, (subject, identity) in enumerate(
        (("b", "b-2"), ("a", "a-1"), ("b", "b-1"), ("a", "a-2"))
    ):
        e_values = [-1.0] * 33
        k_values = [-1.0] * 33
        e_values[12] = 1.0 + index / 10.0
        k_values[12] = 1.5 + index / 10.0
        rows.append({"identity": identity, "subject": subject, "E": e_values, "K": k_values})
    return rows


class Phase5GateBSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_json(CARD_PATH)
        cls.registry = load_json(REGISTRY_PATH)
        cls.records = [_trajectory(cls.card, index) for index in range(4)]
        cls.membership = _membership(cls.records)

    def test_closed_world_trajectory_identity_and_membership_order(self):
        inputs = selector_inputs_from_trajectories(
            self.records,
            expected_membership=self.membership,
            expected_count=4,
            expected_subject_count=2,
        )
        self.assertEqual(len(inputs), 4)
        self.assertEqual(set(inputs[0]), {"identity", "subject", "E", "K"})
        self.assertEqual(len(inputs[0]["E"]), 33)
        self.assertEqual(len(inputs[0]["K"]), 33)
        self.assertEqual(
            inputs[0]["identity"], canonical_identity_string(self.records[0]["identity"])
        )

        reversed_membership = list(reversed(copy.deepcopy(self.membership)))
        with self.assertRaisesRegex(Phase5ContractError, "identity/order|ordinal/order"):
            selector_inputs_from_trajectories(
                self.records, expected_membership=reversed_membership
            )

        empty_identity = copy.deepcopy(self.records)
        empty_identity[0]["identity"]["task"] = ""
        with self.assertRaisesRegex(Phase5ContractError, "identity.task"):
            selector_inputs_from_trajectories(empty_identity)

        forbidden = copy.deepcopy(self.records)
        forbidden[0]["gold"] = "A"
        with self.assertRaises(Phase5ContractError):
            selector_inputs_from_trajectories(forbidden)

    def test_bootstrap_digest_matches_exact_sorted_subject_randrange_stream(self):
        samples = _signals()
        observed = bootstrap_index_digest(samples, replicates=11, seed=7)
        reordered = bootstrap_index_digest(list(reversed(samples)), replicates=11, seed=7)
        self.assertEqual(observed, reordered)

        normalized = sorted(samples, key=lambda row: (row["subject"], row["identity"]))
        groups = {}
        for index, row in enumerate(normalized):
            groups.setdefault(row["subject"], []).append(index)
        rng = random.Random(7)
        draws = []
        for _ in range(11):
            drawn = []
            for subject in sorted(groups):
                indices = groups[subject]
                drawn.extend(indices[rng.randrange(len(indices))] for _ in indices)
            draws.append(drawn)
        expected = hashlib.sha256(canonical_json_bytes(draws)).hexdigest()
        self.assertEqual(observed["draw_index_sha256"], expected)

        contaminated = copy.deepcopy(samples)
        contaminated[0]["outcome"] = 1
        with self.assertRaisesRegex(Phase5ContractError, "extra"):
            bootstrap_index_digest(contaminated, replicates=11, seed=7)

    def test_nonformal_artifacts_panel_hashes_csv_and_recomputation(self):
        bindings = {
            "card_file_sha256": _sha("card"),
            "trajectory_manifest_sha256": _sha("trajectory"),
            "implementation_commit": "a" * 40,
        }
        artifacts = build_selector_artifacts(
            self.records,
            card=self.card,
            registry=self.registry,
            expected_membership=self.membership,
            bindings=bindings,
            formal=False,
            replicates=31,
            seed=20260722,
        )
        report = artifacts["selector_report"]
        panel = artifacts["outcome_panel"]
        freeze = artifacts["selector_freeze"]
        self.assertEqual(len(report["published_ranking"]), 25)
        self.assertTrue(
            all(row["failure_reasons"] for row in report["published_ranking"])
        )
        self.assertEqual(report["window_decision"], "ABSTAIN_NO_POINT_ELIGIBLE")
        self.assertFalse(report["geometry_fields_consumed"])
        self.assertFalse(report["outcome_fields_consumed"])
        self.assertNotIn("15:18", panel["panel_high3"] + panel["blind_low3"])
        self.assertLessEqual(panel["unique_cell_count"], 8)
        self.assertEqual(
            freeze["bootstrap_draw_index_sha256"],
            report["bootstrap"]["draw_index_sha256"],
        )
        csv_text = selector_ranking_csv(report)
        self.assertEqual(len(csv_text.splitlines()), 26)
        self.assertTrue(csv_text.startswith("start,window,score,E,K,point_eligible"))

        receipt = verify_selector_artifacts(
            self.records,
            card=self.card,
            registry=self.registry,
            expected_membership=self.membership,
            bindings=bindings,
            artifacts=artifacts,
            formal=False,
            replicates=31,
            seed=20260722,
        )
        self.assertEqual(receipt["status"], "PASS")
        self.assertFalse(receipt["outcome_values_consumed"])

        changed = copy.deepcopy(artifacts)
        changed["selector_freeze"]["eligible_count"] += 1
        with self.assertRaises(Phase5ContractError):
            verify_selector_artifacts(
                self.records,
                card=self.card,
                registry=self.registry,
                expected_membership=self.membership,
                bindings=bindings,
                artifacts=changed,
                formal=False,
                replicates=31,
                seed=20260722,
            )

    def test_formal_parameters_and_population_are_not_overridable(self):
        bindings = {"trajectory_manifest_sha256": _sha("trajectory")}
        with self.assertRaisesRegex(Phase5ContractError, "R=2000"):
            build_selector_artifacts(
                self.records,
                card=self.card,
                registry=self.registry,
                expected_membership=self.membership,
                bindings=bindings,
                formal=True,
                replicates=31,
                seed=20260722,
            )
        with self.assertRaisesRegex(Phase5ContractError, "record count"):
            build_selector_artifacts(
                self.records,
                card=self.card,
                registry=self.registry,
                expected_membership=self.membership,
                bindings=bindings,
                formal=True,
            )


class Phase5GateBAcquisitionTests(unittest.TestCase):
    def test_repair_smoke_uses_exact_planning_trigger_identities(self):
        pool = [
            {
                "identity": {
                    "task": "mmlu_human_aging",
                    "doc_id": "mmlu_human_aging:validation:18",
                }
            },
            {
                "identity": {
                    "task": "mmlu_marketing",
                    "doc_id": "mmlu_marketing:validation:24",
                }
            },
        ]
        self.assertEqual(_smoke_ordinals(pool), [0, 1])

    def test_b36_uses_native_choice_logits_and_sliced_mismatch_is_non_gating(self):
        native = [1.0, 2.0, 3.0, 4.0]
        sliced = [100.0, 200.0, 300.0, 400.0]
        observed = authoritative_b36_choice_logits(
            native,
            sliced,
            full_head_shape_matches=True,
            full_head_closes=True,
        )
        self.assertIs(observed, native)
        self.assertNotEqual(observed, sliced)
        self.assertEqual(
            stable_choice_probabilities(observed),
            stable_choice_probabilities(native),
        )
        self.assertNotEqual(
            stable_choice_probabilities(observed),
            stable_choice_probabilities(sliced),
        )

        with self.assertRaisesRegex(Phase5AcquisitionError, "full-head logits do not close"):
            authoritative_b36_choice_logits(
                native,
                sliced,
                full_head_shape_matches=True,
                full_head_closes=False,
            )

    def test_boundary_views_apply_final_norm_once_except_native_b36(self):
        calls = []

        def final_norm(value):
            calls.append(value)
            return "norm(%s)" % value

        def lens_space(norm, value, boundary_index, layer_count):
            self.assertEqual(layer_count, 36)
            return value if boundary_index == layer_count else norm(value)

        states = ["B_%d" % index for index in range(37)]
        views = normalized_boundary_views(
            states,
            position=9,
            final_norm=final_norm,
            lens_space_fn=lens_space,
            state_at_position=lambda state, position: "%s@%d" % (state, position),
        )
        self.assertEqual(len(views), 37)
        self.assertEqual(views[0], "norm(B_0@9)")
        self.assertEqual(views[-1], "B_36@9")
        self.assertEqual(calls, ["B_%d@9" % index for index in range(36)])
        with self.assertRaises(Phase5AcquisitionError):
            normalized_boundary_views(
                states[:-1],
                position=0,
                final_norm=final_norm,
                lens_space_fn=lens_space,
                state_at_position=lambda state, position: state,
            )

    def test_formal_retry_requires_proven_zero_forward_and_zero_record(self):
        allowed = {
            "result": "FAILED",
            "forward_calls": 0,
            "records_written": 0,
            "formal_retry_eligible": True,
        }
        self.assertTrue(formal_retry_allowed(allowed))
        for key, value in (("forward_calls", 1), ("records_written", 1)):
            tampered = dict(allowed)
            tampered[key] = value
            self.assertFalse(formal_retry_allowed(tampered))

    def test_strict_writer_is_write_once_and_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = __import__("pathlib").Path(tmp) / "receipt.json"
            write_new_json(path, {"finite": 1.0})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"finite": 2.0})
            with self.assertRaises(ValueError):
                write_new_json(path.with_name("nan.json"), {"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
