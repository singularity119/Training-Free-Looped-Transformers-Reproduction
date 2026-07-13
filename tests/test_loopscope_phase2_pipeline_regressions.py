import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from tflt import eval_runner
from tflt.loopscope.phase2_analysis import (
    Phase2AnalysisError,
    classify_nca,
    exact_mcnemar_p,
    classify_h1,
    choice_output,
    holm_step_down,
    make_analysis_input_manifest,
    required_calibration_cell_ids,
    required_full_cell_ids,
    validate_analysis_input_manifest,
    validate_analysis_report,
)
from tflt.loopscope.phase2_schema import (
    CALIBRATION_IDENTITY_NAMESPACE,
    DIRECT_PROBE_SCORE_SOURCE,
    FULL_LM_EVAL_SCORE_SOURCE,
    FIXED_HORIZON_CONTRAST_IDS,
    PHASE2_ANALYSIS_SCHEMA_VERSION,
    PRIMARY_CONTRAST_IDS,
    SchemaError,
    atomic_write_new_json,
    b2_logical_cells,
    make_calibration_label_sidecar,
    make_identity_manifest,
    make_hashed_manifest,
    protocol_cell_id,
    make_unseal_authorization,
    make_unseal_receipt,
    validate_calibration_label_sidecar,
    validate_unseal_authorization,
    validate_unseal_receipt,
)
from tflt.loopscope.phase2_trajectory import (
    TrajectoryError,
    build_b1_admission_proof,
    run_phase2_trajectory_probe,
    validate_b1_admission_proof,
    validate_probe_aggregate,
)
from tflt.loopscope.schema import attach_manifest_sha256


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/qwen17_mmlu_phase2_h1_v2.json"


def identity(index):
    return {"task": "mmlu_subject", "doc_id": str(index), "doc_hash": "%064x" % (index + 1)}


def source_ref(name, digest="a" * 64):
    return {"path": "%s.json" % name, "sha256": digest}


def cell_ref(cell_id, index):
    return {"cell_id": cell_id, "path": "%s.json" % cell_id, "sha256": "%064x" % (index + 1)}


def analysis_sources(card):
    calibration = [cell_ref(cell_id, index) for index, cell_id in enumerate(required_calibration_cell_ids(card))]
    full = [cell_ref(cell_id, index + 100) for index, cell_id in enumerate(required_full_cell_ids(card))]
    return {
        "card": source_ref("card", card["manifest_sha256"]),
        "calibration_identity_manifest": source_ref("calibration_identity"),
        "full_identity_manifest": source_ref("full_identity", "b" * 64),
        "calibration_baseline": source_ref("calibration_baseline", "c" * 64),
        "calibration_cells": calibration,
        "full_final_output_cells": full,
        "calibration_labels": source_ref("calibration_labels", "d" * 64),
        "unseal_authorization": source_ref("unseal_authorization", "e" * 64),
        "unseal_receipt": source_ref("unseal_receipt", "f" * 64),
    }


def provenance():
    return {
        "authorization_manifest_sha256": "1" * 64,
        "attempt_manifest_sha256": "2" * 64,
        "receipt_manifest_sha256": "3" * 64,
        "executor_thread_id": "019f5c1d-7355-7130-81e2-125b4f1e21c0",
        "command_sha256": "4" * 64,
        "environment_sha256": "5" * 64,
    }


def choice_row(sample_identity, scores, gold=0):
    top = max(range(4), key=lambda index: scores[index])
    return {
        "doc_id": sample_identity["doc_id"],
        "doc_hash": sample_identity["doc_hash"],
        "filtered_resps": list(scores),
        "target": gold,
        "metrics": {"acc,none": int(top == gold)},
    }


def baseline_probe_row(sample_identity):
    boundaries = []
    for window in ("11:14", "12:15", "13:16"):
        start, end = (int(value) for value in window.split(":"))
        boundaries.append(
            {
                "window": window,
                "inclusive_boundary": "B_%d_to_B_%d" % (start, end + 1),
                "native_boundary_formula": "B_N-B_%d" % (end + 1),
                "baseline_update_norm": 1.0,
                "native_continuation_norm": 1.0,
                "baseline_nca": {
                    "value": 1.0,
                    "residual_norm": 1.0,
                    "native_continuation_norm": 1.0,
                    "valid": True,
                    "invalid_reason": None,
                    "dtype": "float32",
                    "provenance": "baseline_single_forward_B_(b+1)-B_a",
                    "body_call_t": None,
                },
            }
        )
    return {
        "sample_identity": sample_identity,
        "answer_position": 7,
        "final_output": choice_output(
            [4.0, 3.0, 2.0, 1.0],
            identity=sample_identity,
            score_source=DIRECT_PROBE_SCORE_SOURCE,
        ),
        "window_boundaries": boundaries,
    }


def trajectory_probe_row(sample_identity, cell):
    steps = []
    for body_call_t in range(cell["k"]):
        steps.append(
            {
                "body_call_t": body_call_t,
                "residual_norm": 1.0,
                "state_norm": 2.0,
                "relative_activity": 0.5,
                "residual_ratio_to_previous": None if body_call_t == 0 else 1.0,
                "adjacent_residual_cosine": None if body_call_t == 0 else 1.0,
                "adjacent_residual_cosine_valid": None if body_call_t == 0 else True,
                "nca": None if body_call_t == 0 else {
                    "value": 1.0,
                    "residual_norm": 1.0,
                    "native_continuation_norm": 1.0,
                    "valid": True,
                    "invalid_reason": None,
                    "dtype": "float32",
                    "provenance": "actual_loop_repeated_step",
                    "body_call_t": body_call_t,
                },
                "valid": True,
            }
        )
    return {
        "sample_identity": sample_identity,
        "answer_position": 7,
        "position_rule": "final_pre_answer_prompt_token",
        "boundary_identity": "native_continuation=B_N-B_(b+1)",
        "body_call_indexing": "zero_based_t=0..K-1",
        "steps": steps,
        "repeated_step_nca_all_valid": True,
        "valid": True,
        "errors": [],
        "event_counts": {
            "wrapper_forward": 1,
            "bypass_false": 1,
            "body_call": cell["k"],
            "operator_body_calls": cell["k"],
            "g_minus_x": cell["k"],
            "looped_hidden_vs_input": 1,
            "stash_pass": 1,
            "identity_forward": 3,
        },
        "final_output": choice_output(
            [4.0, 3.0, 2.0, 1.0],
            identity=sample_identity,
            score_source=DIRECT_PROBE_SCORE_SOURCE,
        ),
    }


class Phase2PipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = json.loads(CARD_PATH.read_text(encoding="utf-8"))

    def _rehash(self, payload):
        payload.pop("manifest_sha256", None)
        attach_manifest_sha256(payload)
        return payload

    def test_large_discordance_exact_mcnemar_is_stable(self):
        for left, right in ((0, 1024), (500, 600), (7000, 7042), (7042, 7000)):
            value = exact_mcnemar_p(left, right)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertEqual(exact_mcnemar_p(7000, 7042), exact_mcnemar_p(7042, 7000))

    def test_analysis_input_is_source_only_and_closes_cell_sets_and_statistics(self):
        payload = make_analysis_input_manifest(self.card, analysis_sources(self.card), provenance())
        validate_analysis_input_manifest(payload, self.card)

        forged = copy.deepcopy(payload)
        forged["primary_contrasts"] = {"forged": {"mcnemar_raw_p": 0.0}}
        missing = copy.deepcopy(payload)
        missing["sources"]["full_final_output_cells"].pop()
        rogue = copy.deepcopy(payload)
        rogue["sources"]["calibration_cells"].append(cell_ref("rogue_cell", 999))
        wrong_seed = copy.deepcopy(payload)
        wrong_seed["statistics_contract"]["nca_bootstrap"]["seed"] = 1
        wrong_replicates = copy.deepcopy(payload)
        wrong_replicates["statistics_contract"]["full_paired_bootstrap"]["replicates"] = 1999
        for candidate in (forged, missing, rogue, wrong_seed, wrong_replicates):
            with self.assertRaises(SchemaError):
                validate_analysis_input_manifest(self._rehash(candidate), self.card)

    def test_unseal_requires_card_sealed_hashes_authorization_and_receipt(self):
        identities = [identity(index) for index in range(512)]
        manifest = make_identity_manifest(
            self.card,
            identity_namespace=CALIBRATION_IDENTITY_NAMESPACE,
            split="phase1_frozen_validation",
            identities=identities,
        )
        sealed_hashes = ["%064x" % (1000 + index) for index in range(16)]
        authorization = make_unseal_authorization(
            self.card,
            manifest,
            sealed_artifact_sha256=sealed_hashes,
            authorization_id="phase2-h1-v2-calibration-unseal-1",
            authorized_by_thread="019f5149-35f6-7383-b3ec-3376ba413d2b",
            issued_at_utc="2026-07-14T00:00:00Z",
        )
        validate_unseal_authorization(authorization, self.card, manifest, sealed_hashes)
        labels = make_calibration_label_sidecar(
            self.card,
            manifest,
            authorization,
            gold_indices=[index % 4 for index in range(512)],
            producer={
                "producer_kind": "authorized_calibration_label_unseal",
                "attempt_manifest_sha256": "1" * 64,
                "receipt_manifest_sha256": "2" * 64,
                "command_sha256": "3" * 64,
                "environment_sha256": "4" * 64,
                "revision_report_sha256": "5" * 64,
            },
        )
        validate_calibration_label_sidecar(labels, self.card, manifest, authorization)
        receipt = make_unseal_receipt(
            self.card,
            manifest,
            authorization,
            labels,
            sealed_artifact_sha256=sealed_hashes,
            accessed_by_thread="019f5132-cd06-7281-be3b-a01a06f7a757",
            accessed_at_utc="2026-07-14T00:01:00Z",
        )
        validate_unseal_receipt(receipt, self.card, manifest, authorization, labels, sealed_hashes)

        changed = list(sealed_hashes)
        changed[-1] = "9" * 64
        with self.assertRaises(SchemaError):
            validate_unseal_authorization(authorization, self.card, manifest, changed)
        forged_receipt = copy.deepcopy(receipt)
        forged_receipt["label_sidecar_sha256"] = "8" * 64
        with self.assertRaises(SchemaError):
            validate_unseal_receipt(
                self._rehash(forged_receipt), self.card, manifest, authorization, labels, sealed_hashes
            )

    def test_b1_producer_path_emits_k1_and_prefix_proof_without_vectors(self):
        identities = [identity(index) for index in range(4)]
        baseline = [
            choice_output(
                [4.0, 3.0, 2.0, 1.0],
                identity=item,
                score_source=DIRECT_PROBE_SCORE_SOURCE,
            )
            for item in identities
        ]
        k1 = {window: copy.deepcopy(baseline) for window in ("11:14", "12:15", "13:16")}
        captures = {}
        for window in ("11:14", "12:15", "13:16"):
            for protocol, k, alpha in (
                ("shared_k2_anchor", 2, 1.0),
                ("fixed_step", 3, 1.5),
                ("fixed_step", 4, 2.0),
            ):
                cell_id = "%s_%s_k%d_a%s" % (
                    protocol, window.replace(":", "_"), k, str(alpha).replace(".", "p").replace("p0", "")
                )
                captures[cell_id] = [
                    {
                        "sample_identity": item,
                        "states": [[1.0, 2.0], [2.0, 3.0]][:k],
                        "residuals": [[0.1, 0.2], [0.2, 0.3]][:k],
                    }
                    for item in identities
                ]
        proof = build_b1_admission_proof(self.card, baseline, k1, captures)
        validate_b1_admission_proof(proof, self.card)
        self.assertTrue(proof["k1_equivalence"]["all_match"])
        self.assertTrue(proof["fixed_step_prefix_consistency"]["all_match"])
        serialized = json.dumps(proof, sort_keys=True)
        self.assertNotIn('"states"', serialized)
        self.assertNotIn('"residuals"', serialized)
        self.assertIn("build_b1_admission_proof(", inspect.getsource(run_phase2_trajectory_probe))

    def test_b1_aggregate_closes_cell_order_and_proof_identity(self):
        identities = [identity(index) for index in range(4)]
        baseline_outputs = [baseline_probe_row(item)["final_output"] for item in identities]
        k1 = {window: copy.deepcopy(baseline_outputs) for window in ("11:14", "12:15", "13:16")}
        captures = {}
        for window in ("11:14", "12:15", "13:16"):
            for protocol, k, alpha in (
                ("shared_k2_anchor", 2, 1.0),
                ("fixed_step", 3, 1.5),
                ("fixed_step", 4, 2.0),
            ):
                cell_id = protocol_cell_id(protocol, window, k, alpha)
                captures[cell_id] = [
                    {
                        "sample_identity": item,
                        "states": [[1.0, 2.0], [2.0, 3.0]],
                        "residuals": [[0.1, 0.2], [0.2, 0.3]],
                    }
                    for item in identities
                ]
        proof = build_b1_admission_proof(self.card, baseline_outputs, k1, captures)
        producer = {
            "producer_kind": "gate_b_remote_b1_smoke_probe",
            "attempt_manifest_sha256": "1" * 64,
            "receipt_manifest_sha256": "2" * 64,
            "command_sha256": "3" * 64,
            "environment_sha256": "4" * 64,
            "revision_report_sha256": "5" * 64,
        }
        report = make_hashed_manifest(
            {
                "schema_version": "loopscope.phase2-b1-smoke-aggregate.v1",
                "artifact_kind": "b1_smoke_scalar_aggregate",
                "card_manifest_sha256": self.card["manifest_sha256"],
                "revision": self.card["science"]["revision"],
                "sample_count": 4,
                "probe_pool": {"source": "fake-test"},
                "warnings": [],
                "session_contract": {
                    "model_load_count": 1,
                    "no_loop_boundary_passes_per_sample": 1,
                    "logical_loop_cell_count": 18,
                    "k1_admission_cell_count": 3,
                    "native_vectors_persisted": False,
                },
                "baseline_examples": [baseline_probe_row(item) for item in identities],
                "loop_cells": [
                    {
                        "cell": cell,
                        "sample_count": 4,
                        "trajectory_samples": [trajectory_probe_row(item, cell) for item in identities],
                    }
                    for cell in b2_logical_cells(self.card)
                ],
                "b1_admission_proof_sha256": proof["manifest_sha256"],
                "producer": producer,
                "vectors_persisted": False,
            }
        )
        validate_probe_aggregate(report, self.card, b1_proof=proof)

        reordered = copy.deepcopy(report)
        reordered["loop_cells"] = list(reversed(reordered["loop_cells"]))
        with self.assertRaises((TrajectoryError, SchemaError)):
            validate_probe_aggregate(self._rehash(reordered), self.card, b1_proof=proof)
        wrong_identity = copy.deepcopy(report)
        wrong_identity["loop_cells"][0]["trajectory_samples"][0]["sample_identity"] = identities[1]
        with self.assertRaises((TrajectoryError, SchemaError)):
            validate_probe_aggregate(self._rehash(wrong_identity), self.card, b1_proof=proof)

    def test_logged_samples_join_is_order_independent_and_exact(self):
        first, second = identity(1), identity(2)
        result = {
            "samples": {
                "mmlu_subject": [
                    choice_row(second, [1.0, 4.0, 0.0, -1.0], gold=1),
                    choice_row(first, [4.0, 1.0, 0.0, -1.0], gold=0),
                ]
            }
        }
        rows = eval_runner._join_phase2_logged_samples(result, [first, second])
        self.assertEqual([row["sample_identity"] for row in rows], [first, second])
        self.assertTrue(all(row["choice_score_source"] == FULL_LM_EVAL_SCORE_SOURCE for row in rows))

        duplicate = copy.deepcopy(result)
        duplicate["samples"]["mmlu_subject"].append(choice_row(first, [4, 1, 0, -1], gold=0))
        with self.assertRaises(ValueError):
            eval_runner._join_phase2_logged_samples(duplicate, [first, second])
        missing_scores = copy.deepcopy(result)
        missing_scores["samples"]["mmlu_subject"][0].pop("filtered_resps")
        with self.assertRaises(ValueError):
            eval_runner._join_phase2_logged_samples(missing_scores, [first, second])

    def test_reversed_nca_interval_is_not_accepted_as_available(self):
        primary = {}
        for window in ("11:14", "12:15", "13:16"):
            for k in (2, 3, 4):
                primary["%s_k%d" % (window, k)] = {
                    "valid_fraction": 1.0,
                    "overall_interval": {"available": True, "interval": [0.2, 0.1]},
                }
        paired = {
            "k%d" % k: {
                "paired_valid_fraction": 1.0,
                "difference_interval": {"available": True, "interval": [0.1, 0.2]},
            }
            for k in (2, 3, 4)
        }
        with self.assertRaises(Phase2AnalysisError):
            classify_nca(primary, paired, {})

    def test_analysis_report_is_closed_world_and_recomputes_p_holm_and_labels(self):
        bootstrap = {"replicates": 2000, "seed": 20260710, "paired_index": True}

        def base_contrast():
            return {
                "sample_count": 14042,
                "delta_acc_pp": 0.0,
                "paired_ci_pp": [0.0, 0.0],
                "bootstrap": dict(bootstrap),
                "transitions": {
                    "right_to_right": 14042,
                    "wrong_to_right": 0,
                    "right_to_wrong": 0,
                    "wrong_to_wrong": 0,
                },
                "right_to_right": 14042,
                "wrong_to_right": 0,
                "right_to_wrong": 0,
                "wrong_to_wrong": 0,
                "mcnemar_raw_p": 1.0,
            }

        primary_specs = []
        for window, stem in (("11:14", "11_14"), ("12:15", "12_15"), ("13:16", "13_16")):
            k2 = protocol_cell_id("shared_k2_anchor", window, 2, 1.0)
            k3 = protocol_cell_id("fixed_step", window, 3, 1.5)
            k4 = protocol_cell_id("fixed_step", window, 4, 2.0)
            primary_specs.extend(
                (("fs_%s_k2_k3" % stem, k2, k3), ("fs_%s_k3_k4" % stem, k3, k4))
            )
        fixed_specs = [
            (
                "fh_12_15_k1_k2",
                protocol_cell_id("baseline_no_loop", "none", 1, 1.0),
                protocol_cell_id("shared_k2_anchor", "12:15", 2, 1.0),
            ),
            (
                "fh_12_15_k2_k3",
                protocol_cell_id("shared_k2_anchor", "12:15", 2, 1.0),
                protocol_cell_id("fixed_horizon", "12:15", 3, 1.0),
            ),
            (
                "fh_12_15_k3_k4",
                protocol_cell_id("fixed_horizon", "12:15", 3, 1.0),
                protocol_cell_id("fixed_horizon", "12:15", 4, 1.0),
            ),
        ]

        def family(specs, frozen_ids):
            raw = {identifier: 1.0 for identifier in frozen_ids}
            adjusted = holm_step_down(raw, frozen_ids)
            result = {}
            for identifier, left, right in specs:
                cell = base_contrast()
                cell.update(
                    {
                        "contrast_id": identifier,
                        "left_cell_id": left,
                        "right_cell_id": right,
                        "holm_adjusted_p": adjusted[identifier]["adjusted_p"],
                        "holm_significant": adjusted[identifier]["significant"],
                        "holm_rank": adjusted[identifier]["holm_rank"],
                    }
                )
                result[identifier] = cell
            return result

        primary = family(primary_specs, PRIMARY_CONTRAST_IDS)
        fixed = family(fixed_specs, FIXED_HORIZON_CONTRAST_IDS)
        mechanism = {}
        for identifier in list(PRIMARY_CONTRAST_IDS) + list(FIXED_HORIZON_CONTRAST_IDS):
            mechanism[identifier] = {
                "sample_count": 14042,
                "mean_js_nats": 0.0,
                "mean_entropy_delta_nats": 0.0,
                "mean_top_margin_delta_raw": 0.0,
                "mean_correct_margin_delta_raw": 0.0,
                "top1_retained_fraction": 1.0,
                "transitions": dict((primary.get(identifier) or fixed[identifier])["transitions"]),
                "wrong_overconfidence": {
                    "definition": "final_wrong_pairs_with_entropy_down_and_top_margin_up",
                    "denominator_scope": "wrong_to_wrong_plus_right_to_wrong",
                    "count": 0,
                    "denominator": 0,
                    "fraction": None,
                    "subgroups": {
                        "wrong_to_wrong": {"count": 0, "denominator": 0, "fraction": None},
                        "right_to_wrong": {"count": 0, "denominator": 0, "fraction": None},
                    },
                    "role": "mechanism_diagnostic_not_H1_decision_rule",
                },
            }

        median_interval = {
            "available": True,
            "median": 0.2,
            "interval": [0.1, 0.3],
            "replicates": 10000,
            "seed": 0,
            "pointwise": True,
            "multiplicity_adjustment": "none",
            "role": "secondary_diagnostic",
        }
        paired_interval = {
            "available": True,
            "difference": 0.0,
            "interval": [-0.1, 0.1],
            "paired_index": True,
            "replicates": 10000,
            "seed": 0,
            "pointwise": True,
            "multiplicity_adjustment": "none",
            "role": "secondary_diagnostic",
        }
        independent_interval = {
            "available": True,
            "difference": -0.1,
            "interval": [-0.2, 0.0],
            "stratified_within_frozen_subgroup": True,
            "left_count": 10,
            "right_count": 10,
            "replicates": 10000,
            "seed": 0,
            "pointwise": True,
            "multiplicity_adjustment": "none",
            "role": "secondary_diagnostic",
        }
        nca_primary = {
            "%s_k%d" % (window, k): {
                "denominator": 512,
                "valid_count": 512,
                "valid_fraction": 1.0,
                "overall_interval": copy.deepcopy(median_interval),
            }
            for window in ("11:14", "12:15", "13:16")
            for k in (2, 3, 4)
        }
        nca_paired = {
            "k%d" % k: {
                "denominator": 512,
                "paired_valid_count": 512,
                "paired_valid_fraction": 1.0,
                "difference_interval": copy.deepcopy(paired_interval),
            }
            for k in (2, 3, 4)
        }
        nca_native = {
            "12:15_k%d" % k: {
                "valid_subgroup_counts": {
                    "right_to_right": 492,
                    "wrong_to_right": 10,
                    "right_to_wrong": 0,
                    "wrong_to_wrong": 10,
                },
                "right_to_right_interval": copy.deepcopy(median_interval),
                "wrong_to_right_minus_wrong_to_wrong_interval": copy.deepcopy(independent_interval),
                "resampling": "stratified_within_frozen_subgroup",
            }
            for k in (2, 3, 4)
        }
        nca_decision = classify_nca(nca_primary, nca_paired, nca_native)
        cumulative = {
            "phase1_k2_vs_baseline_12_15": base_contrast(),
            "fixed_step_k4_vs_k2_12_15": base_contrast(),
            "role": "secondary_sign_guard_and_context_only",
        }
        h1 = classify_h1(
            primary,
            d24_12_15_pp=0.0,
            d24_12_15_ci_pp=[0.0, 0.0],
            phase1_k2_vs_baseline_pp=0.0,
        )
        source_hashes = {
            "card": self.card["manifest_sha256"],
            "calibration_identity_manifest": "1" * 64,
            "full_identity_manifest": "2" * 64,
            "calibration_baseline": "3" * 64,
            "calibration_cells": {
                cell_id: "%064x" % (200 + index)
                for index, cell_id in enumerate(required_calibration_cell_ids(self.card))
            },
            "full_final_output_cells": {
                cell_id: "%064x" % (400 + index)
                for index, cell_id in enumerate(required_full_cell_ids(self.card))
            },
            "calibration_labels": "4" * 64,
            "unseal_authorization": "5" * 64,
            "unseal_receipt": "6" * 64,
        }
        trajectory_summary = {}
        for cell in b2_logical_cells(self.card):
            steps = []
            for body_call_t in range(cell["k"]):
                steps.append(
                    {
                        "body_call_t": body_call_t,
                        "residual_norm_median": 1.0,
                        "state_norm_median": 2.0,
                        "relative_activity_median": 0.5,
                        "residual_ratio_to_previous_median": None if body_call_t == 0 else 1.0,
                        "adjacent_residual_cosine_median": None if body_call_t == 0 else 0.5,
                        "adjacent_residual_cosine_valid_count": 0 if body_call_t == 0 else 512,
                    }
                )
            trajectory_summary[cell["cell_id"]] = {
                "cell": dict(cell),
                "sample_count": 512,
                "valid_sample_count": 512,
                "valid_sample_fraction": 1.0,
                "steps": steps,
                "role": "calibration_512_mechanism_evidence_only",
            }
        report = make_hashed_manifest(
            {
                "schema_version": PHASE2_ANALYSIS_SCHEMA_VERSION,
                "card_id": self.card["card_id"],
                "card_manifest_sha256": self.card["manifest_sha256"],
                "analysis_input_manifest_sha256": "7" * 64,
                "analysis_input_file_sha256": "8" * 64,
                "source_artifacts": source_hashes,
                "identity_closure": {
                    "calibration": {
                        "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
                        "identity_manifest_sha256": "9" * 64,
                        "ordered_identity_sha256": "a" * 64,
                        "sample_count": 512,
                        "natural_order": "canonical_manifest_list_order_zero_based",
                    },
                    "full": {
                        "identity_namespace": "loopscope.phase2.full.mmlu14042.v1",
                        "identity_manifest_sha256": "b" * 64,
                        "ordered_identity_sha256": "c" * 64,
                        "sample_count": 14042,
                        "natural_order": "canonical_manifest_list_order_zero_based",
                    },
                },
                "revision": self.card["science"]["revision"],
                "statistics_contract": copy.deepcopy(self.card["statistics"]),
                "primary_contrasts": primary,
                "fixed_horizon_contrasts": fixed,
                "cumulative_context": cumulative,
                "mechanism_diagnostics": mechanism,
                "calibration_trajectory_summary": trajectory_summary,
                "nca": {
                    "primary_cells": nca_primary,
                    "paired_12_15_minus_13_16": nca_paired,
                    "native_fidelity_cells": nca_native,
                    "decision": nca_decision,
                    "role": "independent_secondary_direction_diagnosis",
                },
                "h1": h1,
                "execution_provenance": provenance(),
                "independent_decisions": True,
            }
        )
        validate_analysis_report(report, self.card)
        validate_analysis_report(json.loads(json.dumps(report, sort_keys=True)), self.card)

        forged = copy.deepcopy(report)
        forged["primary_contrasts"]["fs_12_15_k2_k3"]["mcnemar_raw_p"] = 0.0
        reversed_interval = copy.deepcopy(report)
        reversed_interval["primary_contrasts"]["fs_12_15_k2_k3"]["paired_ci_pp"] = [1.0, -1.0]
        unknown = copy.deepcopy(report)
        unknown["unregistered_result"] = True
        for candidate in (forged, reversed_interval, unknown):
            with self.assertRaises((SchemaError, Phase2AnalysisError)):
                validate_analysis_report(self._rehash(candidate), self.card)

    def test_atomic_analysis_writer_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.json"
            atomic_write_new_json(target, {"value": 1})
            self.assertEqual(json.loads(target.read_text()), {"value": 1})
            with self.assertRaises(FileExistsError):
                atomic_write_new_json(target, {"value": 2})


if __name__ == "__main__":
    unittest.main()
