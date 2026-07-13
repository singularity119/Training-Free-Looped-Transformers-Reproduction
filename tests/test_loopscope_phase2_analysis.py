import math
import json
import unittest
from pathlib import Path
from unittest import mock

from tflt.loopscope.phase2_analysis import (
    Phase2AnalysisError,
    analyze_phase2_evidence,
    apply_primary_holm,
    bootstrap_paired_median_difference,
    bootstrap_stratified_median_difference,
    choice_output,
    compare_choice_outputs,
    classify_h1,
    classify_nca,
    collapse_nca_sample_cell,
    exact_mcnemar_p,
    holm_step_down,
    jensen_shannon,
    paired_accuracy_contrast,
    paired_nca_cells,
    summarize_nca_cell,
    transition_counts,
    wrong_overconfidence,
    required_calibration_cell_ids,
    required_full_cell_ids,
)
from tflt.loopscope.phase2_schema import (
    CALIBRATION_IDENTITY_NAMESPACE,
    DIRECT_PROBE_SCORE_SOURCE,
    FULL_IDENTITY_NAMESPACE,
    FULL_LM_EVAL_SCORE_SOURCE,
    PRIMARY_CONTRAST_IDS,
    b2_logical_cells,
    protocol_cell_id,
)


ROOT = Path(__file__).resolve().parents[1]


def identity(index):
    return {"task": "mmlu_x", "doc_id": str(index), "doc_hash": ("%064x" % (index + 1))}


def record(index, correct):
    return {"sample_identity": identity(index), "correctness": bool(correct)}


def primary_family(default_delta=0.0, default_p=1.0):
    return {
        name: {
            "delta_acc_pp": default_delta,
            "holm_adjusted_p": default_p,
            "wrong_to_right": 0,
            "right_to_wrong": 0,
        }
        for name in PRIMARY_CONTRAST_IDS
    }


def available_interval(low=0.1, high=0.2):
    return {"available": True, "interval": [low, high]}


class Phase2AnalysisTests(unittest.TestCase):
    def test_choice_semantics(self):
        output = choice_output([0, 0, 0, 0], identity=identity(1), gold_index=0, evaluator_acc=True)
        self.assertEqual(output["top1_label"], "A")
        self.assertAlmostEqual(output["entropy_nats"], math.log(4.0), places=6)
        self.assertEqual(output["top_margin_raw"], 0.0)
        self.assertEqual(output["correct_margin_raw"], 0.0)
        self.assertAlmostEqual(jensen_shannon(output["choice_probabilities"], output["choice_probabilities"]), 0.0)
        stable = choice_output([1000, 0, -1000, -2000], identity=identity(2))
        self.assertEqual(stable["top1_label"], "A")
        comparison = compare_choice_outputs(output, output)
        self.assertEqual(comparison["js_nats"], 0.0)
        self.assertTrue(comparison["top1_retained"])
        with self.assertRaises(Phase2AnalysisError):
            choice_output([1, 0, 0, 0], identity=identity(3), gold_index=0, evaluator_acc=False)

    def test_transitions_and_mcnemar_edges(self):
        counts = transition_counts([True, False, True, False], [True, True, False, False])
        self.assertEqual(counts, {"right_to_right": 1, "wrong_to_right": 1, "right_to_wrong": 1, "wrong_to_wrong": 1})
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p(0, 1), 1.0)
        self.assertEqual(exact_mcnemar_p(0, 5), 0.0625)
        self.assertEqual(exact_mcnemar_p(0, 6), 0.03125)
        with self.assertRaises(Phase2AnalysisError):
            exact_mcnemar_p(True, 1)

    def test_holm_ties_and_family_closure(self):
        raw = {name: 0.01 for name in PRIMARY_CONTRAST_IDS}
        result = holm_step_down(raw)
        self.assertEqual([result[name]["holm_rank"] for name in PRIMARY_CONTRAST_IDS], list(range(1, 7)))
        adjusted = [result[name]["adjusted_p"] for name in PRIMARY_CONTRAST_IDS]
        self.assertEqual(adjusted, sorted(adjusted))
        with self.assertRaises(Phase2AnalysisError):
            holm_step_down(dict(list(raw.items())[:-1]))
        family = {name: {"mcnemar_raw_p": raw[name]} for name in PRIMARY_CONTRAST_IDS}
        closed = apply_primary_holm(family)
        self.assertEqual(set(closed), set(PRIMARY_CONTRAST_IDS))
        self.assertTrue(all("holm_adjusted_p" in cell for cell in closed.values()))

    def test_paired_identity_and_bootstrap(self):
        left = [record(i, value) for i, value in enumerate([False, True, False, True])]
        right = [record(i, value) for i, value in enumerate([True, True, False, False])]
        result = paired_accuracy_contrast(left, right, bootstrap_replicates=50, bootstrap_seed=9)
        self.assertEqual(result["delta_acc_pp"], 0.0)
        with self.assertRaises(Exception):
            paired_accuracy_contrast(left, list(reversed(right)), bootstrap_replicates=10)
        paired = bootstrap_paired_median_difference([0, 0, 100], [0, 50, 50], replicates=50, seed=0)
        self.assertTrue(paired["paired_index"])
        self.assertEqual(paired["difference"], -50)
        stratified = bootstrap_stratified_median_difference(
            {"a": [1, 2], "b": [3, 4]}, {"a": [0, 1], "b": [2, 3]}, replicates=20, seed=0
        )
        self.assertTrue(stratified["stratified_resampling"])

    def test_nca_all_step_validity(self):
        good = collapse_nca_sample_cell(
            [{"body_call_t": 1, "valid": True, "value": 0.2}, {"body_call_t": 2, "valid": True, "value": 0.4}],
            3,
        )
        self.assertTrue(good["valid"])
        self.assertAlmostEqual(good["value"], 0.3)
        bad = collapse_nca_sample_cell(
            [{"body_call_t": 1, "valid": True, "value": 0.2}, {"body_call_t": 2, "valid": False, "value": None}],
            3,
        )
        self.assertFalse(bad["valid"])
        self.assertFalse(collapse_nca_sample_cell([], 1)["valid"])

    def test_nca_frozen_denominator_and_paired_indices(self):
        rows = [
            {
                "sample_identity": identity(i),
                "repeated_steps": [{"body_call_t": 1, "valid": True, "value": 0.2}],
            }
            for i in range(512)
        ]
        summary = summarize_nca_cell(rows, k=2, bootstrap_replicates=20)
        self.assertEqual(summary["denominator"], 512)
        self.assertEqual(summary["valid_fraction"], 1.0)
        paired = paired_nca_cells(rows, rows, k=2, bootstrap_replicates=20)
        self.assertEqual(paired["paired_valid_fraction"], 1.0)
        self.assertTrue(paired["difference_interval"]["paired_index"])
        with self.assertRaises(Phase2AnalysisError):
            summarize_nca_cell(rows[:-1], k=2, bootstrap_replicates=2)

    def test_h1_labels_and_boundaries(self):
        family = primary_family()
        family["fs_12_15_k2_k3"].update(delta_acc_pp=0.3, holm_adjusted_p=0.01, wrong_to_right=5)
        self.assertEqual(classify_h1(family, d24_12_15_pp=0, d24_12_15_ci_pp=[-0.1, 0.1], phase1_k2_vs_baseline_pp=0.2)["h1_outcome"], "REFINEMENT_SUPPORTED")
        family["fs_12_15_k2_k3"].update(delta_acc_pp=-0.3, holm_adjusted_p=0.01, wrong_to_right=1, right_to_wrong=5)
        self.assertEqual(classify_h1(family, d24_12_15_pp=1, d24_12_15_ci_pp=[0, 2], phase1_k2_vs_baseline_pp=0.2)["h1_outcome"], "PERTURBATION")
        family = primary_family()
        self.assertEqual(classify_h1(family, d24_12_15_pp=-1, d24_12_15_ci_pp=[-2, -0.01], phase1_k2_vs_baseline_pp=0.2)["h1_outcome"], "TRANSIENT_ONLY")
        family["fs_12_15_k2_k3"].update(delta_acc_pp=0.1, holm_adjusted_p=0.05)
        self.assertEqual(classify_h1(family, d24_12_15_pp=0, d24_12_15_ci_pp=[-1, 1], phase1_k2_vs_baseline_pp=0.2)["h1_outcome"], "SUGGESTIVE")
        family = primary_family()
        self.assertEqual(classify_h1(family, d24_12_15_pp=-1, d24_12_15_ci_pp=[-1, 0], phase1_k2_vs_baseline_pp=0.2)["h1_outcome"], "INCONCLUSIVE")

    def test_nca_priority_and_point95_boundary(self):
        primary = {}
        for window in ("11:14", "12:15", "13:16"):
            for k in (2, 3, 4):
                primary["%s_k%d" % (window, k)] = {"valid_fraction": 0.95, "overall_interval": available_interval()}
        paired = {"k%d" % k: {"paired_valid_fraction": 0.95, "difference_interval": available_interval()} for k in (2, 3, 4)}
        result = classify_nca(primary, paired, {})
        self.assertEqual(result["nca_diagnosis"], "NCA_DIRECTION_SUPPORTED")
        self.assertEqual(result["native_fidelity_check"], "NOT_EVALUABLE")
        native = {
            "12:15_k%d" % k: {
                "valid_subgroup_counts": {"right_to_right": 10, "wrong_to_right": 10, "wrong_to_wrong": 10},
                "right_to_right_interval": available_interval(0.01, 0.2),
                "wrong_to_right_minus_wrong_to_wrong_interval": available_interval(-0.2, 0.0),
            }
            for k in (2, 3, 4)
        }
        native_result = classify_nca(primary, paired, native)
        self.assertEqual(native_result["nca_diagnosis"], "NCA_NATIVE_FIDELITY_ONLY")
        self.assertEqual(native_result["native_fidelity_check"], "SUPPORTED")
        primary["11:14_k2"]["valid_fraction"] = 0.949
        self.assertEqual(classify_nca(primary, paired, {})["nca_diagnosis"], "NCA_INCONCLUSIVE")

    def test_wrong_overconfidence_denominator(self):
        before = [
            {"correctness": False, "entropy_nats": 1.0, "top_margin_raw": 0.1},
            {"correctness": True, "entropy_nats": 1.0, "top_margin_raw": 0.1},
            {"correctness": False, "entropy_nats": 1.0, "top_margin_raw": 0.1},
        ]
        after = [
            {"correctness": False, "entropy_nats": 0.5, "top_margin_raw": 0.2},
            {"correctness": False, "entropy_nats": 0.5, "top_margin_raw": 0.2},
            {"correctness": True, "entropy_nats": 0.5, "top_margin_raw": 0.2},
        ]
        result = wrong_overconfidence(before, after)
        self.assertEqual(result["denominator"], 2)
        self.assertEqual(result["count"], 2)

    def test_analysis_rebuilds_contrast_and_nca_from_sample_sidecars(self):
        card = json.loads(
            (ROOT / "configs/loopscope/qwen17_mmlu_phase2_h1_v2.json").read_text()
        )
        full_identities = [identity(index) for index in range(4)]

        def full_rows(cell_id):
            rows = []
            for index, sample_identity in enumerate(full_identities):
                scores = [4.0, 3.0, 2.0, 1.0]
                if index == 0 and cell_id == protocol_cell_id("shared_k2_anchor", "12:15", 2, 1.0):
                    scores = [3.0, 4.0, 2.0, 1.0]
                rows.append(
                    choice_output(
                        scores,
                        identity=sample_identity,
                        score_source=FULL_LM_EVAL_SCORE_SOURCE,
                        gold_index=0,
                        evaluator_acc=(scores[0] == max(scores)),
                    )
                )
            return rows

        full_cells = {
            cell_id: {"samples": full_rows(cell_id), "manifest_sha256": "%064x" % (100 + offset)}
            for offset, cell_id in enumerate(required_full_cell_ids(card))
        }
        calibration_identities = [identity(index + 1000) for index in range(512)]
        calibration_cells = {}
        for offset, cell in enumerate(b2_logical_cells(card)):
            samples = []
            for sample_identity in calibration_identities:
                final_output = choice_output(
                    [4.0, 3.0, 2.0, 1.0],
                    identity=sample_identity,
                    score_source=DIRECT_PROBE_SCORE_SOURCE,
                )
                steps = [
                    {
                        "body_call_t": 0,
                        "residual_norm": 1.0,
                        "state_norm": 2.0,
                        "relative_activity": 0.5,
                        "residual_ratio_to_previous": None,
                        "adjacent_residual_cosine": None,
                        "adjacent_residual_cosine_valid": None,
                        "nca": None,
                    }
                ]
                steps.extend(
                    {
                        "body_call_t": step,
                        "residual_norm": 1.0,
                        "state_norm": 2.0,
                        "relative_activity": 0.5,
                        "residual_ratio_to_previous": 1.0,
                        "adjacent_residual_cosine": 0.5,
                        "adjacent_residual_cosine_valid": True,
                        "nca": {"body_call_t": step, "valid": True, "value": 0.2},
                    }
                    for step in range(1, cell["k"])
                )
                samples.append(
                    {
                        "sample_identity": sample_identity,
                        "steps": steps,
                        "final_output": final_output,
                        "valid": True,
                    }
                )
            calibration_cells[cell["cell_id"]] = {
                "samples": samples,
                "manifest_sha256": "%064x" % (300 + offset),
            }
        baseline_samples = [
            {
                "sample_identity": sample_identity,
                "final_output": choice_output(
                    [4.0, 3.0, 2.0, 1.0],
                    identity=sample_identity,
                    score_source=DIRECT_PROBE_SCORE_SOURCE,
                ),
            }
            for sample_identity in calibration_identities
        ]
        evidence = {
            "request": {
                "manifest_sha256": "1" * 64,
                "analysis_provenance": {
                    "authorization_manifest_sha256": "2" * 64,
                    "attempt_manifest_sha256": "3" * 64,
                    "receipt_manifest_sha256": "4" * 64,
                    "executor_thread_id": "executor",
                    "command_sha256": "5" * 64,
                    "environment_sha256": "6" * 64,
                },
            },
            "request_file_sha256": "7" * 64,
            "card": card,
            "calibration_identity": {
                "identity_namespace": CALIBRATION_IDENTITY_NAMESPACE,
                "manifest_sha256": "8" * 64,
                "ordered_identity_sha256": "9" * 64,
                "sample_count": 512,
                "natural_order": "canonical_manifest_list_order_zero_based",
                "ordered_sample_identity": calibration_identities,
            },
            "full_identity": {
                "identity_namespace": FULL_IDENTITY_NAMESPACE,
                "manifest_sha256": "a" * 64,
                "ordered_identity_sha256": "b" * 64,
                "sample_count": 14042,
                "natural_order": "canonical_manifest_list_order_zero_based",
            },
            "calibration_baseline": {
                "samples": baseline_samples,
                "manifest_sha256": "c" * 64,
            },
            "calibration_cells": calibration_cells,
            "full_cells": full_cells,
            "labels": {
                "samples": [
                    {"sample_identity": sample_identity, "gold_index": 0}
                    for sample_identity in calibration_identities
                ],
                "manifest_sha256": "d" * 64,
            },
            "authorization": {"manifest_sha256": "e" * 64},
            "unseal_receipt": {"manifest_sha256": "f" * 64},
        }
        interval = {
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
        subgroup_interval = {
            "available": True,
            "difference": 0.0,
            "interval": [-0.1, 0.1],
            "stratified_within_frozen_subgroup": True,
            "left_count": 10,
            "right_count": 10,
            "replicates": 10000,
            "seed": 0,
            "pointwise": True,
            "multiplicity_adjustment": "none",
            "role": "secondary_diagnostic",
        }
        with mock.patch(
            "tflt.loopscope.phase2_analysis.bootstrap_median", return_value=interval
        ), mock.patch(
            "tflt.loopscope.phase2_analysis.bootstrap_paired_median_difference",
            return_value=paired_interval,
        ), mock.patch(
            "tflt.loopscope.phase2_analysis.bootstrap_independent_median_difference",
            return_value=subgroup_interval,
        ), mock.patch(
            "tflt.loopscope.phase2_analysis.validate_analysis_report"
        ):
            report = analyze_phase2_evidence(evidence)
        contrast = report["primary_contrasts"]["fs_12_15_k2_k3"]
        self.assertEqual(contrast["wrong_to_right"], 1)
        self.assertEqual(contrast["right_to_wrong"], 0)
        self.assertEqual(contrast["delta_acc_pp"], 25.0)
        self.assertEqual(contrast["mcnemar_raw_p"], 1.0)
        self.assertNotIn("primary_contrasts", evidence["request"])


if __name__ == "__main__":
    unittest.main()
