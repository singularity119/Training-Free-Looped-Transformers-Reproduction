from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tflt.loopscope.phase4_outcome import CELLS
from tflt.loopscope.phase4_outcome_analysis import (
    P4DError,
    build_statistical_analysis,
    exact_mcnemar_p,
    exact_spearman_permutation,
    joint_category_stratified_bootstrap,
    linear_percentile,
    load_correctness_cell,
    scientific_phase_label,
    validate_evaluator_order_hashes,
    validate_panel_manifest,
)


class Phase4OutcomeAnalysisTests(unittest.TestCase):
    def fixture(self):
        identities = ["id-%02d" % index for index in range(12)]
        categories = ["a"] * 4 + ["b"] * 8
        baseline = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        rows = {
            CELLS[0][0]: baseline,
            CELLS[1][0]: [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            CELLS[2][0]: [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            CELLS[3][0]: [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
            CELLS[4][0]: [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
            CELLS[5][0]: [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            CELLS[6][0]: [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            CELLS[7][0]: [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        }
        windows = [cell[2] for cell in CELLS[1:]]
        scores = {window: float(20 - index) for index, window in enumerate(windows)}
        ranks = {window: index + 1 for index, window in enumerate(windows)}
        return identities, categories, rows, scores, ranks

    def build(self):
        identities, categories, rows, scores, ranks = self.fixture()
        return build_statistical_analysis(
            identities=identities,
            categories=categories,
            correctness_by_cell=rows,
            selector_scores=scores,
            selector_ranks=ranks,
            expected_count=12,
            replicates=40,
            seed=20260718,
        )

    def test_exact_membership_count_duplicate_and_order_fail_closed(self):
        identities, categories, rows, scores, ranks = self.fixture()
        with self.assertRaises(P4DError):
            build_statistical_analysis(
                identities=identities[:-1], categories=categories[:-1],
                correctness_by_cell=rows, selector_scores=scores, selector_ranks=ranks,
                expected_count=12, replicates=4,
            )
        duplicate = list(identities)
        duplicate[-1] = duplicate[0]
        with self.assertRaises(P4DError):
            build_statistical_analysis(
                identities=duplicate, categories=categories,
                correctness_by_cell=rows, selector_scores=scores, selector_ranks=ranks,
                expected_count=12, replicates=4,
            )
        items = list(rows.items())
        reordered = dict([items[0], items[2], items[1], *items[3:]])
        with self.assertRaises(P4DError):
            build_statistical_analysis(
                identities=identities, categories=categories,
                correctness_by_cell=reordered, selector_scores=scores, selector_ranks=ranks,
                expected_count=12, replicates=4,
            )

    def test_result_identity_order_is_restored_and_cross_cell_reorder_fails_closed(self):
        docs = [
            {
                "question_id": index,
                "category": "a",
                "src": "src",
                "question": "q%d" % index,
                "options": ["x", "y"],
            }
            for index in range(2)
        ]
        from tflt.loopscope.phase4_schema import canonical_identity

        identities = [
            canonical_identity(
                {
                    "question_id": doc["question_id"],
                    "category": doc["category"],
                    "src": doc["src"],
                    "question": doc["question"],
                    "ordered_options": doc["options"],
                }
            )
            for doc in docs
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(
                json.dumps(
                    {
                        "samples": {
                            "mmlu_pro": [
                                {"doc": docs[1], "exact_match": 1},
                                {"doc": docs[0], "exact_match": 0},
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            values, order_hash = load_correctness_cell(
                path,
                expected_identities=identities,
                expected_categories=["a", "a"],
            )
            self.assertEqual(values, [False, True])
            self.assertEqual(validate_evaluator_order_hashes([order_hash] * 8), order_hash)
            with self.assertRaises(P4DError):
                validate_evaluator_order_hashes([order_hash] * 7 + ["different"])

    def test_joint_category_stratified_same_indices_and_seed_deterministic(self):
        # Category-constant rows prove every replicate preserves the exact 4/8
        # category allocation. Identical cells prove joint index reuse.
        matrix = np.asarray([[1] * 8] * 4 + [[0] * 8] * 8, dtype=np.int8)
        categories = ["a"] * 4 + ["b"] * 8
        first = joint_category_stratified_bootstrap(
            matrix, categories, replicates=20, seed=7, return_replicates=True
        )
        second = joint_category_stratified_bootstrap(
            matrix, categories, replicates=20, seed=7, return_replicates=True
        )
        self.assertEqual(first["index_stream_sha256"], second["index_stream_sha256"])
        self.assertEqual(first["_replicate_cell_accuracy"], second["_replicate_cell_accuracy"])
        self.assertTrue(all(row == [1.0 / 3.0] * 8 for row in first["_replicate_cell_accuracy"]))
        self.assertTrue(first["same_sampled_indices_all_eight_cells"])
        self.assertTrue(first["category_counts_preserved_each_replicate"])

    def test_gain_enrichment_formulas_and_percentile_interpolation(self):
        analysis = self.build()
        by_role = {row["role"]: [] for row in analysis["cells"]}
        for row in analysis["cells"]:
            by_role[row["role"]].append(row["gain_fraction"])
        high = sum(by_role["blind_high"]) / 3.0
        low = sum(by_role["blind_low"]) / 3.0
        self.assertAlmostEqual(analysis["groups"]["G_enrich"]["point_fraction"], high - low)
        self.assertEqual(linear_percentile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(linear_percentile(list(range(41)), 0.025), 1.0)
        self.assertEqual(linear_percentile(list(range(41)), 0.975), 39.0)

    def test_exact_720_permutation_rho(self):
        report = exact_spearman_permutation([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
        self.assertEqual(report["rho"], 1.0)
        self.assertEqual(report["permutation_count"], 720)
        self.assertEqual(report["extreme_count_abs_rho_ge_observed"], 2)
        self.assertAlmostEqual(report["p_value"], 2.0 / 720.0)

    def test_mcnemar_counts_and_exact_p(self):
        analysis = self.build()
        fixed = analysis["cells"][1]
        self.assertEqual(fixed["transitions_vs_baseline"]["wrong_to_correct"], 1)
        self.assertEqual(fixed["transitions_vs_baseline"]["correct_to_wrong"], 0)
        self.assertEqual(fixed["exact_two_sided_mcnemar_p"], 1.0)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(0, 3), 0.25)

    def test_abstain_na_and_phase_label_mapping(self):
        analysis = self.build()
        self.assertIsNone(analysis["abstain"]["selected_window"])
        for key in (
            "selected_vs_baseline", "selected_vs_15_18",
            "selected_competitiveness_margin_0_30pp", "selected_panel_regret",
        ):
            self.assertEqual(analysis["abstain"][key]["status"], "NA_ABSTAIN")
        self.assertEqual(
            scientific_phase_label("ENRICHMENT_SUPPORTED"),
            "PV_EK_TRS_PROSPECTIVE_RANKING_SIGNAL_ONLY",
        )
        self.assertEqual(
            scientific_phase_label("ENRICHMENT_REFUTED"), "PV_EK_TRS_NOT_SUPPORTED"
        )
        self.assertEqual(
            scientific_phase_label("ENRICHMENT_INCONCLUSIVE"), "PV_EK_TRS_INCONCLUSIVE"
        )
        self.assertNotEqual(
            analysis["labels"]["scientific_phase_label"],
            "PV_EK_TRS_PROSPECTIVE_SELECTION_SUPPORTED",
        )

    def test_panel_manifest_wrapper_is_required(self):
        panel = {
            "baseline": "no-loop",
            "fixed_comparator": "15:18",
            "blind_high3": ["6:9", "10:13", "25:28"],
            "blind_low3": ["4:7", "5:8", "22:25"],
            "selected_window": None,
            "abstain": True,
            "variable_width_outcomes_authorized": False,
        }
        from tflt.loopscope.phase4_outcome_analysis import PANEL_MANIFEST_SHA256

        self.assertEqual(
            validate_panel_manifest(
                {"manifest_sha256": PANEL_MANIFEST_SHA256, "panel": panel}
            ),
            panel,
        )
        with self.assertRaises(P4DError):
            validate_panel_manifest({"manifest_sha256": PANEL_MANIFEST_SHA256, **panel})

    def test_forbidden_raw_generated_fields_are_not_persisted(self):
        analysis = self.build()
        forbidden = {
            "question", "gold", "answer", "response", "generated", "cot_content",
            "resps", "filtered_resps", "correctness",
        }

        def keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)

        self.assertFalse(forbidden.intersection(keys(analysis)))
        self.assertTrue(all(value is False for value in analysis["persistence_safety"].values()))


if __name__ == "__main__":
    unittest.main()
