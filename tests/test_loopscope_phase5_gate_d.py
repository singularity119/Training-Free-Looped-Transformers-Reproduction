from __future__ import annotations

import unittest

import numpy as np

from tflt.loopscope.phase5_outcome import CELLS
from tflt.loopscope.phase5_outcome_analysis import (
    TERMINAL_LABEL,
    build_statistical_analysis,
    exact_mcnemar_p,
    exact_spearman,
    linear_percentile,
    subject_stratified_joint_bootstrap,
)


class Phase5GateDContractTests(unittest.TestCase):
    def setUp(self):
        self.identities = [
            {"task": "mmlu_a", "doc_id": str(index), "doc_hash": "%064x" % (index + 1)}
            for index in range(8)
        ]
        self.subjects = ["a"] * 4 + ["b"] * 4
        baseline = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
        self.matrix = np.column_stack([
            baseline,
            np.asarray([0, 1, 1, 1, 0, 1, 0, 1]),
            np.asarray([1, 1, 1, 1, 0, 1, 0, 1]),
            np.asarray([1, 0, 1, 1, 1, 1, 0, 1]),
            np.asarray([0, 1, 1, 1, 1, 1, 0, 1]),
            np.asarray([0, 0, 1, 1, 0, 1, 0, 1]),
            np.asarray([0, 0, 1, 0, 0, 1, 0, 1]),
            np.asarray([0, 0, 1, 1, 0, 1, 0, 0]),
        ])
        ranking_windows = list(BLIND_WINDOWS_FOR_TEST) + [
            "%d:%d" % (start, start + 3) for start in range(4, 29)
            if "%d:%d" % (start, start + 3) not in BLIND_WINDOWS_FOR_TEST
        ]
        self.selector = {
            "published_ranking": [
                {"window": window, "score": float(100 - rank)}
                for rank, window in enumerate(ranking_windows)
            ]
        }
        self.geometry = {
            window: {
                "E": float(10 - index), "K": float(20 - index),
                "hidden_l2_improvement": float(6 - index),
                "hidden_cosine_improvement": float(6 - index),
                "hidden_cosine_distance_improvement": float(6 - index),
            }
            for index, window in enumerate(BLIND_WINDOWS_FOR_TEST)
        }

    def test_bootstrap_stream_is_subject_stratified_joint_and_deterministic(self):
        first = subject_stratified_joint_bootstrap(self.matrix, self.subjects, replicates=31, seed=7)
        second = subject_stratified_joint_bootstrap(self.matrix, self.subjects, replicates=31, seed=7)
        self.assertEqual(first["draw_index_sha256"], second["draw_index_sha256"])
        self.assertTrue(first["same_joint_draw_stream_all_cells_and_contrasts"])
        self.assertEqual(first["standard_error"], "sample_sd_ddof_1")

    def test_analysis_preserves_abstain_and_separates_ranking_from_selection(self):
        analysis = build_statistical_analysis(
            correctness=self.matrix,
            identities=self.identities,
            subjects=self.subjects,
            selector_report=self.selector,
            geometry=self.geometry,
            replicates=31,
            seed=7,
        )
        self.assertEqual([row["cell_id"] for row in analysis["cells"]], [row[0] for row in CELLS])
        self.assertIsNone(analysis["abstain"]["selected_window"])
        self.assertIsNone(analysis["abstain"]["selected_vs_baseline"])
        self.assertEqual(analysis["terminal"]["label"], TERMINAL_LABEL)
        self.assertTrue(analysis["ranking_diagnostics"]["not_selection_success"])
        self.assertEqual(analysis["blind_six_selector_score_vs_gain"]["permutation_count"], 720)

    def test_statistics_edges(self):
        self.assertEqual(linear_percentile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(0, 6), 0.03125)
        result = exact_spearman([6, 5, 4, 3, 2, 1], [6, 5, 4, 3, 2, 1])
        self.assertEqual(result["rho"], 1.0)
        self.assertEqual(result["permutation_count"], 720)

    def test_non_binary_or_wrong_shape_fails_closed(self):
        bad = self.matrix.copy()
        bad[0, 0] = 2
        with self.assertRaisesRegex(Exception, "not binary"):
            subject_stratified_joint_bootstrap(bad, self.subjects, replicates=3, seed=0)
        with self.assertRaisesRegex(Exception, "shape"):
            subject_stratified_joint_bootstrap(self.matrix[:, :7], self.subjects, replicates=3, seed=0)


BLIND_WINDOWS_FOR_TEST = ("4:7", "5:8", "13:16", "8:11", "9:12", "6:9")


if __name__ == "__main__":
    unittest.main()
