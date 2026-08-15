from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tflt.loopscope import phase7_outcome as outcome


class Phase7OutcomeTests(unittest.TestCase):
    def test_exact_332_card_expands_to_35_unique_cells(self) -> None:
        card = outcome.load_card()
        cells = outcome.expand_cells(card)
        self.assertEqual(len(cells), 35)
        self.assertEqual(
            {key: sum(cell["model_key"] == key for cell in cells) for key in outcome.EXPECTED_MODELS},
            {"qwen25_3b": 13, "llama32_3b": 13, "gemma2_2b": 9},
        )
        self.assertEqual(sum(not cell["loop_enabled"] for cell in cells), 3)
        self.assertEqual(len({cell["cell_id"] for cell in cells}), 35)
        qwen = [cell for cell in cells if cell["model_key"] == "qwen25_3b" and cell["loop_enabled"]]
        self.assertEqual(
            [(cell["window_half_open"], cell["k"], cell["cache_strategy"]) for cell in qwen[:4]],
            [("14:18", 2, "first"), ("14:18", 2, "last"), ("14:18", 3, "first"), ("14:18", 3, "last")],
        )

    def test_debug_panel_covers_three_models_and_all_k_cache_paths(self) -> None:
        cells = outcome.expand_cells(outcome.load_card())
        selected = [cells[index] for index in outcome.debug_cell_indices(cells)]
        self.assertEqual(len(selected), 7)
        self.assertEqual({cell["model_key"] for cell in selected}, set(outcome.EXPECTED_MODELS))
        self.assertEqual(sum(not cell["loop_enabled"] for cell in selected), 3)
        self.assertEqual(
            {(cell["k"], cell["cache_strategy"]) for cell in selected if cell["loop_enabled"]},
            {(2, "first"), (2, "last"), (3, "first"), (3, "last")},
        )

    def test_identity_manifest_and_sanitized_projection_keep_only_allowed_fields(self) -> None:
        expected = [
            {"canonical_identity": "mmlu_alpha:test:0", "subject": "alpha", "task_name": "mmlu_alpha", "test_index": 0},
            {"canonical_identity": "mmlu_beta:test:1", "subject": "beta", "task_name": "mmlu_beta", "test_index": 1},
        ]
        normalized = outcome.validate_identity_manifest(expected)
        result = {
            "samples": {
                "mmlu_alpha": [{"doc_id": 0, "metrics": {"acc,none": 1.0}, "doc": {"question": "not persisted"}}],
                "mmlu_beta": [{"doc_id": "1", "acc": False, "target": 2}],
            }
        }
        rows = outcome.project_sanitized_outcomes(result, normalized)
        self.assertEqual(
            rows,
            [
                {
                    "schema_version": outcome.OUTCOME_SCHEMA,
                    "canonical_identity": "mmlu_alpha:test:0",
                    "subject": "alpha",
                    "task_name": "mmlu_alpha",
                    "test_index": 0,
                    "correctness": True,
                },
                {
                    "schema_version": outcome.OUTCOME_SCHEMA,
                    "canonical_identity": "mmlu_beta:test:1",
                    "subject": "beta",
                    "task_name": "mmlu_beta",
                    "test_index": 1,
                    "correctness": False,
                },
            ],
        )
        self.assertTrue(all(set(row) == {"schema_version", "canonical_identity", "subject", "task_name", "test_index", "correctness"} for row in rows))

    def test_subject_stratified_paired_bootstrap_is_reproducible_and_reports_pp(self) -> None:
        baseline = []
        candidate = []
        for index in range(outcome.EXPECTED_SUBJECTS):
            subject = "subject_%02d" % index
            shared = {
                "canonical_identity": "mmlu_%s:test:0" % subject,
                "subject": subject,
                "task_name": "mmlu_%s" % subject,
                "test_index": 0,
            }
            baseline.append({**shared, "correctness": False})
            candidate.append({**shared, "correctness": True})
        first = outcome.subject_stratified_paired_bootstrap(
            baseline=baseline, candidate=candidate, replicates=20, seed=20260803
        )
        second = outcome.subject_stratified_paired_bootstrap(
            baseline=baseline, candidate=candidate, replicates=20, seed=20260803
        )
        self.assertEqual(first, second)
        self.assertEqual(first["ci_percentage_points"], [100.0, 100.0])

    def test_pool_groups_are_complete_and_bounded(self) -> None:
        cells = outcome.expand_cells(outcome.load_card())[:7]
        groups = outcome._pool_groups(cells, 3)
        self.assertEqual(groups, [[0, 3, 6], [1, 4], [2, 5]])

    def test_static_receipt_reads_the_safe_records_envelope(self) -> None:
        full_cells = outcome.expand_cells(outcome.load_card())
        cells = [dict(full_cells[index]) for index in outcome.debug_cell_indices(full_cells)]
        for index, cell in enumerate(cells):
            cell["index"] = index
        records = [
            {"canonical_identity": "mmlu_abstract_algebra:test:%d" % index, "subject": "abstract_algebra", "task_name": "mmlu_abstract_algebra", "test_index": index}
            for index in range(2)
        ] + [
            {"canonical_identity": "mmlu_anatomy:test:%d" % index, "subject": "anatomy", "task_name": "mmlu_anatomy", "test_index": index}
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "inputs").mkdir()
            (root / "manifest").mkdir()
            identity_path = root / "inputs" / "debug_test_manifest.json"
            identity_path.write_text(json.dumps({"records": records}), encoding="utf-8")
            receipt = outcome._write_static_receipt(
                root,
                {"mode": "debug", "cells": cells, "identity_manifest": str(identity_path)},
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["record_count"], 4)

    def test_prepare_formal_freezes_exact_35_cell_panel_before_forwards(self) -> None:
        records = []
        next_index = {"subject_%02d" % index: 0 for index in range(outcome.EXPECTED_SUBJECTS)}
        for ordinal in range(outcome.EXPECTED_POPULATION):
            subject = "subject_%02d" % (ordinal % outcome.EXPECTED_SUBJECTS)
            test_index = next_index[subject]
            next_index[subject] += 1
            records.append(
                {
                    "canonical_identity": "mmlu_%s:test:%d" % (subject, test_index),
                    "subject": subject,
                    "task_name": "mmlu_%s" % subject,
                    "test_index": test_index,
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "formal"
            with mock.patch.object(outcome, "validate_git", return_value={"commit": "abc", "clean": True}), mock.patch.object(
                outcome, "build_canonical_test_manifest", return_value=records
            ):
                result = outcome.prepare_run(
                    mode="formal", run_root=root, expected_commit="abc", cache_dir="/cache"
                )
            self.assertEqual(result["cell_count"], 35)
            self.assertTrue((root / "manifest" / "static_preoutcome_receipt.json").is_file())
            self.assertEqual(outcome.verify_static(root)["status"], "PASS")

    def test_fresh_analysis_verifier_recomputes_scientific_projection(self) -> None:
        scientific = {
            "schema_version": "loopscope.phase7.gate-e-combined-analysis.v1",
            "panel_cell_count": 35,
            "population": {"records": 14042, "subjects": 57},
            "primary_metric": "standard_accuracy",
            "multiple_comparison_status": "exploratory_nominal_single_cell_results_non_confirmatory",
            "models": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            (root / "manifest").mkdir()
            (root / "manifest" / "run_manifest.json").write_text(
                json.dumps({"schema_version": outcome.RUN_SCHEMA, "gate": "E"}), encoding="utf-8"
            )
            (root / "analysis" / "combined_analysis.json").write_text(
                json.dumps({"scientific": scientific}), encoding="utf-8"
            )
            with mock.patch.object(outcome, "build_scientific_projection", return_value=scientific):
                receipt = outcome.verify_analysis(root)
            self.assertEqual(receipt["status"], "PASS")
            self.assertTrue((root / "analysis" / "fresh_verifier_receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
