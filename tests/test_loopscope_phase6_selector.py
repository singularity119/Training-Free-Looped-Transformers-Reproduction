import copy
import unittest

from tflt.loopscope import phase5_relative_biphasic as v1
from tflt.loopscope.phase6_schema import (
    TRAJECTORY_SCHEMA_VERSION,
    Phase6ContractError,
    validate_selector_freeze,
)
from tflt.loopscope.phase6_selector import (
    KNOWN_OUTCOME_REGISTRY,
    PUBLIC_ROW_FIELDS,
    analyze_selector,
    build_selector_freeze,
    compute_rate_statistics,
    enumerate_candidates,
    freeze_panel,
    project_selector_records,
    select_scientific_state,
    v1_compatibility_view,
)


def _trajectory_records():
    records = []
    for index in range(6):
        h_values = [
            8.0
            - 0.025 * boundary
            + 0.007 * ((boundary + index) % 5)
            + 0.001 * index
            for boundary in range(37)
        ]
        d_values = [
            0.02 * boundary
            + 0.005 * ((2 * boundary + index) % 7)
            + 0.0005 * index
            for boundary in range(37)
        ]
        d_values[-1] = 0.0
        records.append(
            {
                "schema_version": TRAJECTORY_SCHEMA_VERSION,
                "model_repo": "Qwen/Qwen3-4B-Instruct-2507",
                "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                "dataset_repo": "TIGER-Lab/MMLU-Pro",
                "dataset_revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                "split": "test",
                "canonical_identity": "sample-%02d" % index,
                "category": "category-%d" % (index % 2),
                "prompt_sha256": "a" * 64,
                "generated_completion_sha256": "b" * 64,
                "generation_length": 10,
                "replay_length": 15,
                "anchor_token_index": 4,
                "answer_span_start_offset": 20,
                "answer_span_end_offset": 21,
                "answer_match_count": 1,
                "selected_match_ordinal": 0,
                "answer_first_token_index": 5,
                "answer_span_extractor_sha256": "c" * 64,
                "generated_id_text_aligner_sha256": "d" * 64,
                "generation_count": 1,
                "replay_count": 1,
                "replay_incremental_step_count": 5,
                "replay_step_trace_sha256": "3" * 64,
                "replay_argmax_matches_generated": index != 0,
                "loop_insertions": 0,
                "H": h_values,
                "D": d_values,
                "hidden_rms_l2_to_final": [1.0 + index] * 36 + [0.0],
                "hidden_cosine_to_final": [0.5] * 36 + [1.0],
                "hidden_cosine_distance_to_final": [0.5] * 36 + [0.0],
                "adjacent_angular_distance": [0.5] * 36,
                "provenance": {
                    "producer_version": "test",
                    "card_sha256": "e" * 64,
                    "renderer_manifest_sha256": "f" * 64,
                    "tokenizer_manifest_sha256": "0" * 64,
                    "generation_ids_sha256": "1" * 64,
                    "replay_ids_sha256": "2" * 64,
                    "boundary_capture": "raw_B0_through_B36_final_norm_pre_hook",
                    "final_norm_path": "model.norm",
                    "lm_head_path": "lm_head",
                },
            }
        )
    return records


def _decision_row(width, start, score, *, eligible=True, stable=True):
    return {
        "width": width,
        "start": start,
        "NewEligible": eligible,
        "RateStable": stable,
        "S_RATE": score,
    }


def _panel_rows():
    rows = []
    for index, candidate in enumerate(enumerate_candidates(), start=1):
        score = float(index)
        rows.append(
            {
                **candidate,
                "G_H": score,
                "G_K": score,
                "S_RATE": score,
                "selected": False,
            }
        )
    return rows


def _public_panel_rows():
    rows = []
    for index, candidate in enumerate(enumerate_candidates(), start=1):
        score = 100.0 if candidate["window"] == "15:18" else float(index)
        selected = candidate["window"] == "15:18"
        rows.append(
            {
                **candidate,
                "Scorable": True,
                "NetPositive": True,
                "SoftRelativeStable": True,
                "BiphasicStable": True,
                "NewEligible": True,
                "G_H": score,
                "G_K": score,
                "S_RATE": score,
                "RateStable": True,
                "rate_SE_H": 0.1,
                "rate_SE_K": 0.1,
                "rate_c95": 1.0,
                "rate_LCB_H": score - 0.1,
                "rate_LCB_K": score - 0.1,
                "ranking_candidate": True,
                "point_top_tie": False,
                "selected": selected,
                "display_rank": 1 if selected else index + 1,
                "selection_frequency": 1.0 if selected else None,
                "best_q": 1,
                "best_tau": candidate["start"] + 1,
                "tau_pair_frequency": 1.0,
                "v1_failure_reasons": [],
                "v2_failure_reasons": [] if selected else ["NOT_POINT_TOP"],
            }
        )
    return rows


class CandidateEnumerationTests(unittest.TestCase):
    def test_exact_42_candidate_domain(self):
        candidates = enumerate_candidates()
        self.assertEqual(len(candidates), 42)
        self.assertEqual(
            {width: sum(row["width"] == width for row in candidates) for width in (3, 4, 5, 6)},
            {3: 12, 4: 11, 5: 10, 6: 9},
        )
        self.assertEqual(
            {
                width: [row["start"] for row in candidates if row["width"] == width]
                for width in (3, 4, 5, 6)
            },
            {
                3: list(range(11, 23)),
                4: list(range(11, 22)),
                5: list(range(11, 21)),
                6: list(range(11, 20)),
            },
        )
        self.assertEqual(candidates[0]["window"], "11:14")
        self.assertEqual(candidates[-1]["window"], "19:25")


class AdapterAndAnalysisTests(unittest.TestCase):
    def test_projection_makes_hidden_diagnostics_inaccessible(self):
        records = _trajectory_records()
        projected = project_selector_records(records)
        self.assertTrue(
            all(set(row) == {"identity", "category", "H", "D"} for row in projected)
        )
        records[0]["H"][0] = -999.0
        self.assertNotEqual(projected[0]["H"][0], records[0]["H"][0])

    def test_analysis_public_rows_v1_compatibility_and_hidden_independence(self):
        records = _trajectory_records()
        first = analyze_selector(records, replicates=7, seed=20260801, formal=False)
        changed = copy.deepcopy(records)
        for record in changed:
            record["hidden_rms_l2_to_final"] = [2.0] * 36 + [0.0]
            record["hidden_cosine_to_final"] = [0.25] * 36 + [1.0]
            record["hidden_cosine_distance_to_final"] = [0.75] * 36 + [0.0]
            record["adjacent_angular_distance"] = [0.2] * 36
        second = analyze_selector(changed, replicates=7, seed=20260801, formal=False)
        self.assertEqual(first, second)
        self.assertEqual(first["candidate_count"], 42)
        self.assertTrue(all(tuple(row) == PUBLIC_ROW_FIELDS for row in first["rows"]))

        legacy_samples = [
            {
                "identity": row["canonical_identity"],
                "subject": row["category"],
                "H": row["H"],
                "D": row["D"],
            }
            for row in records
        ]
        direct = v1.analyze_model(
            legacy_samples,
            "qwen4",
            replicates=7,
            seed=20260801,
            enforce_population=False,
        )
        self.assertEqual(
            first["v1_compatibility"]["view"],
            v1_compatibility_view(direct["rows"]),
        )

    def test_formal_recipe_cannot_be_changed(self):
        with self.assertRaisesRegex(
            Phase6ContractError, "formal Phase 6 bootstrap contract differs"
        ):
            analyze_selector(
                _trajectory_records(), replicates=7, seed=20260801, formal=True
            )


class ScientificStateTests(unittest.TestCase):
    def test_rate_score_and_rate_stable_statistics(self):
        stable = compute_rate_statistics(
            2.0,
            3.0,
            [1.9, 2.0, 2.1, 2.0, 2.0],
            [2.9, 3.0, 3.1, 3.0, 3.0],
        )
        self.assertAlmostEqual(stable["S_RATE"], 6.0**0.5)
        self.assertTrue(stable["RateStable"])
        unstable = compute_rate_statistics(
            0.01,
            0.01,
            [-1.0, 1.0, -1.0, 1.0, 0.0],
            [-1.0, 1.0, -1.0, 1.0, 0.0],
        )
        self.assertFalse(unstable["RateStable"])

    def test_no_rate_stable_eligible_abstain(self):
        rows = [_decision_row(3, 11, None, eligible=False, stable=False)]
        decision = select_scientific_state(rows, {}, 5)
        self.assertEqual(
            decision["decision"], "ABSTAIN_NO_RATE_STABLE_ELIGIBLE"
        )

    def test_top_tie_abstain(self):
        rows = [
            _decision_row(3, 11, 2.0),
            _decision_row(3, 12, 2.0),
        ]
        rates = {
            (3, 11): {"H": [2.0] * 5, "K": [2.0] * 5},
            (3, 12): {"H": [2.0] * 5, "K": [2.0] * 5},
        }
        decision = select_scientific_state(rows, rates, 5)
        self.assertEqual(decision["decision"], "ABSTAIN_NO_UNIQUE_TOP1")
        self.assertTrue(all(row["point_top_tie"] for row in rows))

    def test_frequency_abstain(self):
        rows = [
            _decision_row(3, 11, 2.0),
            _decision_row(3, 12, 1.0),
        ]
        rates = {
            (3, 11): {"H": [2.0, 0.5, 0.5, 0.5, 0.5], "K": [2.0] * 5},
            (3, 12): {"H": [1.0] * 5, "K": [1.0] * 5},
        }
        decision = select_scientific_state(rows, rates, 5)
        self.assertEqual(decision["decision"], "ABSTAIN_RATE_RANK_UNSTABLE")
        self.assertEqual(decision["selection_frequency"], 0.2)

    def test_selected_window(self):
        rows = [
            _decision_row(3, 11, 2.0),
            _decision_row(3, 12, 1.0),
        ]
        rates = {
            (3, 11): {"H": [2.0] * 5, "K": [2.0] * 5},
            (3, 12): {"H": [1.0] * 5, "K": [1.0] * 5},
        }
        decision = select_scientific_state(rows, rates, 5)
        self.assertEqual(decision["decision"], "SELECTED_WINDOW")
        self.assertEqual(decision["selected_key"], [3, 11])
        self.assertEqual(decision["selection_frequency"], 1.0)


class PanelFreezeTests(unittest.TestCase):
    def test_hidden_diagnostics_cannot_change_high3_or_low3(self):
        rows = _panel_rows()
        baseline = freeze_panel(
            {
                "selector_decision": "ABSTAIN_NO_RATE_STABLE_ELIGIBLE",
                "selected_key": None,
                "rows": rows,
            }
        )
        changed = copy.deepcopy(rows)
        for index, row in enumerate(changed):
            row["hidden_rms_l2_to_final"] = [float(index)] * 37
            row["hidden_cosine_to_final"] = [float(-index)] * 37
            row["adjacent_angular_distance"] = [float(index)] * 36
        observed = freeze_panel(
            {
                "selector_decision": "ABSTAIN_NO_RATE_STABLE_ELIGIBLE",
                "selected_key": None,
                "rows": changed,
            }
        )
        self.assertEqual(observed["high3"], baseline["high3"])
        self.assertEqual(observed["low3"], baseline["low3"])
        self.assertEqual(observed["panel_cells"], baseline["panel_cells"])

    def test_registry_exclusion_disjoint_groups_and_selected_dedup(self):
        rows = _panel_rows()
        selected = next(
            row
            for row in rows
            if row["width"] == 3 and row["window"] == "15:18"
        )
        selected["selected"] = True
        panel = freeze_panel(
            {
                "selector_decision": "SELECTED_WINDOW",
                "selected_key": [3, 15],
                "rows": rows,
            }
        )
        self.assertNotIn("15:18", panel["diagnostic_universe"])
        self.assertNotIn("22:25", panel["diagnostic_universe"])
        candidate_registry = {
            row["window"] for row in rows if row["window"] in KNOWN_OUTCOME_REGISTRY
        }
        self.assertTrue(candidate_registry.isdisjoint(panel["diagnostic_universe"]))
        self.assertEqual(len(panel["high3"]), 3)
        self.assertEqual(len(panel["low3"]), 3)
        self.assertFalse(set(panel["high3"]) & set(panel["low3"]))
        self.assertEqual(
            panel["selected_known_outcome_status"],
            "SELECTED_KNOWN_OUTCOME_WINDOW",
        )
        comparator = next(
            cell for cell in panel["cell_roles"] if cell["cell"] == "15:18"
        )
        self.assertEqual(comparator["roles"], ["fixed_comparator", "selected"])
        self.assertEqual(len(panel["panel_cells"]), 8)
        self.assertEqual(panel["loop_cell_count"], 7)

    def test_panel_is_independent_of_eligibility(self):
        rows = _panel_rows()
        for row in rows:
            row["NewEligible"] = False
            row["RateStable"] = False
        panel = freeze_panel(
            {
                "selector_decision": "ABSTAIN_NO_RATE_STABLE_ELIGIBLE",
                "selected_key": None,
                "rows": rows,
            }
        )
        self.assertEqual(len(panel["high3"]), 3)
        self.assertEqual(len(panel["low3"]), 3)

    def test_underpopulated_diagnostic_universe_blocks(self):
        rows = _panel_rows()[:5]
        with self.assertRaisesRegex(
            Phase6ContractError, "BLOCK_DIAGNOSTIC_PANEL_UNDERPOPULATED"
        ):
            freeze_panel(
                {
                    "selector_decision": "ABSTAIN_NO_RATE_STABLE_ELIGIBLE",
                    "selected_key": None,
                    "rows": rows,
                }
            )

    def test_closed_selector_freeze_passes_parent_schema(self):
        payload = build_selector_freeze(
            {
                "selector_decision": "SELECTED_WINDOW",
                "selected_key": [3, 15],
                "selection_frequency": 1.0,
                "point_top_key": [3, 15],
                "record_count": 12032,
                "category_count": 14,
                "candidate_count": 42,
                "candidate_counts_by_width": {
                    "3": 12,
                    "4": 11,
                    "5": 10,
                    "6": 9,
                },
                "bootstrap": {"replicates": 2000, "seed": 20260801},
                "rows": _public_panel_rows(),
            },
            card_sha256="a" * 64,
            input_manifest_sha256="b" * 64,
        )
        validate_selector_freeze(payload)
        self.assertEqual(payload["selected_window"], "15:18")
        self.assertNotIn("15:18", payload["high3"])
        self.assertNotIn("15:18", payload["low3"])
        self.assertEqual(len(payload["panel_cells"]), 8)


if __name__ == "__main__":
    unittest.main()
