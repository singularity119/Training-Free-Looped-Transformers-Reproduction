import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tflt.loopscope.phase3_analysis import analyze_window_signals
from tflt.loopscope.phase3_p3c import (
    BLIND_WINDOWS,
    HISTORICAL_ALLOWLIST_SCHEMA,
    HISTORICAL_WINDOWS,
    P3CError,
    TEST_METADATA_SCHEMA,
    _validate_descriptive_overlap_closure,
    _expected_pair_id_map,
    _pair_id_from_test_record,
    _validate_allowlist_for_c2,
    _verify_result_sample_content,
    describe_population_overlap,
    exact_paired_comparison,
    make_historical_alignment,
    make_missing12_predictions,
    make_test_metadata_record,
    materialize_c1,
    retrospective_variant_decision,
    safe_test_dataset_fields,
    scan_blind_outcome_existence,
    validate_frozen_analysis_contract,
    validate_test_metadata_records,
    write_new_json,
)
from tflt.loopscope.phase3_schema import canonical_record_key, load_phase3_card
from tflt.loopscope.phase3_verifier import synthetic_signal_records


ROOT = Path(__file__).resolve().parents[1]
ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64


class ForbiddenFieldTrap(dict):
    def __getitem__(self, key):
        if key in {"target", "gold", "answer", "correctness"}:
            raise AssertionError("forbidden field was accessed: %s" % key)
        return super().__getitem__(key)


def metadata_record(task, subject, index, *, doc_hash=ZERO_SHA, content_hash=ONE_SHA):
    return {
        "schema_version": TEST_METADATA_SCHEMA,
        "identity": {
            "task": task,
            "doc_id": str(index),
            "doc_hash": doc_hash,
        },
        "subject": subject,
        "split": "test",
        "sanitized_content_sha256": content_hash,
    }


def decision_analysis(
    *,
    top1=12,
    frequency=0.80,
    eligible=(12,),
    decision="SELECTED",
    ranking=(12, 11, 13),
):
    variants = {}
    for variant in ("CONSENSUS", "FLANK", "SHIFT"):
        variants[variant] = {
            "scopes": {
                "retrospective": {
                    "point_top1_start": top1,
                    "point_top1_window": None if top1 is None else "%d:%d" % (top1, top1 + 3),
                    "selection_frequency": frequency,
                    "window_decision": decision,
                    "point_eligible_starts": list(eligible),
                    "published_ranking": [
                        {"rank": rank, "start": start, "window": "%d:%d" % (start, start + 3)}
                        for rank, start in enumerate(ranking, start=1)
                    ],
                }
            }
        }
    return {"variants": variants}


class Phase3P3CTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_phase3_card(ROOT / "configs/loopscope/phase3_card.json")
        cls.analysis = analyze_window_signals(
            synthetic_signal_records(),
            cls.card,
            replicates=7,
            seed=20260716,
            enforce_population=False,
        )

    def test_safe_projection_never_accesses_forbidden_fields(self):
        row = ForbiddenFieldTrap(
            question="q",
            choices=["a", "b", "c", "d"],
            subject="abstract_algebra",
            target=3,
            gold=3,
            answer="D",
            correctness=True,
        )
        self.assertEqual(
            safe_test_dataset_fields(row, "abstract_algebra"),
            {
                "question": "q",
                "choices": ["a", "b", "c", "d"],
                "subject": "abstract_algebra",
            },
        )

    def test_safe_projection_rejects_wrong_shape_and_subject(self):
        with self.assertRaises(P3CError):
            safe_test_dataset_fields(
                {"question": "q", "choices": ["a"], "subject": "s"}, "s"
            )
        with self.assertRaises(P3CError):
            safe_test_dataset_fields(
                {"question": "q", "choices": ["a", "b", "c", "d"], "subject": "x"},
                "s",
            )

    def test_test_metadata_exact_14042_57_subject_closure(self):
        records = []
        for subject_index in range(57):
            subject = "subject_%02d" % subject_index
            task = "mmlu_%s" % subject
            count = 246 if subject_index < 56 else 266
            for index in range(count):
                records.append(metadata_record(task, subject, index))
        records.sort(key=canonical_record_key)
        normalized = validate_test_metadata_records(
            records,
            self.card,
            expected_ordered_identity_sha256=None,
        )
        self.assertEqual(len(normalized), 14042)
        with self.assertRaisesRegex(P3CError, "14,042"):
            validate_test_metadata_records(
                records[:-1],
                self.card,
                expected_ordered_identity_sha256=None,
            )

    def test_test_metadata_rejects_order_duplicate_and_hash_drift(self):
        records = [
            metadata_record("mmlu_b", "b", 0),
            metadata_record("mmlu_a", "a", 0),
        ]
        with self.assertRaisesRegex(P3CError, "canonical order"):
            validate_test_metadata_records(
                records,
                self.card,
                enforce_frozen_counts=False,
                expected_ordered_identity_sha256=None,
            )
        duplicate = [records[1], copy.deepcopy(records[1])]
        with self.assertRaisesRegex(P3CError, "duplicate"):
            validate_test_metadata_records(
                duplicate,
                self.card,
                enforce_frozen_counts=False,
                expected_ordered_identity_sha256=None,
            )
        with self.assertRaisesRegex(P3CError, "planning freeze"):
            validate_test_metadata_records(
                [records[1]],
                self.card,
                enforce_frozen_counts=False,
                expected_ordered_identity_sha256="f" * 64,
            )

    def test_identity_overlap_fails_and_content_overlap_is_descriptive(self):
        validation = [
            {
                "identity": {"task": "mmlu_s", "doc_id": "v", "doc_hash": ZERO_SHA},
                "split": "validation",
                "sanitized_content_sha256": ZERO_SHA,
            }
        ]
        test = [
            {
                "identity": {"task": "mmlu_t", "doc_id": "t", "doc_hash": ONE_SHA},
                "split": "test",
                "sanitized_content_sha256": ONE_SHA,
            }
        ]
        receipt = describe_population_overlap(
            validation, test, self.card, enforce_frozen_counts=False
        )
        self.assertEqual(receipt["identity_intersection_count"], 0)
        self.assertEqual(receipt["sanitized_content_intersection_count"], 0)
        same_identity = copy.deepcopy(test)
        same_identity[0]["identity"] = copy.deepcopy(validation[0]["identity"])
        with self.assertRaisesRegex(P3CError, "identity intersection"):
            describe_population_overlap(
                validation, same_identity, self.card, enforce_frozen_counts=False
            )
        same_content = copy.deepcopy(test)
        same_content[0]["sanitized_content_sha256"] = ZERO_SHA
        descriptive = describe_population_overlap(
            validation, same_content, self.card, enforce_frozen_counts=False
        )
        self.assertEqual(descriptive["sanitized_content_intersection_count"], 1)
        self.assertEqual(descriptive["sanitized_content_overlap_hashes"], [ZERO_SHA])
        self.assertEqual(
            descriptive["sanitized_content_overlaps"][0]["validation_identities"],
            [validation[0]["identity"]],
        )
        self.assertEqual(
            descriptive["sanitized_content_overlaps"][0]["test_identities"],
            [same_content[0]["identity"]],
        )

    def test_c1_c0_validator_accepts_descriptive_content_overlap(self):
        validation = [
            {
                "identity": {"task": "mmlu_s", "doc_id": "v", "doc_hash": ZERO_SHA},
                "split": "validation",
                "sanitized_content_sha256": ZERO_SHA,
            }
        ]
        test = [
            {
                "identity": {"task": "mmlu_t", "doc_id": "t", "doc_hash": ONE_SHA},
                "split": "test",
                "sanitized_content_sha256": ZERO_SHA,
            }
        ]
        closure = describe_population_overlap(
            validation, test, self.card, enforce_frozen_counts=False
        )
        self.assertEqual(
            _validate_descriptive_overlap_closure(
                closure, expected_validation_count=1, expected_test_count=1
            ),
            1,
        )

    def test_blind_scan_uses_existence_metadata_and_rejects_any_value_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "clean"
            clean.mkdir()
            (clean / "command_args.json").write_text(
                json.dumps({"window": "12:15"}), encoding="utf-8"
            )
            with patch("tflt.loopscope.phase3_p3c.BLIND_SEARCH_ROOT", root.resolve()):
                receipt = scan_blind_outcome_existence(root)
                self.assertFalse(receipt["outcome_values_read"])
                blind_dir = root / "run" / "window-0-3"
                blind_dir.mkdir(parents=True)
                (blind_dir / "results.json").write_text(
                    "this content must never be parsed", encoding="utf-8"
                )
                with self.assertRaisesRegex(P3CError, "blind-12"):
                    scan_blind_outcome_existence(root)

    def test_c1_entry_point_has_no_historical_or_outcome_root_argument(self):
        signature = inspect.signature(materialize_c1)
        self.assertNotIn("phase1_root", signature.parameters)
        self.assertNotIn("outcome_root", signature.parameters)
        source = inspect.getsource(materialize_c1)
        self.assertNotIn("results.json", source)
        self.assertNotIn("PHASE1_ROOT", source)

    def test_frozen_analysis_closes_all_support_rank_and_scopes(self):
        validate_frozen_analysis_contract(self.analysis, self.card)
        changed = copy.deepcopy(self.analysis)
        changed["variants"]["SHIFT"]["scopes"]["deployment"]["published_ranking"][0][
            "rank"
        ] = 2
        with self.assertRaisesRegex(P3CError, "rank closure"):
            validate_frozen_analysis_contract(changed, self.card)
        changed = copy.deepcopy(self.analysis)
        changed["variants"]["FLANK"]["support_starts"] = list(range(5, 21))
        with self.assertRaisesRegex(P3CError, "support"):
            validate_frozen_analysis_contract(changed, self.card)

    def test_write_once_and_manifest_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            write_new_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"value": 2})
        changed = copy.deepcopy(self.analysis)
        changed["outcome_fields_consumed"] = True
        with self.assertRaisesRegex(P3CError, "outcome"):
            validate_frozen_analysis_contract(changed, self.card)

    def test_retrospective_080_boundary_false_positives_and_priority(self):
        with patch("tflt.loopscope.phase3_p3c.validate_frozen_analysis_contract"):
            decision = retrospective_variant_decision(decision_analysis(), self.card)
            self.assertEqual(decision["selected_variant"], "CONSENSUS")
            self.assertEqual(decision["science_label"], "RETROSPECTIVE_TOP1_RECOVERY")

            below = retrospective_variant_decision(
                decision_analysis(frequency=0.799999999999), self.card
            )
            self.assertIsNone(below["selected_variant"])
            self.assertEqual(below["science_label"], "RETROSPECTIVE_SHORTLIST_ONLY")

            harmful = retrospective_variant_decision(
                decision_analysis(eligible=(4, 6, 12)), self.card
            )
            self.assertIsNone(harmful["selected_variant"])
            self.assertEqual(harmful["science_label"], "RETROSPECTIVE_SHORTLIST_ONLY")

    def test_all_retrospective_labels(self):
        with patch("tflt.loopscope.phase3_p3c.validate_frozen_analysis_contract"):
            recovered = retrospective_variant_decision(decision_analysis(), self.card)
            shortlist = retrospective_variant_decision(
                decision_analysis(top1=11, ranking=(11, 12, 13)), self.card
            )
            unsupported = retrospective_variant_decision(
                decision_analysis(top1=11, ranking=(11, 13, 14)), self.card
            )
            inconclusive = retrospective_variant_decision(
                decision_analysis(), self.card, evidence_complete=False
            )
        self.assertEqual(recovered["science_label"], "RETROSPECTIVE_TOP1_RECOVERY")
        self.assertEqual(shortlist["science_label"], "RETROSPECTIVE_SHORTLIST_ONLY")
        self.assertEqual(unsupported["science_label"], "RETROSPECTIVE_NOT_SUPPORTED")
        self.assertEqual(inconclusive["science_label"], "RETROSPECTIVE_INCONCLUSIVE")
        self.assertTrue(shortlist["stop_h3"])
        self.assertTrue(unsupported["stop_h3"])
        self.assertTrue(inconclusive["stop_h3"])

    def test_exact_pairing_requires_same_14042_identity_order(self):
        baseline = {"id-%05d" % index: index % 2 == 0 for index in range(14042)}
        candidate = dict(baseline)
        result = exact_paired_comparison(
            baseline,
            candidate,
            bootstrap_replicates=1,
            bootstrap_seed=20260710,
        )
        self.assertEqual(result["sample_alignment"]["matched_count"], 14042)
        self.assertTrue(result["sample_alignment"]["exact_match"])
        reversed_candidate = dict(reversed(list(candidate.items())))
        with self.assertRaisesRegex(P3CError, "identities/order"):
            exact_paired_comparison(
                baseline,
                reversed_candidate,
                bootstrap_replicates=1,
                bootstrap_seed=20260710,
            )
        missing = dict(list(candidate.items())[:-1])
        with self.assertRaisesRegex(P3CError, "identities/order"):
            exact_paired_comparison(
                baseline,
                missing,
                bootstrap_replicates=1,
                bootstrap_seed=20260710,
            )

    def test_historical_allowlist_rejects_blind_extra_missing_and_reorder(self):
        exact = {
            "schema_version": HISTORICAL_ALLOWLIST_SCHEMA,
            "cells": [{"cell": "baseline"}]
            + [{"cell": window} for window in HISTORICAL_WINDOWS],
            "results_values_read": False,
        }
        _validate_allowlist_for_c2(exact)
        for changed in (
            {**exact, "cells": exact["cells"][:-1]},
            {
                **exact,
                "cells": exact["cells"] + [{"cell": BLIND_WINDOWS[0]}],
            },
            {**exact, "cells": [exact["cells"][0], exact["cells"][2], exact["cells"][1]] + exact["cells"][3:]},
        ):
            with self.assertRaisesRegex(P3CError, r"baseline\+13"):
                _validate_allowlist_for_c2(changed)
        with self.assertRaisesRegex(P3CError, "premature outcome"):
            _validate_allowlist_for_c2({**exact, "results_values_read": True})

    def test_result_sample_content_closes_identity_and_rejects_tamper(self):
        safe = {"question": "q", "choices": ["a", "b", "c", "d"], "subject": "s"}
        record = make_test_metadata_record("mmlu_s", "s", 0, safe)
        expected = _expected_pair_id_map([record])
        payload = {
            "samples": {
                "mmlu_s": [
                    {
                        "doc_id": 0,
                        "doc_hash": record["identity"]["doc_hash"],
                        "doc": safe,
                        "acc,none": 1,
                    }
                ]
            }
        }
        _verify_result_sample_content(payload, expected)
        changed = copy.deepcopy(payload)
        changed["samples"]["mmlu_s"][0]["doc"]["question"] = "tampered"
        with self.assertRaisesRegex(P3CError, "identity hash"):
            _verify_result_sample_content(changed, expected)
        changed = copy.deepcopy(payload)
        changed["samples"]["mmlu_s"].append(copy.deepcopy(changed["samples"]["mmlu_s"][0]))
        with self.assertRaisesRegex(P3CError, "duplicate"):
            _verify_result_sample_content(changed, expected)

    def test_phase1_pair_id_is_canonical_decimal(self):
        record = metadata_record("mmlu_s", "s", 7)
        self.assertEqual(_pair_id_from_test_record(record), "mmlu_s:7")
        changed = copy.deepcopy(record)
        changed["identity"]["doc_id"] = "07"
        with self.assertRaisesRegex(P3CError, "canonical decimal"):
            _pair_id_from_test_record(changed)

    def test_missing12_predictions_freeze_only_selected_variant_support(self):
        starts = [1, 2, 3, 5, 7, 10, 16, 19, 21, 23]
        analysis = {
            "variants": {
                "SHIFT": {
                    "support_starts": list(range(1, 24)),
                    "scopes": {
                        "blind_enrichment": {
                            "published_ranking": [
                                {
                                    "rank": rank,
                                    "start": start,
                                    "window": "%d:%d" % (start, start + 3),
                                }
                                for rank, start in enumerate(starts, start=1)
                            ]
                        }
                    },
                }
            }
        }
        decision = {
            "selected_variant": "SHIFT",
            "science_label": "RETROSPECTIVE_TOP1_RECOVERY",
        }
        predictions = make_missing12_predictions(analysis, decision, self.card)
        self.assertEqual(predictions["blind12_universe"], list(BLIND_WINDOWS))
        self.assertEqual(predictions["frozen_top1"], "1:4")
        self.assertEqual(predictions["frozen_top3"], ["1:4", "2:5", "3:6"])
        self.assertFalse(predictions["blind12_outcome_values_consumed"])
        stopped = make_missing12_predictions(
            analysis,
            {"selected_variant": None, "science_label": "RETROSPECTIVE_NOT_SUPPORTED"},
            self.card,
        )
        self.assertEqual(stopped["prediction_status"], "NO_VARIANT_STOP_H3")
        self.assertEqual(stopped["scorable_ranking"], [])

    def test_historical_alignment_uses_frozen_ranks_and_all_13_outcomes(self):
        summaries = {
            window: {
                "accuracy_fraction": 0.5 + index / 10000.0,
                "delta_accuracy_pp_direct": index / 100.0,
            }
            for index, window in enumerate(HISTORICAL_WINDOWS)
        }
        paired = {
            window: {
                "discordant_count": index,
                "mcnemar": {"p_value": 1.0},
            }
            for index, window in enumerate(HISTORICAL_WINDOWS)
        }
        alignment = make_historical_alignment(self.analysis, summaries, paired)
        self.assertEqual(set(alignment), {"CONSENSUS", "FLANK", "SHIFT"})
        self.assertTrue(all(len(rows) == 13 for rows in alignment.values()))
        flank_22 = next(row for row in alignment["FLANK"] if row["window"] == "22:25")
        self.assertFalse(flank_22["selector_scorable"])
        self.assertIsNone(flank_22["frozen_selector_rank"])
        shift_12 = next(row for row in alignment["SHIFT"] if row["window"] == "12:15")
        self.assertTrue(shift_12["selector_scorable"])
        self.assertIsNotNone(shift_12["frozen_selector_rank"])


if __name__ == "__main__":
    unittest.main()
