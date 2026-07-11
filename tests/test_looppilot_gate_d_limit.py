import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from tflt.looppilot.limit import (
    GATE_D_TASKS,
    assert_gate_d_signal_payload_safe,
    build_decision_id,
    validate_gate_d_config,
    verify_gate_d_records,
)
from tflt.looppilot.schema import write_json_once


ROOT = Path(__file__).resolve().parents[1]


def frozen_config():
    return json.loads(
        (ROOT / "configs/looppilot/gate_d_limit.json").read_text(encoding="utf-8")
    )


def key(task="mmlu_abstract_algebra", index=0):
    return "%s|1|%s|%s|test:%d" % (task, "a" * 64, "b" * 64, index)


def sample(arm, task="mmlu_abstract_algebra", index=0):
    return {
        "schema_version": 1,
        "arm": arm,
        "doc_key": key(task, index),
        "task": task,
        "task_version": "1",
        "renderer_hash": "a" * 64,
        "doc_hash": "b" * 64,
        "occurrence_id": "test:%d" % index,
        "revision_closure_id": "c" * 64,
        "subject": task.removeprefix("mmlu_"),
        "choices": ["a", "b", "c", "d"],
        "gold": 0,
        "prediction": 0,
        "choice_loglikelihoods": [-1.0, -2.0, -3.0, -4.0],
    }


def signal(task="mmlu_abstract_algebra", index=0, choice=0):
    doc_key = key(task, index)
    decision_id = build_decision_id(
        doc_key, "LOOP_K2", "always_loop_engineering_control"
    )
    return {
        "schema_version": 1,
        "doc_key": doc_key,
        "task": task,
        "task_version": "1",
        "renderer_hash": "a" * 64,
        "doc_hash": "b" * 64,
        "occurrence_id": "test:%d" % index,
        "choice_index": choice,
        "decision_id": decision_id,
        "action": "LOOP_K2",
        "decision_reason": "always_loop_engineering_control",
        "controller_input_fields": ["r0_median", "c0", "n0_median"],
        "question_token_count": 10,
        "r0_median": 0.1,
        "r0_p90": 0.2,
        "r0_max": 0.3,
        "c0": 0.5,
        "n0_median": 0.01,
        "n0_p90": 0.02,
        "n0_max": 0.03,
        "q1": 0.8,
        "residual_cosine": 0.9,
        "operator_body_calls": 2,
        "k_used": 2,
        "wrapper_restored": True,
    }


def decision(task="mmlu_abstract_algebra", index=0):
    row = signal(task, index, 0)
    return {
        "schema_version": 1,
        "doc_key": row["doc_key"],
        "task": task,
        "task_version": "1",
        "renderer_hash": "a" * 64,
        "doc_hash": "b" * 64,
        "occurrence_id": "test:%d" % index,
        "decision_id": row["decision_id"],
        "action": row["action"],
        "decision_reason": row["decision_reason"],
        "controller_input_fields": row["controller_input_fields"],
        "r0_median": row["r0_median"],
        "c0": row["c0"],
        "n0_median": row["n0_median"],
        "choice_indices": [0, 1, 2, 3],
    }


class GateDConfigTest(unittest.TestCase):
    def test_checked_in_config_is_exactly_frozen(self):
        config = frozen_config()
        validate_gate_d_config(config)
        self.assertEqual(tuple(config["tasks"]), GATE_D_TASKS)
        self.assertEqual(config["limit"], 5)
        self.assertEqual(config["total_docs"], 20)
        self.assertEqual(config["controller"], "always_loop")

    def test_scientific_or_full_eval_drift_is_rejected(self):
        for field, value in (
            ("limit", None),
            ("limit", 6),
            ("batch_size", "auto"),
            ("tasks", ["mmlu"]),
            ("controller", None),
            ("model_revision", "f" * 40),
        ):
            config = frozen_config()
            config[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    validate_gate_d_config(config)

    def test_signal_and_decision_payloads_reject_leakage_and_nonfinite(self):
        for forbidden in ("gold", "answer", "target", "correctness", "full_logits"):
            row = signal()
            row[forbidden] = 0
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    assert_gate_d_signal_payload_safe(row)
        row = signal()
        row["r0_median"] = math.inf
        with self.assertRaises(ValueError):
            assert_gate_d_signal_payload_safe(row)

    def test_json_artifacts_are_exclusive_create(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            write_json_once(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                write_json_once(path, {"ok": False})


class GateDJoinTest(unittest.TestCase):
    def setUp(self):
        self.baseline = [sample("baseline")]
        self.loop = [sample("always_loop")]
        self.signals = [signal(choice=i) for i in range(4)]
        self.decisions = [decision()]

    def verify(self):
        return verify_gate_d_records(
            self.baseline,
            self.loop,
            self.signals,
            self.decisions,
            expected_doc_count=1,
            expected_signal_count=4,
        )

    def test_closed_join_and_terminal_flags(self):
        report, summary = self.verify()
        self.assertEqual(report["coverage"], 1.0)
        self.assertTrue(report["four_choice_same_decision"])
        self.assertTrue(summary["body_call_closed"])
        self.assertTrue(summary["signals_finite_and_in_range"])
        self.assertFalse(summary["threshold_selected"])
        self.assertFalse(summary["ranking_computed"])
        self.assertFalse(summary["limit_for_scientific_claim"])

    def test_missing_duplicate_or_extra_keys_fail(self):
        variants = [
            ([], self.loop, self.signals, self.decisions),
            (self.baseline * 2, self.loop, self.signals, self.decisions),
            (self.baseline, self.loop, self.signals[:-1], self.decisions),
            (self.baseline, self.loop, self.signals, self.decisions * 2),
        ]
        for rows in variants:
            with self.subTest(lengths=[len(group) for group in rows]):
                with self.assertRaises(ValueError):
                    verify_gate_d_records(*rows, expected_doc_count=1, expected_signal_count=4)

    def test_revision_metadata_and_four_choice_decision_mismatch_fail(self):
        changed_loop = copy.deepcopy(self.loop)
        changed_loop[0]["revision_closure_id"] = "d" * 64
        with self.assertRaises(ValueError):
            verify_gate_d_records(
                self.baseline,
                changed_loop,
                self.signals,
                self.decisions,
                expected_doc_count=1,
                expected_signal_count=4,
            )
        changed_signals = copy.deepcopy(self.signals)
        changed_signals[-1]["decision_id"] = "e" * 64
        with self.assertRaises(ValueError):
            verify_gate_d_records(
                self.baseline,
                self.loop,
                changed_signals,
                self.decisions,
                expected_doc_count=1,
                expected_signal_count=4,
            )

    def test_body_call_restore_and_forbidden_fields_fail(self):
        for field, value in (
            ("operator_body_calls", 1),
            ("k_used", 1),
            ("wrapper_restored", False),
            ("gold", 0),
        ):
            changed = copy.deepcopy(self.signals)
            changed[0][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    verify_gate_d_records(
                        self.baseline,
                        self.loop,
                        changed,
                        self.decisions,
                        expected_doc_count=1,
                        expected_signal_count=4,
                    )


if __name__ == "__main__":
    unittest.main()
