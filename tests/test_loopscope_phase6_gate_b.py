import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/loopscope/run_qwen4_phase6_gate_b.py"
SPEC = importlib.util.spec_from_file_location("phase6_gate_b", SCRIPT)
gate_b = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate_b)


def _raw(question_id, category, text):
    return {
        "question_id": question_id,
        "category": category,
        "src": "unit",
        "question": text,
        "options": ["one", "two", "three"],
    }


class GateBCanonicalIdentityTests(unittest.TestCase):
    def test_safe_rows_sort_by_numeric_question_id_then_content_hash(self):
        rows = gate_b._safe_rows(
            [
                _raw("10", "biology", "later"),
                _raw("2", "math", "same-id-b"),
                _raw("2", "physics", "same-id-a"),
            ]
        )
        self.assertEqual([int(row["question_id"]) for row in rows], [2, 2, 10])
        self.assertEqual(
            rows[:2],
            sorted(rows[:2], key=lambda row: row["safe_content_sha256"]),
        )
        self.assertTrue(all("answer" not in row for row in rows))
        self.assertTrue(all("cot_content" not in row for row in rows))

    def test_forbidden_payloads_fail_closed(self):
        payload = {"safe": {"rows": [{"canonical_identity": "x"}]}}
        gate_b._scan_forbidden_keys(payload)
        for key in ("answer", "cot_content", "gold", "accuracy", "logits"):
            contaminated = copy.deepcopy(payload)
            contaminated["safe"][key] = "forbidden"
            with self.assertRaisesRegex(gate_b.GateBError, "forbidden persisted key"):
                gate_b._scan_forbidden_keys(contaminated)

    def test_gate_c_plan_is_import_only_and_preserves_two_pass_contract(self):
        card = gate_b.load_json(gate_b.CARD_PATH)
        plan = gate_b.gate_c_runtime_plan(card)
        self.assertEqual(plan["status"], "GATE_C_RUNTIME_PLAN_IMPORTABLE")
        self.assertFalse(plan["gate_c_executed"])
        self.assertFalse(plan["model_or_data_loaded"])
        self.assertFalse(plan["model_forward_executed"])
        self.assertFalse(plan["generation_or_replay_executed"])
        self.assertFalse(plan["pass_2"]["use_cache"])
        self.assertTrue(plan["pass_2"]["output_hidden_states"])
        self.assertEqual(plan["pass_2"]["loop_insertions"], 0)
        self.assertEqual(plan["decoder_layers"], 36)
        self.assertIn("pre_hook", plan["boundary_capture"])

    def test_parser_exposes_only_gate_b_modes_and_import_plan(self):
        parser = gate_b.build_parser()
        for command in ("acquire", "verify"):
            parsed = parser.parse_args(
                [command, "--run-root", "/tmp/write-once", "--expected-commit", "a" * 40]
            )
            self.assertEqual(parsed.command, command)
        self.assertEqual(parser.parse_args(["dry-plan"]).command, "dry-plan")

    def test_frozen_registry_and_provenance_paths(self):
        self.assertEqual(
            gate_b.KNOWN_REGISTRY,
            ("no-loop", "15:18", "6:9", "10:13", "25:28", "4:7", "5:8", "22:25"),
        )
        self.assertIn("phase4-p4b-20260716T204154Z", str(gate_b.PHASE4_PANEL_PATH))
        self.assertEqual(
            gate_b.PHASE4_PANEL_SHA256,
            "02148127ef7b634106b1ee1044f591012d3543563e94344583480a7c3061eb79",
        )
        self.assertEqual(
            gate_b.PHASE4_SELECTOR_SHA256,
            "a4cc48a9b82b3d89b962cdefb55b6e4133c08a0aff3431f0de72c3750fc062c8",
        )


if __name__ == "__main__":
    unittest.main()
