import unittest
from pathlib import Path

from tflt.loopscope import phase6_mmlu5_gate_l as gate_l


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/phase6_mmlu5_prefix_card.json"
TRAJECTORY_SCHEMA_PATH = ROOT / "configs/loopscope/phase6_mmlu5_trajectory_schema.json"


class GateLMMLU5Tests(unittest.TestCase):
    def test_dry_plan_is_gate_l_only_and_gate_m_is_not_executed(self):
        result = gate_l.dry_run()
        self.assertEqual(result["status"], "DRY_RUN_VALID")
        self.assertEqual(result["gate"], "L")
        self.assertEqual(result["smoke_count"], 4)
        self.assertEqual(result["partition"], "debug")
        self.assertFalse(result["formal_acquisition"])
        self.assertFalse(result["selector_executed"])
        self.assertFalse(result["outcome_read"])
        self.assertEqual(
            result["gate_m_formal_runtime_dry_plan"]["status"],
            "GATE_M_FORMAL_RUNTIME_DRY_PLAN_IMPORTABLE",
        )
        self.assertFalse(result["gate_m_formal_runtime_dry_plan"]["formal_acquisition_executed"])

    def test_import_only_plan_and_preferred_membership_are_closed(self):
        plan = gate_l.import_only_dry_plan(CARD_PATH, TRAJECTORY_SCHEMA_PATH)
        self.assertEqual(plan["status"], "IMPORT_ONLY_DRY_PLAN")
        self.assertEqual(plan["records"], 4)
        self.assertFalse(plan["model_weights_loaded"])
        self.assertFalse(plan["forward_executed"])
        self.assertEqual(len(gate_l.PREFERRED_SMOKE_IDENTITIES), 4)
        self.assertEqual(
            len({row["doc_id"] for row in gate_l.PREFERRED_SMOKE_IDENTITIES}),
            4,
        )

    def test_debug_launcher_is_bounded_and_does_not_enter_follow_up(self):
        text = gate_l.build_sbatch_text(
            run_root=Path("/tmp/phase6-gate-l-mmlu5-test"),
            expected_commit=gate_l.AUTHORIZED_BASE,
        )
        self.assertIn("#SBATCH --partition=debug", text)
        self.assertIn("#SBATCH --time=00:29:00", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("#SBATCH --mem=64G", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertNotIn("selector", text.lower())
        self.assertNotIn("outcome", text.lower())


if __name__ == "__main__":
    unittest.main()
