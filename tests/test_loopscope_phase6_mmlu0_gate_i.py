import math
import unittest
from pathlib import Path

from tflt.loopscope import phase6_mmlu0_gate_i as gate_i
from tflt.loopscope.phase6_mmlu0_schema import (
    BOUNDARY_COUNT,
    CHOICE_SURFACE_MANIFEST_SHA256,
    build_trajectory_record,
    choice_logits_to_probabilities,
    load_json,
    validate_trajectory_record,
)


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "configs/loopscope/phase6_mmlu0_prefix_card.json"


class GateITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = load_json(CARD_PATH)

    def test_dry_run_is_frozen_and_non_executing(self):
        result = gate_i.dry_run()
        self.assertEqual(result["status"], "DRY_RUN_VALID")
        self.assertEqual(result["gate"], "I")
        self.assertEqual(result["smoke_count"], 4)
        self.assertEqual(result["partition"], "debug")
        self.assertFalse(result["selector_executed"])
        self.assertFalse(result["loop_executed"])
        self.assertFalse(result["outcome_read"])
        self.assertFalse(result["formal_acquisition"])

    def test_debug_launcher_has_exact_bounded_resources(self):
        text = gate_i.build_sbatch_text(
            run_root=Path("/hpc2hdd/home/xhuang225/workspaces/gate-i-test"),
            expected_commit=gate_i.AUTHORIZED_BASE,
        )
        self.assertIn("#SBATCH --partition=debug", text)
        self.assertIn("#SBATCH --time=00:29:00", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("#SBATCH --mem=64G", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertNotIn("formal", text.lower())
        self.assertNotIn("selector", text.lower())

    def test_raw_adjacent_angles_are_kept_separate_from_normalized_geometry(self):
        probabilities = [
            choice_logits_to_probabilities([1.0, 0.2, -0.1, -0.4])
            for _ in range(BOUNDARY_COUNT)
        ]
        normalized = [[1.0, 0.0] for _ in range(BOUNDARY_COUNT)]
        raw = [[1.0, 0.0], [0.0, 1.0]] + [[1.0, 0.0] for _ in range(BOUNDARY_COUNT - 2)]
        provenance = {
            "contract_version": self.card["renderer"]["contract_version"],
            "semantic_contract_sha256": self.card["renderer"]["semantic_contract_sha256"],
            "task_source_manifest_sha256": self.card["task"]["lm_eval_task_source"]["default_source_manifest_sha256"],
            "choice_surface_manifest_sha256": CHOICE_SURFACE_MANIFEST_SHA256,
        }
        record = build_trajectory_record(
            {"task": "mmlu", "doc_id": "synthetic-raw-angle", "doc_hash": "a" * 64},
            "synthetic_subject",
            "b" * 64,
            provenance,
            probabilities,
            normalized,
            self.card,
            angular_hidden_states=raw,
        )
        validate_trajectory_record(record, self.card)
        self.assertAlmostEqual(record["boundaries"][0]["hidden_cosine_to_final"], 1.0)
        self.assertAlmostEqual(record["adjacent_angular_distance"][0]["angle_over_pi"], 0.5)
        self.assertTrue(all(math.isfinite(row["angle_over_pi"]) for row in record["adjacent_angular_distance"]))

    def test_scheduler_acceptance_is_debug_completed_only(self):
        row = {
            "JobIDRaw": "123456",
            "JobName": "loopscope-p6-i-smoke",
            "Partition": "debug",
            "State": "COMPLETED",
            "ExitCode": "0:0",
            "Elapsed": "00:01:00",
            "NodeList": "gpu-node",
            "AllocTRES": "cpu=8,mem=64G,gres/gpu=1",
            "ReqTRES": "cpu=8,mem=64G,gres/gpu=1",
        }
        self.assertEqual(gate_i._validate_scheduler([row], "123456"), row)
        invalid = dict(row, Partition="formal")
        with self.assertRaises(gate_i.GateIMMLU0Error):
            gate_i._validate_scheduler([invalid], "123456")


if __name__ == "__main__":
    unittest.main()
