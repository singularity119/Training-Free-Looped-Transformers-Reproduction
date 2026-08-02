import json
import tempfile
import unittest
from pathlib import Path

from tflt.loopscope import phase6_mmlu5_gate_o as gate_o
from tflt.loopscope.phase3_p3c import safe_test_doc_sha256
from tflt.loopscope.phase3_schema import sanitized_content_sha256


class GateOTests(unittest.TestCase):
    def test_dry_run_closes_five_cell_contract(self):
        result = gate_o.dry_run()
        self.assertEqual(result["status"], "DRY_RUN_VALID")
        self.assertEqual(result["gate"], "O")
        self.assertEqual(result["cell_count"], 5)
        self.assertEqual(
            [row["cell_id"] for row in result["cells"]],
            ["no_loop", "14:16", "13:16", "15:18", "12:16"],
        )
        self.assertEqual(result["decode_mode"], "full")
        self.assertEqual(result["formal_array"], "0-4%5")
        self.assertFalse(result["outcome_read"])

    def test_formal_manifest_has_exact_cells_and_full_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = gate_o.build_launch_manifest(
                mode="formal",
                run_root=Path(temporary) / "run",
                expected_commit=gate_o.ADMISSION_BASE,
            )
            self.assertEqual(
                [(row["cell_id"], row["window"]) for row in manifest["cells"]],
                [("no_loop", None), ("14:16", "14:16"), ("13:16", "13:16"), ("15:18", "15:18"), ("12:16", "12:16")],
            )
            self.assertEqual(manifest["scheduler"]["partition"], "emergency_gpu")
            self.assertEqual(manifest["scheduler"]["qos"], "emergency_gpu")
            for row in manifest["cells"][1:]:
                argv = row["eval_argv"]
                self.assertIn("--decode-mode", argv)
                self.assertEqual(argv[argv.index("--decode-mode") + 1], "full")
            gate_o.validate_launch_manifest(manifest, mode="formal")

            root = Path(temporary) / "run"
            (root / "manifest").mkdir(parents=True)
            gate_o._write_new_json(root / "manifest/launch_manifest.json", manifest)
            sbatch = gate_o.build_sbatch_text(
                mode="formal",
                run_root=root,
                expected_commit=gate_o.ADMISSION_BASE,
            )
            self.assertIn("#SBATCH --array=0-4%5", sbatch)
            self.assertIn("#SBATCH --partition=emergency_gpu", sbatch)
            self.assertIn("#SBATCH --qos=emergency_gpu", sbatch)
            self.assertIn("#SBATCH --gres=gpu:a800:1", sbatch)
            self.assertNotIn("--decode-mode", sbatch)

    def test_identity_projection_does_not_need_outcome_fields(self):
        safe = {
            "question": "Which symbol is first?",
            "choices": ["A", "B", "C", "D"],
            "subject": "synthetic",
        }
        expected = [
            {
                "identity": {
                    "task": "mmlu_synthetic",
                    "doc_id": "0",
                    "doc_hash": safe_test_doc_sha256(safe),
                },
                "subject": "synthetic",
                "split": "test",
                "sanitized_content_sha256": sanitized_content_sha256(safe["question"], safe["choices"]),
            }
        ]
        projected = gate_o.project_identity_only(
            {
                "samples": {
                    "mmlu_synthetic": [
                        {"doc_id": 0, "doc": safe, "target": 2}
                    ]
                },
                "results": {"mmlu_synthetic": {"acc,none": 0.0}},
            },
            expected,
        )
        self.assertEqual(projected[0]["identity"], expected[0]["identity"])
        self.assertNotIn("target", projected[0])

    def test_subject_stratified_bootstrap_is_deterministic(self):
        matrix = [[0, 1, 0, 1, 0], [1, 1, 1, 0, 1], [0, 0, 1, 1, 1], [1, 0, 1, 1, 0]]
        subjects = ["a", "a", "b", "b"]
        first = gate_o.subject_stratified_paired_bootstrap(matrix, subjects, replicates=20, seed=7)
        second = gate_o.subject_stratified_paired_bootstrap(matrix, subjects, replicates=20, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["replicates"], 20)
        self.assertEqual(len(first["cell_gain_summaries"]), 4)
        self.assertTrue(first["same_joint_draw_stream_all_cells_and_contrasts"])

    def test_scheduler_requires_exact_five_completed_array_tasks(self):
        rows = [
            {
                "JobID": "12345_%d" % index,
                "State": "COMPLETED",
                "ExitCode": "0:0",
                "ElapsedRaw": "10",
                "AllocTRES": "gres/gpu:a800=1",
                "Partition": "emergency_gpu",
                "NodeList": "a%d" % index,
            }
            for index in range(5)
        ]
        self.assertEqual(len(gate_o.validate_scheduler_rows(rows, "12345")), 5)
        with self.assertRaises(gate_o.GateOError):
            gate_o.validate_scheduler_rows(rows[:4], "12345")


if __name__ == "__main__":
    unittest.main()
