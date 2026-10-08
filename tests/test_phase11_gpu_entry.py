"""The GPU entry point must be inspectable without installed model libraries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ENTRY = Path(__file__).resolve().parents[1] / "scripts" / "loopscope" / "check_phase11_gpu.py"


class GPUEntryTests(unittest.TestCase):
    def invoke(self, *args):
        env = dict(os.environ)
        env.pop("SLURM_JOB_ID", None)
        env.pop("SLURM_JOB_PARTITION", None)
        # -S removes site packages: CLI paths cannot accidentally need torch,
        # transformers or lm_eval merely to render a plan.
        return subprocess.run([sys.executable, "-S", str(ENTRY), *args], env=env,
                              text=True, capture_output=True, timeout=30)

    def test_help_without_heavy_libraries(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dry-run", result.stdout)

    def test_dry_run_is_synthetic_and_creates_no_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fresh"
            result = self.invoke("--run-root", str(root), "--commit", "synthetic-checkout", "--dry-run")
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["scope"], "PREFLIGHT_ONLY")
            self.assertFalse(plan["target_gold_loaded"])
            self.assertEqual(plan["generation"]["min_new_tokens"], 3)
            self.assertEqual(plan["generation"]["max_new_tokens"], 3)
            self.assertEqual(plan["generation"]["num_beams"], 1)
            self.assertEqual(plan["window"], [15, 18])
            self.assertEqual((plan["cache_strategy"], plan["decode_mode"]), ("first", "full"))
            self.assertEqual(len(plan["cases"]), 11)
            self.assertIn("K3-current_t-lambda0.1", plan["cases"])
            self.assertIn("K3-current_t-lambda0.9", plan["cases"])
            self.assertFalse(root.exists())

    def test_real_path_rejects_missing_debug_job_before_import_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fresh"
            result = self.invoke("--run-root", str(root), "--commit", "synthetic-checkout")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("real Slurm debug allocation", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
