import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/loopscope/run_phase11_canary_batch.py"
SPEC = importlib.util.spec_from_file_location("canary_batch", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.job = dict(dataset="mmlu_pro", manifest="full.json", pool="full-pool.json",
                        cell_id="online", run_root="fresh", canonical_index=100,
                        population_count=12032)

    def test_formal_preserves_full_pool_singleton_shard(self):
        command = MODULE.command(self.job, "FORMAL_TEST", "revision")
        self.assertEqual(command[command.index("--shard") + 1], "100/12032")
        self.assertEqual(command[command.index("--pool") + 1], "full-pool.json")
        self.assertNotIn("--engineering-generate-steps", command)

    def test_synthetic_three_real_generation_steps(self):
        command = MODULE.command(self.job, "PREFLIGHT_ONLY", "revision")
        self.assertNotIn("--shard", command)
        self.assertEqual(command[command.index("--engineering-generate-steps") + 1], "3")
        self.job["dataset"] = "arc_challenge"
        self.assertNotIn("--engineering-generate-steps", MODULE.command(self.job, "PREFLIGHT_ONLY", "revision"))
