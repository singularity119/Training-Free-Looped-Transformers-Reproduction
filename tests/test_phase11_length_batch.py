import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/loopscope"))
SPEC = importlib.util.spec_from_file_location("length_batch", ROOT / "scripts/loopscope/phase11_length_batch.py")
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


class LengthBatchTests(unittest.TestCase):
    def test_child_uses_length_extension_and_retains_random_order(self):
        job = dict(membership="frozen.json", pool="safe-pool.json", cell_id="gpqa_main-q4-native",
                   run_root="fresh", indices=[4, 404, 25])
        cmd = BATCH.command(job, "revision")
        self.assertIn("scripts/loopscope/phase11_length_pilot.py", cmd)
        self.assertEqual(cmd[cmd.index("--indices") + 1], "4,404,25")
        self.assertNotIn("--synthetic", cmd)
        self.assertNotIn("scripts/loopscope/run_phase11_accuracy.py", cmd)
        job["synthetic"] = True
        cmd = BATCH.command(job, "revision")
        self.assertIn("--synthetic", cmd)
        self.assertNotIn("--indices", cmd)


if __name__ == "__main__":
    unittest.main()
