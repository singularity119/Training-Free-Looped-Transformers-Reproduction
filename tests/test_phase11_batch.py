import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/loopscope/run_phase11_batch.py"
SPEC = importlib.util.spec_from_file_location("phase11_batch", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.job = dict(dataset="gpqa_main", manifest="full.json", pool="full-pool.json",
                        cell_id="online", run_root="fresh", population_count=448)

    def test_formal_index_batch_preserves_full_pool(self):
        self.job["indices_file"] = "fresh-indices.json"
        command = MODULE.command(self.job, "FORMAL_TEST", "revision")
        self.assertEqual(command[command.index("--indices-file") + 1], "fresh-indices.json")
        self.assertEqual(command[command.index("--pool") + 1], "full-pool.json")
        self.assertNotIn("--shard", command)
        self.assertNotIn("--engineering-generate-steps", command)

    def test_canonical_singleton_compatibility(self):
        self.job["canonical_index"] = 100
        command = MODULE.command(self.job, "FORMAL_TEST", "revision")
        self.assertEqual(command[command.index("--shard") + 1], "100/448")

    def test_synthetic_generation_still_uses_three_steps(self):
        self.job["indices_file"] = "fresh-indices.json"
        command = MODULE.command(self.job, "PREFLIGHT_ONLY", "revision")
        self.assertIn("--indices-file", command)
        self.assertEqual(command[command.index("--engineering-generate-steps") + 1], "3")
        self.job["dataset"] = "arc_challenge"
        self.assertNotIn("--engineering-generate-steps", MODULE.command(self.job, "PREFLIGHT_ONLY", "revision"))

    def test_thread_policy_uses_at_most_eight_cpu_threads(self):
        for packing, count in ((1, 8), (2, 4), (4, 2), (6, 1), (8, 1)):
            with self.subTest(packing=packing):
                env = MODULE.thread_environment(packing, {"OMP_NUM_THREADS": "99", "CUDA_VISIBLE_DEVICES": "GPU-allocated"})
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    self.assertEqual(env[name], str(count))
                self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-allocated")
                self.assertLessEqual(count * packing, 8)

    def test_inline_indices_are_fresh_and_original_numbered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.job.update(run_root=str(root / "fresh"), indices=[4, 18, 447])
            jobs = MODULE.prepare_jobs([self.job], root)
            declaration = json.loads(Path(jobs[0]["indices_file"]).read_text())
            self.assertEqual(declaration, {"schema": "loopscope.phase11.index_batch.v1",
                "dataset": "gpqa_main", "cell_id": "online", "population_count": 448, "indices": [4, 18, 447]})
            with self.assertRaises(FileExistsError):
                MODULE.prepare_jobs([self.job], root)

    def test_same_cell_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = dict(self.job, run_root=str(root / "one"), indices=[4, 18])
            second = dict(self.job, run_root=str(root / "two"), indices=[18, 20])
            with self.assertRaisesRegex(ValueError, "repeats canonical"):
                MODULE.prepare_jobs([first, second], root)


if __name__ == "__main__":
    unittest.main()
