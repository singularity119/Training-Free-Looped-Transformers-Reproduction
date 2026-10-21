import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/loopscope"))
import phase11_qwen3_nothink_launch as launcher
from tflt.loopscope.phase11_qwen3_nothink import CELLS


class LaunchTests(unittest.TestCase):
    def test_sixteen_workers_cover_three_cells_and_unmasked_devices(self):
        plan = launcher.make_plan("/fresh/runs", "/fresh/pool", "source", 16)
        pairs = [(job["cell_key"], i) for job in plan["registry"]["jobs"] for i in job["indices"]]
        self.assertEqual(len(pairs), 1344)
        self.assertEqual(set(pairs), {(key, i) for key in CELLS for i in range(448)})
        self.assertEqual(plan["registry"]["schema"], "loopscope.phase11.qwen3_nothink.full_registry.v1")
        spawned = []
        class Process:
            def __init__(self, sampler):
                self.returncode = None if sampler else 0
            def poll(self):
                return self.returncode
            def terminate(self):
                self.returncode = 0
            def wait(self):
                return self.returncode
        def popen(command, **kwargs):
            spawned.append((command, kwargs))
            return Process(command[0] == "nvidia-smi")
        env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8"}
        inventory = "\n".join(f"GPU-{i}, NVIDIA H20" for i in range(8))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(launcher.subprocess, "check_output", side_effect=[inventory, "8\n"]), \
                    patch.object(launcher.subprocess, "Popen", side_effect=popen):
                launcher.launch(path, "/model", "/admission", root / "launch", "actual", 60)
            self.assertTrue((root / "launch/node-0/complete.json").exists())
            env["RANK"] = "8"
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(launcher.subprocess, "check_output", side_effect=[inventory, "8\n"]), \
                    patch.object(launcher.subprocess, "Popen", side_effect=popen):
                launcher.launch(path, "/model", "/admission", root / "launch", "actual", 60)
            self.assertTrue((root / "launch/node-1/complete.json").exists())
        workers = [(cmd, kw) for cmd, kw in spawned if cmd[0] != "nvidia-smi"]
        self.assertEqual(len(workers), 16)
        self.assertEqual({kw["env"]["D9_WORKER_RANK"] for cmd, kw in workers}, {str(i) for i in range(16)})
        self.assertTrue(all("RANK" not in kw["env"] and "WORLD_SIZE" not in kw["env"] for cmd, kw in workers))
        self.assertEqual({kw["env"]["CUDA_VISIBLE_DEVICES"] for cmd, kw in workers}, {f"GPU-{i}" for i in range(8)})


if __name__ == "__main__":
    unittest.main()
