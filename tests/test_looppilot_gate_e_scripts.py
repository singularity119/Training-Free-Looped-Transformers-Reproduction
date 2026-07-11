import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GateEScriptContractTest(unittest.TestCase):
    def test_gpu_array_contract_and_sequential_arms(self):
        text = (ROOT / "scripts/looppilot/gate_e_gpu_array.sbatch").read_text()
        for line in (
            "#SBATCH --job-name=looppilot-gate-e",
            "#SBATCH --partition=i64m1tga40u",
            "#SBATCH --gres=gpu:a40:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=06:00:00",
        ):
            self.assertIn(line, text)
        runner = (ROOT / "scripts/looppilot/run_gate_e_shard.py").read_text()
        self.assertLess(runner.index('"baseline_a"'), runner.index('"baseline_b"'))
        self.assertLess(runner.index('"baseline_b"'), runner.index('"always_loop"'))

    def test_cpu_analysis_contract(self):
        text = (ROOT / "scripts/looppilot/gate_e_analysis.sbatch").read_text()
        for line in (
            "#SBATCH --partition=i64m512u",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=02:00:00",
        ):
            self.assertIn(line, text)
        self.assertNotIn("--gres", text)

    def test_submitter_has_one_dynamic_array_and_ambiguous_submit_guard(self):
        gpu = (ROOT / "scripts/looppilot/submit_gate_e_gpu.sh").read_text()
        self.assertEqual(gpu.count("sbatch --parsable"), 1)
        self.assertIn('array="0-7%$concurrency"', gpu)
        self.assertNotIn("0-3%2", gpu)
        self.assertNotIn("4-7%2", gpu)
        self.assertIn("/opt/slurm/bin/sinfo", gpu)
        self.assertIn("/opt/slurm/bin/scontrol show node -o", gpu)
        self.assertIn("/opt/slurm/bin/squeue", gpu)
        self.assertIn("gpu-capacity-snapshot-", gpu)
        self.assertIn("gpu-array-submit-started.txt", gpu)
        self.assertIn("SubmitLine", gpu)
        self.assertIn("job_id", gpu)
        self.assertNotIn("scancel", gpu)
        analysis = (ROOT / "scripts/looppilot/submit_gate_e_analysis.sh").read_text()
        self.assertEqual(analysis.count("sbatch --parsable"), 1)
        self.assertIn("gpu-array-terminal-verified.txt", analysis)
        self.assertNotIn("batch-b-terminal-verified.txt", analysis)

    def test_capacity_parser_counts_only_eligible_scheduler_visible_a40s(self):
        text = (ROOT / "scripts/looppilot/submit_gate_e_gpu.sh").read_text()
        code = text.split("<<'PY_CAPACITY'\n", 1)[1].split("\nPY_CAPACITY", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sinfo = root / "sinfo.txt"
            nodes = root / "nodes.txt"
            queue = root / "queue.txt"
            snapshot = root / "snapshot.json"
            sinfo.write_text("raw sinfo\n")
            queue.write_text("raw squeue\n")
            nodes.write_text(
                "NodeName=gpu-a Partitions=i64m1tga40u State=MIXED CfgTRES=cpu=64,gres/gpu:a40=8 AllocTRES=cpu=8,gres/gpu:a40=3\n"
                "NodeName=gpu-b Partitions=other State=IDLE CfgTRES=cpu=64,gres/gpu:a40=8 AllocTRES=\n"
                "NodeName=gpu-c Partitions=i64m1tga40u State=DRAIN CfgTRES=cpu=64,gres/gpu:a40=8 AllocTRES=\n"
            )
            completed = subprocess.run(
                [sys.executable, "-", str(sinfo), str(nodes), str(queue), str(snapshot), "20260712T000000Z"],
                input=code,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "5")
            payload = json.loads(snapshot.read_text())
            self.assertEqual(payload["aggregate_free_a40"], 5)
            self.assertEqual(payload["concurrency"], 5)
            self.assertEqual(payload["eligible_nodes"][0]["node"], "gpu-a")
            self.assertIn("raw_outputs", payload)

    def test_capacity_parser_fails_on_negative_free_a40(self):
        text = (ROOT / "scripts/looppilot/submit_gate_e_gpu.sh").read_text()
        code = text.split("<<'PY_CAPACITY'\n", 1)[1].split("\nPY_CAPACITY", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("sinfo.txt", "queue.txt"):
                (root / name).write_text("raw\n")
            (root / "nodes.txt").write_text(
                "NodeName=gpu-a Partitions=i64m1tga40u State=MIXED CfgTRES=gres/gpu:a40=2 AllocTRES=gres/gpu:a40=3\n"
            )
            completed = subprocess.run(
                [sys.executable, "-", str(root / "sinfo.txt"), str(root / "nodes.txt"), str(root / "queue.txt"), str(root / "snapshot.json"), "20260712T000001Z"],
                input=code,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
