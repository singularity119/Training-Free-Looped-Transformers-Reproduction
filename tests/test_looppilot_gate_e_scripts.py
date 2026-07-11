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

    def test_submitters_have_ambiguous_submit_guard_and_exact_arrays(self):
        gpu = (ROOT / "scripts/looppilot/submit_gate_e_gpu.sh").read_text()
        self.assertIn("0-3%2", gpu)
        self.assertIn("4-7%2", gpu)
        self.assertIn("SubmitLine", gpu)
        self.assertIn("job_id", gpu)
        self.assertNotIn("scancel", gpu)
        analysis = (ROOT / "scripts/looppilot/submit_gate_e_analysis.sh").read_text()
        self.assertEqual(analysis.count("sbatch --parsable"), 1)


if __name__ == "__main__":
    unittest.main()
