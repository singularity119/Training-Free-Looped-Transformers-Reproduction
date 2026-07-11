import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GateCScriptContractTest(unittest.TestCase):
    def test_sbatch_is_exact_single_debug_a40_job(self):
        text = (ROOT / "scripts/looppilot/gate_c_probe.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --job-name=looppilot-gate-c", text)
        self.assertIn("#SBATCH --partition=debug", text)
        self.assertIn("#SBATCH --gres=gpu:a40:1", text)
        self.assertIn("#SBATCH --cpus-per-task=8", text)
        self.assertIn("#SBATCH --mem=64G", text)
        self.assertIn("#SBATCH --time=00:30:00", text)
        self.assertNotIn("--array", text)
        self.assertIn("HF_HUB_OFFLINE=1", text)
        self.assertIn("HF_DATASETS_OFFLINE=1", text)

    def test_submitter_has_one_sbatch_and_write_once_root_guard(self):
        text = (ROOT / "scripts/looppilot/submit_gate_c_probe.sh").read_text(encoding="utf-8")
        self.assertEqual(text.count("sbatch --parsable"), 1)
        self.assertIn('test ! -e "$run_root"', text)
        self.assertIn("prepare_gate_c_probe.py", text)
        self.assertNotIn("scancel", text)

    def test_profile_and_probe_config_match(self):
        profile = (ROOT / "configs/profiles/hpc2-looppilot-gate-c.toml").read_text(
            encoding="utf-8"
        )
        config = json.loads(
            (ROOT / "configs/looppilot/gate_c_probe.json").read_text(encoding="utf-8")
        )
        self.assertIn('partition = "debug"', profile)
        self.assertIn('gres = "gpu:a40:1"', profile)
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["tasks"], [
            "mmlu_abstract_algebra",
            "mmlu_anatomy",
            "mmlu_astronomy",
            "mmlu_business_ethics",
        ])


if __name__ == "__main__":
    unittest.main()
