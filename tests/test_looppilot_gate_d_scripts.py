import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GateDScriptContractTest(unittest.TestCase):
    def test_sbatch_is_exact_single_debug_a40_job_and_sequential(self):
        text = (ROOT / "scripts/looppilot/gate_d_limit.sbatch").read_text(encoding="utf-8")
        for line in (
            "#SBATCH --job-name=looppilot-gate-d",
            "#SBATCH --partition=debug",
            "#SBATCH --gres=gpu:a40:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=00:30:00",
        ):
            self.assertIn(line, text)
        self.assertNotIn("--array", text)
        self.assertIn("run_gate_d_limit.py", text)
        self.assertIn("verify_gate_d_limit.py", text)
        self.assertLess(text.index("run_gate_d_limit.py"), text.index("verify_gate_d_limit.py"))
        self.assertIn("HF_HUB_OFFLINE=1", text)
        self.assertIn("HF_DATASETS_OFFLINE=1", text)

    def test_submitter_has_one_sbatch_and_write_once_root_guard(self):
        text = (ROOT / "scripts/looppilot/submit_gate_d_limit.sh").read_text(encoding="utf-8")
        self.assertEqual(text.count("sbatch --parsable"), 1)
        self.assertIn('test ! -e "$run_root"', text)
        self.assertIn("prepare_gate_d_limit.py", text)
        self.assertNotIn("scancel", text)

    def test_profile_and_limit_config_match_handoff(self):
        profile = (ROOT / "configs/profiles/hpc2-looppilot-gate-d.toml").read_text(
            encoding="utf-8"
        )
        config = json.loads(
            (ROOT / "configs/looppilot/gate_d_limit.json").read_text(encoding="utf-8")
        )
        self.assertIn('partition = "debug"', profile)
        self.assertIn('gres = "gpu:a40:1"', profile)
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["limit"], 5)
        self.assertEqual(config["total_docs"], 20)

    def test_qwen_model_is_preloaded_with_gate_c_torch_dtype_contract(self):
        text = (ROOT / "scripts/looppilot/run_gate_d_limit.py").read_text(encoding="utf-8")
        self.assertIn("AutoModelForCausalLM.from_pretrained(", text)
        self.assertIn("torch_dtype=torch.float16", text)
        hflm_call = text[text.index("lm = GateDHFLM(") : text.index("model = _underlying_hf_model")]
        self.assertNotIn("dtype=", hflm_call)


if __name__ == "__main__":
    unittest.main()
