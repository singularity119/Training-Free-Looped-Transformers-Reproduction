import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LoopPilotConfigTest(unittest.TestCase):
    def test_phase_one_config_and_revisions_are_frozen(self):
        config = json.loads((ROOT / "configs/looppilot/qwen17_mmlu_phase1.json").read_text(encoding="utf-8"))
        revisions = json.loads((ROOT / "configs/looppilot/resolved_revisions.json").read_text(encoding="utf-8"))
        self.assertEqual(config["model"], "Qwen/Qwen3-1.7B-Base")
        self.assertEqual(config["window"], [12, 15])
        self.assertEqual(config["k"], 2)
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["model_revision"], revisions["model"]["commit_hash"])
        self.assertEqual(config["tokenizer_revision"], revisions["tokenizer"]["commit_hash"])

    def test_validator_accepts_checked_in_config(self):
        completed = subprocess.run(
            [sys.executable, "scripts/looppilot/validate_phase1_config.py"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(completed.stdout.strip()), 64)


if __name__ == "__main__":
    unittest.main()
