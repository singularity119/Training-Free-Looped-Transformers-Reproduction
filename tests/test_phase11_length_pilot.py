"""Verify the length extension reaches the original producer without recipe drift."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tflt.loopscope.phase11_accuracy import generation_record
from tflt.loopscope.phase11_panel import load_config

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/loopscope"))
SPEC = importlib.util.spec_from_file_location("phase11_length_pilot", REPO / "scripts/loopscope/phase11_length_pilot.py")
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


class LengthPilotTests(unittest.TestCase):
    def invoke(self, directory, capacity=262144):
        root = Path(directory)
        member = root / "membership.json"
        member.write_text(json.dumps(PILOT.membership()))
        args = PILOT.parser().parse_args(["acquire", "--membership", str(member), "--pool", "synthetic.json",
            "--cell-id", "gpqa_main-q4-native", "--synthetic", "--run-root", str(root / "run"), "--commit", "source"])
        row = {"task": "gpqa_main", "identity": "synthetic", "subject": "synthetic", "input_ids": [1, 2],
               "choice_labels": ["A", "B"]}
        cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
        torch = SimpleNamespace(cuda=cuda, set_grad_enabled=lambda x: None, manual_seed=lambda x: None)
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=capacity))
        record = generation_record(row, "Final answer: (A)", [11, 12, 151645], [151645, 151643], 8192, 0)
        with patch.dict(os.environ, {"SLURM_JOB_ID": "1", "SLURM_JOB_PARTITION": "debug"}), \
             patch.object(PILOT, "load_pool", return_value={"rows": [row]}), \
             patch.object(PILOT, "source_provenance", return_value={"source_head": "source"}), \
             patch.object(PILOT, "load_runtime", return_value=(torch, model, None, None, [151645, 151643], {})), \
             patch.object(PILOT, "ForwardTiming", return_value=SimpleNamespace(summary=lambda: {}, close=lambda: None)), \
             patch.object(PILOT, "acquire_row", return_value=(record, {})) as acquire, redirect_stdout(io.StringIO()):
            if capacity != 262144:
                with self.assertRaisesRegex(ValueError, "native model context differs"):
                    PILOT.acquire(args)
                acquire.assert_not_called()
                self.assertTrue((root / "run/failure.json").exists())
            else:
                PILOT.acquire(args)
                call = acquire.call_args.args
                self.assertEqual(call[-2:], (8192, 3))
                self.assertEqual(call[8]["max_new_tokens"], 8192)
                self.assertEqual(call[7]["cache_strategy"], "first")
                self.assertEqual(call[7]["decode_mode"], "full")
                summary = json.loads((root / "run/summary.json").read_text())
                self.assertEqual(summary["count"], 1)
                self.assertFalse(summary["target_gold_loaded"])
                self.assertEqual(load_config()["max_new_tokens"], 2048)

    def test_8192_and_native_eos_reach_existing_row_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            self.invoke(directory)

    def test_model_context_mismatch_prevents_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            self.invoke(directory, 8192)


if __name__ == "__main__":
    unittest.main()
