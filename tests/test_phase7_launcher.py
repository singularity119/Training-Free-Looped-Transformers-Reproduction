import importlib.util
import io
import json
import unittest
from pathlib import Path

from tflt.loopscope.phase7_schema import Phase7ContractError


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/loopscope/run_phase7_base_trajectory.py"
SPEC = importlib.util.spec_from_file_location("run_phase7_base_trajectory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeDevice:
    type = "cuda"

    def __str__(self):
        return "cuda:0"


class FakeCuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeTensor:
    def __init__(self, device="cpu"):
        self.device = device

    def to(self, device):
        return FakeTensor(str(device))


class FakeTorch:
    def __init__(self, available=True):
        self.cuda = FakeCuda(available)

    def device(self, name):
        if name != "cuda:0":
            raise AssertionError("unexpected device")
        return FakeDevice()

    def is_tensor(self, value):
        return isinstance(value, FakeTensor)


class FakeModel:
    def __init__(self):
        self._parameters = [FakeTensor(), FakeTensor()]

    def to(self, device):
        self._parameters = [value.to(device) for value in self._parameters]
        return self

    def parameters(self):
        return iter(self._parameters)


class Phase7LauncherTests(unittest.TestCase):
    def test_cache_dir_is_propagated_only_when_given(self):
        without_cache = MODULE._pretrained_load_kwargs(None)
        self.assertNotIn("cache_dir", without_cache)
        with_cache = MODULE._pretrained_load_kwargs("/audited/hf_home")
        self.assertEqual(with_cache["cache_dir"], "/audited/hf_home")
        self.assertTrue(with_cache["local_files_only"])
        self.assertFalse(with_cache["trust_remote_code"])

    def test_cuda_model_and_tensor_inputs_share_one_device(self):
        fake_torch = FakeTorch(available=True)
        device = MODULE._require_cuda_device(fake_torch)
        model = MODULE._bind_model_to_device(FakeModel(), device)
        bound = MODULE._bind_tensor_inputs(
            fake_torch,
            {"input_ids": FakeTensor(), "attention_mask": FakeTensor(), "metadata": "safe"},
            device,
        )
        self.assertEqual({parameter.device for parameter in model.parameters()}, {"cuda:0"})
        self.assertEqual(bound["input_ids"].device, "cuda:0")
        self.assertEqual(bound["attention_mask"].device, "cuda:0")
        self.assertEqual(bound["metadata"], "safe")

    def test_runtime_fails_closed_without_cuda(self):
        with self.assertRaisesRegex(Phase7ContractError, "BLOCK_CUDA_UNAVAILABLE"):
            MODULE._require_cuda_device(FakeTorch(available=False))

    def test_progress_counter_contains_only_safe_integer_metadata(self):
        stream = io.StringIO()
        MODULE._emit_progress(25, 765, stream=stream)
        self.assertEqual(
            stream.getvalue(),
            "event=phase7_progress completed_records=25 total_records=765\n",
        )

    def test_progress_counter_does_not_change_jsonl_serialization_or_order(self):
        records = [{"canonical_identity": "first"}, {"canonical_identity": "second"}]
        before = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in records
        )
        stream = io.StringIO()
        MODULE._emit_progress(2, 2, stream=stream)
        after = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in records
        )
        self.assertEqual(after, before)
        self.assertEqual([row["canonical_identity"] for row in records], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
