"""CPU checks for resource shape, cache evidence and source admission."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tflt.loopscope.phase11_panel import load_config
from tflt.loopscope.phase11_adapter import token_metadata

ENTRY = Path(__file__).resolve().parents[1] / "scripts/loopscope/measure_phase11_resources.py"
sys.path.insert(0, str(ENTRY.parent))
SPEC = importlib.util.spec_from_file_location("phase11_resources_entry", ENTRY)
RESOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESOURCE)


class Tokenizer:
    all_special_ids = [0, 1]
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [0, 2, 3, 4, 5, 6, 1]


class IDs:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def tolist(self):
        return self.rows


class Cache:
    def __init__(self, length, bad_layer=False):
        self.length, self.bad_layer = length, bad_layer

    def get_seq_length(self, layer=0):
        return self.length - int(self.bad_layer and layer == 1)


class Model:
    config = SimpleNamespace(num_hidden_layers=2)
    device = "cuda"

    def __init__(self, limit=2048, bad_layer=False):
        self.pre, self.post = None, None
        self.limit, self.bad_layer = limit, bad_layer
        self.options = None

    def register_forward_pre_hook(self, hook, with_kwargs):
        self.pre = hook
        return SimpleNamespace(remove=lambda: setattr(self, "pre", None))

    def register_forward_hook(self, hook, with_kwargs):
        self.post = hook
        return SimpleNamespace(remove=lambda: setattr(self, "post", None))

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=True):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                      past_key_values=past_key_values, use_cache=use_cache)
        if self.pre:
            self.pre(self, (), kwargs)
        before = 0 if past_key_values is None else past_key_values.length
        output = SimpleNamespace(past_key_values=Cache(before + input_ids.shape[1], self.bad_layer))
        if self.post:
            self.post(self, (), kwargs, output)
        return output

    def generate(self, input_ids, attention_mask, **options):
        self.options = options
        cache = self.forward(input_ids=input_ids, attention_mask=attention_mask).past_key_values
        for _ in range(self.limit - 1):
            cache = self.forward(input_ids=IDs([[2]]), past_key_values=cache).past_key_values
        return SimpleNamespace(shape=(1, input_ids.shape[1] + self.limit))


def fake_torch():
    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                           memory_allocated=lambda: 10, memory_reserved=lambda: 12,
                           max_memory_allocated=lambda: 20, max_memory_reserved=lambda: 24)
    return SimpleNamespace(cuda=cuda, tensor=lambda rows, **kwargs: IDs(rows),
                           long="long", ones_like=lambda ids: ids)


class ResourceTests(unittest.TestCase):
    def test_bound_generation_shape_and_online_family(self):
        config = load_config()
        for dataset, length in (("mmlu_pro", 2860), ("gpqa_main", 2819)):
            plan = RESOURCE.resource_plan(dataset, "OnlineK3", config)
            self.assertEqual(plan["prompt_tokens"], length)
            self.assertEqual((plan["min_new_tokens"], plan["max_new_tokens"]), (2048, 2048))
            self.assertEqual(plan["expected_final_cache_length"], length + 2047)
            self.assertEqual((plan["cell"]["k"], plan["cell"]["direction_policy"], plan["cell"]["strength"]),
                             (3, "current_t", 0.9))
            self.assertEqual(plan["scope"], "PREFLIGHT_ONLY")

    def test_arc_shape_is_full_candidate_not_generation(self):
        plan = RESOURCE.resource_plan("arc_challenge", "LoopK3", load_config())
        self.assertEqual((plan["prompt_tokens"], plan["arc_continuation_tokens"], plan["arc_candidate_count"]),
                         (1242, 46, 4))
        self.assertIsNone(plan["max_new_tokens"])
        self.assertEqual(plan["expected_decode_forwards"], 0)

    def test_synthetic_tokens_exclude_special_and_padding(self):
        tokens = RESOURCE.ordinary_tokens(Tokenizer(), 2860)
        self.assertEqual(len(tokens), 2860)
        self.assertTrue(set(tokens).isdisjoint({0, 1}))

    def measure(self, model):
        config = load_config()
        plan = RESOURCE.resource_plan("mmlu_pro", "Native", config)
        stream, stages = io.StringIO(), {}
        with redirect_stdout(io.StringIO()):
            result = RESOURCE.measure_case(fake_torch(), model, Tokenizer(), None, plan, config, stream, stages)
        return result, stream, stages

    def test_native_real_call_options_and_scalar_all_layer_cache_trace(self):
        model = Model()
        result, stream, stages = self.measure(model)
        self.assertEqual(model.options, {"do_sample": False, "num_beams": 1, "use_cache": True,
                                        "min_new_tokens": 2048, "max_new_tokens": 2048})
        self.assertEqual(result["generated_token_count"], 2048)
        self.assertEqual((result["prefill_count"], result["decode_count"]), (1, 2047))
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(rows), 2048)
        self.assertEqual(rows[-1]["cache_lengths_by_layer"], [4907, 4907])
        self.assertEqual(stages["decode"]["forward_count"], 2047)
        self.assertIsNone(model.pre)
        self.assertIsNone(model.post)

    def test_early_stop_cannot_claim_2048_resource_shape(self):
        with self.assertRaisesRegex(ValueError, "actually reach the fixed 2048-token limit"):
            self.measure(Model(limit=3))

    def test_arc_runs_complete_native_token_requests_without_generation(self):
        model = Model()

        class Adapter:
            def _loglikelihood_tokens(self, requests, disable_tqdm):
                self.requests = requests
                self.phase11_positions = []
                for _, context, continuation in requests:
                    metadata = token_metadata(context, continuation, 262144)
                    self.phase11_positions.append(metadata)
                    model.forward(input_ids=IDs([list(metadata["actual_tokens"])]), use_cache=False)
                return [(-12.0, False)] * len(requests)

        adapter, stages, stream = Adapter(), {}, io.StringIO()
        plan = RESOURCE.resource_plan("arc_challenge", "Native", load_config())
        with redirect_stdout(io.StringIO()):
            result = RESOURCE.measure_case(fake_torch(), model, Tokenizer(), adapter, plan, load_config(), stream, stages)
        self.assertEqual([(len(c), len(a)) for _, c, a in adapter.requests], [(1242, 46)] * 4)
        self.assertEqual(result["candidate_input_tokens"], [1287] * 4)
        self.assertEqual(result["scored_continuation_tokens"], 184)
        self.assertEqual(stages["candidate_scoring"]["forward_count"], 4)
        self.assertIsNone(model.options)
        self.assertIsNone(adapter.phase11_identity)
        self.assertIsNone(adapter.phase11_runtime)

    def test_bad_layer_cache_is_detected_and_hooks_released(self):
        model = Model(bad_layer=True)
        with self.assertRaisesRegex(ValueError, "per-layer resource cache growth"):
            self.measure(model)
        self.assertIsNone(model.pre)
        self.assertIsNone(model.post)

    def test_source_mismatch_rejected_before_write_or_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fresh"
            args = RESOURCE.parser().parse_args(["--dataset", "mmlu_pro", "--family", "Native",
                       "--run-root", str(root), "--commit", "not-source-HEAD", "--dry-run"])
            with self.assertRaisesRegex(ValueError, "source Git HEAD differs"):
                RESOURCE.run(args)
            self.assertFalse(root.exists())

    def test_dry_run_has_no_side_effect_or_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fresh"
            args = RESOURCE.parser().parse_args(["--dataset", "gpqa_main", "--family", "Native",
                        "--run-root", str(root), "--commit", "expected", "--dry-run"])
            output = io.StringIO()
            with patch.object(RESOURCE, "source_provenance", return_value={"source_head": "expected"}), \
                 patch.object(RESOURCE, "load_runtime") as load, redirect_stdout(output):
                self.assertEqual(RESOURCE.run(args), 0)
            self.assertFalse(json.loads(output.getvalue())["model_loading"])
            self.assertFalse(root.exists())
            load.assert_not_called()

    def test_help_without_model_libraries(self):
        env = dict(os.environ, PYTHONPATH=str(ENTRY.parents[2] / "src"))
        result = subprocess.run([sys.executable, "-S", str(ENTRY), "--help"], env=env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--family", result.stdout)


if __name__ == "__main__":
    unittest.main()
