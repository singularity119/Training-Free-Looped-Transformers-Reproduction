import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("d5_base_under_test", ROOT / "scripts/loopscope/phase11_d5_base.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def job(root, cid=base.CURRENT, family="p10-base-mmlu", repeat=1):
    commit = (base.P8 if cid == base.NATIVE else base.P8G if cid == base.LOOP else base.P10C) if family.endswith("mmlu") else base.P10D
    return {"family": family, "repeat": repeat, "cell_id": cid, "pool": "original-pool.json",
            "manifest": "original-manifest.json", "indices": [0, 1], "run_root": str(root),
            "source_commit": "controller-source", "producer_source": str(ROOT), "producer_commit": commit,
            "seed": {1: 20261014, 2: 20261015}[repeat], "synthetic": False, "engineering": True}


ROWS = [{"identity": "input:0", "prompt": "Question0 Answer:", "subject": "s", "doc_index": 0},
        {"identity": "input:1", "prompt": "Question1 Answer:", "subject": "s", "doc_index": 1}]


class FakeRuntime:
    env = {"test_runtime": True}
    manager = nullcontext()
    def __init__(self, fail=None):
        self.scored = []
        self.fail = fail
    def synchronize(self):
        pass
    def peak(self):
        return {"peak_allocated_bytes": 10, "peak_reserved_bytes": 20, "oom_count": 0}
    def score(self, row, index, dataset):
        self.scored.append(index)
        if self.fail == index:
            raise RuntimeError("interrupted question")
        return [-1., -2., -3., -4.], [{"position": 1, "input_length": 2,
            "continuation_length": 1, "left_truncated_tokens": 0} for _ in range(4)], {"contexts": 4}


class D5BaseTests(unittest.TestCase):
    def test_per_cell_source_and_new_repeat_seeds(self):
        for cid, expected in ((base.NATIVE, "p8"), (base.LOOP, "p8g"), (base.CURRENT, "p10")):
            for repeat in (1, 2):
                value = job("unused", cid, repeat=repeat)
                self.assertEqual(base.validate_job(value), (expected, [0, 1]))
        value = job("unused")
        value["seed"] = 20261002
        with self.assertRaisesRegex(ValueError, "effective seed"):
            base.validate_job(value)
        value = job("unused", base.NATIVE)
        value["producer_commit"] = base.P10C
        with self.assertRaisesRegex(ValueError, "producer commit"):
            base.validate_job(value)

    def test_compute_seed_covers_python_numpy_torch_cuda(self):
        torch = SimpleNamespace(manual_seed=Mock(), cuda=SimpleNamespace(manual_seed_all=Mock()))
        numpy = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        with patch.dict("sys.modules", {"numpy": numpy}), patch.object(base.random, "seed") as python_seed:
            base.seed_compute(torch, 20261015)
        python_seed.assert_called_once_with(20261015)
        numpy.random.seed.assert_called_once_with(20261015)
        torch.manual_seed.assert_called_once_with(20261015)
        torch.cuda.manual_seed_all.assert_called_once_with(20261015)

    def test_phase10_seeds_before_model_load_and_before_computation(self):
        value = job("unused")
        events = []
        torch = SimpleNamespace(set_grad_enabled=Mock(), initial_seed=lambda: 20261014,
            cuda=SimpleNamespace(synchronize=Mock(), reset_peak_memory_stats=Mock(),
                                 max_memory_allocated=lambda: 10, max_memory_reserved=lambda: 20))
        def load_runtime(*args):
            events.append("load_model")
            return torch, object(), object(), object(), None, {}
        legacy = SimpleNamespace(load_runtime=load_runtime, make_loop_config=Mock(return_value={}))
        wrapper = SimpleNamespace(looped_model=lambda *args: nullcontext())
        adapter = SimpleNamespace(score_choices=Mock())
        with patch.dict("sys.modules", {"torch": torch, "tflt.wrapper": wrapper,
                                        "tflt.loopscope.phase9_adapter": adapter}), \
             patch.dict(base.os.environ, {"SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": "debug"}), \
             patch.object(base.importlib.metadata, "version", return_value="0.4.11"), \
             patch.object(base, "load_file", return_value=legacy), \
             patch.object(base, "seed_compute", side_effect=lambda *args: events.append("seed")):
            runtime = base.HistoricalRuntime(value, "p10", "mmlu", {"arm": "Online"})
        self.assertEqual(events, ["seed", "load_model", "seed"])
        self.assertEqual(runtime.env["effective_rng_seeds"], {key: 20261014 for key in
            ("python", "numpy", "torch", "cuda_all_devices")})

    def test_interrupted_question_recovers_without_recomputing_success(self):
        with tempfile.TemporaryDirectory() as temp:
            value = job(temp)
            cell = {"cell_id": base.CURRENT}
            first, second = FakeRuntime(fail=1), FakeRuntime()
            with patch.object(base, "load_inputs", return_value=("mmlu", cell, ROWS)):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    base.run(value, runtime_factory=lambda *args: first)
                saved = Path(temp) / "complete/000000.json"
                original, timestamp = saved.read_bytes(), saved.stat().st_mtime_ns
                self.assertEqual(base.run(value, runtime_factory=lambda *args: second), 0)
                self.assertEqual(second.scored, [1])
                self.assertEqual(saved.read_bytes(), original)
                self.assertEqual(saved.stat().st_mtime_ns, timestamp)
                factory = Mock(side_effect=AssertionError("completed run must not load model"))
                self.assertEqual(base.run(value, runtime_factory=factory), 0)
                factory.assert_not_called()
            records = [json.loads(line) for line in (Path(temp) / "records.jsonl").read_text().splitlines()]
            self.assertEqual([r["canonical_index"] for r in records], [0, 1])
            self.assertTrue(all(r["seed"] == r["effective_seed"] == 20261014 for r in records))
            self.assertEqual((Path(temp) / "scores.jsonl").read_bytes(), (Path(temp) / "records.jsonl").read_bytes())

    def test_checkpoint_identity_or_repeat_mismatch_is_rejected(self):
        value = job("unused")
        validator = base.record_validator(value, ROWS, "mmlu")
        record = {**base.binding(value), "canonical_index": 0, "identity": "input:0",
                  "scores": [-1.] * 4, "continuation_token_lengths": [1] * 4}
        telemetry = {**base.binding(value), "canonical_index": 0, "identity": "input:0"}
        validator(0, record, telemetry)
        telemetry["repeat"] = 2
        with self.assertRaisesRegex(ValueError, "binding"):
            validator(0, record, telemetry)

    def test_phase10_engineering_routes_original_validation_scope(self):
        from tflt.loopscope import phase10_panel, phase10_accuracy
        value = job("unused")
        cell = {"cell_id": base.CURRENT, "new_in_phase10": True, "model": base.MODEL,
                "revision": base.REVISION, "dtype": "bfloat16", "cache_strategy": "first",
                "window": [15, 18], "k": 2, "arm": "Online"}
        with patch.object(base, "read_json", return_value={}), \
             patch.object(phase10_panel, "validate_score_manifest", return_value={base.CURRENT: cell}) as manifest, \
             patch.object(phase10_accuracy, "load_pool", return_value=ROWS) as pool:
            self.assertEqual(base.load_inputs(value, "p10", [0]), ("mmlu", cell, ROWS))
        self.assertEqual(manifest.call_args.kwargs["scope"], "PREFLIGHT_ONLY")
        self.assertEqual(pool.call_args.args[-1], "PREFLIGHT_ONLY")

    def test_synthetic_base_inputs_are_not_fake_scored(self):
        value = job("unused")
        value["synthetic"] = True
        with self.assertRaisesRegex(ValueError, "synthetic Base inputs are unsupported"):
            base.load_inputs(value, "p10", [0])


if __name__ == "__main__":
    unittest.main()
