"""Local torch-free checks of the protected opt-in and request context."""

import json
from pathlib import Path
import tempfile
import unittest

from tflt.config import LoopConfig
from tflt.strategies import run_loop
from tflt.loopscope.phase8_runtime import (
    AnswerResidualCollector, Phase8Runtime, load_basis, save_basis,
)


class FakeDelta:
    ndim = 3
    shape = (1, 3, 2)

    def __getitem__(self, index):
        return (2.0, 3.0)


class Phase8RuntimeTests(unittest.TestCase):
    def config(self, **kwargs):
        return LoopConfig("synthetic", (1, 2), **kwargs)

    def test_default_and_zero_callback_preserve_euler(self):
        operator = lambda x: 2 * x + 3
        for k in (2, 4):
            original = run_loop(operator, 1.0, self.config(k=k))
            seen = []
            def zero(delta, t):
                seen.append((delta, t))
                return delta
            observed = run_loop(operator, 1.0, self.config(k=k, residual_transform=zero))
            self.assertEqual(original, observed)
            self.assertEqual([t for _, t in seen], list(range(k)))
            self.assertEqual(seen[0][0], 4.0)

    def test_callback_runs_after_residual_before_euler_update(self):
        seen = []
        def transform(delta, t):
            seen.append((delta, t))
            return delta if t == 0 else 0.5 * delta
        value = run_loop(lambda x: 2 * x, 2., self.config(residual_transform=transform))
        self.assertEqual(seen, [(2., 0), (3., 1)])
        self.assertEqual(value, 3.75)

    def test_k1_never_calls_callback(self):
        def forbidden(delta, t):
            self.fail("K1 must retain native branch")
        self.assertEqual(run_loop(lambda x: x + 4, 2., self.config(k=1, residual_transform=forbidden)), 6.)

    def test_optin_only_accepts_euler_block(self):
        for override in ({"strategy": "naive"}, {"iteration_mode": "layer"}):
            with self.assertRaises(ValueError):
                self.config(residual_transform=lambda x, t: x, **override)
        self.assertIsNone(self.config(strategy="naive").residual_transform)

    def test_request_context_and_exact_zero_path(self):
        delta = FakeDelta()
        runtime = Phase8Runtime([1., 0.], strength=0)
        with self.assertRaises(RuntimeError):
            runtime(delta, 0)
        with runtime.context({"subject": "x", "doc_index": 2}, 1):
            self.assertIs(runtime(delta, 0), delta)
            self.assertIs(runtime(delta, 1), delta)
        self.assertIsNone(runtime._active)
        self.assertEqual([x["position"] for x in runtime.calls], [1, 1])
        self.assertFalse(any(x["applied"] for x in runtime.calls))

    def test_collection_only_path_keeps_raw_vector_and_t(self):
        class Collector:
            def __init__(self):
                self.rows = []
            def collect(self, *args):
                self.rows.append(args)
        collector = Collector()
        runtime = Phase8Runtime(collector=collector)
        delta = FakeDelta()
        with runtime.context("id1", 2):
            self.assertIs(runtime(delta, 1), delta)
        self.assertEqual(collector.rows, [("id1", 1, (2., 3.), 2)])

    def test_collector_declares_bounded_identity_and_t_contract(self):
        collector = AnswerResidualCollector(["id1", "id2"], 2)
        with self.assertRaises(ValueError):
            collector.collect("outside", 1, None, 0)
        with self.assertRaises(ValueError):
            collector.collect("id1", 2, None, 0)
        # Duplicate choice requests must not materialize another row.
        collector.rows[('"id1"', 1)] = object()
        collector.collect("id1", 1, None, 0)
        self.assertEqual(len(collector.rows), 1)
        self.assertEqual(collector.duplicate_calls, 1)

    def test_basis_roundtrip_and_provenance_match(self):
        basis = {"schema": "loopscope-phase8-basis-v1", "k": 2,
                 "metadata": {"model": "synthetic", "window": [1, 2], "k": 2},
                 "direction": [0.6, 0.8]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basis.json"
            save_basis(path, basis)
            self.assertEqual(load_basis(path, basis["metadata"]), basis)
            with self.assertRaises(FileExistsError):
                save_basis(path, basis)
            with self.assertRaises(ValueError):
                load_basis(path, {"k": 4})


if __name__ == "__main__":
    unittest.main()
