import unittest

from tflt.audit import AuditCollector
from tflt.config import LoopConfig
from tflt.wrapper import apply_loop_wrapper


class AddLayer:
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, hidden_states, *args, **kwargs):
        return (hidden_states + self.delta,)


class DummyInner:
    def __init__(self):
        self.layers = [AddLayer(1.0), AddLayer(2.0), AddLayer(4.0)]


class DummyModel:
    def __init__(self):
        self.model = DummyInner()

    def forward(self, x, **kwargs):
        result = x
        for layer in self.model.layers:
            result = layer(result, **kwargs)[0]
        return result


class AuditTest(unittest.TestCase):
    def test_block_audit_counts_body_stash_and_identity(self):
        model = DummyModel()
        collector = AuditCollector()
        cfg = LoopConfig(
            "dummy",
            (0, 1),
            k=2,
            iteration_mode="block",
            strategy="damped_euler",
            decode_mode="full",
            audit_collector=collector,
        )
        handle = apply_loop_wrapper(model, cfg)
        try:
            self.assertEqual(model.forward(0.0), 7.0)
        finally:
            handle.restore()
        counts = collector.summary()["counts"]
        self.assertEqual(counts["block_wrapper_forward"], 1)
        self.assertEqual(counts["operator_body_calls"], 2)
        self.assertEqual(counts["stash_passes"], 1)
        self.assertEqual(counts["identity_layer_forwards"], 1)
        self.assertEqual(counts["bypass_false"], 1)
        self.assertGreater(collector.summary()["tensor_diffs"]["g_minus_x"][0]["max_abs"], 0.0)

    def test_decode_bypass_audit_counts_bypass(self):
        model = DummyModel()
        collector = AuditCollector()
        cfg = LoopConfig(
            "dummy",
            (0, 1),
            k=2,
            iteration_mode="block",
            strategy="damped_euler",
            decode_mode="bypass",
            audit_collector=collector,
        )
        handle = apply_loop_wrapper(model, cfg)
        try:
            self.assertEqual(model.forward(0.0, past_key_value=object(), use_cache=True), 7.0)
        finally:
            handle.restore()
        counts = collector.summary()["counts"]
        self.assertEqual(counts["block_wrapper_forward"], 1)
        self.assertEqual(counts["bypass_true"], 1)
        self.assertEqual(counts.get("operator_body_calls", 0), 0)

    def test_restore_recovers_dummy_output(self):
        model = DummyModel()
        before = model.forward(0.0)
        collector = AuditCollector()
        cfg = LoopConfig("dummy", (0, 1), iteration_mode="block", audit_collector=collector)
        handle = apply_loop_wrapper(model, cfg)
        self.assertNotEqual(model.model.layers[0].__class__.__name__, "AddLayer")
        handle.restore()
        self.assertEqual(model.forward(0.0), before)
        self.assertEqual(model.model.layers[0].__class__.__name__, "AddLayer")


if __name__ == "__main__":
    unittest.main()
