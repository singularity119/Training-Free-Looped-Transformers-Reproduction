import unittest

from tflt.audit import AuditCollector
from tflt.config import LoopConfig
from tflt.wrapper import apply_loop_wrapper


class AddLayer:
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, hidden_states, *args, **kwargs):
        return (hidden_states + self.delta,)


class ShapedValue:
    def __init__(self, value, seq_len):
        self.value = float(value)
        self.seq_len = int(seq_len)
        self.shape = (1, self.seq_len, 1)

    def __add__(self, delta):
        return ShapedValue(self.value + float(delta), self.seq_len)

    __radd__ = __add__

    def __sub__(self, other):
        return ShapedValue(self.value - float(other), self.seq_len)

    def __mul__(self, other):
        return ShapedValue(self.value * float(other), self.seq_len)

    __rmul__ = __mul__

    def __float__(self):
        return self.value

    def __eq__(self, other):
        try:
            return self.value == float(other)
        except Exception:
            return False


class FakeCache:
    def __init__(self, length):
        self.length = int(length)

    def get_seq_length(self, *args):
        return self.length


class FakeCachePosition:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def max(self):
        return max(self.values)

    def numel(self):
        return len(self.values)


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
            self.assertEqual(model.forward(ShapedValue(0.0, 1), past_key_value=object(), use_cache=True), 7.0)
        finally:
            handle.restore()
        counts = collector.summary()["counts"]
        self.assertEqual(counts["block_wrapper_forward"], 1)
        self.assertEqual(counts["bypass_true"], 1)
        self.assertEqual(counts.get("operator_body_calls", 0), 0)
        record = collector.summary()["forward_records"][0]
        self.assertEqual(record["hidden_seq_length"], 1)
        self.assertTrue(record["is_incremental_decode_step"])

    def test_qwen3_prefill_like_cache_does_not_bypass(self):
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
            model.forward(
                ShapedValue(0.0, 8),
                past_key_value=FakeCache(5),
                use_cache=True,
                cache_position=FakeCachePosition(range(8)),
            )
        finally:
            handle.restore()
        summary = collector.summary()
        counts = summary["counts"]
        self.assertEqual(counts["block_wrapper_forward"], 1)
        self.assertEqual(counts["bypass_false"], 1)
        self.assertEqual(counts["operator_body_calls"], 2)
        record = summary["forward_records"][0]
        self.assertEqual(record["hidden_seq_length"], 8)
        self.assertEqual(record["cache_position_length"], 8)
        self.assertFalse(record["is_incremental_decode_step"])

    def test_true_decode_like_cache_bypasses(self):
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
            model.forward(
                ShapedValue(0.0, 1),
                past_key_value=FakeCache(5),
                use_cache=True,
                cache_position=FakeCachePosition([5]),
            )
        finally:
            handle.restore()
        summary = collector.summary()
        counts = summary["counts"]
        self.assertEqual(counts["block_wrapper_forward"], 1)
        self.assertEqual(counts["bypass_true"], 1)
        self.assertEqual(counts.get("operator_body_calls", 0), 0)
        record = summary["forward_records"][0]
        self.assertEqual(record["hidden_seq_length"], 1)
        self.assertEqual(record["cache_position_length"], 1)
        self.assertTrue(record["is_incremental_decode_step"])

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
