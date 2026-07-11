import unittest

from tflt.audit import AuditCollector
from tflt.config import LoopConfig
from tflt.looppilot.controller import AlwaysLoop, NeverLoop
from tflt.wrapper import apply_loop_wrapper


class RecordingLayer:
    def __init__(self, delta):
        self.delta = delta
        self.inputs = []

    def __call__(self, hidden_states, *args, **kwargs):
        self.inputs.append(hidden_states)
        return (hidden_states + self.delta,)


class Inner:
    def __init__(self):
        self.layers = [RecordingLayer(1.0), RecordingLayer(2.0), RecordingLayer(4.0)]


class Model:
    def __init__(self):
        self.model = Inner()

    def forward(self, value):
        for layer in self.model.layers:
            value = layer(value)[0]
        return value


def config(controller=None, collector=None):
    return LoopConfig(
        "qwen3-1.7b-base",
        (0, 1),
        k=2,
        iteration_mode="block",
        strategy="damped_euler",
        alpha=1.0,
        beta=0.0,
        cache_strategy="last",
        decode_mode="full",
        controller=controller,
        audit_collector=collector,
    )


class RecordingSignalCollector:
    def __init__(self):
        self.calls = []

    def observe_operator_call(self, x, y, layer_updates):
        self.calls.append((x, y, list(layer_updates)))

    def controller_probe(self, x, y):
        return {"observed_calls": len(self.calls)}


class LoopPilotWrapperTest(unittest.TestCase):
    def test_none_controller_preserves_legacy_output_and_emits_no_controller_events(self):
        model = Model()
        collector = AuditCollector()
        handle = apply_loop_wrapper(model, config(None, collector))
        try:
            self.assertEqual(model.forward(0.0), 7.0)
        finally:
            handle.restore()
        event_names = [row["event"] for row in collector.events]
        self.assertNotIn("controller_decision", event_names)
        self.assertNotIn("controller_health", event_names)

    def test_always_loop_matches_legacy_and_health_closes_body_calls(self):
        legacy = Model()
        controlled = Model()
        collector = AuditCollector()
        legacy_handle = apply_loop_wrapper(legacy, config())
        controlled_handle = apply_loop_wrapper(controlled, config(AlwaysLoop(), collector))
        try:
            self.assertEqual(controlled.forward(0.0), legacy.forward(0.0))
        finally:
            legacy_handle.restore()
            controlled_handle.restore()
        summary = collector.summary()
        self.assertEqual(summary["counts"]["operator_body_calls"], 2)
        self.assertEqual(summary["counts"]["k_used_total"], 2)
        self.assertTrue(summary["controller_health_closed"])

    def test_never_loop_returns_normal_window_y0_and_stashes_from_x0(self):
        model = Model()
        collector = AuditCollector()
        first, second = model.model.layers[:2]
        handle = apply_loop_wrapper(model, config(NeverLoop(), collector))
        try:
            self.assertEqual(model.forward(0.0), 7.0)
        finally:
            handle.restore()
        self.assertEqual(collector.summary()["counts"]["operator_body_calls"], 1)
        self.assertEqual(collector.summary()["counts"]["k_used_total"], 1)
        self.assertEqual(first.inputs[-1], 0.0)
        self.assertEqual(second.inputs[-1], 1.0)

    def test_controlled_block_exposes_per_layer_updates_to_signal_collector(self):
        model = Model()
        signal_collector = RecordingSignalCollector()
        cfg = config(AlwaysLoop(), AuditCollector())
        object.__setattr__(cfg, "signal_collector", signal_collector)
        object.__setattr__(cfg, "controller_probe", signal_collector.controller_probe)
        handle = apply_loop_wrapper(model, cfg)
        try:
            model.forward(0.0)
        finally:
            handle.restore()
        self.assertEqual(len(signal_collector.calls), 2)
        self.assertEqual(signal_collector.calls[0][2], [1.0, 2.0])
        self.assertEqual(signal_collector.calls[1][2], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
