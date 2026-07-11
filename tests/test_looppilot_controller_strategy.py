import unittest

from tflt.config import LoopConfig
from tflt.looppilot.controller import AlwaysLoop, NeverLoop
from tflt.strategies import run_loop, run_loop_controlled


class CountingOperator:
    def __init__(self):
        self.calls = 0

    def __call__(self, value):
        self.calls += 1
        return 2.0 * value + 2.0


class ControlledStrategyTest(unittest.TestCase):
    def setUp(self):
        self.config = LoopConfig(
            "qwen3-1.7b-base",
            (12, 15),
            k=2,
            iteration_mode="block",
            strategy="damped_euler",
            alpha=1.0,
            beta=0.0,
            cache_strategy="last",
            decode_mode="bypass",
        )

    def test_controller_none_preserves_legacy_path_without_events(self):
        legacy_operator = CountingOperator()
        controlled_operator = CountingOperator()
        events = []

        legacy = run_loop(legacy_operator, 0.0, self.config)
        actual = run_loop_controlled(
            controlled_operator,
            0.0,
            self.config,
            controller=None,
            emit=lambda event, payload: events.append((event, payload)),
        )

        self.assertEqual(actual, legacy)
        self.assertEqual(controlled_operator.calls, legacy_operator.calls)
        self.assertEqual(events, [])

    def test_always_loop_matches_legacy_k2_and_counts_two_body_calls(self):
        legacy_operator = CountingOperator()
        controlled_operator = CountingOperator()
        events = []

        legacy = run_loop(legacy_operator, 0.0, self.config)
        actual = run_loop_controlled(
            controlled_operator,
            0.0,
            self.config,
            controller=AlwaysLoop(),
            emit=lambda event, payload: events.append((event, payload)),
        )

        self.assertEqual(actual, legacy)
        self.assertEqual(controlled_operator.calls, 2)
        self.assertEqual(events[-1][0], "controller_health")
        self.assertEqual(events[-1][1]["operator_body_calls"], 2)
        self.assertEqual(events[-1][1]["k_used"], 2)

    def test_never_loop_returns_y0_not_x0_and_counts_one_body_call(self):
        operator = CountingOperator()
        events = []

        actual = run_loop_controlled(
            operator,
            3.0,
            self.config,
            controller=NeverLoop(),
            emit=lambda event, payload: events.append((event, payload)),
        )

        self.assertEqual(actual, 8.0)
        self.assertNotEqual(actual, 3.0)
        self.assertEqual(operator.calls, 1)
        self.assertEqual(events[-1][1]["operator_body_calls"], 1)
        self.assertEqual(events[-1][1]["k_used"], 1)

    def test_controlled_path_rejects_non_frozen_loop_contract(self):
        bad = LoopConfig("m", (12, 15), k=3, strategy="damped_euler")
        with self.assertRaises(ValueError):
            run_loop_controlled(CountingOperator(), 0.0, bad, controller=AlwaysLoop())

    def test_decision_uses_probe_built_after_first_required_body_call(self):
        operator = CountingOperator()

        class ObservingController:
            def decide(self, probe=None):
                self.test.assertEqual(operator.calls, 1)
                self.test.assertEqual(probe, {"x0": 3.0, "y0": 8.0})
                return NeverLoop().decide(probe)

        controller = ObservingController()
        controller.test = self
        actual = run_loop_controlled(
            operator,
            3.0,
            self.config,
            controller=controller,
            probe=lambda x0, y0: {"x0": x0, "y0": y0},
        )
        self.assertEqual(actual, 8.0)
        self.assertEqual(operator.calls, 1)


if __name__ == "__main__":
    unittest.main()
