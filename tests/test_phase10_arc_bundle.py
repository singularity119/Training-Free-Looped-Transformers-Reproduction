import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/loopscope/run_phase10_arc_bundle.py"
SPEC = importlib.util.spec_from_file_location("phase10_arc_bundle", SCRIPT)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def cell(name, policy=None, strength=None, arm="Online", model="model", k=2):
    return {"cell_id": name, "model": model, "window": [12, 15], "k": k,
            "arm": arm, "direction_policy": policy, "strength": strength}


class ArcBundleBudgetTests(unittest.TestCase):
    def test_forecast_uses_measured_policy_and_conservative_loop_then_two_workers(self):
        cells = [cell("fixed-first", "fixed_t0", .1), cell("current-first", "current_t", .1),
                 cell("fixed-remaining", "fixed_t0", .2),
                 cell("current-remaining", "current_t", .2), cell("loop", arm="Loop")]
        value = bundle.forecast_remaining(cells, {"fixed-first": 10., "current-first": 20.}, 2)
        self.assertEqual(value["durations_seconds"],
                         {"fixed-remaining": 10., "current-remaining": 20., "loop": 20.})
        self.assertEqual(value["makespan_seconds"], 30.)
        self.assertAlmostEqual(value["forecast_gpu_hours"], 60. / 3600.)
        self.assertEqual(set(value["order"]), {"fixed-remaining", "current-remaining", "loop"})

    def test_missing_model_measurement_cannot_borrow_other_model_cost(self):
        cells = [cell("measured", "fixed_t0", .1),
                 cell("other", "fixed_t0", .2, model="other-model")]
        with self.assertRaises(ValueError):
            bundle.forecast_remaining(cells, {"measured": 10.}, 2)

    def test_two_gpu_idle_time_is_charged_in_admission(self):
        value = bundle.budget_admission(1., 20 * 3600., 2, 3600., 49 * 3600.)
        self.assertTrue(value["admitted"])
        self.assertGreaterEqual(value["projected_gpu_hours"], 1. + 40. + 2.4)
        self.assertEqual(value["remaining_walltime_seconds"], 29 * 3600.)

    def test_remaining_forecast_above_hard_budget_stops(self):
        value = bundle.budget_admission(1., 3600., 2, 41 * 3600., 49 * 3600.)
        self.assertFalse(value["admitted"])
        self.assertGreater(value["projected_gpu_hours"], 100.)

    def test_safe_gpu_hour_total_still_stops_when_allocation_time_is_short(self):
        value = bundle.budget_admission(.5, 1000., 2, 1900., 3000.)
        self.assertLess(value["projected_gpu_hours"], 100.)
        self.assertFalse(value["admitted"])
        self.assertGreater(value["required_walltime_seconds"], value["remaining_walltime_seconds"])


if __name__ == "__main__":
    unittest.main()
