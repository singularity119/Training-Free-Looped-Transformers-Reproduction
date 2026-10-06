import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/loopscope/run_phase10_arc_bundle.py"
SPEC = importlib.util.spec_from_file_location("phase10_arc_bundle", SCRIPT)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def cell(name, policy=None, strength=None, arm="Online", model="model", k=2):
    return {"cell_id": name, "model": model, "window": [12, 15], "k": k,
            "arm": arm, "direction_policy": policy, "strength": strength}


class ArcBundleBudgetTests(unittest.TestCase):
    def test_eight_workers_run_first9_and_remaining59_without_slot_overlap(self):
        barrier = threading.Barrier(8)

        class RecordingBundle(bundle.Bundle):
            def __init__(self):
                super().__init__(SimpleNamespace(gpu_count=8), {}, list(map(str, range(8))))
                self.running_slots = set()
                self.calls = []

            def cell(self, cell_id, slot, deadline):
                with self.lock:
                    if slot in self.running_slots:
                        raise AssertionError("two cells overlap on one GPU")
                    self.running_slots.add(slot)
                    self.calls.append((cell_id, self.gpu_ids[slot]))
                if int(cell_id) < 8:
                    barrier.wait(timeout=5)
                with self.lock:
                    self.running_slots.remove(slot)
                return {"cell_id": cell_id, "status": "CELL_VERIFIED"}

        runner = RecordingBundle()
        self.assertTrue(runner.stage(list(map(str, range(9))), float("inf")))
        self.assertTrue(runner.stage(list(map(str, range(9, 68))), float("inf")))
        self.assertEqual(len(runner.calls), 68)
        self.assertEqual({value[0] for value in runner.calls}, set(map(str, range(68))))
        self.assertEqual({value[1] for value in runner.calls[:8]}, set(map(str, range(8))))
        self.assertFalse(runner.running_slots)

    def test_eight_gpu_forecast_charges_idle_tail_and_full_allocation(self):
        cells = [cell("first", "fixed_t0", .1)] + [
            cell(str(i), "fixed_t0", .2) for i in range(9)]
        value = bundle.forecast_remaining(cells, {"first": 10.}, 8)
        self.assertEqual(value["makespan_seconds"], 20.)
        self.assertAlmostEqual(value["forecast_gpu_hours"], 160. / 3600.)
        admission = bundle.budget_admission(.2, 3600., 8, 10 * 3600., 12 * 3600.)
        self.assertFalse(admission["admitted"])
        self.assertGreater(admission["projected_gpu_hours"], 100.)

    def test_nonpositive_or_noninteger_worker_count_is_rejected(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                bundle.validate_gpu_count(value)

    def test_real_launcher_needs_no_compute_node_slurm_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            script = repo / "scripts/loopscope/run_phase10_arc_bundle.py"
            script.parent.mkdir(parents=True)
            script.write_text("import json,sys; print(json.dumps(sys.argv[1:]))\n")
            launcher = SCRIPT.with_name("phase10_arc_bundle.sbatch")
            command = ["/bin/bash", str(launcher), str(repo), sys.executable,
                       "manifest", "pool", "plan", "run", "synthetic", "PREFLIGHT_ONLY",
                       "2", ".065", ".48333333333333334", "indices"]
            env = {**os.environ, "PATH": str(repo / "no-executables"),
                   "SLURM_JOB_ID": "123", "SLURM_JOB_START_TIME": "1700000000"}
            env.pop("DRY_RUN", None)
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(result.stdout)
            self.assertEqual(args[args.index("--allocation-start-epoch") + 1], "1700000000")
            self.assertIn("source=SLURM_JOB_START_TIME", result.stderr)
            env.pop("SLURM_JOB_START_TIME")
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("Slurm allocation start epoch missing", result.stderr)

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
