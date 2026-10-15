import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tflt.loopscope.phase11_accuracy import generation_record
from tflt.loopscope.phase11_checkpoint import RecordStore
from tflt.loopscope.phase11_panel import load_config, write_json_once

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/loopscope"
sys.path.insert(0, str(SCRIPTS))
import phase11_d6 as d6
from build_phase11_d6_plan import build
from phase11_d6_batch import child_command, validate_task


def sample(key="cot_native"):
    cell = d6.selected_cells()[key]
    job = {k: cell[k] for k in ("cell_key", "cell_id", "prompt_mode", "damping_scope")}
    job.update(seed=20261015, synthetic=False, engineering=False, pool="/frozen/pool",
               run_root="/fresh/cell", source_commit="commit", indices=[0])
    metadata = d6.binding(job, "FORMAL_TEST", load_config(), cell, [0])
    row = {"task": "gpqa_main", "identity": "sample:0", "subject": "subject", "choices": ["one", "two", "three", "four"],
           "choice_labels": ["A", "B", "C", "D"], "input_ids": [12, 13]}
    record = generation_record(row, "Final answer: (A)", [24, 151645], [151645, 151643], 8192, 0)
    tag = {k: metadata[k] for k in d6.TAG_FIELDS}
    record["d6_binding"] = tag
    telemetry = {"prefill_count": 1, "decode_count": 1, "prefill_input_tokens": 2,
        "generated_token_count": 2, "cache_growth_closed": True, "incremental_generation": True,
        "d6_binding": tag, "identity": row["identity"], "canonical_index": 0}
    if cell["arm"] == "Online":
        multiplier = 1 if cell["damping_scope"] == "prefill_only" else 2
        telemetry["online"] = {"decode_fit_count": 0, "directions_released": True,
            "forward_count": 2, "callback_count": 2 * cell["k"],
            "callbacks_by_t": {str(t): 2 for t in range(cell["k"])},
            "damping_scope": cell["damping_scope"], "prefill_callback_count": cell["k"],
            "decode_callback_count": cell["k"],
            "prefill_applied_by_t": {str(t): int(cell["direction_policy"] == "current_t" or t > 0)
                                     for t in range(cell["k"])},
            "decode_applied_by_t": {str(t): int(cell["direction_policy"] == "current_t" or t > 0) *
                int(cell["damping_scope"] == "full") for t in range(cell["k"])},
            "applied_by_t": {str(t): multiplier * int(cell["direction_policy"] == "current_t" or t > 0)
                             for t in range(cell["k"])}}
    return job, metadata, {"rows": [row]}, record, telemetry


class ProducerTests(unittest.TestCase):
    def test_panel_and_plan_close_12_by448_exact_membership(self):
        recipe, formal, debug = build("/source", "/new/runs", "commit", "/inputs", "/synthetic")
        self.assertEqual(len(recipe["cells"]), 12)
        self.assertEqual(sum(validate_task(t) for t in formal["tasks"]), 5376)
        pairs = [(j["cell_key"], i) for t in formal["tasks"] for j in t["jobs"] for i in j["indices"]]
        self.assertEqual(len(set(pairs)), 5376)
        roots = [j["run_root"] for t in formal["tasks"] for j in t["jobs"]]
        self.assertEqual(len(roots), len(set(roots)))
        self.assertEqual(len(debug["tasks"][0]["jobs"]), 12)
        self.assertTrue(all(j["synthetic"] and j["engineering"] for j in debug["tasks"][0]["jobs"]))
        for cell in recipe["cells"]:
            self.assertEqual(cell["seed"], 20261015)
            self.assertEqual(cell["input_seed"], 20261008)
            self.assertEqual(sum(len(r["indices"]) for r in cell["raw_roots"]), 448)
            self.assertEqual(cell["cell"]["decode_mode"], "full")
            self.assertEqual(cell["cell"]["cache_strategy"], "first")

    def test_plan_and_job_reject_wrong_science_or_duplicate_membership(self):
        _, formal, _ = build("/source", "/new/runs", "commit", "/inputs", "/synthetic")
        task = copy.deepcopy(formal["tasks"][0])
        task["jobs"].append(dict(task["jobs"][0], run_root="/different"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_task(task)
        job, _, _, _, _ = sample()
        for field, value in (("seed", True), ("seed", 20261008), ("prompt_mode", "direct"), ("damping_scope", "full")):
            with self.subTest(field=field, value=value):
                bad = dict(job, **{field: value})
                with self.assertRaises(ValueError), patch.object(d6, "load_pool") as pool:
                    d6.load_job(bad)
                pool.assert_not_called()

    def test_all_scopes_preserve_strict_raw_and_actual_callback_validation(self):
        for key in d6.selected_cells():
            with self.subTest(key=key):
                _, metadata, pool, record, telemetry = sample(key)
                d6.validate_bundle(0, record, telemetry, metadata, pool)
                bad = copy.deepcopy(record)
                bad["d6_binding"]["seed"] = 20261008
                with self.assertRaisesRegex(ValueError, "binding"):
                    d6.validate_bundle(0, bad, telemetry, metadata, pool)
                if metadata["cell"]["arm"] == "Online":
                    bad = copy.deepcopy(telemetry)
                    bad["online"]["applied_by_t"]["1"] += 1
                    with self.assertRaisesRegex(ValueError, "scope"):
                        d6.validate_bundle(0, record, bad, metadata, pool)

    def test_success_resume_never_loads_model_or_overwrites_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, metadata, pool, record, telemetry = sample("cot_current_k2_prefill")
            job["run_root"] = metadata["run_root"] = tmp
            write_json_once(Path(tmp) / "command_args.json", metadata)
            store = RecordStore(tmp, [0], lambda i, r, t: d6.validate_bundle(i, r, t, metadata, pool))
            store.commit(0, record, telemetry)
            saved = (Path(tmp) / "complete/000000.json").read_bytes()
            job_file = Path(tmp) / "job.json"
            job_file.write_text(json.dumps(job))
            args = d6.parser().parse_args(["--job-file", str(job_file)])
            with patch.object(d6, "load_job", return_value=("FORMAL_TEST", load_config(), metadata["cell"], pool, [0])), \
                    patch.object(d6, "source_provenance", return_value={}), \
                    patch.dict("os.environ", {"SLURM_JOB_ID": "123"}), patch.object(d6, "load_runtime") as heavy:
                self.assertEqual(d6.run(args), 0)
            heavy.assert_not_called()
            self.assertEqual((Path(tmp) / "complete/000000.json").read_bytes(), saved)
            self.assertTrue(json.loads((Path(tmp) / "summary.json").read_text())["recovered_from_atomic_records"])

    def test_effective_seed_covers_python_numpy_torch_cuda(self):
        torch = SimpleNamespace(manual_seed=Mock(), cuda=SimpleNamespace(manual_seed_all=Mock()))
        numpy = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        with patch.dict("sys.modules", {"numpy": numpy}), patch.object(d6.random, "seed") as python_seed:
            actual = d6.seed_compute(torch)
        for callback in (python_seed, numpy.random.seed, torch.manual_seed, torch.cuda.manual_seed_all):
            callback.assert_called_once_with(20261015)
        self.assertEqual(actual, {k: 20261015 for k in ("python", "numpy", "torch", "cuda_all_devices")})

    def test_child_is_an_independent_d6_process_with_its_own_source(self):
        command, env = child_command({"producer_source": "/new/source"}, Path("/job.json"), Path("/controller"))
        self.assertEqual(command[1:], ["/controller/scripts/loopscope/phase11_d6.py", "--job-file", "/job.json"])
        self.assertEqual(env["PYTHONPATH"], "/new/source/src")


if __name__ == "__main__":
    unittest.main()
