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
import phase11_d7 as d7


def sample(key="native"):
    cell = d7.selected_cells()[key]
    job = {k: cell[k] for k in ("cell_key", "cell_id", "prompt_mode", "damping_scope")}
    job.update(seed=20261016, synthetic=False, engineering=False, pool="/frozen/pool",
               run_root="/fresh/cell", source_commit="commit", indices=[0])
    metadata = d7.binding(job, "FORMAL_TEST", load_config(), cell, [0])
    row = {"task": "mmlu_pro", "identity": "sample:0", "subject": "subject", "choices": ["one", "two", "three", "four"],
           "choice_labels": ["A", "B", "C", "D"], "input_ids": [12, 13]}
    record = generation_record(row, "Final answer: (A)", [24, 151645], [151645, 151643], 8192, 0)
    tag = {k: metadata[k] for k in d7.TAG_FIELDS}
    record["d7_binding"] = tag
    telemetry = {"prefill_count": 1, "decode_count": 1, "prefill_input_tokens": 2,
        "generated_token_count": 2, "cache_growth_closed": True, "incremental_generation": True,
        "d7_binding": tag, "identity": row["identity"], "canonical_index": 0}
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
    def test_plan_and_job_reject_wrong_science_or_duplicate_membership(self):
        job, _, _, _, _ = sample()
        for field, value in (("seed", True), ("seed", 20261008), ("prompt_mode", "direct"), ("damping_scope", "full")):
            with self.subTest(field=field, value=value):
                bad = dict(job, **{field: value})
                with self.assertRaises(ValueError), patch.object(d7, "load_pool") as pool:
                    d7.load_job(bad)
                pool.assert_not_called()

    def test_all_scopes_preserve_strict_raw_and_actual_callback_validation(self):
        for key in d7.selected_cells():
            with self.subTest(key=key):
                _, metadata, pool, record, telemetry = sample(key)
                d7.validate_bundle(0, record, telemetry, metadata, pool)
                bad = copy.deepcopy(record)
                bad["d7_binding"]["seed"] = 20261008
                with self.assertRaisesRegex(ValueError, "binding"):
                    d7.validate_bundle(0, bad, telemetry, metadata, pool)
                if metadata["cell"]["arm"] == "Online":
                    bad = copy.deepcopy(telemetry)
                    bad["online"]["applied_by_t"]["1"] += 1
                    with self.assertRaisesRegex(ValueError, "scope"):
                        d7.validate_bundle(0, record, bad, metadata, pool)

    def test_success_resume_never_loads_model_or_overwrites_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, metadata, pool, record, telemetry = sample("current_t_k2_l05")
            job["run_root"] = metadata["run_root"] = tmp
            write_json_once(Path(tmp) / "command_args.json", metadata)
            store = RecordStore(tmp, [0], lambda i, r, t: d7.validate_bundle(i, r, t, metadata, pool))
            store.commit(0, record, telemetry)
            saved = (Path(tmp) / "complete/000000.json").read_bytes()
            job_file = Path(tmp) / "job.json"
            job_file.write_text(json.dumps(job))
            args = d7.parser().parse_args(["--job-file", str(job_file)])
            with patch.object(d7, "load_job", return_value=("FORMAL_TEST", load_config(), metadata["cell"], pool, [0])), \
                    patch.object(d7, "source_provenance", return_value={}), \
                    patch.dict("os.environ", {"SLURM_JOB_ID": "123"}), patch.object(d7, "load_runtime") as heavy:
                self.assertEqual(d7.run(args), 0)
            heavy.assert_not_called()
            self.assertEqual((Path(tmp) / "complete/000000.json").read_bytes(), saved)
            self.assertTrue(json.loads((Path(tmp) / "summary.json").read_text())["recovered_from_atomic_records"])

    def test_effective_seed_covers_python_numpy_torch_cuda(self):
        torch = SimpleNamespace(manual_seed=Mock(), cuda=SimpleNamespace(manual_seed_all=Mock()))
        numpy = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        with patch.dict("sys.modules", {"numpy": numpy}), patch.object(d7.random, "seed") as python_seed:
            actual = d7.seed_compute(torch)
        for callback in (python_seed, numpy.random.seed, torch.manual_seed, torch.cuda.manual_seed_all):
            callback.assert_called_once_with(20261016)
        self.assertEqual(actual, {k: 20261016 for k in ("python", "numpy", "torch", "cuda_all_devices")})



if __name__ == "__main__":
    unittest.main()
