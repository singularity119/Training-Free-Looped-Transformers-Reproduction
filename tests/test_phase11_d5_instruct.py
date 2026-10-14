import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tflt.loopscope.phase11_accuracy import arc_record, generation_record
from tflt.loopscope.phase11_checkpoint import RecordStore
from tflt.loopscope.phase11_panel import load_config, write_json_once

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "loopscope"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("d5_instruct", SCRIPTS / "phase11_d5_instruct.py")
d5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d5)


def sample(dataset="arc_challenge", arm="Native"):
    suffix = "native" if arm == "Native" else "w15-18-k2-online-fixed-t0-lambda0.9"
    cell = d5.selected_cells(dataset)[dataset + "-q4-" + suffix]
    family = "p11-instruct-arc" if dataset == "arc_challenge" else "p11-instruct-gpqa"
    job = {"family": family, "repeat": 1, "cell_id": cell["cell_id"], "pool": "/frozen/pool",
           "indices": [0], "run_root": "/fresh/repeat1", "source_commit": "commit", "synthetic": False,
           "seed": 20261014}
    metadata = d5.binding(job, dataset, "FORMAL_TEST", load_config(), cell, [0])
    row = {"task": dataset, "identity": "sample:0", "subject": "subject", "choices": ["one", "two"],
           "choice_labels": ["A", "B"], "input_ids": [12, 13]}
    if dataset == "arc_challenge":
        record = arc_record(row, [-1., -2.], [{"continuation_length": 1}] * 2, 0)
        telemetry = {"candidate_count": 2, "left_truncated_count": 0, "incremental_generation": False}
    else:
        record = generation_record(row, "Final answer: (A)", [24, 151645], [151645, 151643], 8192, 0)
        telemetry = {"prefill_count": 1, "decode_count": 1, "prefill_input_tokens": 2,
                     "generated_token_count": 2, "cache_growth_closed": True, "incremental_generation": True}
    tag = {key: metadata[key] for key in ("family", "repeat", "cell_id", "source_commit", "seed")}
    record["d5_binding"] = tag
    telemetry.update(d5_binding=tag, identity=row["identity"], canonical_index=0)
    if arm == "Online":
        telemetry["online"] = {"decode_fit_count": 0, "directions_released": True}
    return job, metadata, {"rows": [row]}, record, telemetry


class D5InstructTests(unittest.TestCase):
    def test_exact_subset_and_original_cells(self):
        self.assertEqual(len(d5.selected_cells("arc_challenge")), 3)
        self.assertEqual(len(d5.selected_cells("gpqa_main")), 5)
        self.assertNotIn("arc_challenge-q4-w15-18-k3-loop", d5.selected_cells("arc_challenge"))
        self.assertNotIn("gpqa_main-q4-w15-18-k2-online-fixed-t0-lambda0.9", d5.selected_cells("gpqa_main"))

    def test_record_binding_scoring_and_lifecycle(self):
        for dataset, arm in (("arc_challenge", "Native"), ("arc_challenge", "Online"), ("gpqa_main", "Native")):
            with self.subTest(dataset=dataset, arm=arm):
                _, metadata, pool, record, telemetry = sample(dataset, arm)
                d5.validate_bundle(0, record, telemetry, metadata, pool)
                for field, value in (("repeat", 2), ("seed", 1), ("cell_id", "wrong")):
                    bad = copy.deepcopy(record)
                    bad["d5_binding"][field] = value
                    with self.assertRaisesRegex(ValueError, "binding"):
                        d5.validate_bundle(0, bad, telemetry, metadata, pool)
        _, metadata, pool, record, telemetry = sample("gpqa_main")
        telemetry["decode_count"] = 0
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            d5.validate_bundle(0, record, telemetry, metadata, pool)

    def test_atomic_success_resume_skips_model_and_recovers_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            job, metadata, pool, record, telemetry = sample()
            job["run_root"] = tmp
            metadata["run_root"] = tmp
            write_json_once(Path(tmp) / "command_args.json", metadata)
            store = RecordStore(tmp, [0], lambda i, r, t: d5.validate_bundle(i, r, t, metadata, pool))
            store.commit(0, record, telemetry)
            original = (Path(tmp) / "complete/000000.json").read_bytes()
            job_file = Path(tmp) / "job.json"
            job_file.write_text(json.dumps(job))
            args = d5.parser().parse_args(["--job-file", str(job_file)])
            with patch.object(d5, "load_job", return_value=("arc_challenge", "FORMAL_TEST", load_config(),
                    metadata["cell"], pool, [0])), patch.object(d5, "source_provenance", return_value={}), \
                    patch.dict("os.environ", {"SLURM_JOB_ID": "123"}), patch.object(d5, "load_runtime") as heavy:
                self.assertEqual(d5.run(args), 0)
                heavy.assert_not_called()
            self.assertEqual((Path(tmp) / "complete/000000.json").read_bytes(), original)
            self.assertTrue(json.loads((Path(tmp) / "summary.json").read_text())["recovered_from_atomic_records"])

    def test_reject_invalid_cell_and_unfrozen_seed_before_pool(self):
        job, _, _, _, _ = sample()
        job["cell_id"] = "arc_challenge-q4-w15-18-k3-loop"
        with self.assertRaisesRegex(ValueError, "subset"):
            d5.load_job(job)
        job, _, _, _, _ = sample()
        job["seed"] = True
        with self.assertRaisesRegex(ValueError, "seed"):
            d5.load_job(job)


if __name__ == "__main__":
    unittest.main()
