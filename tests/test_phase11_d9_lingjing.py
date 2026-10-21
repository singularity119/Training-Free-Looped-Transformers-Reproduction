import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import copy
import json
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/loopscope"))
from phase11_d9_lingjing import make_plan, topology, child_environment, launch
from phase11_d9_lingjing_worker import allocation, direct_full_waiver, metadata as actual_metadata
from tflt.loopscope.phase11_d8 import CELLS, actual_seed
import analyze_phase11_d9 as scorer
from phase11_d8 import metadata
from tflt.loopscope.phase11_d8 import validate_job


class LingjingD9Tests(unittest.TestCase):
    def test_user_direct_full_is_explicit_and_keeps_actual_revision_unverified(self):
        provenance = {"mode": "USER_DESIGNATED_OSS_UNVERIFIED_REVISION", "model_mount": "/oss/model",
                      "actual_revision": "UNVERIFIED_OSS_COPY", "actual_tokenizer_revision": "UNVERIFIED_OSS_COPY"}
        admission = {"direct_full_user_authorization": True, "model_provenance": provenance,
                     "preflight_passed": False, "capacity": {"headroom_fraction": None}}
        self.assertTrue(direct_full_waiver(admission, "/oss/model"))
        self.assertFalse(direct_full_waiver(dict(admission, direct_full_user_authorization=False), "/oss/model"))
        self.assertFalse(direct_full_waiver(admission, "/other/model"))
        job = make_plan("/cloud/formal", "/cloud/pool", "source")["registry"]["jobs"][0]
        cell = validate_job(job)
        pool = {"think_open_id": 1, "think_close_id": 2, "d9_model_provenance": provenance}
        meta = actual_metadata(job, cell, pool)
        self.assertEqual(meta["revision"], "UNVERIFIED_OSS_COPY")
        self.assertEqual(meta["cell"]["revision"], "UNVERIFIED_OSS_COPY")
        self.assertNotEqual(meta["recipe_reference_revision"], "UNVERIFIED_OSS_COPY")
        self.assertFalse(admission["preflight_passed"])
        self.assertIsNone(admission["capacity"]["headroom_fraction"])

    def test_eight_workers_cover_original_three_cells_without_duplicates(self):
        plan = make_plan("/cloud/fresh", "/cloud/thinking-pool.json", "source-revision")
        jobs = plan["registry"]["jobs"]
        pairs = [(j["cell_key"], i) for j in jobs for i in j["indices"]]
        self.assertEqual(len(pairs), 1344)
        self.assertEqual(set(pairs), {(key, i) for key in CELLS for i in range(448)})
        self.assertEqual(len(jobs), 24)
        self.assertEqual(plan["total_gpus"], 8)
        for job in jobs:
            self.assertEqual(job["indices"], sorted(job["indices"]))
            self.assertEqual([actual_seed(i) for i in job["indices"]], [20261018+i for i in job["indices"]])

    def test_two_node_leaders_and_child_gpu_isolation(self):
        for total in (8, 16):
            plan = make_plan("/cloud/fresh", "/cloud/pool", "revision", total)
            for rank in range(total):
                env = {"RANK": str(rank), "LOCAL_RANK": str(rank%8), "WORLD_SIZE": str(total), "LOCAL_WORLD_SIZE": "8"}
                self.assertEqual(topology(env, plan), (rank, rank%8, rank//8))
        child = child_environment(dict(env, MASTER_ADDR="host", TORCHELASTIC_RUN_ID="id"), "GPU-uuid", 8, 11)
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "GPU-uuid")
        self.assertEqual(child["D9_WORKER_RANK"], "11")
        self.assertFalse({"RANK", "WORLD_SIZE", "MASTER_ADDR", "TORCHELASTIC_RUN_ID"} & child.keys())

    def test_preflight_uses_own_synthetic_scope(self):
        plan = make_plan("/cloud/smoke", "/cloud/synthetic", "revision", 1, True)
        self.assertEqual(len(plan["registry"]["jobs"]), 3)
        self.assertTrue(all(j["synthetic"] and j["engineering"] and j["indices"] == [0]
                            for j in plan["registry"]["jobs"]))
        # Exercise the actual eight-card rank routing before formal deployment.
        eight = make_plan("/cloud/smoke8", "/cloud/synthetic", "revision", 8, True)
        self.assertEqual(eight["gpus_per_node"], 8)
        self.assertEqual(len(eight["registry"]["jobs"]), 24)
        self.assertTrue(all(j["synthetic"] and j["engineering"] and j["indices"] == [0]
                            for j in eight["registry"]["jobs"]))
        self.assertEqual(topology({"RANK": "7", "LOCAL_RANK": "7", "WORLD_SIZE": "8",
                                   "LOCAL_WORLD_SIZE": "8"}, eight), (7, 7, 0))
        sixteen = make_plan("/cloud/smoke16", "/cloud/synthetic", "revision", 16, True)
        self.assertEqual(len(sixteen["registry"]["jobs"]), 48)
        self.assertEqual(topology({"RANK": "8", "LOCAL_RANK": "0", "WORLD_SIZE": "16",
                                   "LOCAL_WORLD_SIZE": "8"}, sixteen), (8, 0, 1))

    def test_allocation_records_platform_without_fake_slurm(self):
        self.assertEqual(allocation({"D9_PLATFORM_JOB_ID": "real-job-id"})["platform"], "lingjing")
        with self.assertRaisesRegex(ValueError, "real Lingjing"):
            allocation({})
        with self.assertRaisesRegex(ValueError, "Lingjing allocation"):
            allocation({"D9_PLATFORM_JOB_ID": "id", "SLURM_JOB_ID": "invented"})

    def test_closure_restores_all_canonical_indices_and_rejects_a_missing_item(self):
        registry = make_plan("/cloud/fresh", "/cloud/pool", "revision")["registry"]
        pool = {"rows": [{"identity": f"synthetic:{i}"} for i in range(448)],
                "test_identities": [f"synthetic:{i}" for i in range(448)],
                "think_open_id": 1, "think_close_id": 2}
        facts = {"/cloud/pool": pool}
        for job in registry["jobs"]:
            root = Path(job["run_root"])
            facts[str(root / "summary.json")] = {"status": "RAW_COMPLETE", "count": len(job["indices"]),
                "metadata": metadata(job, validate_job(job), pool)}
            for i in job["indices"]:
                facts[str(root / "complete" / f"{i:06d}.json")] = {"record": {
                    "identity": f"synthetic:{i}", "canonical_index": i, "generated_token_count": 10,
                    "end_reason": "eos", "actual_seed": actual_seed(i), "final_span": {"prediction_index": 0}},
                    "telemetry": {"elapsed_seconds": 1, "peak_allocated_bytes": 1, "peak_reserved_bytes": 1}}
        with patch.object(scorer, "read", side_effect=lambda p: facts[str(p)]), \
             patch.object(scorer, "validate_pool", side_effect=lambda p: p), \
             patch.object(scorer, "validate_bundle"):
            data, _ = scorer.load_all(registry, object())
            self.assertEqual([r["canonical_index"] for r in data[CELLS[0]]], list(range(448)))
            broken = copy.deepcopy(registry)
            last = broken["jobs"][-1]
            last["indices"] = last["indices"][:-1]
            facts[str(Path(last["run_root"]) / "summary.json")] = {
                "status": "RAW_COMPLETE", "count": len(last["indices"]),
                "metadata": metadata(last, validate_job(last), pool)}
            with self.assertRaisesRegex(ValueError, "448 identities missing"):
                scorer.load_all(broken, object())

    def test_gold_cannot_be_passed_to_prediction_closure(self):
        with self.assertRaisesRegex(ValueError, "gold is allowed only"):
            scorer.run(SimpleNamespace(registry="registry", closure=None, gold="sealed-gold"))

    def test_both_platform_leaders_spawn_their_eight_workers_and_track_memory(self):
        class Process:
            def __init__(self, sampler=False):
                self.sampler = sampler
                self.returncode = None if sampler else 0
            def poll(self):
                return self.returncode
            def terminate(self):
                self.returncode = 0
            def wait(self):
                return self.returncode
        spawned = []
        def popen(command, **kwargs):
            spawned.append((command, kwargs))
            return Process(command[0] == "nvidia-smi")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_plan("/cloud/fresh", "/cloud/pool", "revision", 16)
            plan_file = root / "plan.json"
            plan_file.write_text(json.dumps(plan))
            for outer in (0, 8):
                env = {"RANK": str(outer), "LOCAL_RANK": "0", "WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8",
                       "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
                with patch.dict(os.environ, env, clear=True), patch("phase11_d9_lingjing.subprocess.Popen", side_effect=popen):
                    launch(plan_file, "/model", "/admission", root / "launch", "real-job", 60)
            workers = [(c, k) for c, k in spawned if c[0] != "nvidia-smi"]
            self.assertEqual(len(workers), 16)
            self.assertEqual({int(k["env"]["D9_WORKER_RANK"]) for c, k in workers}, set(range(16)))
            self.assertTrue((root / "launch/node-0/complete.json").exists())
            self.assertTrue((root / "launch/node-1/complete.json").exists())


if __name__ == "__main__":
    unittest.main()
