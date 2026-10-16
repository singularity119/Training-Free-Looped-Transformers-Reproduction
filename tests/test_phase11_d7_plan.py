"""Input-only D7 membership and ordered execution contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/loopscope"))
from build_phase11_d7_plan import build, canary_indices, chunks, POPULATION
from phase11_d7_batch import FIRST_KEYS, LOOP_KEYS, validate_plan_barrier, validate_first_closure, child_command


class D7PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pool = self.root / "pool.json"
        self.pool.write_text(json.dumps({"rows": [{"input_ids": [1]} for _ in range(POPULATION)],
                                         "target_gold_loaded": False}))
        self.prefix = str(self.root / "runs")

    def make(self, kind, **kwargs):
        return build(kind, "/source", self.prefix, "commit", str(self.pool),
                     synthetic_pool="/synthetic.json", **kwargs)

    def save(self, name, content):
        file = self.root / name
        file.write_text(json.dumps(content))
        return str(file)

    def completed_canary(self):
        _, plan = self.make("canary", tasks=3)
        for cell in plan["cells"]:
            raw = cell["raw_roots"][0]
            root = Path(raw["run_root"])
            root.mkdir(parents=True)
            metadata = {k: cell[k] for k in ("cell_key", "cell_id", "pool", "source_commit",
                                             "seed", "prompt_mode", "damping_scope")}
            metadata.update(indices=raw["indices"], run_root=str(root))
            (root / "summary.json").write_text(json.dumps({"status": "RAW_COMPLETE", "count": 4,
                "metadata": metadata, "target_gold_loaded": False}))
        return self.save("canary-plan.json", plan)

    def completed_first(self):
        _, first = self.make("first", canary_plan=self.completed_canary(), packing=3)
        receipt = {"schema": "loopscope.phase11.d7.first_batch_raw_closure.v1",
                   "status": "FIRST_BATCH_RAW_CLOSED_GOLD_UNREAD", "target_gold_loaded": False,
                   "record_count": 36096, "cell_count": 3, "cell_keys": list(FIRST_KEYS),
                   "cells": copy.deepcopy(first["cells"])}
        return first, receipt

    def test_canary_quantiles_preserve_canonical_ties(self):
        pool = json.loads(self.pool.read_text())
        self.assertEqual(canary_indices(pool), [3007, 6015, 9023, 12031])
        pool["rows"][0]["input_ids"] = [1, 2]
        self.assertEqual(canary_indices(pool), [0, 3008, 6016, 9024])
        _, plan = self.make("canary", tasks=3)
        self.assertEqual(plan["scheduled_record_count"], 12)
        self.assertEqual(len(plan["tasks"]), 3)
        self.assertTrue(all(t["packing"] == 1 and len(t["jobs"]) == 1 for t in plan["tasks"]))
        self.assertEqual({c["cell_key"] for c in plan["cells"]}, set(FIRST_KEYS))

    def test_debug_is_sequential_synthetic_all_five(self):
        _, plan = self.make("debug", packing=5)
        task = plan["tasks"][0]
        self.assertEqual((task["partition"], task["packing"]), ("debug", 1))
        self.assertEqual(plan["record_count"], 5)
        self.assertTrue(all(j["indices"] == [0] and j["synthetic"] and j["engineering"]
                            for j in task["jobs"]))
        self.assertEqual({j["cell_key"] for j in task["jobs"]}, set(FIRST_KEYS + LOOP_KEYS))

    def test_first_reuses_successful_canary_without_repeat(self):
        canary = self.completed_canary()
        _, plan = self.make("first", canary_plan=canary, packing=4, shard_size=256)
        excluded = set(json.loads(Path(canary).read_text())["canary_indices"])
        self.assertEqual(plan["record_count"], 36096)
        self.assertEqual(plan["scheduled_record_count"], 36084)
        self.assertEqual(len(plan["tasks"]), 8)
        for task in plan["tasks"]:
            for job in task["jobs"]:
                self.assertGreaterEqual(len(job["indices"]), 64)
                self.assertFalse(set(job["indices"]) & excluded)
                self.assertEqual(job["seed"], 20261016)
        for cell in plan["cells"]:
            seen = [i for r in cell["raw_roots"] for i in r["indices"]]
            self.assertEqual(sorted(seen), list(range(POPULATION)))

    def test_last_requires_bound_raw_closure_at_build_and_execution(self):
        with self.assertRaises(ValueError):
            self.make("last")
        first, receipt = self.completed_first()
        first_path, receipt_path = self.save("first.json", first), self.save("closure.json", receipt)
        _, final = self.make("last", first_plan=first_path, first_closure=receipt_path, packing=2)
        self.assertEqual((final["record_count"], final["scheduled_record_count"]), (60160, 24064))
        self.assertEqual(len(final["cells"]), 5)
        self.assertEqual({j["cell_key"] for t in final["tasks"] for j in t["jobs"]}, set(LOOP_KEYS))
        validate_plan_barrier(final)
        receipt["cells"][0]["pool"] = "/other-pool"
        self.save("closure.json", receipt)
        with self.assertRaises(ValueError):
            validate_plan_barrier(final)
        with self.assertRaises(ValueError):
            validate_first_closure({**receipt, "record_count": 12}, first["cells"])

    def test_unfinished_canary_cannot_expand(self):
        _, canary = self.make("canary", tasks=3)
        with self.assertRaises(FileNotFoundError):
            self.make("first", canary_plan=self.save("canary.json", canary))
        with self.assertRaises(ValueError):
            self.make("canary", tasks=3, packing=2)
        with self.assertRaises(ValueError):
            self.make("first", tasks=9)

    def test_short_tail_is_merged_and_driver_is_d7(self):
        self.assertEqual([len(x) for x in chunks(list(range(260)), 256)], [260])
        cmd, env = child_command({"producer_source": "/producer"}, Path("/job.json"), Path("/controller"))
        self.assertEqual(cmd[1], "/controller/scripts/loopscope/phase11_d7.py")
        self.assertEqual(env["PYTHONPATH"], "/producer/src")


if __name__ == "__main__":
    unittest.main()
