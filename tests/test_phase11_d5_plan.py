import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/loopscope"
sys.path.insert(0, str(SCRIPTS))
from build_phase11_d5_plan import build
from phase11_d5_batch import validate_task


class PlanTest(unittest.TestCase):
    def test_full_fresh_membership(self):
        recipe, formal, debug = build("/stage", "/runs/new", "source")
        self.assertEqual(sum(validate_task(t) for t in formal["tasks"]), 102796)
        roots = [j["run_root"] for t in formal["tasks"] for j in t["jobs"]]
        self.assertEqual(len(roots), len(set(roots)))
        self.assertEqual(sum(len(t["jobs"]) for t in debug["tasks"]), 28)
        for j in recipe["cells"]:
            self.assertEqual(j["seed"], {1: 20261014, 2: 20261015}[j["repeat"]])
        self.assertTrue(all(j["engineering"] for t in debug["tasks"] for j in t["jobs"]))

    def test_duplicate_rejected(self):
        _, formal, _ = build("/stage", "/runs/new", "source")
        task = formal["tasks"][0]
        task["jobs"].append(dict(task["jobs"][0], run_root="/another-root"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_task(task)


if __name__ == "__main__":
    unittest.main()
