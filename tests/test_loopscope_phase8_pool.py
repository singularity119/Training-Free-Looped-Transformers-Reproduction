import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tflt.loopscope.phase8_pool import (
    DATASET_REVISION, SEED, _load_safe_dataset, identity,
    select_calibration, select_debug, write_debug_pool,
)


class Phase8PoolTests(unittest.TestCase):
    def test_proportional_allocation_and_one_shared_rng(self):
        counts = {"zebra": 13, "algebra": 7, "biology": 10}
        rows = select_calibration(counts, count=11)
        # Integer quotas: 2,3,4; two remainder seats go to zebra/biology.
        expected_quota = {"algebra": 2, "biology": 4, "zebra": 5}
        rng = random.Random(SEED)
        expected = [(s, i) for s in sorted(counts)
                    for i in sorted(rng.sample(range(counts[s]), expected_quota[s]))]
        self.assertEqual([(r["subject"], r["doc_index"]) for r in rows], expected)
        self.assertEqual(rows, select_calibration(dict(reversed(list(counts.items()))), count=11))
        self.assertTrue(all(DATASET_REVISION in row["identity"] for row in rows))

    def test_remainder_tie_is_subject_order(self):
        rows = select_calibration({"c": 10, "a": 10, "b": 10}, count=5)
        self.assertEqual({s: sum(r["subject"] == s for r in rows) for s in ("a", "b", "c")},
                         {"a": 2, "b": 2, "c": 1})

    def test_debug_uses_both_tokenizers_and_excludes_fit(self):
        rows = [{"identity": identity("math", i), "subject": "math", "doc_index": i,
                 "prompt_token_lengths": {"m1": 100 if i == 0 else i,
                                          "m2": 90 if i == 4 else i}}
                for i in range(10)]
        selected = select_debug(rows, rows[:4], ["m1", "m2"])
        self.assertEqual([r["doc_index"] for r in selected], [0, 1, 2, 3, 4, 9, 8, 7])
        self.assertEqual([r["role"] for r in selected], ["fit"] * 4 + ["verify"] * 4)

    def test_target_column_projection_before_row_access(self):
        calls = []
        class Dataset:
            def __init__(self, split, columns=None):
                self.split = split
                self.info = SimpleNamespace(download_checksums={"hf://datasets/cais/mmlu@" + DATASET_REVISION + "/" + split: {}})
                self.column_names = columns or ["question", "subject", "choices", "answer"]
                self.cache_files = [{"filename": "/cache/" + split + ".arrow"}]
            def select_columns(self, columns):
                calls.append(("projection", self.split, columns))
                return Dataset(self.split, columns)
            def __len__(self):
                return 5
            def __iter__(self):
                raise AssertionError("loader must not materialize source rows")
        def load_dataset(*args, **kwargs):
            calls.append(("load", kwargs))
            return Dataset(kwargs["split"])
        fake = SimpleNamespace(DatasetDict=dict, DownloadConfig=lambda **kw: kw,
                               DownloadMode=SimpleNamespace(REUSE_DATASET_IF_EXISTS="reuse"),
                               load_dataset=load_dataset)
        backend = SimpleNamespace(_config_value=lambda config, name: config.get(name))
        task = SimpleNamespace(config={"dataset_path": "cais/mmlu", "dataset_name": "math"})
        with patch.dict("sys.modules", {"datasets": fake}):
            datasets, evidence = _load_safe_dataset(task, backend, "/cache")
        loads = [call[1] for call in calls if call[0] == "load"]
        self.assertEqual([c["split"] for c in loads], ["validation", "dev"])
        self.assertTrue(all(c["revision"] == DATASET_REVISION for c in loads))
        self.assertTrue(all(c["download_config"]["local_files_only"] for c in loads))
        self.assertNotIn("answer", datasets["validation"].column_names)
        self.assertIn("answer", datasets["dev"].column_names)
        self.assertEqual(evidence["subject"], "math")

    def test_write_identity_manifest_contains_no_other_target_prompts(self):
        bundle = {"dataset": {"split": "validation"}, "seed": SEED,
                  "calibration_identities": [identity("math", i) for i in range(512)],
                  "rows": [{"prompt": "debug"}] * 8}
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "fresh")
            write_debug_pool(bundle, output)
            manifest = json.loads((Path(output) / "calibration_identities.json").read_text())
            self.assertEqual(set(manifest), {"dataset", "seed", "count", "identities"})
            self.assertEqual(manifest["count"], 512)
            with self.assertRaises(FileExistsError):
                write_debug_pool(bundle, output)


if __name__ == "__main__":
    unittest.main()
