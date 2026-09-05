import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tflt.loopscope.phase8_test_pool import (
    DATASET_REPO, DATASET_REVISION, SEED, _load_safe_dataset,
    test_identity, validate_test_pool, write_test_pool,
)


class TestPoolTests(unittest.TestCase):
    def test_target_column_projection_before_row_access(self):
        calls = []
        source_mode = "urls"
        class Dataset:
            def __init__(self, split, columns=None):
                self.split = split
                self.info = SimpleNamespace(download_checksums={"hf://datasets/cais/mmlu@" + DATASET_REVISION + "/" + split: {}})
                self.column_names = columns or ["question", "subject", "choices", "answer"]
                self.cache_files = [{"filename": "/cache/" + split + ".arrow"}]
                if source_mode != "urls":
                    self.info.download_checksums = {}
                    version = DATASET_REVISION if source_mode == "directory" else "other-version"
                    self.cache_files = [{"filename": "/cache/" + version + "/" + split + ".arrow"}]
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
        self.assertEqual([c["split"] for c in loads], ["test", "dev"])
        self.assertTrue(all(c["revision"] == DATASET_REVISION for c in loads))
        self.assertTrue(all(c["download_config"]["local_files_only"] for c in loads))
        self.assertNotIn("answer", datasets["test"].column_names)
        self.assertIn("answer", datasets["dev"].column_names)
        self.assertEqual(evidence["subject"], "math")
        source_mode = "directory"
        with patch.dict("sys.modules", {"datasets": fake}):
            _, evidence = _load_safe_dataset(task, backend, "/cache")
        self.assertEqual(evidence["splits"]["test"]["revision_evidence"], "loaded_cache_version_directory")
        source_mode = "wrong"
        with patch.dict("sys.modules", {"datasets": fake}), self.assertRaises(ValueError):
            _load_safe_dataset(task, backend, "/cache")

    def fixture(self):
        counts = {f"subject_{i:02}": 246 + (i < 20) for i in range(57)}
        rows = [{"identity": test_identity(s, i), "subject": s, "doc_index": i,
                 "split": "test", "question": "question", "choices": ["a", "b", "c", "d"],
                 "prompt": "prompt Answer:", "prompt_token_lengths": {"m1": 3, "m2": 4}}
                for s, n in counts.items() for i in range(n)]
        return {"schema_version": "loopscope.phase8.test_pool.v1", "status": "FORMAL_TEST_GOLD_FREE",
                "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "split": "test"},
                "seed": SEED, "rows": rows, "test_identities": [r["identity"] for r in rows],
                "source_evidence": {"subject_counts": counts, "loaded_splits": ["test", "dev"],
                    "target_gold_loaded": False, "tokenizers": [{"model": "m1"}, {"model": "m2"}]}}

    def test_complete_canonical_identity_and_gold_free_fields(self):
        bundle = self.fixture()
        validate_test_pool(bundle)
        self.assertIn(":test:", bundle["rows"][0]["identity"])
        row = bundle["rows"][0]
        for change in ({"split": "validation"}, {"identity": "wrong"}, {"answer": 0},
                       {"prompt_token_lengths": {"m1": 3}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_test_pool(dict(bundle, rows=[dict(row, **change)] + bundle["rows"][1:]))
        with self.assertRaises(ValueError):
            validate_test_pool(dict(bundle, rows=bundle["rows"][1:]))
        with self.assertRaises(ValueError):
            validate_test_pool(dict(bundle, rows=list(reversed(bundle["rows"]))))

    def test_write_once(self):
        bundle = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fresh"
            write_test_pool(bundle, str(root))
            self.assertEqual(json.loads((root / "test_identities.json").read_text())["count"], 14042)
            self.assertEqual(json.loads((root / "test_pool.json").read_text())["rows"], bundle["rows"])
            with self.assertRaises(FileExistsError):
                write_test_pool(bundle, str(root))


if __name__ == "__main__":
    unittest.main()
