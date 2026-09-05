import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tflt.loopscope.phase8_pool import (
    DATASET_REPO, DATASET_REVISION, SEED, build_calibration_pool,
    build_debug_pool, identity, select_calibration, select_calibration_rows,
    write_calibration_pool,
)


class CalibrationPoolTests(unittest.TestCase):
    def setUp(self):
        self.members = select_calibration({"algebra": 600, "biology": 931})
        self.rows = [{"identity": identity(s, i), "subject": s, "doc_index": i,
                      "split": "validation", "question": "safe question",
                      "choices": ["a", "b", "c", "d"], "prompt": "safe prompt"}
                     for s, n in (("algebra", 600), ("biology", 931)) for i in range(n)]
        self.manifest = {"dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION,
                                     "split": "validation"},
                         "seed": SEED, "count": 512,
                         "identities": [r["identity"] for r in self.members]}

    def test_formal_selection_exactly_replays_b_order(self):
        selected = select_calibration_rows(list(reversed(self.rows)), self.members, self.manifest)
        self.assertEqual([r["identity"] for r in selected], self.manifest["identities"])
        self.assertEqual(len(selected), 512)
        self.assertTrue(all(r["role"] == "fit" for r in selected))
        for field, wrong in (("seed", 0), ("count", 511), ("dataset", {}),
                             ("identities", list(reversed(self.manifest["identities"])))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                select_calibration_rows(self.rows, self.members, dict(self.manifest, **{field: wrong}))

    def test_missing_row_and_duplicate_do_not_pass(self):
        selected_id = self.members[0]["identity"]
        with self.assertRaises(ValueError):
            select_calibration_rows([r for r in self.rows if r["identity"] != selected_id],
                                    self.members, self.manifest)
        with self.assertRaises(ValueError):
            select_calibration_rows(self.rows + [self.rows[0]], self.members, self.manifest)

    def test_shared_builder_entrypoints(self):
        with patch("tflt.loopscope.phase8_pool._build_pool", return_value={}) as builder:
            build_debug_pool({}, "/cache")
            builder.assert_called_once_with({}, "/cache")
            builder.reset_mock()
            build_calibration_pool({}, "/cache", self.manifest)
            builder.assert_called_once_with({}, "/cache", self.manifest)

    def test_export_persists_only_ordered512_and_is_write_once(self):
        rows = select_calibration_rows(self.rows, self.members, self.manifest)
        bundle = {"schema_version": "loopscope.phase8.calibration_pool.v1",
                  "status": "FORMAL_CALIBRATION", "dataset": self.manifest["dataset"],
                  "seed": SEED, "calibration_identities": self.manifest["identities"], "rows": rows}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fresh"
            write_calibration_pool(bundle, str(root))
            saved = json.loads((root / "calibration_pool.json").read_text())
            self.assertEqual(saved["rows"], rows)
            self.assertEqual(json.loads((root / "calibration_identities.json").read_text()), self.manifest)
            self.assertEqual(sorted(p.name for p in root.iterdir()),
                             ["calibration_identities.json", "calibration_pool.json"])
            with self.assertRaises(FileExistsError):
                write_calibration_pool(bundle, str(root))
            with self.assertRaises(ValueError):
                write_calibration_pool(dict(bundle, rows=self.rows), str(Path(tmp) / "bad"))


if __name__ == "__main__":
    unittest.main()
