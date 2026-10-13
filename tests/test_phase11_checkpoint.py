import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tflt.loopscope.phase11_checkpoint import RecordStore


def validator(index, record, fact):
    if record["canonical_index"] != index or record["identity"] != f"synthetic:{index}":
        raise ValueError("identity contract differs")
    if record["cell_id"] != "synthetic-cell" or fact["canonical_index"] != index or fact["prefill_count"] != 1:
        raise ValueError("cell/mechanism contract differs")


def bundle(index):
    return {"record": {"canonical_index": index, "identity": f"synthetic:{index}", "cell_id": "synthetic-cell"},
            "telemetry": {"canonical_index": index, "prefill_count": 1}}


def commit(store, index):
    value = bundle(index)
    store.commit(index, value["record"], value["telemetry"])


class CheckpointTests(unittest.TestCase):
    def test_question_boundary_restart_isolates_half_files_and_skips_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RecordStore(root, [0, 1], validator)
            store.mark_in_progress(0)
            commit(store, 0)
            success = root / "complete/000000.json"
            original, mtime = success.read_bytes(), success.stat().st_mtime_ns
            store.mark_in_progress(1)
            partial = root / "complete/.000001.json.interrupted.tmp"
            partial.write_bytes(b'{"record":')
            restarted = RecordStore(root, [0, 1], validator)
            self.assertEqual(list(restarted.done), [0])
            self.assertEqual(restarted.pending, [1])
            self.assertEqual(success.read_bytes(), original)
            self.assertEqual(success.stat().st_mtime_ns, mtime)
            self.assertFalse(partial.exists())
            self.assertEqual(len(restarted.quarantined), 1)
            preserved = Path(restarted.quarantined[0]["preserved_path"])
            self.assertEqual(preserved.read_bytes(), b'{"record":')
            self.assertEqual(restarted.quarantined[0]["reason"], "uncommitted temporary file")
            restarted.mark_in_progress(1)
            commit(restarted, 1)
            records, facts = restarted.finalize_records()
            self.assertEqual([json.loads(line)["canonical_index"] for line in records.read_text().splitlines()], [0, 1])
            self.assertEqual([json.loads(line)["canonical_index"] for line in facts.read_text().splitlines()], [0, 1])
            self.assertEqual(success.read_bytes(), original)
            final_bytes, final_mtime = records.read_bytes(), records.stat().st_mtime_ns
            restarted.finalize_records()
            self.assertEqual(records.read_bytes(), final_bytes)
            self.assertEqual(records.stat().st_mtime_ns, final_mtime)
            with self.assertRaises(FileExistsError):
                commit(restarted, 0)

    def test_stale_checkpoint_is_rebuilt_from_complete_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RecordStore(root, [1, 0], validator)
            commit(store, 1)
            (root / "checkpoint.json").write_text(json.dumps({"done": [0], "pending": [], "in_progress": [1]}))
            restarted = RecordStore(root, [1, 0], validator)
            state = json.loads((root / "checkpoint.json").read_text())
            self.assertEqual(list(restarted.done), [1])
            self.assertEqual(state["done"], [1])
            self.assertEqual(state["pending"], [0])
            self.assertEqual(state["in_progress"], [])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                restarted.finalize_records()
            commit(restarted, 0)
            records, _ = restarted.finalize_records()
            self.assertEqual([json.loads(line)["canonical_index"] for line in records.read_text().splitlines()], [1, 0])

    def test_complete_bundle_survives_crash_before_checkpoint_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RecordStore(root, [0], validator)
            store.mark_in_progress(0)
            with patch.object(store, "_save_state", side_effect=RuntimeError("simulated state-update crash")):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    commit(store, 0)
            self.assertTrue((root / "complete/000000.json").exists())
            self.assertEqual(json.loads((root / "checkpoint.json").read_text())["done"], [])
            restarted = RecordStore(root, [0], validator)
            self.assertEqual(list(restarted.done), [0])
            self.assertEqual(restarted.pending, [])
            restarted.finalize_records()

    def test_incomplete_bundle_preserved_with_reason_contract_error_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RecordStore(root, [0, 1], validator)
            (root / "complete/000000.json").write_bytes(b'{"record":')
            (root / "complete/000001.json").write_text(json.dumps({"telemetry": {}}))
            recovered = RecordStore(root, [0, 1], validator)
            self.assertEqual(recovered.done, {})
            self.assertEqual(recovered.pending, [0, 1])
            self.assertEqual(len(recovered.quarantined), 2)
            for event in recovered.quarantined:
                self.assertTrue(Path(event["preserved_path"]).exists())
                reason = Path(event["preserved_path"] + ".reason.json")
                self.assertEqual(json.loads(reason.read_text())["reason"], event["reason"])
            bad = bundle(0)
            bad["record"]["identity"] = "wrong-identity"
            (root / "complete/000000.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "identity contract"):
                RecordStore(root, [0, 1], validator)
            self.assertTrue((root / "complete/000000.json").exists())
            with self.assertRaisesRegex(ValueError, "identity contract"):
                store.commit(0, bad["record"], bad["telemetry"])

    def test_publish_failure_preserves_temporary_bytes_and_requires_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = RecordStore(root, [0], validator)
            with patch("tflt.loopscope.phase11_checkpoint.os.replace", side_effect=OSError("simulated publish failure")):
                with self.assertRaisesRegex(OSError, "publish failure"):
                    commit(store, 0)
            self.assertFalse((root / "complete/000000.json").exists())
            restarted = RecordStore(root, [0], validator)
            self.assertEqual(restarted.pending, [0])
            self.assertEqual(len(restarted.quarantined), 1)
            self.assertEqual(json.loads(Path(restarted.quarantined[0]["preserved_path"]).read_text()), bundle(0))
            commit(restarted, 0)
            records, _ = restarted.finalize_records()
            records.write_text(json.dumps({"identity": "different"}) + "\n")
            with self.assertRaisesRegex(ValueError, "existing final output differs"):
                restarted.finalize_records()
            self.assertIn("different", records.read_text())


if __name__ == "__main__":
    unittest.main()
