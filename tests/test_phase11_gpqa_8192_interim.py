import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/loopscope/analyze_phase11_gpqa_8192_interim.py'
spec = importlib.util.spec_from_file_location('interim_runner', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class InterimSnapshotTests(unittest.TestCase):
    def test_old_partial_question_requires_both_flushed_record_and_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'records.jsonl').write_text('{"canonical_index":0}\n{"canonical_index":1}\n{"canonical_index":')
            (root / 'telemetry.jsonl').write_text('{"canonical_index":0}\n{"canonical_index":')
            self.assertEqual(list(runner.root_bundles(root)),
                [({'canonical_index':0}, {'canonical_index':0})])
            self.assertTrue((root / 'records.jsonl').read_text().endswith('{"canonical_index":'))

    def test_checkpoint_snapshot_reads_committed_bundles_once_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / 'complete'
            complete.mkdir()
            value = {'record':{'canonical_index':0}, 'telemetry':{'canonical_index':0}}
            success = complete / '000000.json'
            success.write_text(json.dumps(value))
            half = complete / '.000001.json.crash.tmp'
            half.write_text('{"record":')
            mtime = success.stat().st_mtime_ns
            self.assertEqual(list(runner.root_bundles(root)), [(value['record'], value['telemetry'])])
            self.assertEqual(success.stat().st_mtime_ns, mtime)
            self.assertEqual(half.read_text(), '{"record":')
            self.assertFalse((root / 'checkpoint.json').exists())


if __name__ == '__main__':
    unittest.main()
