import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'scripts/loopscope/observe_phase11_gpqa_8192.py'
spec = importlib.util.spec_from_file_location('d4_observer', path)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class ArrayObserverTests(unittest.TestCase):
    def test_actual_pending_compaction_preserves_exact_children(self):
        ids = [f'12997851_{i}' for i in range(8)]
        self.assertEqual(observer.accounting_ids('12997851_[0-7%8]', ids), ids)
        self.assertEqual(observer.accounting_ids('12997851_[1-2,5%8]', ids), [ids[1], ids[2], ids[5]])
        self.assertEqual(observer.accounting_ids('12997851_[0-7%8]', ['12996845']), [])

    def test_pending_long_arrays_and_running_parent_are_not_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [f'12997851_{i}' for i in range(8)]
            (root / 'batch_summary.json').write_text(json.dumps({'status': 'RUNNING'}))
            (root / 'checkpoint.json').write_text(json.dumps({'done_record_count': 2, 'target_count': 10}))
            jobset = root / 'jobset.json'
            jobs = [{'job_id': i, 'batch_root': str(root), 'checkpoint_mode': 'question_boundary', 'records': 10}
                    for i in ids]
            jobset.write_text(json.dumps({'jobs': jobs, 'active_job_ids': ids,
                'prior_verified_complete_records': 4539, 'unsubmitted_records': 0}))
            queue = subprocess.CompletedProcess([], 0, '\n'.join(f'{i}|PENDING|QOSMaxCpuPerUserLimit' for i in ids), '')
            account = subprocess.CompletedProcess([], 0, '12997851_[0-7%8]|PENDING|0:0|0\n', '')
            with patch.object(observer.subprocess, 'run', side_effect=[queue, account]):
                value = observer.run(jobset)
            self.assertEqual(value['missing_accounting_ids'], [])
            self.assertEqual(value['accounting_state_counts'], {'PENDING': 8})
            self.assertEqual(value['verified_complete_formal_records'], 4539 + 16)
            self.assertEqual(value['failed_parent_jobs'], [])
            self.assertFalse(value['trigger_resume'])


if __name__ == '__main__':
    unittest.main()
