"""Outcome-free Gate F closure tests using the canonical full panel shape."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tflt.loopscope.phase8_gate_f_accuracy import SCHEMA, panel
from tflt.loopscope.phase8_gate_f_verify import verify


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


class GateFVerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.pool = root / 'pool.json'
        cls.manifest = root / 'manifest.json'
        rows = [dict(identity=f'subject-{i % 57}:test:{i // 57}',
                     subject=f'subject-{i % 57}', doc_index=i // 57) for i in range(14042)]
        dataset = dict(repo='synthetic-canonical-test', revision='fixture', split='test')
        write_json(cls.pool, dict(dataset=dataset, rows=rows))
        cells = panel('/fixed-t0', '/fixed-t1')
        write_json(cls.manifest, dict(schema=SCHEMA, scope='FORMAL_TEST', dataset=dataset,
            pool=str(cls.pool), t0_basis_root='/fixed-t0', t1_basis_root='/fixed-t1', cells=cells))
        scores = ''.join(json.dumps(dict(row, scores=[-1., -2., -3., -4.]))+'\n' for row in rows)
        cls.roots = []
        for cell in cells:
            cell_root = root / cell['cell_id']
            cell_root.mkdir()
            cls.roots.append(cell_root)
            metadata = dict(cell=cell, pool=str(cls.pool), manifest=str(cls.manifest),
                source_commit='fixed-f-producer', scope='FORMAL_TEST', start=0, end=14042)
            write_json(cell_root / 'command_args.json', metadata)
            # Four scorer candidates per identity, same count at every Euler step.
            callbacks = {str(t): 4*14042 for t in range(cell['k'])}
            applied = {str(t): (0 if t == 0 else 4*14042) for t in range(cell['k'])}
            write_json(cell_root / 'summary.json', dict(status='SCORES_COMPLETE',
                metadata=metadata, target_gold_loaded=False, count=14042,
                positions=dict(callback_by_t=callbacks, applied_by_t=applied)))
            write_json(cell_root / 'env.json', dict(source_commit='fixed-f-producer',
                model_revision=cell['revision'], model_dtype='torch.'+cell['dtype']))
            (cell_root / 'scores.jsonl').write_text(scores, encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def check(self, roots, full_panel=True):
        # Only bypass pool origin validation: shard identities, panel recipe, row
        # counts, finite scores, intervention counts and full closure run for real.
        with patch('tflt.loopscope.phase8_gate_f_verify.validate_test_pool'):
            return verify(roots, self.pool, self.manifest, full_panel=full_panel)

    def test_complete_sixteen_cells_close_224672_raw_scores(self):
        result = self.check(self.roots)
        self.assertEqual(result['status'], 'FULL_PANEL_CLOSED')
        self.assertEqual(result['cell_count'], 16)
        self.assertEqual(result['sample_count'], 224672)
        self.assertEqual(result['subject_count'], 57)
        self.assertEqual(set(result['counts_by_cell'].values()), {14042})
        self.assertIs(result['target_gold_loaded'], False)

    def test_missing_cell_cannot_close_full_panel(self):
        with self.assertRaisesRegex(ValueError, 'all 16 cells'):
            self.check(self.roots[:-1])

    def test_overlapping_shard_cannot_be_counted_twice(self):
        with self.assertRaisesRegex(ValueError, 'overlapping or duplicate'):
            self.check([self.roots[0], self.roots[0]], full_panel=False)

    def test_fitting_at_t0_does_not_authorize_t0_intervention(self):
        path = self.roots[0] / 'summary.json'
        original = path.read_text()
        summary = json.loads(original)
        summary['positions']['applied_by_t']['0'] = 4*14042
        try:
            write_json(path, summary)
            with self.assertRaisesRegex(ValueError, 'wrong intervention timing'):
                self.check([self.roots[0]], full_panel=False)
        finally:
            path.write_text(original)

    def test_later_steps_must_all_apply_fixed_direction(self):
        path = self.roots[0] / 'summary.json'
        original = path.read_text()
        summary = json.loads(original)
        summary['positions']['applied_by_t']['1'] -= 4
        try:
            write_json(path, summary)
            with self.assertRaisesRegex(ValueError, 'wrong intervention timing'):
                self.check([self.roots[0]], full_panel=False)
        finally:
            path.write_text(original)


if __name__ == '__main__':
    unittest.main()
