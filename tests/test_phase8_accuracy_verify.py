import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tflt.loopscope.phase8_accuracy import panel
from tflt.loopscope.phase8_accuracy_verify import verify, validate_manifest


class AccuracyVerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/loopscope/phase8_model_windows.json').read_text())
        self.cells = panel(config)
        self.rows = [{'identity': f'test:s:{i}', 'subject': f's{i % 57}', 'doc_index': i}
                     for i in range(14042)]
        self.pool = self.root / 'pool.json'
        self.manifest = self.root / 'manifest.json'
        self.bundle = {'dataset': {'split': 'test'}, 'rows': self.rows}
        self.package = {'schema': 'loopscope.phase8.accuracy_manifest.v1',
                        'dataset': self.bundle['dataset'], 'pool': str(self.pool), 'cells': self.cells}
        self.write(self.pool, self.bundle)
        self.write(self.manifest, self.package)
        # Pool builder tests own the real pool schema. This fixture isolates shard closure.
        mock = patch('tflt.loopscope.phase8_accuracy_verify.validate_test_pool')
        mock.start()
        self.addCleanup(mock.stop)

    def write(self, path, obj):
        path.write_text(json.dumps(obj))

    def shard(self, cell=0, start=0, end=3, name='a'):
        root = self.root / name
        root.mkdir()
        metadata = dict(cell=self.cells[cell], scope='FORMAL_TEST', start=start, end=end,
                        source_commit='example-commit', pool=str(self.pool), manifest=str(self.manifest))
        self.write(root / 'command_args.json', metadata)
        self.write(root / 'env.json', dict(source_commit=metadata['source_commit'],
                                          model_revision=self.cells[cell]['revision'],
                                          model_dtype='torch.'+self.cells[cell]['dtype']))
        self.write(root / 'summary.json', dict(status='SCORES_COMPLETE', metadata=metadata,
                                             count=end-start, target_gold_loaded=False))
        with (root / 'scores.jsonl').open('w') as handle:
            for row in self.rows[start:end]:
                handle.write(json.dumps(dict(row, scores=[-1.0, -2.0, -3.0, -4.0]))+'\n')
        return root

    def test_shards_close_and_overlap_fails(self):
        a, b = self.shard(), self.shard(start=3, end=5, name='b')
        result = verify([a, b], self.pool, self.manifest)
        self.assertEqual(result['sample_count'], 5)
        self.assertEqual(result['status'], 'SHARDS_CLOSED')
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            verify([a, a], self.pool, self.manifest)
        with self.assertRaisesRegex(ValueError, '18 cells'):
            verify([a, b], self.pool, self.manifest, True)

    def test_label_nonfinite_and_missing_scores_fail(self):
        a = self.shard()
        path = a / 'scores.jsonl'
        lines = path.read_text().splitlines()
        row = json.loads(lines[0])
        row['answer'] = 0
        path.write_text(json.dumps(row)+'\n'+'\n'.join(lines[1:])+'\n')
        with self.assertRaisesRegex(ValueError, 'no outcomes'):
            verify([a], self.pool, self.manifest)
        del row['answer']
        row['scores'][0] = float('nan')
        path.write_text(json.dumps(row)+'\n'+'\n'.join(lines[1:])+'\n')
        with self.assertRaisesRegex(ValueError, 'finite'):
            verify([a], self.pool, self.manifest)
        path.write_text('\n'.join(lines[:2])+'\n')
        with self.assertRaisesRegex(ValueError, 'row count'):
            verify([a], self.pool, self.manifest)

    def test_basis_and_identity_mismatch_fail(self):
        self.package['cells'][2]['basis_path'] = '/wrong/basis.json'
        with self.assertRaisesRegex(ValueError, 'frozen'):
            validate_manifest(self.package, self.bundle, self.pool)
        a = self.shard()
        path = a / 'scores.jsonl'
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(reversed(lines))+'\n')
        with self.assertRaisesRegex(ValueError, 'identity/order'):
            verify([a], self.pool, self.manifest)

    def test_preflight_uses_first_and_longest_rows(self):
        from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION
        from tflt.loopscope.phase8_accuracy import select_rows
        self.bundle['dataset'] = dict(repo=DATASET_REPO, revision=DATASET_REVISION, split='test')
        for i, row in enumerate(self.rows):
            row['prompt_token_lengths'] = {self.cells[0]['model']: i+1}
        self.package['dataset'] = self.bundle['dataset']
        self.write(self.pool, self.bundle)
        self.write(self.manifest, self.package)
        a = self.shard()
        metadata = json.loads((a / 'command_args.json').read_text())
        metadata['scope'] = 'PREFLIGHT_ONLY'
        selected = select_rows(self.bundle, self.cells[0], 'PREFLIGHT_ONLY', 0, 3)
        self.assertEqual([r['doc_index'] for r in selected], [0, 1, 14038, 14039, 14040, 14041])
        self.write(a / 'command_args.json', metadata)
        self.write(a / 'summary.json', dict(status='SCORES_COMPLETE', metadata=metadata,
                                          count=6, target_gold_loaded=False))
        with (a / 'scores.jsonl').open('w') as handle:
            for row in selected:
                handle.write(json.dumps({**{k: row[k] for k in ('identity','subject','doc_index')},
                                         'scores': [-1., -2., -3., -4.]})+'\n')
        self.assertEqual(verify([a], self.pool, self.manifest)['sample_count'], 6)
        with self.assertRaisesRegex(ValueError, 'formal test'):
            verify([a], self.pool, self.manifest, True)

    def test_full_panel_closes(self):
        roots = [self.shard(cell=i, end=14042, name=str(i)) for i in range(18)]
        result = verify(roots, self.pool, self.manifest, True)
        self.assertEqual(result['status'], 'FULL_PANEL_CLOSED')
        self.assertEqual(result['sample_count'], 252756)
        self.assertFalse(result['target_gold_loaded'])


if __name__ == '__main__':
    unittest.main()
