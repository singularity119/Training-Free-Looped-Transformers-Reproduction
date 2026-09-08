import copy
import unittest
from unittest.mock import patch
from tflt.loopscope.phase8_gate_f_accuracy import (
    SCHEMA, panel, logical_panel, validate_manifest, analyze_panel, checked_basis)
from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION


class GateFAccuracyTests(unittest.TestCase):
    def test_sixteen_new_cells_share_exactly_four_t0_paths(self):
        cells = panel('/t0', '/t1')
        self.assertEqual(len(cells), 16)
        self.assertEqual(len({c['cell_id'] for c in cells}), 16)
        self.assertEqual(len(logical_panel()), 36)
        t0 = cells[:12]
        self.assertTrue(all(c['fit_t'] == 0 for c in t0))
        for path in {c['basis_path'] for c in t0}:
            self.assertEqual({c['k'] for c in t0 if c['basis_path'] == path}, {2, 3, 4})
        self.assertEqual(len({c['basis_path'] for c in t0}), 4)
        self.assertEqual([c['basis_path'] for c in cells[12:] if c['arm'] == 'Spectral'],
                         ['/t1/cell-0/basis.json', '/t1/cell-1/basis.json'])
        manifest = dict(schema=SCHEMA, scope='FORMAL_TEST', dataset={}, pool='/pool',
                        t0_basis_root='/t0', t1_basis_root='/t1', cells=cells)
        self.assertEqual(len(validate_manifest(manifest, {'dataset': {}}, '/pool')), 16)
        cells[1]['basis_path'] = '/t0/copied-k3/basis.json'
        with self.assertRaisesRegex(ValueError, '16-cell'):
            validate_manifest(manifest, {'dataset': {}}, '/pool')

    def test_t0_loader_receives_application_k_and_scope(self):
        cell = panel('/t0', '/t1')[2]
        with patch('tflt.loopscope.phase8_gate_f_basis.load_t0_basis', return_value={'basis': 1}) as loader:
            self.assertEqual(checked_basis(cell, 'FORMAL_TEST'), {'basis': 1})
            loader.assert_called_once_with('/t0/cell-0/basis.json', cell, 'FORMAL_CALIBRATION')
            self.assertEqual(cell['k'], 4)

    def test_t1_basis_scope_and_k_stay_specific(self):
        cell = panel('/t0', '/t1')[13]
        metadata = {k: cell[k] for k in ('model', 'revision', 'window', 'k', 'dtype', 'cache_strategy')}
        metadata.update(scope='FORMAL_CALIBRATION', alpha=1., dataset=dict(
            repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'))
        basis = dict(metadata=metadata, rank=1, row_count=512, **{'lambda': .5})
        with patch('tflt.loopscope.phase8_runtime.load_basis', return_value=basis):
            self.assertIs(checked_basis(cell, 'FORMAL_TEST'), basis)
            metadata['k'] = 2
            with self.assertRaisesRegex(ValueError, 't1 basis'):
                checked_basis(cell, 'FORMAL_TEST')
            metadata['k'] = 3
            metadata['scope'] = 'PREFLIGHT_ONLY'
            with self.assertRaisesRegex(ValueError, 'scope recipe'):
                checked_basis(cell, 'FORMAL_TEST')
            self.assertIs(checked_basis(cell, 'PREFLIGHT_ONLY'), basis)

    def test_two_twelve_item_families_and_descriptive_controls(self):
        cells = logical_panel()
        for cell in cells:
            cell['scores'] = [[1, 2, 0, -1]] * 8 if cell['fit_t'] == 0 else [[2, 1, 0, -1]] * 8
            if cell['fit_t'] == 1:
                del cell['fit_t']  # Immutable D/E cells predate explicit fit_t.
        gold, subjects = [0, 0, 1, 1, 1, 1, 1, 1], ['s'] * 4 + ['t'] * 4
        result = analyze_panel(cells, gold, subjects, 100)
        self.assertEqual(len(result['cells']), 36)
        self.assertEqual(len(result['contrasts']), 36)
        for family in ('T0_MINUS_T1_EXPLORATORY', 'T0_MINUS_LOOP_EXPLORATORY'):
            rows = [r for r in result['contrasts'] if r['family'] == family]
            self.assertEqual(len(rows), 12)
            for row in rows:
                self.assertEqual(row['holm_family_size'], 12)
                self.assertEqual(row['delta_pp'], 50)
                self.assertEqual((row['wrong_to_right'], row['right_to_wrong']), (6, 2))
        rows = [r for r in result['contrasts'] if r['family'] == 'T1_MINUS_LOOP_DESCRIPTIVE']
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r['holm_adjusted_p'] is None for r in rows))
        for broken in (cells[:-1], cells[:-1] + [copy.deepcopy(cells[0])]):
            with self.assertRaisesRegex(ValueError, '36 unique'):
                analyze_panel(broken, gold, subjects, 100)
        cells[0]['cache_strategy'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'recipe mismatch'):
            analyze_panel(cells, gold, subjects, 100)


if __name__ == '__main__':
    unittest.main()
