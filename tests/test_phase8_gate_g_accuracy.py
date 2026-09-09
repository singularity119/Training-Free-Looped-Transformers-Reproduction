import copy
import unittest
from unittest.mock import patch
from tflt.loopscope.phase8_gate_g_accuracy import (
    SCHEMA, FAMILIES, panel, logical_panel, validate_manifest, analyze_panel, checked_basis)
from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION


class GateGAccuracyTests(unittest.TestCase):
    def test_nine_new_cells_four_bases_and_new_window(self):
        cells = panel('/basis')
        self.assertEqual(len(cells), 9)
        self.assertEqual(len({c['cell_id'] for c in cells}), 9)
        self.assertTrue(all(c['window'] == [15,18] and c['dtype']=='bfloat16'
                            and c['cache_strategy']=='first' for c in cells))
        spectral = [c for c in cells if c['arm']=='Spectral']
        self.assertEqual(len({c['basis_path'] for c in spectral}), 4)
        self.assertEqual({c['basis_path'] for c in spectral if c['fit_t']==0}, {'/basis/t0/basis.json'})
        self.assertEqual({c['basis_path'] for c in spectral if c['fit_t']==1},
                         {'/basis/k2/basis.json','/basis/k3/basis.json','/basis/k4/basis.json'})
        manifest = dict(schema=SCHEMA, scope='FORMAL_TEST', dataset={}, pool='/pool',
                        basis_root='/basis', cells=cells)
        self.assertEqual(len(validate_manifest(manifest, {'dataset': {}}, '/pool')), 9)
        cells[0]['window'] = [12,15]
        with self.assertRaisesRegex(ValueError, '9-cell'):
            validate_manifest(manifest, {'dataset': {}}, '/pool')

    def test_analysis_cli_requires_g_closure_before_gold(self):
        import importlib.util
        import json
        from pathlib import Path
        import runpy
        import sys
        import tempfile
        from unittest.mock import MagicMock
        scripts = Path(__file__).resolve().parents[1] / 'scripts' / 'loopscope'
        helpers = MagicMock()
        closer = MagicMock()
        with tempfile.TemporaryDirectory() as temp:
            closure = Path(temp)/'closure.json'
            closure.write_text(json.dumps(dict(status='GATE_G_FULL_LOGICAL_PANEL_CLOSED',
                target_gold_loaded=False, source_closure='/fresh', pool='/pool', cells=[])))
            saved = json.loads(closure.read_text())
            helpers.read_json.side_effect = [saved, {'rows':[]}]
            closer.close.return_value = saved
            helpers.load_frozen_gold.side_effect = RuntimeError('GOLD_BOUNDARY_REACHED')
            with patch.dict(sys.modules, {'analyze_phase8_accuracy':helpers,
                                          'close_phase8_gate_g_panel':closer}), patch.object(sys, 'argv',
                    ['analyze', '--closure', str(closure), '--output-dir', str(Path(temp)/'out')]):
                with self.assertRaisesRegex(RuntimeError, 'GOLD_BOUNDARY_REACHED'):
                    runpy.run_path(str(scripts/'analyze_phase8_gate_g_accuracy.py'), run_name='__main__')
            closer.close.assert_called_once_with('/fresh')
            helpers.load_frozen_gold.assert_called_once()

    def test_t0_loader_receives_application_k(self):
        cell = panel('/basis')[8]
        with patch('tflt.loopscope.phase8_gate_g_calibration.load_t0_basis', return_value={'basis':1}) as loader:
            self.assertEqual(checked_basis(cell, 'FORMAL_TEST'), {'basis':1})
            loader.assert_called_once_with('/basis/t0/basis.json', cell, 'FORMAL_CALIBRATION')
            self.assertEqual(cell['k'], 4)

    def test_t1_basis_scope_and_k_stay_specific(self):
        cell = panel('/basis')[4]
        metadata = {k:cell[k] for k in ('model','revision','window','k','dtype','cache_strategy')}
        metadata.update(scope='FORMAL_CALIBRATION', alpha=1., fit_t=1, dataset=dict(
            repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'))
        basis = dict(metadata=metadata, fit_t=1, rank=1, row_count=512, **{'lambda':.5})
        with patch('tflt.loopscope.phase8_runtime.load_basis', return_value=basis):
            self.assertIs(checked_basis(cell, 'FORMAL_TEST'), basis)
            metadata['k'] = 2
            with self.assertRaisesRegex(ValueError, 'K-specific'):
                checked_basis(cell, 'FORMAL_TEST')
            metadata['k'] = 3
            metadata['scope'] = 'PREFLIGHT_ONLY'
            with self.assertRaisesRegex(ValueError, 'scope recipe'):
                checked_basis(cell, 'FORMAL_TEST')
            self.assertIs(checked_basis(cell, 'PREFLIGHT_ONLY'), basis)

    def test_three_separate_three_item_families(self):
        cells = logical_panel()
        for cell in cells:
            cell['scores'] = [[1,2,0,-1]]*8 if cell['fit_t']==0 else [[2,1,0,-1]]*8
        gold, subjects = [0,0,1,1,1,1,1,1], ['s']*4+['t']*4
        result = analyze_panel(cells, gold, subjects, 100)
        self.assertEqual(len(result['cells']), 9)
        self.assertEqual(len(result['contrasts']), 9)
        for family in FAMILIES:
            rows = [r for r in result['contrasts'] if r['family']==family]
            self.assertEqual({r['k'] for r in rows}, {2,3,4})
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(row['holm_family_size'], 3)
                self.assertIsNotNone(row['holm_adjusted_p'])
                self.assertEqual(row['delta_pp'], 0 if family=='T1_MINUS_LOOP_EXPLORATORY' else 50)
        self.assertEqual(result['native_context']['correct'], 10262)
        for broken in (cells[:-1], cells[:-1]+[copy.deepcopy(cells[0])]):
            with self.assertRaisesRegex(ValueError, '9 unique'):
                analyze_panel(broken,gold,subjects,100)
        cells[0]['cache_strategy']='wrong'
        with self.assertRaisesRegex(ValueError,'recipe mismatch'):
            analyze_panel(cells,gold,subjects,100)


if __name__=='__main__': unittest.main()
