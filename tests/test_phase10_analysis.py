import copy
import importlib.util
import sys
import types
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tflt.loopscope.phase10_analysis import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, DIRECTION_FAMILY, SCAN_FAMILY, TRANSFER_FAMILY,
    analyze_panel, exact_mcnemar_p, holm_adjust, score_predictions, select_transfer_lambdas,
    validate_transfer_lambdas, write_report, write_tables,
)
from tflt.loopscope.phase10_panel import build_panel


def synthetic(dataset='mmlu'):
    gold = [0, 1, 2, 0, 1, 2]
    cells = copy.deepcopy(build_panel(dataset)['cells'])
    for cell in cells:
        cell['scores'] = [[-1.0 if choice == label else -5.0 for choice in range(4)] for label in gold]
        if dataset == 'arc_challenge':
            cell['choice_text_lengths'] = [[1, 2, 3, 4] for _ in gold]
    return cells, gold


class Phase10AnalysisTests(unittest.TestCase):
    def test_exact_mcnemar_and_holm(self):
        self.assertAlmostEqual(exact_mcnemar_p(6, 0), 0.03125)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_frozen_families_bootstrap_and_smaller_lambda_tie(self):
        cells, gold = synthetic()
        # Give one strategy its maximum correct count at lambda0.2 and0.4.
        for cell in cells:
            if cell['cell_id'].startswith('q4-w15-18-k2-online-fixed-t0-') and cell['strength'] not in (0.2, 0.4):
                cell['scores'][0] = [-5.0, -1.0, -5.0, -5.0]
        result = analyze_panel('mmlu', cells, gold, ['large'] * 4 + ['small'] * 2)
        self.assertEqual(result['families'][SCAN_FAMILY]['size'], 126)
        self.assertEqual(result['families'][DIRECTION_FAMILY]['size'], 45)
        self.assertEqual(len(result['contrasts']), 171)
        policy_rows = [row for row in result['contrasts'] if row['family'] == DIRECTION_FAMILY]
        self.assertEqual(sum(row['k'] == 2 for row in policy_rows), 18)
        self.assertEqual(sum(row['k'] == 3 for row in policy_rows), 27)
        self.assertEqual({row['comparison'] for row in policy_rows},
                         {'current_t-fixed_t0', 'current_t-lag1', 'lag1-fixed_t0'})
        self.assertTrue(all(row['comparison'] == 'current_t-fixed_t0' for row in policy_rows if row['k'] == 2))
        self.assertEqual(len(result['cells']), 68)
        self.assertEqual(len(result['display_rows']), 86)
        self.assertEqual(sum(row['alias_of'] is not None for row in result['display_rows']), 18)
        self.assertEqual(result['bootstrap']['replicates'], BOOTSTRAP_REPLICATES)
        self.assertEqual(result['bootstrap']['seed'], BOOTSTRAP_SEED)
        transfer = result['transfer_lambdas']
        selected = validate_transfer_lambdas(transfer)
        self.assertEqual(selected['q4-w15-18-k2-fixed-t0']['strength'], 0.2)
        self.assertEqual([row['strength'] for key, row in selected.items() if key != 'q4-w15-18-k2-fixed-t0'], [0.1] * 6)
        row = next(row for row in result['contrasts'] if row['treatment'] == 'q4-w15-18-k2-online-fixed-t0-lambda0.1'
                   and row['reference'].endswith('-loop'))
        self.assertEqual((row['wrong_to_right'], row['right_to_wrong'], row['net_corrections']), (0, 1, -1))
        self.assertEqual(row['bootstrap']['subject_count'], 2)
        self.assertAlmostEqual(row['delta_pp'], -100 / 6)
        self.assertAlmostEqual(row['subject_macro_delta_pp'], -12.5)
        self.assertFalse(row['supports_positive_gain'])
        for family in (SCAN_FAMILY, DIRECTION_FAMILY):
            rows = [r for r in result['contrasts'] if r['family'] == family]
            self.assertEqual([r['holm_adjusted_p'] for r in rows], holm_adjust([r['mcnemar_exact_p'] for r in rows]))

    def test_arc_infers_acc_norm_and_keeps_acc_descriptive_with_fourteen_transfers(self):
        mmlu, gold = synthetic()
        transfer = analyze_panel('mmlu', mmlu, gold, ['a'] * 3 + ['b'] * 3)['transfer_lambdas']
        cells, gold = synthetic('arc_challenge')
        # Demonstrate length-normalized argmax differs from raw full-text LL.
        gold[0] = 1
        for cell in cells:
            cell['scores'][0] = [-2.0, -3.0, -20.0]  # variable choices allowed
            cell['choice_text_lengths'][0] = [1, 3, 1]
        result = analyze_panel('arc_challenge', cells, gold, transfer_lambdas=transfer)
        self.assertEqual(set(result['families']), {SCAN_FAMILY, DIRECTION_FAMILY, TRANSFER_FAMILY})
        self.assertEqual(result['families'][TRANSFER_FAMILY]['size'], 14)
        self.assertEqual(len(result['contrasts']), 185)
        self.assertEqual(result['cells'][0]['correct'], 6)
        self.assertEqual(result['cells'][0]['acc_correct'], 5)
        self.assertEqual(result['primary_metric'], 'acc_norm')
        with self.assertRaisesRegex(ValueError, 'MMLU cell statistics'):
            select_transfer_lambdas(result['cells'])
        self.assertEqual(result['contrasts'][0]['bootstrap']['method'], 'item_paired_bootstrap')
        self.assertNotIn('subject_macro_delta_pp', result['contrasts'][0])
        self.assertNotIn('subject_count', result['contrasts'][0]['bootstrap'])
        self.assertEqual(score_predictions([[-2, -3]], [[1, 3]]), [1])
        self.assertEqual(score_predictions([[-2, -3]]), [0])
        with self.assertRaises(ValueError):
            analyze_panel('arc_challenge', cells, gold)
        bad = copy.deepcopy(cells)
        bad[0]['choice_text_lengths'][0] = [1, 4, 1]
        with self.assertRaises(ValueError):
            analyze_panel('arc_challenge', bad, gold, transfer_lambdas=transfer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_tables(path / 'tables', result)
            write_report(path / 'report.md', result)
            self.assertEqual(len((path / 'tables/display_cells.csv').read_text().splitlines()), 87)
            with self.assertRaises(FileExistsError):
                write_tables(path / 'tables', result)

    def test_normal_cli_reloads_verifier_roots_before_label_input(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/loopscope/analyze_phase10.py'
        spec = importlib.util.spec_from_file_location('phase10_analysis_cli_test', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        panel = build_panel('mmlu')
        panel['sample_count_per_cell'] = 2  # synthetic fixture, no real records
        cell_ids = {cell['cell_id'] for cell in panel['cells']}
        closure = {'schema': 'loopscope.phase10.score_verification.v1',
                   'status': 'FULL_PHASE10_SCORE_PANEL_CLOSED', 'target_gold_loaded': False,
                   'dataset': 'mmlu', 'scope': 'FORMAL_TEST', 'cell_count': 68,
                   'sample_count': 136, 'counts_by_cell': {cell_id: 2 for cell_id in cell_ids},
                   'manifest': '/fake/manifest.json', 'pool': '/fake/pool.json',
                   'roots': [{'root': '/fake/' + cell['cell_id'], 'cell_id': cell['cell_id']}
                             for cell in panel['cells']]}
        aligned = {'schema': 'loopscope.phase10.aligned_scores.v1', 'dataset': 'mmlu',
                   'target_gold_loaded': False, 'cells': {cell_id: [[-1, -3, -4, -5]] * 2 for cell_id in cell_ids},
                   'identities': ['synthetic0', 'synthetic1'], 'subjects': ['s0', 's1'],
                   'choice_lengths': [[1, 1, 1, 1]] * 2}
        calls = []
        def fake_verify(roots, manifest, pool, scope, config_path, **kwargs):
            calls.append((roots, kwargs))
            return closure, aligned
        fake_module = types.SimpleNamespace(verify=fake_verify)
        with patch.object(module, 'build_panel', return_value=panel), patch.dict(sys.modules, {'verify_phase10_scores': fake_module}):
            cells, loaded = module.load_closed_scores(closure)
            self.assertEqual(len(cells), 68)
            self.assertEqual(loaded['identities'], aligned['identities'])
            self.assertEqual(len(calls[0][0]), 59)
            self.assertTrue(calls[0][1]['full_panel'])
            self.assertTrue(calls[0][1]['include_reuse'])
            broken = {**closure, 'status': 'FULL_PHASE10_NEW_SCORE_PANEL_CLOSED'}
            with self.assertRaises(ValueError):
                module.load_closed_scores(broken)
            self.assertEqual(len(calls), 1)

    def test_analysis_rejects_alias_double_counting_and_missing_subjects(self):
        cells, gold = synthetic()
        with self.assertRaises(ValueError):
            analyze_panel('mmlu', cells + [cells[-1]], gold, ['x'] * len(gold))
        with self.assertRaises(ValueError):
            analyze_panel('mmlu', cells, gold)
        bad = copy.deepcopy(cells)
        bad[0]['strength'] = 0.1
        with self.assertRaises(ValueError):
            analyze_panel('mmlu', bad, gold, ['x'] * len(gold))
        with self.assertRaises(ValueError):
            select_transfer_lambdas({'dataset': 'arc_challenge', 'cells': []})


if __name__ == '__main__':
    unittest.main()
