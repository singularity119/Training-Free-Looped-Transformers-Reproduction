import math
import unittest
import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from tflt.loopscope.phase8_accuracy_stats import (
    analyze_panel, cell_accuracy, exact_mcnemar_p, holm_adjust,
    paired_contrast, score_predictions, stratified_paired_bootstrap,
)


class Phase8AccuracyStatsTests(unittest.TestCase):
    def test_raw_scores_first_tie_and_macro_weighting(self):
        scores = [[0, 0, -1, -2], [0, 1, 0, 0], [0, 2, 1, -1], [0, 2, 1, 0]]
        result = cell_accuracy(scores, [0, 1, 1, 2], ['a', 'a', 'a', 'b'])
        self.assertEqual(result['correct'], 3)
        self.assertEqual(result['micro_accuracy'], .75)
        self.assertEqual(result['subject_macro_accuracy'], .5)
        self.assertEqual(score_predictions(scores), [0, 1, 1, 1])
        with self.assertRaises(ValueError):
            score_predictions([[0, 1, math.nan, 2]])

    def test_exact_mcnemar_known_binomial_tails(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1)
        self.assertAlmostEqual(exact_mcnemar_p(6, 0), 1 / 32)
        self.assertAlmostEqual(exact_mcnemar_p(5, 1), 14 / 64)
        self.assertAlmostEqual(exact_mcnemar_p(1, 5), 14 / 64)
        self.assertEqual(exact_mcnemar_p(7000, 7000), 1)
        self.assertTrue(0 < exact_mcnemar_p(6600, 7000) < 1)

    def test_holm_original_order_and_monotonicity(self):
        self.assertEqual(holm_adjust([.04, .01, .03, .2]), [.09, .04, .09, .2])
        self.assertEqual(holm_adjust([.6, .7]), [1, 1])

    def test_paired_counts_and_subject_fixed_resampling(self):
        # Within each subject every difference is constant. Stratified
        # resampling must preserve the point estimate exactly (pooled won't).
        reference = [0, 0, 0, 1]
        treatment = [1, 1, 1, 0]
        result = paired_contrast(reference, treatment, ['a', 'a', 'a', 'b'], 50)
        self.assertEqual((result['wrong_to_right'], result['right_to_wrong']), (3, 1))
        self.assertEqual(result['delta_pp'], 50)
        self.assertEqual(result['subject_macro_delta_pp'], 0)
        self.assertEqual(result['bootstrap']['ci_low_pp'], 50)
        self.assertEqual(result['bootstrap']['ci_high_pp'], 50)

    def test_bootstrap_distribution_and_reproducibility(self):
        # n=2 within one subject, one gain and one loss: exact statistic
        # distribution has masses .25/.5/.25 at -100/0/100 pp.
        one = stratified_paired_bootstrap([-1, 1], ['a', 'a'])
        two = stratified_paired_bootstrap([-1, 1], ['a', 'a'])
        self.assertEqual(one, two)
        self.assertEqual((one['ci_low_pp'], one['ci_high_pp']), (-100, 100))

    def test_cli_rejects_incomplete_closure_before_gold_loader(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/loopscope/analyze_phase8_accuracy.py'
        spec = importlib.util.spec_from_file_location('phase8_analysis_cli', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closure = root / 'closure.json'
            closure.write_text(json.dumps({'status': 'PARTIAL', 'target_gold_loaded': False}))
            with patch.object(module, 'load_frozen_gold') as loader, patch('sys.argv', [
                    str(path), '--closure', str(closure), '--output-dir', str(root / 'output'),
                    '--cache-dir', str(root / 'cache')]):
                with self.assertRaisesRegex(ValueError, 'saved full pre-outcome'):
                    module.main()
                loader.assert_not_called()
                self.assertFalse((root / 'output').exists())

    def test_all_panel_cells_families_and_exploratory_context(self):
        cells = []
        for model in ('q4', 'q17'):
            cells.append(dict(cell_id=model + '-native', model=model, window=None,
                              k=None, arm='Native', scores=[[1, 0, 0, 0]]))
            for window in ([12, 15], [6, 9]):
                for k in (2, 4):
                    for arm in ('Loop', 'Spectral'):
                        cells.append(dict(cell_id=f'{model}-{window[0]}-{k}-{arm}',
                                          model=model, window=window, k=k, arm=arm,
                                          scores=[[1, 0, 0, 0]]))
        result = analyze_panel(cells, [0], ['a'])
        self.assertEqual(len(result['cells']), 18)
        self.assertEqual(len(result['contrasts']), 24)
        for family in ('K2_PRIMARY', 'K4_SECONDARY'):
            rows = [r for r in result['contrasts'] if r['family'] == family]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r['holm_adjusted_p'] == 1 for r in rows))
        rows = [r for r in result['contrasts'] if r['family'] == 'EXPLORATORY_NATIVE_CONTEXT']
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(r['holm_adjusted_p'] is None for r in rows))


if __name__ == '__main__':
    unittest.main()
