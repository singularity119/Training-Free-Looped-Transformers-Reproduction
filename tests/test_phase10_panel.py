import copy
import json
from pathlib import Path
import tempfile
import unittest

from tflt.loopscope.phase10_panel import (
    STRENGTHS, build_panel, load_config, new_cells, score_manifest,
    validate_config, validate_panel, validate_score_manifest, write_manifest,
)
from tflt.loopscope.phase9_accuracy import logical_panel
from tflt.loopscope.phase9_gate_e_accuracy import new_cells as legacy_lag1_cells


class Phase10PanelTests(unittest.TestCase):
    def test_exact_panel_membership_aliases_and_new_counts(self):
        for dataset, new_count, count in [('mmlu', 59, 14042), ('arc_challenge', 68, 1172)]:
            panel = build_panel(dataset)
            self.assertEqual(validate_panel(panel)['cell_count'], 68)
            self.assertEqual(len(panel['cells']), 68)
            self.assertEqual(len(panel['display_rows']), 86)
            self.assertEqual(len(new_cells(dataset)), new_count)
            self.assertEqual(panel['new_configuration_sample_records'], new_count * count)
            self.assertEqual(panel['validated_reuse_cell_count'], 0)
            self.assertFalse(panel['target_gold_loaded'])
            by_id = {row['cell_id']: row for row in panel['cells']}
            aliases = [row for row in panel['display_rows'] if row['alias_of']]
            self.assertEqual(len(aliases), 18)
            for row in aliases:
                original = by_id[row['alias_of']]
                self.assertEqual(row['k'], 2)
                self.assertEqual(row['direction_policy'], 'lag1')
                self.assertEqual(original['direction_policy'], 'fixed_t0')
                self.assertEqual((row['window'], row['model'], row['strength']),
                                 (original['window'], original['model'], original['strength']))
            online = [row for row in panel['cells'] if row['arm'] == 'Online']
            self.assertEqual(len(online), 63)
            current = [row for row in online if row['direction_policy'] == 'current_t']
            self.assertEqual(len(current), 27)
            self.assertTrue(all(row['new_in_phase10'] for row in current))
            self.assertFalse(any(row['alias_of'] for row in panel['display_rows'] if row['direction_policy'] == 'current_t'))
            self.assertEqual({row['strength'] for row in online}, set(STRENGTHS))
            self.assertEqual({(row['model_index'], tuple(row['window']), row['k']) for row in online},
                             {(0, (15, 18), 2), (0, (15, 18), 3), (1, (12, 15), 2)})
            self.assertEqual({row['arm'] for row in panel['cells']}, {'Native', 'Loop', 'Online'})

    def test_nine_expected_reuse_mappings_match_legacy_cell_recipes(self):
        legacy = {row['cell_id']: row for row in logical_panel() + legacy_lag1_cells()}
        panel = build_panel('mmlu')
        current = {row['cell_id']: row for row in panel['cells']}
        self.assertEqual(len(panel['reuse_mapping']), 9)
        for mapping in panel['reuse_mapping']:
            old, new = legacy[mapping['source_cell_id']], current[mapping['cell_id']]
            for name in ('model', 'revision', 'dtype', 'cache_strategy', 'window', 'boundaries', 'k'):
                self.assertEqual(new[name], old[name])
            self.assertTrue(mapping['roots'])
            self.assertTrue(all(path.startswith('/hpc2hdd/') for path in mapping['roots']))
            self.assertTrue(mapping['source_manifest_schema'].startswith('loopscope.phase9.'))
            self.assertEqual(mapping['source_score_file'], 'scores.jsonl')
            self.assertIn('NOT_RAW_SCORE_VERIFIED', mapping['verification_status'])
            if new['arm'] == 'Online':
                self.assertEqual(new['strength'], 0.5)
                self.assertEqual(new['direction_policy'], 'lag1' if old['arm'] == 'Lag1-Online' else 'fixed_t0')
            else:
                self.assertEqual(new['arm'], old['arm'])
        arc = build_panel('arc_challenge')
        self.assertEqual(arc['reuse_mapping'], [])
        self.assertEqual(arc['dataset_recipe']['revision'], '210d026faf9955653af8916fad021475a3f00453')
        self.assertEqual(arc['dataset_recipe']['max_length'], 32768)

    def test_scoring_manifest_contains_all_runtime_cells_and_only_new_score_cells(self):
        for dataset, score_count in [('mmlu', 59), ('arc_challenge', 68)]:
            value = score_manifest(dataset, Path('/frozen/pool'), 'PREFLIGHT_ONLY')
            self.assertEqual(len(validate_score_manifest(value, pool_path=Path('/frozen/pool'), scope='PREFLIGHT_ONLY')), 68)
            self.assertEqual(value['score_cell_count'], score_count)
            self.assertEqual(len(value['score_cells']), score_count)
            self.assertTrue(all(cell['new_in_phase10'] for cell in value['score_cells']))
            wrong = copy.deepcopy(value)
            wrong['cells'][0]['revision'] = 'different'
            with self.assertRaises(ValueError):
                validate_score_manifest(wrong)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            write_manifest(path, build_panel('mmlu'))
            self.assertEqual(json.loads(path.read_text())['display_row_count'], 86)
            with self.assertRaises(FileExistsError):
                write_manifest(path, {})

    def test_frozen_science_rejects_changed_grid_or_added_k(self):
        for field, replacement in [('strengths', [0.1, 1.0]), ('direction_policies', ['fixed_t0']), ('alpha', 0.5)]:
            config = load_config()
            config[field] = replacement
            with self.assertRaises(ValueError):
                validate_config(config)
        config = load_config()
        config['cells'][1]['k'] = 4
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == '__main__':
    unittest.main()
