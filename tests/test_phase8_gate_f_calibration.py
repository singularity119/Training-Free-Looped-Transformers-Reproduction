"""Gate F runner checks: fixed pool selection and saved-source end-to-end closure."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/loopscope'))
spec = importlib.util.spec_from_file_location('gate_f_runner', ROOT / 'scripts/loopscope/run_phase8_gate_f_calibration.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CalibrationRunnerTests(unittest.TestCase):
    def pool(self):
        from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION
        model = 'Qwen/Qwen3-4B-Base'
        identities = ['validation:%d' % i for i in range(512)]
        return dict(dataset=dict(repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'),
                    calibration_identities=identities,
                    rows=[dict(identity=i, prompt='outcome-free prompt', prompt_token_lengths={model: 512-n})
                          for n, i in enumerate(identities)])

    def test_preflight_is_four_even_when_longest_overlap_initial(self):
        pool = self.pool()
        selected = runner.select_rows(pool, 'Qwen/Qwen3-4B-Base', 'PREFLIGHT_ONLY')
        self.assertEqual([r['identity'] for r in selected], pool['calibration_identities'][:4])
        self.assertEqual(runner.select_rows(pool, 'unused', 'FORMAL_CALIBRATION'), pool['rows'])

    def test_saved_t0_fit_save_cross_k_load(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is required for saved-residual tensor integration')
        from tflt.loopscope.phase8_gate_f_basis import load_t0_basis
        config = json.loads((ROOT / 'configs/loopscope/phase8_model_windows.json').read_text())['models'][0]
        pool = self.pool()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pool_path, saved_path, output = base / 'pool.json', base / 'residuals.pt', base / 'f'
            pool_path.write_text(json.dumps(pool))
            metadata = dict(scope='FORMAL_CALIBRATION', model=config['model'], revision=config['observed_revision'],
                            window=[12, 15], k=2, dtype='bfloat16', cache_strategy='first', alpha=1.,
                            dataset=pool['dataset'], source_commit='historical-C-producer',
                            run_root='historical-C-cell-0', pool=str(pool_path))
            matrix = torch.tensor([[float(i+1), 2., 1.] for i in range(512)])
            torch.save(dict(metadata=metadata, identities=pool['calibration_identities'],
                            positions={'0': [10]*512, '1': [10]*512},
                            matrices={'0': matrix, '1': matrix * .5}), saved_path)
            argv = ['--mode', 't0-from-saved', '--cell-index', '0', '--pool', str(pool_path),
                    '--saved-residuals', str(saved_path), '--run-root', str(output),
                    '--commit', 'new-F-producer', '--scope', 'FORMAL_CALIBRATION']
            with patch.dict(os.environ, {'SLURM_JOB_ID': 'cpu-test'}):
                self.assertEqual(runner.main(argv), 0)
            loaded = [load_t0_basis(output / 'basis.json', dict(metadata, k=k)) for k in (2,3,4)]
            self.assertEqual(loaded[0]['metadata']['source_commit'], 'historical-C-producer')
            self.assertEqual(loaded[0]['fit_provenance']['source_commit'], 'new-F-producer')
            self.assertEqual(loaded[0]['direction'], loaded[2]['direction'])
            summary = json.loads((output / 'summary.json').read_text())
            self.assertFalse(summary['target_gold_loaded'])
            self.assertEqual(summary['cross_k_load_verified'], [2,3,4])
            preflight = base / 'preflight'
            preflight_argv = list(argv)
            preflight_argv[preflight_argv.index(str(output))] = str(preflight)
            preflight_argv[preflight_argv.index('FORMAL_CALIBRATION')] = 'PREFLIGHT_ONLY'
            with patch.dict(os.environ, {'SLURM_JOB_ID': 'debug-test', 'SLURM_JOB_PARTITION': 'debug'}):
                self.assertEqual(runner.main(preflight_argv), 0)
            sampled = load_t0_basis(preflight / 'basis.json', dict(metadata, k=4), 'PREFLIGHT_ONLY')
            self.assertEqual(sampled['row_count'], 4)
            self.assertEqual(sampled['source_identities'], pool['calibration_identities'][:4])
            sampled_summary = json.loads((preflight / 'summary.json').read_text())
            self.assertEqual(sampled_summary['source_metadata']['scope'], 'FORMAL_CALIBRATION')
            with patch.dict(os.environ, {'SLURM_JOB_ID': 'cpu-test'}):
                with self.assertRaises(FileExistsError):
                    runner.main(argv)


if __name__ == '__main__':
    unittest.main()
