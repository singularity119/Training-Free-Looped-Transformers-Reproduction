"""Torch-free Gate E recipe checks; real tensors are checked in HPC preflight."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from tflt.config import LoopConfig
from tflt.strategies import run_loop
from tflt.loopscope.phase8_gate_e_calibration import verify_metadata
from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION
from tflt.loopscope.phase8_runtime import AnswerResidualCollector, Phase8Runtime, save_basis, load_basis

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gate_e_calibration_runner', ROOT / 'scripts/loopscope/run_phase8_gate_e_calibration.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GateECalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/loopscope/phase8_model_windows.json').read_text())
        model = self.config['models'][0]
        self.metadata = dict(scope='FORMAL_CALIBRATION', model=model['model'],
            revision=model['observed_revision'], window=[12, 15], k=3,
            dtype='bfloat16', cache_strategy='first', alpha=1.,
            dataset=dict(repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'),
            source_commit='producer', run_root='/run', pool='/pool.json')

    def test_two_windows_only_and_no_legacy_basis(self):
        for window in ([12, 15], [13, 16]):
            metadata = dict(self.metadata, window=window)
            self.assertEqual(verify_metadata(metadata, 'FORMAL_CALIBRATION', '/pool.json', self.config),
                             ('Qwen/Qwen3-4B-Base', tuple(window), 3))
        for field, value in [('k', 2), ('k', 4), ('model', 'Qwen/Qwen3-1.7B-Base'),
                             ('window', [6, 9]), ('dtype', 'float16'), ('cache_strategy', 'last')]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                verify_metadata(dict(self.metadata, **{field: value}), 'FORMAL_CALIBRATION', '/pool.json', self.config)

    def test_debug_and_formal_preserve_original_pool_order(self):
        rows = [dict(identity=str(i), prompt_token_lengths={'model': i}) for i in range(512)]
        pool = dict(rows=rows, calibration_identities=[r['identity'] for r in rows])
        self.assertEqual(runner.select_rows(pool, 'model', 'FORMAL_CALIBRATION'), rows)
        self.assertEqual([r['identity'] for r in runner.select_rows(pool, 'model', 'PREFLIGHT_ONLY')], ['0', '1', '510', '511'])
        pool['calibration_identities'].reverse()
        with self.assertRaises(ValueError):
            runner.select_rows(pool, 'model', 'FORMAL_CALIBRATION')

    def test_dry_run_maps_both_cells_to_k3(self):
        for index, window in enumerate(([12, 15], [13, 16])):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(runner.main(['--cell-index', str(index), '--pool', '/pool.json',
                    '--run-root', '/run', '--commit', 'producer', '--scope', 'PREFLIGHT_ONLY', '--dry-run']), 0)
            result = json.loads(stream.getvalue())
            self.assertEqual((result['window'], result['k']), (window, 3))
            self.assertEqual(result['model']['model'], 'Qwen/Qwen3-4B-Base')

    def test_three_steps_and_two_interventions_use_h_one_third(self):
        seen = []
        def transform(delta, t):
            seen.append((delta, t))
            return delta if t == 0 else .5 * delta
        config = LoopConfig('synthetic', (12, 15), k=3, alpha=1., cache_strategy='first',
                            residual_transform=transform)
        # Constant native residual 3: increments 1, 0.5, 0.5 at h=1/3.
        self.assertAlmostEqual(run_loop(lambda x: x + 3, 0., config), 2.)
        self.assertEqual(seen, [(3., 0), (3., 1), (3., 2)])
        collector = AnswerResidualCollector(['one', 'two'], 3)
        for key in collector.keys:
            for t in range(3):
                collector.rows[key, t] = object()
        self.assertEqual(collector.summary()['rows_by_t'], {'0': 2, '1': 2, '2': 2})
        self.assertEqual(collector.summary()['row_count'], 6)
        with self.assertRaises(ValueError):
            collector.collect('one', 3, None, 0)

    def test_k3_basis_roundtrip_requires_exact_metadata(self):
        basis = dict(schema='loopscope-phase8-basis-v1', k=3, metadata=self.metadata, direction=[.6, .8])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'basis.json'
            save_basis(path, basis)
            self.assertEqual(load_basis(path, self.metadata), basis)
            with self.assertRaises(ValueError):
                load_basis(path, dict(self.metadata, k=2))
            with self.assertRaises(FileExistsError):
                save_basis(path, basis)


if __name__ == '__main__':
    unittest.main()
