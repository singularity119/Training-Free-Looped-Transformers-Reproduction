"""CPU-only contract checks; tensor execution is covered by the HPC preflight."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from tflt.loopscope.phase8_calibration_verify import verify_metadata
from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION


class CalibrationVerifierContractTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] /
                                 'configs/loopscope/phase8_model_windows.json').read_text())
        model = self.config['models'][0]
        self.metadata = dict(scope='FORMAL_CALIBRATION', model=model['model'],
                             revision=model['observed_revision'], window=[12, 15], k=2,
                             dtype=model['historical_runtime_dtype'], cache_strategy='first',
                             alpha=1.0, dataset=dict(repo=DATASET_REPO, revision=DATASET_REVISION,
                                                     split='validation'),
                             source_commit='producer-revision', run_root='/run/cell', pool='/pool.json')

    def check_metadata(self, metadata):
        return verify_metadata(metadata, 'FORMAL_CALIBRATION', Path('/pool.json'), self.config)

    def test_canonical_recipe(self):
        self.assertEqual(self.check_metadata(self.metadata), ('Qwen/Qwen3-4B-Base', (12, 15), 2))

    def test_smoke_and_wrong_cell_rejected(self):
        for field, value in [('scope', 'SMOKE_ONLY'), ('dtype', 'float16'), ('window', [6, 9]),
                             ('k', 3), ('revision', 'other'), ('pool', '/other.json')]:
            with self.subTest(field=field):
                changed = dict(self.metadata, **{field: value})
                with self.assertRaises(ValueError):
                    self.check_metadata(changed)

    def test_test_split_rejected(self):
        changed = copy.deepcopy(self.metadata)
        changed['dataset']['split'] = 'test'
        with self.assertRaises(ValueError):
            self.check_metadata(changed)

    def test_import_and_help_without_torch(self):
        subprocess.run([sys.executable, '-c',
                        "import sys; sys.modules['torch'] = None; "
                        "import tflt.loopscope.phase8_calibration_verify"], check=True)
        script = Path(__file__).resolve().parents[1] / 'scripts/loopscope/verify_phase8_calibration.py'
        result = subprocess.run([sys.executable, str(script), '--help'], check=True,
                                capture_output=True, text=True)
        self.assertIn('--require-all', result.stdout)


if __name__ == '__main__':
    unittest.main()
