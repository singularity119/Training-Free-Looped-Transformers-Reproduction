"""Gate G new-window source admission, cross-K mapping and pool selection."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/loopscope'))
from run_phase8_gate_g_calibration import select_rows, main
from tflt.loopscope.phase8_gate_g_calibration import (MODEL, REVISION, SCHEMA,
    validate_source_metadata, validate_source_rows, load_t0_basis)
from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION


def metadata():
    return dict(scope='FORMAL_CALIBRATION', model=MODEL, revision=REVISION,
                window=[15,18], k=2, dtype='bfloat16', cache_strategy='first', alpha=1.,
                dataset=dict(repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'),
                source_commit='committed-G-source', run_root='new-G-root/k2', pool='C-fixed-pool')


class CalibrationTests(unittest.TestCase):
    def test_new_window_only(self):
        validate_source_metadata(metadata())
        for change in (dict(window=[12,15]), dict(k=3), dict(alpha=.5)):
            with self.assertRaises(ValueError):
                validate_source_metadata(dict(metadata(), **change))

    def test_fixed_pool_long_tail(self):
        ids = [f'validation:{i}' for i in range(512)]
        pool = dict(calibration_identities=ids,
                    rows=[dict(identity=x, prompt_token_lengths={MODEL:i}) for i,x in enumerate(ids)])
        self.assertEqual([r['identity'] for r in select_rows(pool, MODEL, 'PREFLIGHT_ONLY')],
                         [ids[0],ids[1],ids[510],ids[511]])
        self.assertEqual(select_rows(pool, MODEL, 'FORMAL_CALIBRATION'), pool['rows'])
        pool['rows'][2]['identity'] = ids[0]
        with self.assertRaises(ValueError):
            select_rows(pool, MODEL, 'FORMAL_CALIBRATION')

    def test_shared_t0_metadata_loads_identically(self):
        ids = [f'validation:{i}' for i in range(512)]
        basis = dict(schema=SCHEMA, fit_t=0, source_k=2, applies_to_k=[2,3,4], rank=1,
                     **{'lambda':.5}, algorithm='cpu_float64_reduced_svd_uncentered_unnormalized',
                     metadata=metadata(), source_identities=ids, row_count=512,
                     source_positions=[10]*512, direction=[1.,0.])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'basis.json'
            path.write_text(json.dumps(basis))
            loaded = [load_t0_basis(path, dict(metadata(),k=k)) for k in (2,3,4)]
            self.assertEqual(loaded[0],loaded[2])
            with self.assertRaises(ValueError):
                load_t0_basis(path, dict(metadata(),window=[12,15]))
        saved = dict(metadata=metadata(), identities=ids, matrices={'0':None,'1':None},
                     positions={'0':[10]*512,'1':[10]*512})
        validate_source_rows(saved,metadata(),ids)
        saved['positions']['1'] = [11]*512
        with self.assertRaises(ValueError):
            validate_source_rows(saved,metadata(),ids)

    def test_dry_run_freezes_new_recipe_without_torch(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['--pool','pool','--run-root','new-root','--commit','commit',
                                   '--scope','PREFLIGHT_ONLY','--dry-run']),0)
        value = json.loads(output.getvalue())
        self.assertEqual(value['window'],[15,18])
        self.assertEqual(value['model']['model'],MODEL)


if __name__ == '__main__':
    unittest.main()
