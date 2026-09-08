"""Torch-free identity, schema and cross-K application checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tflt.loopscope.phase8_gate_f_basis import (
    ALGORITHM, SCHEMA, load_t0_basis, validate_source_metadata, validate_source_rows)
from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION
from tflt.loopscope.phase8_runtime import save_basis


class GateFBasisTests(unittest.TestCase):
    def setUp(self):
        self.identities = ['sample-%d' % i for i in range(512)]
        self.metadata = dict(scope='FORMAL_CALIBRATION', model='Qwen/Qwen3-4B-Base',
            revision='906bfd4b4dc7f14ee4320094d8b41684abff8539', window=[12, 15], k=2,
            dtype='bfloat16', cache_strategy='first', alpha=1.,
            dataset=dict(repo=DATASET_REPO, revision=DATASET_REVISION, split='validation'),
            source_commit='original-producer', run_root='/original/cell', pool='/pool.json')
        self.saved = dict(metadata=self.metadata, identities=self.identities,
            positions={'0': [20]*512, '1': [20]*512}, matrices={'0': None, '1': None})
        self.basis = dict(schema=SCHEMA, fit_t=0, source_k=2, applies_to_k=[2, 3, 4],
            metadata=self.metadata, rank=1, **{'lambda': .5}, algorithm=ALGORITHM,
            source_identities=self.identities, source_positions=[20]*512, row_count=512,
            direction=[.6, .8])

    def load(self, basis, cell=None, scope='FORMAL_CALIBRATION'):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'basis.json'
            save_basis(path, basis)
            return load_t0_basis(path, cell or self.metadata, scope)

    def test_one_basis_loads_unchanged_across_three_k(self):
        for k in (2, 3, 4):
            loaded = self.load(self.basis, dict(self.metadata, k=k, fit_t=0))
            self.assertEqual(loaded, self.basis)
            self.assertEqual(loaded['metadata']['k'], 2)
            self.assertNotIn('k', loaded)

    def test_formal_source_identity_order_and_positions(self):
        self.assertEqual(validate_source_rows(self.saved, self.metadata, self.identities), self.identities)
        for change in ('order', 'position', 'missing'):
            saved = copy.deepcopy(self.saved)
            if change == 'order': saved['identities'].reverse()
            if change == 'position': saved['positions']['1'][0] = 21
            if change == 'missing': saved['matrices'].pop('1')
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_source_rows(saved, self.metadata, self.identities)

    def test_source_recipe_rejects_non_k2_and_wrong_dtype(self):
        for field, value in [('k', 3), ('dtype', 'float16'), ('window', [6, 9])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_source_metadata(dict(self.metadata, **{field: value}))

    def test_application_mismatch_and_legacy_schema_rejected(self):
        for field, value in [('window', [13, 16]), ('k', 1), ('fit_t', 1), ('dtype', 'float16')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(self.basis, dict(self.metadata, **{field: value}))
        for field, value in [('schema', 'loopscope-phase8-basis-v1'), ('source_k', 4),
                             ('applies_to_k', [2]), ('direction', [1., 1.]), ('k', 2)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(dict(self.basis, **{field: value}))

    def test_preflight_cannot_load_as_formal(self):
        basis = dict(self.basis, metadata=dict(self.metadata, scope='PREFLIGHT_ONLY'),
                     source_identities=['sample-0'], source_positions=[20], row_count=1)
        self.assertEqual(self.load(basis, scope='PREFLIGHT_ONLY'), basis)
        with self.assertRaises(ValueError):
            self.load(basis)


if __name__ == '__main__':
    unittest.main()
