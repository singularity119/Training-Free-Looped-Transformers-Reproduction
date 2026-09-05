"""Decision-relevant verifier failures without model/data dependencies."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/loopscope/verify_phase8_debug.py'
spec = importlib.util.spec_from_file_location('phase8_debug_verifier', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DebugVerifierTests(unittest.TestCase):
    def fixture(self, root):
        def put(path, data):
            path.write_text(json.dumps(data))
        name='w12-15-k2'
        (root/name).mkdir()
        recipe=dict(source_commit='test-revision',model={'model':'synthetic'},cells=[[[12,15],2]])
        put(root/'summary.json',dict(status='DEBUG_COMPLETE',scope='SMOKE_ONLY',recipe=recipe,
            oom_count=0,elapsed_seconds=60,fit_identity_count=4,verified_identity_count=4,
            semantic_checks={'t0_unchanged':True},cell_names=[name]))
        put(root/'env.json',dict(partition='debug',job_id='test-job',source_commit='test-revision'))
        native=[dict(identity='verify%d'%i,scores=[1.,2.,3.,4.],adapter_scores=[1.,2.,3.,4.]) for i in range(4)]
        put(root/'native.json',native)
        metadata=dict(scope='SMOKE_ONLY',model='synthetic',window=[12,15])
        put(root/name/'basis.json',dict(schema='loopscope-phase8-basis-v1',k=2,fit_t=1,row_count=4,
            direction=[1.,0.],metadata=metadata,source_identities=['fit%d'%i for i in range(4)]))
        (root/name/'residual_t1.pt').touch()
        results=[]
        for row in native:
            results.append(dict(identity=row['identity'],original=row['scores'],none=row['scores'],
                zero=row['scores'],spectral=[1.,2.,3.,3.5],
                calls=[dict(identity=row['identity'],t=t,applied=t>=1) for t in range(2)],
                positions=[dict(position=9,input_length=10,continuation_length=1)]*4))
        put(root/name/'debug.json',dict(metadata=metadata,collector={'rows_by_t':{'0':4,'1':4},
            'row_count':8},basis_roundtrip=True,results=results))
        return root/name/'debug.json'

    def test_complete_debug_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fixture(root)
            self.assertEqual(module.verify(root)['status'],'VERIFIED_DEBUG')

    def test_missing_timestep_and_changed_zero_score_rejected(self):
        for change in ('count','score'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                path=self.fixture(root)
                data=json.loads(path.read_text())
                if change=='count':
                    data['collector']['rows_by_t']['1']=3
                else:
                    data['results'][0]['zero'][0]=0
                path.write_text(json.dumps(data))
                with self.assertRaises(AssertionError):
                    module.verify(root)


if __name__=='__main__':
    unittest.main()
