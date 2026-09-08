import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tflt.loopscope.phase8_gate_e_accuracy import panel, validate_manifest, analyze_panel, SCHEMA
from tflt.loopscope.phase8_accuracy_verify import verify


class GateEAccuracyTests(unittest.TestCase):
    def test_panel_and_scope(self):
        cells = panel('/calibration')
        self.assertEqual(len(cells),4)
        self.assertEqual([c['basis_path'] for c in cells if c['arm']=='Spectral'],
                         ['/calibration/cell-0/basis.json','/calibration/cell-1/basis.json'])
        self.assertEqual({c['k'] for c in cells},{3})
        package=dict(schema=SCHEMA,scope='FORMAL_TEST',dataset={},pool='/pool',
                     basis_root='/calibration',cells=cells)
        self.assertEqual(len(validate_manifest(package,{'dataset':{}},'/pool')),4)
        cells[0]['k']=4
        with self.assertRaisesRegex(ValueError,'four-cell'):
            validate_manifest(package,{'dataset':{}},'/pool')

    def test_preflight_basis_never_admitted_formally(self):
        from tflt.loopscope.phase8_gate_e_accuracy import checked_basis, MODEL, REVISION
        from tflt.loopscope.phase8_pool import DATASET_REPO, DATASET_REVISION
        cell=panel('/calibration')[1]
        basis=dict(metadata=dict(scope='PREFLIGHT_ONLY',model=MODEL,revision=REVISION,
                    window=[12,15],k=3,dtype='bfloat16',cache_strategy='first',alpha=1.,
                    dataset=dict(repo=DATASET_REPO,revision=DATASET_REVISION,split='validation')),
                   rank=1,**{'lambda':.5},row_count=6)
        with patch('tflt.loopscope.phase8_runtime.load_basis',return_value=basis):
            self.assertIs(checked_basis(cell,'PREFLIGHT_ONLY'),basis)
            with self.assertRaisesRegex(ValueError,'scope recipe'):
                checked_basis(cell,'FORMAL_TEST')
            basis['metadata']['scope']='FORMAL_CALIBRATION'
            basis['row_count']=512
            self.assertIs(checked_basis(cell,'FORMAL_TEST'),basis)
            basis['metadata']['k']=4
            with self.assertRaisesRegex(ValueError,'scope recipe'):
                checked_basis(cell,'FORMAL_TEST')

    def test_two_contrasts_and_count_identities(self):
        cells = panel('/calibration')
        for c in cells:
            c['scores'] = [[2,1,0,-1]]*8 if c['arm']=='Loop' else [[1,2,0,-1]]*8
        result=analyze_panel(cells,[0,0,1,1,1,1,1,1],['s']*4+['t']*4,100)
        self.assertEqual(len(result['contrasts']),2)
        for row in result['contrasts']:
            self.assertEqual(row['family'],'K3_EXPLORATORY')
            self.assertEqual(row['holm_family_size'],2)
            self.assertEqual((row['wrong_to_right'],row['right_to_wrong']),(6,2))
            self.assertEqual(row['delta_pp'],50)
            self.assertEqual(row['treatment_correct']-row['reference_correct'],4)
        self.assertTrue(result['interpretation']['prior_K2_K4_outcomes_known'])
        cells[0]['k']=4
        with self.assertRaisesRegex(ValueError,'four Gate E'):
            analyze_panel(cells,[0]*8,['s']*8,100)

    def test_full_four_cell_closure_and_missing_cell(self):
        with tempfile.TemporaryDirectory() as tmp, patch('tflt.loopscope.phase8_accuracy_verify.validate_test_pool'):
            root=Path(tmp)
            rows=[dict(identity=f's:{i}',subject=f's{i%57}',doc_index=i) for i in range(14042)]
            pool=root/'pool.json'; manifest=root/'manifest.json'
            pool.write_text(json.dumps(dict(dataset={},rows=rows)))
            cells=panel('/calibration')
            manifest.write_text(json.dumps(dict(schema=SCHEMA,scope='FORMAL_TEST',dataset={},pool=str(pool),basis_root='/calibration',cells=cells)))
            roots=[]
            for i,cell in enumerate(cells):
                d=root/str(i); d.mkdir(); roots.append(d)
                metadata=dict(cell=cell,scope='FORMAL_TEST',start=0,end=14042,source_commit='test',pool=str(pool),manifest=str(manifest))
                (d/'command_args.json').write_text(json.dumps(metadata))
                (d/'summary.json').write_text(json.dumps(dict(status='SCORES_COMPLETE',metadata=metadata,target_gold_loaded=False,count=14042)))
                (d/'env.json').write_text(json.dumps(dict(source_commit='test',model_revision=cell['revision'],model_dtype='torch.bfloat16')))
                (d/'scores.jsonl').write_text(''.join(json.dumps(dict(row,scores=[0.,1.,2.,3.]))+'\n' for row in rows))
            result=verify(roots,pool,manifest,True,gate_e=True)
            self.assertEqual(result['sample_count'],56168)
            self.assertEqual(result['cell_count'],4)
            self.assertFalse(result['target_gold_loaded'])
            with self.assertRaisesRegex(ValueError,'all 4 cells'):
                verify(roots[:3],pool,manifest,True,gate_e=True)
            with self.assertRaisesRegex(ValueError,'overlapping'):
                verify(roots+[roots[0]],pool,manifest,True,gate_e=True)
