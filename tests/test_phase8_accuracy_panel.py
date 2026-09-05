import json
from pathlib import Path
import unittest
from tflt.loopscope.phase8_accuracy import panel,select_rows
from tflt.loopscope.phase8_pool import DATASET_REPO,DATASET_REVISION

class PanelTests(unittest.TestCase):
    def test_frozen_membership_basis_pairs(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/loopscope/phase8_model_windows.json').read_text())
        cells=panel(config)
        self.assertEqual(len(cells),18)
        self.assertEqual(len({c['cell_id'] for c in cells}),18)
        self.assertEqual(sum(c['arm']=='Native' for c in cells),2)
        spectral=[c for c in cells if c['arm']=='Spectral']
        self.assertEqual(len({c['basis_path'] for c in spectral}),8)
        for c in spectral:
            counterpart=next(x for x in cells if x['cell_id']==c['cell_id'].replace('spectral','loop'))
            self.assertIsNone(counterpart['basis_path'])
            self.assertEqual((c['window'],c['k']),(counterpart['window'],counterpart['k']))
    def test_canonical_shards_and_upper_tail(self):
        rows=[{'identity':str(i),'prompt_token_lengths':{'model':i+1}} for i in range(14042)]
        pool={'dataset':dict(repo=DATASET_REPO,revision=DATASET_REVISION,split='test'),'rows':rows}
        cell={'model':'model'}
        self.assertEqual(select_rows(pool,cell,'FORMAL_TEST',512,1024),rows[512:1024])
        self.assertEqual([r['identity'] for r in select_rows(pool,cell,'PREFLIGHT_ONLY',0,14042)],['0','1','14038','14039','14040','14041'])
        with self.assertRaises(ValueError): select_rows(pool,cell,'FORMAL_TEST',512,512)

if __name__=='__main__': unittest.main()
