import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('calibration_runner',Path(__file__).resolve().parents[1]/'scripts/loopscope/run_phase8_calibration.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class SelectionTest(unittest.TestCase):
    def test_preflight_keeps_canonical_order_and_upper_tail(self):
        rows=[dict(identity=str(i),prompt_token_lengths={'model':i}) for i in range(512)]
        pool=dict(rows=rows,calibration_identities=[r['identity'] for r in rows])
        self.assertEqual([r['identity'] for r in m.select_rows(pool,'model','PREFLIGHT_ONLY')],['0','1','510','511'])
        self.assertEqual(m.select_rows(pool,'model','FORMAL_CALIBRATION'),rows)
        pool['calibration_identities'].reverse()
        with self.assertRaises(ValueError):m.select_rows(pool,'model','FORMAL_CALIBRATION')
