#!/usr/bin/env python3
"""Freeze the 18 Gate D cells and check their eight admitted C basis mappings."""
import argparse
import json
from pathlib import Path
from tflt.loopscope.phase8_accuracy import panel,checked_basis,write_json
from tflt.loopscope.phase8_test_pool import validate_test_pool

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[2]
    config=json.loads((repo/'configs/loopscope/phase8_model_windows.json').read_text())
    pool=json.loads(a.pool.read_text())
    validate_test_pool(pool)
    cells=panel(config)
    for cell in cells:
        if cell['arm']=='Spectral': checked_basis(cell)
    write_json(a.output,dict(schema='loopscope.phase8.accuracy_manifest.v1',dataset=pool['dataset'],pool=str(a.pool),cells=cells))
if __name__=='__main__': main()
