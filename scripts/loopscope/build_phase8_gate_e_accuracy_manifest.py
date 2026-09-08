#!/usr/bin/env python3
"""Freeze four Gate E K3 cells with two scope-matched calibration bases."""
import argparse
import json
from pathlib import Path
from tflt.loopscope.phase8_accuracy import write_json
from tflt.loopscope.phase8_gate_e_accuracy import panel, checked_basis, SCHEMA
from tflt.loopscope.phase8_test_pool import validate_test_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--basis-root',type=Path,required=True)
    parser.add_argument('--scope',choices=('PREFLIGHT_ONLY','FORMAL_TEST'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    pool = json.loads(args.pool.read_text())
    validate_test_pool(pool)
    cells = panel(args.basis_root)
    for cell in cells:
        if cell['arm'] == 'Spectral':
            checked_basis(cell,args.scope)
    write_json(args.output,dict(schema=SCHEMA,scope=args.scope,dataset=pool['dataset'],
                               pool=str(args.pool),basis_root=str(args.basis_root),cells=cells))


if __name__ == '__main__':
    main()
